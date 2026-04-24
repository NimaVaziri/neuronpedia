import gc
import gzip
import os
import threading
import time
from collections import OrderedDict
from importlib.metadata import version as pkg_version
from typing import Any

import psutil
import requests
import torch
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from neuronpedia_graph.explorer import ExploreCircuitsJob, build_explore_circuits_response
from neuronpedia_graph.grouping import auto_group_nodes, format_grouping_response
from neuronpedia_graph.scorer import BatchGraphScorer
from pydantic import BaseModel, ValidationError
from starlette.concurrency import run_in_threadpool
from transformers import AutoTokenizer

load_dotenv()

BACKEND = os.getenv("BACKEND", "circuit-tracer")

if BACKEND == "circuit-tracer":
    from circuit_tracer import attribute
    from circuit_tracer.graph import prune_graph
    from circuit_tracer.replacement_model import ReplacementModel
    from circuit_tracer.utils.create_graph_files import (
        build_model,
        create_nodes,
        create_used_nodes_and_edges,
    )
    from circuit_tracer.utils.salient_logits import compute_salient_logits
elif BACKEND == "lm-saes-crm":
    from neuronpedia_graph.crm_backend import (
        forward_pass_crm,
        generate_graph_crm,
        load_crm_model,
    )


LIMIT_TOKENS = int(os.getenv("TOKEN_LIMIT", 64))
DEFAULT_MAX_FEATURE_NODES = int(os.getenv("MAX_FEATURE_NODES", 10000))
MAX_SCORER_CACHE_ITEMS = int(os.getenv("MAX_SCORER_CACHE_ITEMS", 8))
MAX_GRAPH_CACHE_ITEMS = int(os.getenv("MAX_GRAPH_CACHE_ITEMS", 8))
PREBUILD_SCORER_ON_GENERATE = os.getenv("PREBUILD_SCORER_ON_GENERATE", "").lower() == "true"
OFFLOAD = None
UPDATE_INTERVAL = int(os.getenv("UPDATE_INTERVAL", 1000))

SECRET_KEY = os.getenv("SECRET")
if not SECRET_KEY:
    raise ValueError(
        "SECRET environment variable not set. Please create a .env file with SECRET=<your_secret_key>"
    )

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN environment variable not set. Please create a .env file with HF_TOKEN=<your_huggingface_token>"
    )


def get_device() -> torch.device:
    """Determine the appropriate device for model loading."""
    device_env = os.environ.get("DEVICE")
    if device_env:
        return torch.device(device_env)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_model_dtype() -> torch.dtype | None:
    """
    Parse MODEL_DTYPE environment variable into torch dtype.
    Default is float32.
    """
    model_dtype_env = os.environ.get("MODEL_DTYPE", "bfloat16")

    dtype_mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    return dtype_mapping.get(model_dtype_env)


app = FastAPI()
from starlette.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
# GZipMiddleware disabled — it buffers StreamingResponse bodies, breaking SSE
# app.add_middleware(GZipMiddleware, minimum_size=1000)

transcoders: Any = None
model: Any = None
request_lock = threading.Lock()
_tokenizer: Any = None
_tokenizer_lock = threading.Lock()

TRANSCODER_SET_TO_SOURCE_URL_ARRAYS = {
    "gemma": [
        "https://neuronpedia.org/gemma-2-2b/gemmascope-transcoder-16k",
        "https://huggingface.co/google/gemma-scope-2b-pt-transcoders",
    ],
    "mwhanna/qwen3-4b-transcoders": [
        "https://neuronpedia.org/qwen3-4b/transcoder-hp",
        "https://huggingface.co/mwhanna/qwen3-4b-transcoders",
    ],
    "mntss/clt-gemma-2-2b-2.5M": [
        "https://neuronpedia.org/gemma-2-2b/clt-hp",
        "https://huggingface.co/mntss/clt-gemma-2-2b-2.5M",
    ],
    "mwhanna/gemma-scope-2-4b-it/transcoder_all/width_262k_l0_small_affine": [
        "https://neuronpedia.org/gemma-3-4b-it/gemmascope-transcoder-262k",
        "https://huggingface.co/mwhanna/gemma-scope-2-4b-it/transcoder_all/width_262k_l0_small_affine",
        "https://huggingface.co/google/gemma-scope-2-4b-it/transcoder_all",
    ],
}

TLENS_MODEL_ID_TO_NP_MODEL_ID = {
    "google/gemma-2-2b": "gemma-2-2b",
    "google/gemma-3-4b-it": "gemma-3-4b-it",
    "meta-llama/Llama-3.2-1B": "llama3.1-8b",
    "Qwen/Qwen3-4B": "qwen3-4b",
}

GENERATOR_INFO = {
    "name": "circuit-tracer by Hanna & Piotrowski",
    "version": pkg_version("circuit-tracer"),
    "url": "https://github.com/safety-research/circuit-tracer",
}

loaded_model_arg = os.getenv("MODEL_ID")
print(f"Model: {loaded_model_arg}")
if not loaded_model_arg:
    raise ValueError(
        "TransformerLens model name is required. Please specify a model as a command line argument. Valid models: "
        + ", ".join(TLENS_MODEL_ID_TO_NP_MODEL_ID.keys())
    )

device = get_device()
model_dtype = get_model_dtype()

# CRM backend: lm-saes with Lorsa + Transcoders
crm_model: Any = None
crm_replacement_modules: Any = None
crm_sae_metadata: Any = None

if BACKEND == "lm-saes-crm":
    print(f"[CRM] Loading CRM backend for model: {loaded_model_arg}")
    crm_model, crm_replacement_modules, crm_sae_metadata = load_crm_model()
    model = crm_model
else:
    # Circuit-tracer backend (default)
    transcoder_set = os.getenv("TRANSCODER_SET")
    print(f"Transcoder set: {transcoder_set}")
    if not transcoder_set:
        raise ValueError("Transcoder set is required. Please specify a transcoders set.")

    def check_is_nnsight_model(model_id: str) -> bool:
        return model_id.startswith("google/gemma-3-")

    is_nnsight_model = check_is_nnsight_model(loaded_model_arg)
    lazy_encoder = is_nnsight_model or os.getenv("LAZY_ENCODER", "").lower() == "true"

    model = ReplacementModel.from_pretrained(
        loaded_model_arg,
        transcoder_set,
        device=device,
        dtype=model_dtype,
        lazy_encoder=lazy_encoder,
        lazy_decoder=True,
        backend="nnsight" if is_nnsight_model else "transformerlens",
    )


def get_shared_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        with _tokenizer_lock:
            if _tokenizer is None:
                _tokenizer = AutoTokenizer.from_pretrained(model.cfg.tokenizer_name)
    return _tokenizer


def _prune_lru_cache(cache: "OrderedDict[str, Any]", max_items: int, label: str):
    while max_items >= 0 and len(cache) > max_items:
        evicted_slug, evicted_value = cache.popitem(last=False)
        del evicted_value
        print(f"Evicted {label} cache entry '{evicted_slug}'")


def _cache_graph_artifacts(slug: str, scorer: Any, graph_data: dict[str, Any]):
    if MAX_SCORER_CACHE_ITEMS > 0:
        _scorer_cache.pop(slug, None)
        _scorer_cache[slug] = scorer
        _prune_lru_cache(_scorer_cache, MAX_SCORER_CACHE_ITEMS, "scorer")

    if MAX_GRAPH_CACHE_ITEMS > 0:
        _graph_cache.pop(slug, None)
        _graph_cache[slug] = graph_data
        _prune_lru_cache(_graph_cache, MAX_GRAPH_CACHE_ITEMS, "graph")


def _get_cached_scorer(slug: str):
    scorer = _scorer_cache.get(slug)
    if scorer is not None:
        _scorer_cache.move_to_end(slug)
        if slug in _graph_cache:
            _graph_cache.move_to_end(slug)
    return scorer


def printMemory():
    if torch.cuda.is_available():
        current_memory = torch.cuda.memory_allocated() / (1024**3)
        print(f"GPU memory usage: {current_memory:.2f} GB")
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage_gb = memory_info.rss / (1024**3)
        print(f"CPU memory usage: {memory_usage_gb:.2f} GB")


async def verify_secret_key(x_secret_key: str = Header(None)):
    if not x_secret_key:
        raise HTTPException(status_code=400, detail="x-secret-key header missing")
    if x_secret_key != SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid x-secret-key")
    return x_secret_key


class GraphGenerationRequest(BaseModel):
    prompt: str
    model_id: str
    batch_size: int = 48
    max_n_logits: int = 10
    desired_logit_prob: float = 0.95
    node_threshold: float = 0.8
    edge_threshold: float = 0.98
    slug_identifier: str
    max_feature_nodes: int = DEFAULT_MAX_FEATURE_NODES
    signed_url: str | None = None
    user_id: str | None = None
    compress: bool = False
    enable_qk_tracing: bool = False
    qk_top_fraction: float = 0.6
    qk_topk: int = 10


class ForwardPassRequest(BaseModel):
    prompt: str
    max_n_logits: int = 10
    desired_logit_prob: float = 0.95


class SteerFeature(BaseModel):
    layer: int
    index: int
    token_active_position: int
    steer_position: int | None = None
    steer_generated_tokens: bool = False
    delta: float | None = None
    ablate: bool = False


class SteerRequest(BaseModel):
    model_id: str
    prompt: str
    features: list[SteerFeature]
    n_tokens: int = 10
    top_k: int = 5
    temperature: float = 0.0
    freq_penalty: float = 0
    seed: int | None = None
    freeze_attention: bool = False


class SteerBatchRequest(BaseModel):
    model_id: str
    prompt: str
    feature_sets: list[list[SteerFeature]]
    n_tokens: int = 10
    top_k: int = 5
    temperature: float = 0.0
    freq_penalty: float = 0
    seed: int | None = None
    freeze_attention: bool = False


@app.get("/check-busy")
async def check_busy():
    """Check if the server is currently busy processing a request."""
    is_busy = request_lock.locked()
    return {"busy": is_busy}


def get_topk(logits: torch.Tensor, tokenizer, k: int = 5):
    raw_logits = logits[0, -1, :]
    probs = torch.softmax(raw_logits, dim=-1)
    topk = torch.topk(probs, k)
    return [
        (tokenizer.decode([topk.indices[i]]), topk.values[i].item(), raw_logits[topk.indices[i]].item())
        for i in range(k)
    ]


def _validate_steer_features(features: list[SteerFeature], sequence_length: int):
    for feature in features:
        if feature.ablate and feature.delta is not None:
            raise HTTPException(status_code=400, detail="When ablate is True, delta must be None")
        if not feature.ablate and feature.delta is None:
            raise HTTPException(status_code=400, detail="When ablate is False, delta must be provided")
        if feature.steer_generated_tokens and feature.steer_position is not None:
            raise HTTPException(
                status_code=400,
                detail="When steer_generated_tokens is True, position must be None",
            )
        if not feature.steer_generated_tokens and feature.steer_position is None:
            raise HTTPException(
                status_code=400,
                detail="When steer_generated_tokens is False, position must be provided",
            )
        if feature.steer_position is not None and (
            feature.steer_position < 0 or feature.steer_position >= sequence_length
        ):
            raise HTTPException(status_code=400, detail="Position is out of bounds")


def _build_intervention_tuples(
    features: list[SteerFeature],
    activations,
    sequence_length: int,
):
    intervention_tuples = []
    for f in features:
        if f.steer_generated_tokens:
            intervention_tuples.append(
                (
                    f.layer,
                    slice(sequence_length, None, None),
                    f.index,
                    0 if f.ablate else activations[(f.layer, f.token_active_position, f.index)] + f.delta,
                )
            )
        else:
            intervention_tuples.append(
                (
                    f.layer,
                    f.steer_position,
                    f.index,
                    0 if f.ablate else activations[(f.layer, f.token_active_position, f.index)] + f.delta,
                )
            )
    return intervention_tuples


def _compute_default_run(req_data, sequence_length: int):
    if req_data.seed is not None:
        torch.manual_seed(req_data.seed)
    default_tokenized = model.generate(
        req_data.prompt,
        do_sample=True,
        use_past_kv_cache=False,
        verbose=False,
        stop_at_eos=True,
        max_new_tokens=req_data.n_tokens,
        temperature=req_data.temperature,
        freq_penalty=req_data.freq_penalty,
        return_type="tokens",
    )[0]
    default_tokenized_str_tokens = [
        model.tokenizer.decode([token]) for token in default_tokenized
    ]
    default_generation = "".join(default_tokenized_str_tokens)

    with torch.inference_mode():
        default_logits = model(default_tokenized.unsqueeze(0))
        if default_logits.dim() == 2:
            default_logits = default_logits.unsqueeze(0)
        topk_default_by_token = []
        for i in range(len(default_tokenized_str_tokens)):
            if i < sequence_length - 1:
                topk_default_by_token.append(
                    {"token": default_tokenized_str_tokens[i], "top_logits": []}
                )
                continue
            topk_default = get_topk(
                default_logits[:, : i + 1, :], model.tokenizer, req_data.top_k
            )
            topk_default_by_token.append(
                {
                    "token": default_tokenized_str_tokens[i],
                    "top_logits": [
                        {"token": token, "prob": prob, "logit": logit}
                        for token, prob, logit in topk_default
                    ],
                }
            )

    return default_tokenized_str_tokens, default_generation, topk_default_by_token


def _compute_steered_run(req_data, sequence_length: int, intervention_tuples, default_tokenized_str_tokens):
    if req_data.seed is not None:
        torch.manual_seed(req_data.seed)
    steered_tokenized, steered_logits, _ = model.feature_intervention_generate(
        req_data.prompt,
        intervention_tuples,
        freeze_attention=req_data.freeze_attention,
        do_sample=True,
        verbose=False,
        stop_at_eos=True,
        max_new_tokens=req_data.n_tokens + 1,
        temperature=req_data.temperature,
        freq_penalty=req_data.freq_penalty,
        return_type="tokens",
    )
    steered_tokenized = steered_tokenized[0]
    steered_tokenized_str_tokens = [
        model.tokenizer.decode([token]) for token in steered_tokenized
    ]
    steered_generation = "".join(steered_tokenized_str_tokens)

    if steered_logits.dim() == 2:
        steered_logits = steered_logits.unsqueeze(0)

    topk_steered_by_token = []
    for i in range(len(default_tokenized_str_tokens)):
        if i < sequence_length - 1:
            topk_steered_by_token.append(
                {"token": steered_tokenized_str_tokens[i], "top_logits": []}
            )
            continue
        gen_idx = i - (sequence_length - 1)
        topk_steered = get_topk(
            steered_logits[:, : gen_idx + 1, :], model.tokenizer, req_data.top_k
        )
        topk_steered_by_token.append(
            {
                "token": steered_tokenized_str_tokens[i],
                "top_logits": [
                    {"token": token, "prob": prob, "logit": logit}
                    for token, prob, logit in topk_steered
                ],
            }
        )

    return {
        "STEERED_LOGITS_BY_TOKEN": topk_steered_by_token,
        "STEERED_GENERATION": steered_generation,
    }


@app.post("/steer", dependencies=[Depends(verify_secret_key)])
async def steer_handler(req: Request):
    """Handle steer requests"""
    print("========== Steer Start ==========")
    print(
        f"Thread {threading.get_ident()}: Received request. Attempting to acquire lock."
    )
    if not request_lock.acquire(blocking=False):
        print(
            f"Thread {threading.get_ident()}: Lock acquisition failed (busy). Rejecting request."
        )
        return JSONResponse(
            content={"error": "Server busy, please try again later."}, status_code=503
        )

    print(f"Thread {threading.get_ident()}: Lock acquired.")
    try:
        request_body = await req.json()
        req_data = SteerRequest.model_validate(request_body)

        if req_data.model_id != loaded_model_arg:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{req_data.model_id}' is not available. Only '{loaded_model_arg}' is currently loaded.",
            )

        sequence_length = len(model.tokenizer(req_data.prompt).input_ids)
        _validate_steer_features(req_data.features, sequence_length)

        print(f"Received steer request: {req_data}")

        _, activations = model.get_activations(req_data.prompt, sparse=True)
        intervention_tuples = _build_intervention_tuples(req_data.features, activations, sequence_length)
        default_tokenized_str_tokens, default_generation, topk_default_by_token = _compute_default_run(
            req_data, sequence_length
        )
        steered_result = _compute_steered_run(
            req_data, sequence_length, intervention_tuples, default_tokenized_str_tokens
        )

        print(f"Default generation: {default_generation}")
        print(f"Steered generation: {steered_result['STEERED_GENERATION']}")

        response = {
            "DEFAULT_LOGITS_BY_TOKEN": topk_default_by_token,
            "STEERED_LOGITS_BY_TOKEN": steered_result["STEERED_LOGITS_BY_TOKEN"],
            "DEFAULT_GENERATION": default_generation,
            "STEERED_GENERATION": steered_result["STEERED_GENERATION"],
        }

        return response

    finally:
        if request_lock.locked():
            print(f"Thread {threading.get_ident()}: Releasing lock in finally block.")
            request_lock.release()
        else:
            print(
                f"Thread {threading.get_ident()}: Lock was not held by current path in finally block (already released or never acquired)."
            )


@app.post("/steer-batch", dependencies=[Depends(verify_secret_key)])
async def steer_batch_handler(req: Request):
    """Handle multiple steer requests for the same prompt in a single request."""
    print("========== Steer Batch Start ==========")
    if not request_lock.acquire(blocking=False):
        return JSONResponse(
            content={"error": "Server busy, please try again later."}, status_code=503
        )

    try:
        request_body = await req.json()
        req_data = SteerBatchRequest.model_validate(request_body)

        if req_data.model_id != loaded_model_arg:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{req_data.model_id}' is not available. Only '{loaded_model_arg}' is currently loaded.",
            )

        sequence_length = len(model.tokenizer(req_data.prompt).input_ids)
        for features in req_data.feature_sets:
            _validate_steer_features(features, sequence_length)

        _, activations = model.get_activations(req_data.prompt, sparse=True)
        default_tokenized_str_tokens, default_generation, topk_default_by_token = _compute_default_run(
            req_data, sequence_length
        )

        steered_results = []
        for features in req_data.feature_sets:
            intervention_tuples = _build_intervention_tuples(features, activations, sequence_length)
            steered_results.append(
                _compute_steered_run(req_data, sequence_length, intervention_tuples, default_tokenized_str_tokens)
            )

        return {
            "DEFAULT_LOGITS_BY_TOKEN": topk_default_by_token,
            "DEFAULT_GENERATION": default_generation,
            "STEERED_RESULTS": steered_results,
        }
    finally:
        if request_lock.locked():
            request_lock.release()


@app.post("/forward-pass", dependencies=[Depends(verify_secret_key)])
async def forward_pass_handler(req: Request):
    """Handle forward pass requests to get salient logits"""
    print("========== Forward Pass Start ==========")

    print(
        f"Thread {threading.get_ident()}: Received request. Attempting to acquire lock."
    )
    if not request_lock.acquire(blocking=False):
        print(
            f"Thread {threading.get_ident()}: Lock acquisition failed (busy). Rejecting request."
        )
        return JSONResponse(
            content={"error": "Server busy, please try again later."}, status_code=503
        )

    print(f"Thread {threading.get_ident()}: Lock acquired.")
    try:
        request_body = await req.json()
        req_data = ForwardPassRequest.model_validate(request_body)
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid request body", "details": e.errors()},
        )
    finally:
        if request_lock.locked():
            print(
                f"Thread {threading.get_ident()}: Releasing lock in validation finally block."
            )
            request_lock.release()

    try:
        print(f"Received forward pass request: prompt='{req_data.prompt}'")

        if BACKEND == "lm-saes-crm":
            return forward_pass_crm(
                req_data.prompt,
                crm_model,
                max_n_logits=req_data.max_n_logits,
                desired_logit_prob=req_data.desired_logit_prob,
            )

        # Circuit-tracer backend
        tokens = model.tokenizer.encode(req_data.prompt, add_special_tokens=False)
        if tokens and tokens[0] != model.tokenizer.bos_token_id:
            tokens = model.tokenizer.encode(req_data.prompt, add_special_tokens=True)
        print(f"Tokens: {tokens}")

        input_ids = torch.tensor([tokens]).to(get_device())

        with torch.no_grad():
            output = model(input_ids)
            if hasattr(output, "logits"):
                output = output.logits

            logits = output[0, -1, :]

            if hasattr(model, "unembed"):
                unembed_matrix = model.unembed.W_U
            elif hasattr(model, "lm_head"):
                unembed_matrix = model.lm_head.weight
            else:
                raise AttributeError(
                    "Model has neither 'unembed' nor 'lm_head' attribute"
                )

            logit_indices, logit_probs, _ = compute_salient_logits(
                logits,
                unembed_matrix,
                max_n_logits=req_data.max_n_logits,
                desired_logit_prob=req_data.desired_logit_prob,
            )

        results = []
        for idx, prob in zip(logit_indices.tolist(), logit_probs.tolist()):
            token = model.tokenizer.decode([idx])
            results.append(
                {"token": token, "token_id": idx, "probability": float(prob)}
            )

        response = {
            "prompt": req_data.prompt,
            "input_tokens": [model.tokenizer.decode([token]) for token in tokens],
            "salient_logits": results,
            "total_salient_tokens": len(results),
            "cumulative_probability": float(logit_probs.sum()),
        }

        print(
            f"Found {len(results)} salient tokens with cumulative prob: {response['cumulative_probability']:.4f}"
        )

        return response

    except Exception as e:
        print(f"Error in forward pass: {str(e)}")
        return {"error": f"Forward pass failed: {str(e)}"}

    finally:
        if request_lock.locked():
            print(f"Thread {threading.get_ident()}: Releasing lock in finally block.")
            request_lock.release()
        else:
            print(
                f"Thread {threading.get_ident()}: Lock was not held by current path in finally block (already released or never acquired)."
            )


@app.post("/generate-graph", dependencies=[Depends(verify_secret_key)])
async def generate_graph(req: Request):
    print(
        f"Thread {threading.get_ident()}: Received request. Attempting to acquire lock."
    )
    if not request_lock.acquire(blocking=False):
        print(
            f"Thread {threading.get_ident()}: Lock acquisition failed (busy). Rejecting request."
        )
        return JSONResponse(
            content={"error": "Server busy, please try again later."}, status_code=503
        )

    print(f"Thread {threading.get_ident()}: Lock acquired.")
    try:
        try:
            request_body = await req.json()
            req_data = GraphGenerationRequest.model_validate(request_body)
        except ValidationError as e:
            print(f"Thread {threading.get_ident()}: Validation error. Releasing lock.")
            request_lock.release()
            raise HTTPException(
                status_code=400,
                detail={"error": "Invalid request body", "details": e.errors()},
            )
        except Exception as e:
            print(
                f"Thread {threading.get_ident()}: JSON parsing error. Releasing lock."
            )
            request_lock.release()
            print(f"Error getting/parsing JSON: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        prompt = req_data.prompt
        tlens_model_id = req_data.model_id
        if tlens_model_id is None or tlens_model_id != loaded_model_arg:
            request_lock.release()
            raise HTTPException(
                status_code=400,
                detail=f"Model '{tlens_model_id}' is not available. Only '{loaded_model_arg}' is currently loaded.",
            )

        batch_size = req_data.batch_size
        max_n_logits = req_data.max_n_logits
        desired_logit_prob = req_data.desired_logit_prob
        node_threshold = req_data.node_threshold
        edge_threshold = req_data.edge_threshold
        slug_identifier = req_data.slug_identifier or f"generated-{int(time.time())}"
        max_feature_nodes = req_data.max_feature_nodes
        print(
            f"Thread {threading.get_ident()}: Processing request for prompt: '{prompt[:50]}...' with parameters:"
        )
        print(f"  model_id: {tlens_model_id}")
        print(f"  batch_size: {batch_size}")
        print(f"  max_n_logits: {max_n_logits}")
        print(f"  desired_logit_prob: {desired_logit_prob}")
        print(f"  node_threshold: {node_threshold}")
        print(f"  edge_threshold: {edge_threshold}")
        print(f"  slug_identifier: {slug_identifier}")
        print(f"  max_feature_nodes: {max_feature_nodes}")
        print(f"  backend: {BACKEND}")

        def _blocking_graph_generation_task():
            print(
                f"Thread {threading.get_ident()} (worker): Starting blocking graph generation."
            )
            _total_start_time = time.time()

            try:
                tokens = model.tokenizer.encode(prompt, add_special_tokens=False)
                print(
                    f"Thread {threading.get_ident()} (worker): {len(tokens)} Tokens: {tokens}"
                )
                if len(tokens) > LIMIT_TOKENS:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Prompt exceeds token limit ({len(tokens)} > {LIMIT_TOKENS})",
                    )
            except Exception as e:
                print(
                    f"Thread {threading.get_ident()} (worker): Tokenization error: {e}"
                )
                raise HTTPException(status_code=500, detail="Failed to tokenize prompt")

            if BACKEND == "lm-saes-crm":
                return generate_graph_crm(
                    prompt,
                    crm_model,
                    crm_replacement_modules,
                    crm_sae_metadata,
                    slug_identifier=slug_identifier,
                    max_n_logits=max_n_logits,
                    desired_logit_prob=desired_logit_prob,
                    batch_size=batch_size,
                    max_feature_nodes=max_feature_nodes,
                    node_threshold=node_threshold,
                    edge_threshold=edge_threshold,
                    signed_url=req_data.signed_url,
                    user_id=req_data.user_id,
                    compress=req_data.compress,
                    enable_qk_tracing=req_data.enable_qk_tracing,
                    qk_top_fraction=req_data.qk_top_fraction,
                    qk_topk=req_data.qk_topk,
                )

            print(f"Thread {threading.get_ident()} (worker): Prompt: '{prompt}'")

            attribution_start = time.time()
            _graph = attribute(
                prompt,
                model,
                max_n_logits=max_n_logits,
                desired_logit_prob=desired_logit_prob,
                batch_size=batch_size,
                max_feature_nodes=req_data.max_feature_nodes,
                offload=OFFLOAD,
                update_interval=UPDATE_INTERVAL,
            )
            attribution_time_ms = (time.time() - attribution_start) * 1000
            print(
                f"Thread {threading.get_ident()} (worker): Attribution Time: {attribution_time_ms:.2f}ms"
            )

            _prune_device = "cuda" if torch.cuda.is_available() else device
            _graph.to(_prune_device)

            _node_mask, _edge_mask, _cumulative_scores = (
                el.cpu() for el in prune_graph(_graph, node_threshold, edge_threshold)
            )
            _graph.to("cpu")

            tokenizer = get_shared_tokenizer()

            _nodes = create_nodes(
                _graph,
                _node_mask,
                tokenizer,
                _cumulative_scores,
            )
            print("nodes created")
            _used_nodes, _used_edges = create_used_nodes_and_edges(
                _graph, _nodes, _edge_mask
            )
            print("used nodes and edges created")
            _output_model = build_model(
                _graph,
                _used_nodes,
                _used_edges,
                slug_identifier,
                TLENS_MODEL_ID_TO_NP_MODEL_ID[tlens_model_id],
                node_threshold,
                tokenizer,
            )
            print("output model created")

            # Optionally pre-build scorer and cache for /build-circuit endpoint.
            # Bulk graph generation does not need this, and skipping it materially
            # reduces both latency and memory pressure.
            output_dict = _output_model.model_dump() if hasattr(_output_model, 'model_dump') else _output_model
            if PREBUILD_SCORER_ON_GENERATE and isinstance(output_dict, dict) and 'nodes' in output_dict and 'links' in output_dict:
                try:
                    scorer = BatchGraphScorer(
                        output_dict['nodes'], output_dict['links'], device_str=str(device),
                    )
                    _cache_graph_artifacts(slug_identifier, scorer, {
                        'nodes': output_dict['nodes'],
                        'links': output_dict['links'],
                    })
                    print(f"Cached scorer '{slug_identifier}' for /build-circuit ({len(output_dict['nodes'])} nodes)")
                except Exception as e:
                    print(f"Failed to cache scorer: {e}")
            elif not PREBUILD_SCORER_ON_GENERATE:
                print("Skipping scorer prebuild for /generate-graph")

            # if signed_url is not provided, we don't upload the file, just return the output model
            if req_data.signed_url is None:
                print("No signed url provided, returning output model")
                return _output_model

            # if signed_url is provided, we upload the file and return a success message
            print(f"Uploading file to url: {req_data.signed_url}")
            current_time_ms = int(time.time() * 1000)
            # Convert to dict to add additional fields
            model_dict = _output_model.model_dump()

            # Add additional metadata fields
            model_dict["metadata"]["info"] = {
                "creator_name": req_data.user_id
                if req_data.user_id
                else "Anonymous (CT)",
                "creator_url": "https://neuronpedia.org",
                "source_urls": TRANSCODER_SET_TO_SOURCE_URL_ARRAYS[transcoder_set],
                "transcoder_set": transcoder_set,
                "generator": GENERATOR_INFO,
                "create_time_ms": current_time_ms,
            }

            model_dict["metadata"]["generation_settings"] = {
                "max_n_logits": max_n_logits,
                "desired_logit_prob": desired_logit_prob,
                "batch_size": batch_size,
                "max_feature_nodes": max_feature_nodes,
            }

            model_dict["metadata"]["pruning_settings"] = {
                "node_threshold": node_threshold,
                "edge_threshold": edge_threshold,
            }

            # Convert back to JSON string
            model_json = json.dumps(model_dict)

            # Handle compression if requested
            compress_time_ms = 0
            if req_data.compress:
                print("Compressing data with gzip (level 3)...")
                compress_start = time.time()
                data_to_upload = gzip.compress(
                    model_json.encode("utf-8"), compresslevel=3
                )
                compress_time_ms = (time.time() - compress_start) * 1000
                headers = {
                    "Content-Type": "application/json",
                    "Content-Encoding": "gzip",
                }
            else:
                data_to_upload = model_json.encode("utf-8")
                headers = {"Content-Type": "application/json"}

            # Track upload size
            upload_size_bytes = len(data_to_upload)

            # Start upload timing
            upload_start = time.time()
            response = requests.put(
                req_data.signed_url,
                data=data_to_upload,
                headers=headers,
            )
            upload_time_ms = (time.time() - upload_start) * 1000

            print(f"Upload response: {response.status_code}")
            # print(f"Upload response: {response.text}")
            if response.status_code != 200:
                return {"error": "Failed to upload file"}

            print(f"File: uploaded successfully to url: {req_data.signed_url}")

            _total_time_ms = time.time() - _total_start_time

            # Log timing summary
            timing_parts = [
                f"attribution_ms={attribution_time_ms:.0f}",
                f"upload_ms={upload_time_ms:.0f}",
                f"upload_size_bytes={upload_size_bytes}",
                f"upload_size_mb={upload_size_bytes / (1024 * 1024):.2f}",
                f"total_ms={_total_time_ms:.0f}",
            ]

            if req_data.compress:
                timing_parts.extend(
                    [
                        f"compress_ms={compress_time_ms:.0f}",
                        f"compression_ratio={len(model_json.encode('utf-8')) / upload_size_bytes:.2f}",
                    ]
                )

            print(
                f"Thread {threading.get_ident()} (worker): Total Time for blocking task: {_total_time_ms=:.2f}s"
            )

            return {
                "success": f"Graph uploaded successfully to url: {req_data.signed_url}"
            }

        try:
            result = await run_in_threadpool(_blocking_graph_generation_task)
            print(f"Thread {threading.get_ident()}: Blocking task completed.")
            return result
        except HTTPException:
            raise
        except Exception as e:
            import traceback

            print(
                f"Thread {threading.get_ident()}: Error during graph generation in worker thread: {e}"
            )
            print("Stack trace:")
            traceback.print_exc()
            raise HTTPException(
                status_code=500, detail="Internal server error during graph generation"
            )

    finally:
        printMemory()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Cleared CUDA cache")

        gc.collect()
        print("Cleared CPU memory")
        if request_lock.locked():
            print(f"Thread {threading.get_ident()}: Releasing lock in finally block.")
            request_lock.release()
        else:
            print(
                f"Thread {threading.get_ident()}: Lock was not held by current path in finally block (already released or never acquired)."
            )


# ============ Circuit Explorer ============


class GroupNodesRequest(BaseModel):
    graph_data: dict
    pinned_ids: list[str]
    prompt: str = ""
    grouping_model: str = "sonnet"


@app.post("/group-nodes", dependencies=[Depends(verify_secret_key)])
async def group_nodes_handler(req: Request):
    """Group pinned nodes into semantic supernodes using LLM."""
    try:
        request_body = await req.json()
        req_data = GroupNodesRequest.model_validate(request_body)
        nodes = req_data.graph_data.get("nodes", [])
        links = req_data.graph_data.get("links", [])
        sn, explanations, member_reasons, grouping_failed, grouping_error = await run_in_threadpool(
            auto_group_nodes, nodes, links, req_data.pinned_ids, req_data.prompt, req_data.grouping_model,
        )
        return format_grouping_response(
            sn,
            explanations,
            member_reasons,
            req_data.grouping_model,
            grouping_failed,
            grouping_error,
        )
    except Exception as e:
        print(f"Error in group-nodes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class ExploreCircuitsRequest(BaseModel):
    model_id: str
    graph_data: dict
    target_logit_node_id: str | None = None
    endpoint_node_ids: list[str] = []
    num_seeds: int = 5
    max_iterations: int = 80
    pc_passes: int = 1
    keep_ratio: float = 0.4
    grouping_model: str = "sonnet"


@app.post("/score-circuit", dependencies=[Depends(verify_secret_key)])
async def score_circuit(req: Request):
    """Score circuit(s) using PyTorch. Pre-processes graph once, scores all pin sets."""
    body = await req.json()
    nodes = body.get('nodes', [])
    links = body.get('links', [])
    pinned_ids_list = body.get('pinnedIdsList', [])

    scorer = BatchGraphScorer(nodes, links)
    results = [scorer.score(pins) for pins in pinned_ids_list]

    return JSONResponse(content={'results': results})


# Cache: slug → pre-built scorer (avoids re-parsing graph data)
_scorer_cache: "OrderedDict[str, BatchGraphScorer]" = OrderedDict()
_graph_cache: "OrderedDict[str, dict]" = OrderedDict()


@app.post("/build-circuit", dependencies=[Depends(verify_secret_key)])
async def build_circuit(req: Request):
    """
    Build a circuit server-side using batched GPU scoring.
    Runs IA greedy + pathway completion entirely on the server.
    Pass graph data via JSON, or pass slug to use a cached pre-built scorer.
    """
    body = await req.json()
    slug = body.get('slug')
    nodes = body.get('nodes')
    links = body.get('links')
    endpoint_ids = body.get('endpointIds', [])
    method = body.get('method', 'ia_pc')  # 'c_only', 'ia', 'ia_pc'
    alpha = body.get('alpha', 0.15)
    device_str = body.get('device', str(device))

    import time as _time
    _t0 = _time.time()

    # Use cached pre-built scorer if slug matches
    if slug and nodes is None:
        scorer = _get_cached_scorer(slug)
    else:
        scorer = None

    if scorer is not None:
        print(f"[build-circuit] Using cached scorer for '{slug}' ({_time.time()-_t0:.3f}s)")
    elif nodes is not None and links is not None:
        scorer = BatchGraphScorer(nodes, links, device_str=device_str)
        if slug:
            _cache_graph_artifacts(slug, scorer, {'nodes': nodes, 'links': links})
    else:
        raise HTTPException(status_code=400, detail="Must provide nodes/links or a valid slug")

    print(f"[build-circuit] Scorer ready, starting build ({_time.time()-_t0:.3f}s)")

    def _do_build():
        if method == 'ia_pc' or method == 'ia':
            return scorer.build_circuit_ia_pc(
                endpoint_ids, alpha=alpha,
                pc_passes=3 if method == 'ia_pc' else 0,
            )
        else:
            return scorer.build_circuit_ia_pc(
                endpoint_ids, alpha=0.0,
                pc_passes=0,
            )

    result = await run_in_threadpool(_do_build)

    print(f"[build-circuit] Done: {result['totalFeatures']} features in {_time.time()-_t0:.3f}s")
    return JSONResponse(content=result)


@app.post("/explore-circuits", dependencies=[Depends(verify_secret_key)])
async def explore_circuits_handler(req: Request):
    """Explore multiple circuits and stream results as SSE events."""
    print("========== Explore Circuits Start ==========")
    request_body = await req.json()
    req_data = ExploreCircuitsRequest.model_validate(request_body)
    job = ExploreCircuitsJob(
        graph_data=req_data.graph_data,
        target_logit_node_id=req_data.target_logit_node_id,
        endpoint_node_ids=req_data.endpoint_node_ids,
        num_seeds=req_data.num_seeds,
        max_iterations=req_data.max_iterations,
        pc_passes=req_data.pc_passes,
        keep_ratio=req_data.keep_ratio,
        grouping_model=req_data.grouping_model,
    )
    return build_explore_circuits_response(job, device_str=str(device))
