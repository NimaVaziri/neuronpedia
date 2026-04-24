"""
Native Python eval runner — builds circuits in a separate process (no model contention),
only uses HTTP for steer calls.
Usage: python eval/eval_runner.py [--prompts verified|all] [--method ia_pc|c_only]

Requires the graph server to be running on STEER_PORT for causal validation steer calls.
"""
import argparse
import concurrent.futures
import json
import multiprocessing
import os
import ssl
import sys
import time
import urllib.request
from datetime import datetime

GRAPH_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if GRAPH_APP_DIR not in sys.path:
    sys.path.insert(0, GRAPH_APP_DIR)

STEER_URL = os.environ.get("STEER_URL", "http://127.0.0.1:5005")
STEER_SECRET = os.environ.get("STEER_SECRET", "SECRET")
MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-2-2b")
GRAPH_BASE_URL = os.environ.get("GRAPH_BASE_URL", "http://localhost:3000")
EVAL_GRAPH_URL_PREFIX = os.environ.get("EVAL_GRAPH_URL_PREFIX", "").rstrip("/")
EVAL_GRAPH_DIR = os.environ.get("EVAL_GRAPH_DIR", "")
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

# ── Eval prompt registry ──

PROMPTS = {
    "capital-france": {"prompt": "The capital of France is called", "target": " Paris", "category": "factual-recall", "graph_url": "/graph-data/capital-france.json"},
    "capital-japan": {"prompt": "The capital of Japan is called", "target": " Tokyo", "category": "factual-recall", "graph_url": "/graph-data/capital-japan.json"},
    "capital-germany": {"prompt": "The capital of Germany is called", "target": " Berlin", "category": "factual-recall", "graph_url": "/graph-data/capital-germany.json"},
    "capital-italy": {"prompt": "The capital of Italy is called", "target": " Rome", "category": "factual-recall", "graph_url": "/graph-data/capital-italy.json"},
    "capital-australia": {"prompt": "The capital of Australia is called", "target": " Canberra", "category": "factual-recall", "graph_url": "/graph-data/capital-australia.json"},
    "dallas-austin": {"prompt": "The capital of the state containing Dallas is", "target": " Austin", "category": "multi-step",
        "graph_url": "https://neuronpedia-attrib.s3.us-east-1.amazonaws.com/user-graphs/cmao6p1xd0000ypedw0qhlhob/gemma-fact-dallas-austin.json"},
    "newyork-albany": {"prompt": "The capital of the state containing New York is", "target": " Albany", "category": "multi-step", "graph_url": "/graph-data/newyork-albany.json"},
    "miami-tallahassee": {"prompt": "The capital of the state containing Miami is", "target": " Tallahassee", "category": "multi-step", "graph_url": "/graph-data/miami-tallahassee.json"},
    "sanfrancisco-sacramento": {"prompt": "The capital of the state containing San Francisco is", "target": " Sacramento", "category": "multi-step", "graph_url": "/graph-data/sanfrancisco-sacramento.json"},
    "seattle-olympia": {"prompt": "The capital of the state containing Seattle is", "target": " Olympia", "category": "multi-step", "graph_url": "/graph-data/seattle-olympia.json"},
    "samsung-won": {"prompt": "The country where Samsung was founded uses the currency called the", "target": " won", "category": "multi-step", "graph_url": "/graph-data/samsung-won.json"},
    "apple-dollar": {"prompt": "The country where Apple was founded uses the currency called the", "target": " dollar", "category": "multi-step", "graph_url": "/graph-data/apple-dollar.json"},
    "mumbai-asia": {"prompt": "The continent of the country containing Mumbai is called", "target": " Asia", "category": "multi-step", "graph_url": "/graph-data/mumbai-asia.json"},
    "eiffel-tower-french": {"prompt": "The language spoken in the country where the Eiffel Tower is located is", "target": " French", "category": "multi-step", "graph_url": "/graph-data/eiffel-tower-french.json"},
    "london-english": {"prompt": "The language spoken in the country where Big Ben is located is", "target": " English", "category": "multi-step", "graph_url": "/graph-data/london-english.json"},
    "colosseum-italian": {"prompt": "The language spoken in the country where the Colosseum is located is", "target": " Italian", "category": "multi-step", "graph_url": "/graph-data/colosseum-italian.json"},
    "tajmahal-hindi": {"prompt": "The language spoken in the country where the Taj Mahal is located is", "target": " Hindi", "category": "multi-step", "graph_url": "/graph-data/tajmahal-hindi.json"},
    "toyota-yen": {"prompt": "The currency used in the country where Toyota is headquartered is called the", "target": " yen", "category": "multi-step", "graph_url": "/graph-data/toyota-yen.json"},
    "bmw-euro": {"prompt": "The currency used in the country where BMW is headquartered is called the", "target": " Euro", "category": "multi-step", "graph_url": "/graph-data/bmw-euro.json"},
    "berlin-german": {"prompt": "The language spoken in the country whose capital is Berlin is", "target": " German", "category": "multi-step", "graph_url": "/graph-data/berlin-german.json"},
    "tokyo-japanese": {"prompt": "The language spoken in the country whose capital is Tokyo is", "target": " Japanese", "category": "multi-step", "graph_url": "/graph-data/tokyo-japanese.json"},
    "bonjour-french": {"prompt": "Bonjour means hello in", "target": " French", "category": "cross-lingual", "graph_url": "/graph-data/bonjour-french.json"},
    "hola-spanish": {"prompt": "Hola means hello in", "target": " Spanish", "category": "cross-lingual", "graph_url": "/graph-data/hola-spanish.json"},
    "ciao-italian": {"prompt": "Ciao means hello in", "target": " Italian", "category": "cross-lingual", "graph_url": "/graph-data/ciao-italian.json"},
    "danke-german": {"prompt": "Danke means thank you in", "target": " German", "category": "cross-lingual", "graph_url": "/graph-data/danke-german.json"},
    "grazie-italian": {"prompt": "Grazie means thank you in", "target": " Italian", "category": "cross-lingual", "graph_url": "/graph-data/grazie-italian.json"},
    "plural-mice": {"prompt": "The plural of mouse is", "target": " mice", "category": "irregular-morphology", "graph_url": "/graph-data/plural-mice.json"},
    "plural-geese": {"prompt": "The plural of goose is", "target": " geese", "category": "irregular-morphology", "graph_url": "/graph-data/plural-geese.json"},
    "plural-teeth": {"prompt": "The plural of tooth is", "target": " teeth", "category": "irregular-morphology", "graph_url": "/graph-data/plural-teeth.json"},
    "plural-feet": {"prompt": "The plural of foot is", "target": " feet", "category": "irregular-morphology", "graph_url": "/graph-data/plural-feet.json"},
    "plural-children": {"prompt": "The plural of child is", "target": " children", "category": "irregular-morphology", "graph_url": "/graph-data/plural-children.json"},
    "plural-men": {"prompt": "The plural of man is", "target": " men", "category": "irregular-morphology", "graph_url": "/graph-data/plural-men.json"},
    "opposite-hot-cold": {"prompt": "The opposite of hot is", "target": " cold", "category": "antonym", "graph_url": "/graph-data/opposite-hot-cold.json"},
    "opposite-big-small": {"prompt": "The opposite of big is", "target": " small", "category": "antonym", "graph_url": "/graph-data/opposite-big-small.json"},
    "opposite-fast-slow": {"prompt": "The opposite of fast is", "target": " slow", "category": "antonym", "graph_url": "/graph-data/opposite-fast-slow.json"},
    "opposite-light-dark": {"prompt": "The opposite of light is", "target": " dark", "category": "antonym", "graph_url": "/graph-data/opposite-light-dark.json"},
    "opposite-up-down": {"prompt": "The opposite of up is", "target": " down", "category": "antonym", "graph_url": "/graph-data/opposite-up-down.json"},
    "raining-umbrella": {"prompt": "If it is raining, you should bring an", "target": " umbrella", "category": "conditional-reasoning", "graph_url": "/graph-data/raining-umbrella.json"},
    "cold-jacket": {"prompt": "If it is cold, you should bring a", "target": " jacket", "category": "conditional-reasoning", "graph_url": "/graph-data/cold-jacket.json"},
    "thirsty-drink": {"prompt": "If you are thirsty, you should", "target": " drink", "category": "conditional-reasoning", "graph_url": "/graph-data/thirsty-drink.json"},
    "hands-wash": {"prompt": "If your hands are dirty, you should wash", "target": " them", "category": "conditional-reasoning", "graph_url": "/graph-data/hands-wash.json"},
    "fire-burn": {"prompt": "If you touch fire, you will get", "target": " burned", "category": "conditional-reasoning", "graph_url": "/graph-data/fire-burn.json"},
    "weather-not-cloudy": {"prompt": "The weather was not cloudy, it was actually", "target": " sunny", "category": "negation", "graph_url": "/graph-data/weather-not-cloudy.json"},
    "store-not-closed": {"prompt": "The store was not closed, it was actually", "target": " open", "category": "negation", "graph_url": "/graph-data/store-not-closed.json"},
    "statement-not-false": {"prompt": "The statement was not false, it was actually", "target": " true", "category": "negation", "graph_url": "/graph-data/statement-not-false.json"},
    "show-tickets": {"prompt": "The show was not sold out, actually they had remaining", "target": " tickets", "category": "negation", "graph_url": "/graph-data/show-tickets.json"},
    "flight-not-domestic": {"prompt": "The flight was not domestic, it was", "target": " international", "category": "negation", "graph_url": "/graph-data/flight-not-domestic.json"},
    "verdict-not-innocent": {"prompt": "The verdict was not innocent, he was found", "target": " guilty", "category": "negation", "graph_url": "/graph-data/verdict-not-innocent.json"},
    "sequence-not-finite": {"prompt": "The sequence of numbers was not finite, it was", "target": " infinite", "category": "negation", "graph_url": "/graph-data/sequence-not-finite.json"},
    "keys-cabinet-are": {"prompt": "The keys to the cabinet", "target": " are", "category": "syntactic-agreement", "graph_url": "/graph-data/keys-cabinet-are.json"},
    "key-cabinets-is": {"prompt": "The key to the cabinets", "target": " is", "category": "syntactic-agreement", "graph_url": "/graph-data/key-cabinets-is.json"},
    "cats-mat-sleep": {"prompt": "The cats on the mat", "target": " are", "category": "syntactic-agreement", "graph_url": "/graph-data/cats-mat-sleep.json"},
    "cat-mats-sleeps": {"prompt": "The cat on the mats", "target": " is", "category": "syntactic-agreement", "graph_url": "/graph-data/cat-mats-sleeps.json"},
    "woman-introduction": {"prompt": "The woman who spoke to the managers introduced", "target": " herself", "category": "syntactic-agreement", "graph_url": "/graph-data/woman-introduction.json"},
    "women-introduction": {"prompt": "The women who spoke to the managers introduced", "target": " themselves", "category": "syntactic-agreement", "graph_url": "/graph-data/women-introduction.json"},
    "books-shelf-are": {"prompt": "The books on the library shelf", "target": " are", "category": "syntactic-agreement", "graph_url": "/graph-data/books-shelf-are.json"},
    "book-shelves-is": {"prompt": "The book on the library shelves", "target": " is", "category": "syntactic-agreement", "graph_url": "/graph-data/book-shelves-is.json"},
    "transitive-abc": {"prompt": "If A is larger than B, and B is larger than C, then A is larger than", "target": " C", "category": "transitive-reasoning", "graph_url": "/graph-data/transitive-abc.json"},
    "transitive-color": {"prompt": "If red is darker than blue, and blue is darker than yellow, then red is darker than", "target": " yellow", "category": "transitive-reasoning", "graph_url": "/graph-data/transitive-color.json"},
    "transitive-people": {"prompt": "If Alice is smarter than Bob, and Bob is smarter than Carol, then Alice is smarter than", "target": " Carol", "category": "transitive-reasoning", "graph_url": "/graph-data/transitive-people.json"},
    "transitive-material": {"prompt": "If iron is stronger than wood, and wood is stronger than paper, then iron is stronger than", "target": " paper", "category": "transitive-reasoning", "graph_url": "/graph-data/transitive-material.json"},
    "transitive-distance": {"prompt": "If a marathon is longer than a mile, and a mile is longer than a sprint, then a marathon is longer than a", "target": " sprint", "category": "transitive-reasoning", "graph_url": "/graph-data/transitive-distance.json"},
}


def _worker_loop(pipe):
    """Persistent worker process — imports scorer once, handles multiple requests."""
    import time as _time
    from neuronpedia_graph.scorer import BatchGraphScorer, prune_graph_for_scoring

    while True:
        try:
            msg = pipe.recv()
        except EOFError:
            break

        if msg is None:  # shutdown signal
            break

        graph = msg["graph"]
        target_token = msg["target"]
        method = msg["method"]
        alpha = msg["alpha"]
        device_str = msg.get("device_str", "cpu")

        # Focus on target
        tl = None
        for n in graph["nodes"]:
            if n["feature_type"] == "logit" and target_token.strip() in (n.get("clerp") or ""):
                tl = n
                break
        for n in graph["nodes"]:
            if n["feature_type"] == "logit" and (tl is None or n["node_id"] != tl["node_id"]):
                n["token_prob"] = 0
        endpoints = [n["node_id"] for n in graph["nodes"] if n["feature_type"] == "embedding"]
        if tl:
            endpoints.append(tl["node_id"])

        # Pre-prune graph to reduce matrix size
        pruned_nodes, pruned_links = prune_graph_for_scoring(graph["nodes"], graph["links"], target_token, keep_ratio=0.4)

        t0 = _time.time()
        scorer = BatchGraphScorer(pruned_nodes, pruned_links, device_str=device_str)
        pc_passes = 3 if method == "ia_pc" else 0
        _alpha = alpha if method != "c_only" else 0.0
        result = scorer.build_circuit_ia_pc(endpoints, alpha=_alpha, max_steps=200, pc_passes=pc_passes)
        elapsed = _time.time() - t0

        pipe.send({
            "pinnedIds": result["pinnedIds"],
            "iaFeatures": result["iaFeatures"],
            "pcFeatures": result["pcFeatures"],
            "totalFeatures": result["totalFeatures"],
            "replacementScore": result["replacementScore"],
            "completenessScore": result["completenessScore"],
            "build_time": elapsed,
        })


class CircuitWorker:
    """Manages a persistent subprocess for circuit building."""

    def __init__(self, device_str: str = "cpu"):
        self.device_str = device_str
        ctx = multiprocessing.get_context("spawn")
        self.parent_conn, child_conn = ctx.Pipe()
        self.process = ctx.Process(target=_worker_loop, args=(child_conn,), daemon=True)
        self.process.start()

    def build(self, graph: dict, target_token: str, method: str, alpha: float) -> dict:
        self.parent_conn.send({
            "graph": graph,
            "target": target_token,
            "method": method,
            "alpha": alpha,
            "device_str": self.device_str,
        })
        if self.parent_conn.poll(timeout=120):
            return self.parent_conn.recv()
        raise RuntimeError("Worker timed out")

    def shutdown(self):
        try:
            self.parent_conn.send(None)
            self.process.join(timeout=5)
        except Exception:
            self.process.kill()


def resolve_graph_url(prompt_id: str, graph_url: str) -> str:
    if EVAL_GRAPH_URL_PREFIX:
        return f"{EVAL_GRAPH_URL_PREFIX}/{prompt_id}.json"
    return graph_url


def sanitize_model_id(model_id: str) -> str:
    tail = model_id.split("/")[-1] if "/" in model_id else model_id
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in tail).strip("-")


def resolve_scorer_device(requested_device: str = "auto", torch_module=None) -> str:
    if requested_device != "auto":
        return requested_device

    if torch_module is None:
        import torch as torch_module

    if torch_module.cuda.is_available():
        return "cuda"

    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


def get_model_eval_results_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "models", sanitize_model_id(MODEL_ID), "results")


def get_model_eval_graphs_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "models", sanitize_model_id(MODEL_ID), "graphs")


def load_graph(prompt_id: str, graph_url: str) -> dict:
    """Load graph from disk when available, otherwise from URL."""
    local_graph_dir = EVAL_GRAPH_DIR or get_model_eval_graphs_dir()
    local_graph_path = os.path.join(local_graph_dir, f"{prompt_id}.json")
    if os.path.exists(local_graph_path):
        with open(local_graph_path, "r", encoding="utf-8") as f:
            return json.load(f)

    graph_url = resolve_graph_url(prompt_id, graph_url)
    if graph_url.startswith("http"):
        url = graph_url
    else:
        url = f"{GRAPH_BASE_URL}{graph_url}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, context=SSL_CTX) as resp:
        return json.loads(resp.read().decode("utf-8"))


def focus_on_target(graph: dict, target_token: str) -> dict:
    """Zero non-target logit weights."""
    target_logit = None
    for n in graph["nodes"]:
        if n["feature_type"] == "logit" and target_token.strip() in (n.get("clerp") or ""):
            target_logit = n
            break
    nodes = []
    for n in graph["nodes"]:
        if n["feature_type"] == "logit" and (target_logit is None or n["node_id"] != target_logit["node_id"]):
            nodes.append({**n, "token_prob": 0})
        else:
            nodes.append(n)
    return {**graph, "nodes": nodes}


def get_endpoints(graph: dict, target_token: str) -> list[str]:
    """Get embedding + target logit node IDs."""
    eps = [n["node_id"] for n in graph["nodes"] if n["feature_type"] == "embedding"]
    tl = None
    for n in graph["nodes"]:
        if n["feature_type"] == "logit" and target_token.strip() in (n.get("clerp") or ""):
            tl = n
            break
    if tl:
        eps.append(tl["node_id"])
    return eps


def call_steer(prompt: str, features: list[dict]) -> dict:
    """Call the graph server's /steer endpoint."""
    body = json.dumps({
        "model_id": MODEL_ID, "prompt": prompt, "features": features,
        "n_tokens": 1, "top_k": 50, "temperature": 0, "freq_penalty": 0,
        "seed": 42, "freeze_attention": False,
    }).encode()
    req = urllib.request.Request(
        f"{STEER_URL}/steer", data=body, method="POST",
        headers={"Content-Type": "application/json", "x-secret-key": STEER_SECRET},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def call_steer_batch(prompt: str, feature_sets: list[list[dict]]) -> dict:
    """Call the graph server's batched /steer-batch endpoint."""
    body = json.dumps({
        "model_id": MODEL_ID,
        "prompt": prompt,
        "feature_sets": feature_sets,
        "n_tokens": 1,
        "top_k": 50,
        "temperature": 0,
        "freq_penalty": 0,
        "seed": 42,
        "freeze_attention": False,
    }).encode()
    req = urllib.request.Request(
        f"{STEER_URL}/steer-batch",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "x-secret-key": STEER_SECRET},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def find_target_prob(logits_by_token: list, target_token: str) -> float:
    best = 0.0
    for entry in logits_by_token:
        if not entry or "top_logits" not in entry:
            continue
        for l in entry["top_logits"]:
            if l["token"] == target_token and l["prob"] > best:
                best = l["prob"]
    return best


def node_to_steer_feature(node_id: str, graph: dict) -> dict | None:
    node = next((n for n in graph["nodes"] if n["node_id"] == node_id), None)
    return node_to_steer_feature_data(node, node_id)


def node_to_steer_feature_data(node: dict | None, node_id: str | None = None) -> dict | None:
    if not node or node["feature_type"] != "cross layer transcoder":
        return None
    layer = int(node["layer"])
    resolved_node_id = node_id or node["node_id"]
    index = int(resolved_node_id.split("_")[1])
    return {
        "layer": layer, "index": index,
        "token_active_position": node["ctx_idx"],
        "steer_position": node["ctx_idx"],
        "delta": None, "ablate": True, "steer_generated_tokens": False,
    }


def validate_circuit(graph: dict, pinned_ids: list[str], target_token: str) -> dict:
    """Run necessity + sufficiency tests via steer endpoint."""
    prompt = graph["metadata"]["prompt"].replace("<bos>", "")
    pinned_id_set = set(pinned_ids)
    node_by_id = {n["node_id"]: n for n in graph["nodes"]}

    pinned_features = [f for f in (node_to_steer_feature_data(node_by_id.get(pid), pid) for pid in pinned_ids) if f]
    complement_features = [
        f
        for f in (
            node_to_steer_feature_data(n)
            for n in graph["nodes"]
            if n["feature_type"] == "cross layer transcoder" and n["node_id"] not in pinned_id_set
        )
        if f
    ]

    feature_sets = []
    indices = {}
    if pinned_features:
        indices["necessity"] = len(feature_sets)
        feature_sets.append(pinned_features)
    if complement_features:
        indices["sufficiency"] = len(feature_sets)
        feature_sets.append(complement_features)

    batched = call_steer_batch(prompt, feature_sets)
    baseline_prob = find_target_prob(batched["DEFAULT_LOGITS_BY_TOKEN"], target_token)
    batched_results = batched.get("STEERED_RESULTS", [])

    nec_prob = baseline_prob
    if "necessity" in indices:
        nec_prob = find_target_prob(
            batched_results[indices["necessity"]]["STEERED_LOGITS_BY_TOKEN"],
            target_token,
        )

    suf_prob = 0.0
    if "sufficiency" in indices:
        suf_prob = find_target_prob(
            batched_results[indices["sufficiency"]]["STEERED_LOGITS_BY_TOKEN"],
            target_token,
        )

    necessity = baseline_prob - nec_prob
    sufficiency = suf_prob / baseline_prob if baseline_prob > 0 else 0

    return {
        "baseline": baseline_prob, "necessity": necessity,
        "sufficiency": sufficiency, "nec_prob": nec_prob, "suf_prob": suf_prob,
    }


def choose_keep_ratio(
    graph: dict,
    feature_budget: int,
    min_keep_ratio: float,
    max_keep_ratio: float,
) -> float:
    n_features = sum(1 for n in graph["nodes"] if n["feature_type"] == "cross layer transcoder")
    if n_features <= 0:
        return max_keep_ratio
    raw_ratio = feature_budget / n_features
    return max(min_keep_ratio, min(max_keep_ratio, raw_ratio))


def build_circuit_for_prompt(
    raw_graph: dict,
    target_token: str,
    method: str,
    alpha: float,
    feature_budget: int,
    min_keep_ratio: float,
    max_keep_ratio: float,
    device_str: str,
    scorer_cls,
    prune_fn,
) -> dict:
    graph = focus_on_target(raw_graph, target_token)
    endpoints = get_endpoints(graph, target_token)
    keep_ratio = choose_keep_ratio(graph, feature_budget, min_keep_ratio, max_keep_ratio)

    t0 = time.time()
    pruned_nodes, pruned_links = prune_fn(graph["nodes"], graph["links"], target_token, keep_ratio=keep_ratio)
    scorer = scorer_cls(pruned_nodes, pruned_links, device_str=device_str)
    pc_passes = 3 if method == "ia_pc" else 0
    _alpha = alpha if method != "c_only" else 0.0
    circuit = scorer.build_circuit_ia_pc(endpoints, alpha=_alpha, max_steps=200, pc_passes=pc_passes)
    build_time = time.time() - t0

    return {
        "circuit": circuit,
        "build_time": build_time,
        "device": device_str,
        "keep_ratio": keep_ratio,
        "pruned_nodes": len(pruned_nodes),
        "pruned_links": len(pruned_links),
    }


def graph_size_stats(graph: dict) -> dict:
    """Count raw attribution graph size before eval-time pruning."""
    graph_nodes = len(graph["nodes"])
    graph_links = len(graph["links"])
    graph_feature_nodes = sum(
        1 for n in graph["nodes"]
        if n["feature_type"] == "cross layer transcoder"
    )
    return {
        "graph_nodes": graph_nodes,
        "graph_links": graph_links,
        "graph_feature_nodes": graph_feature_nodes,
    }


def run_validation_timed(graph: dict, pinned_ids: list[str], target_token: str) -> tuple[dict, float]:
    t0 = time.time()
    return validate_circuit(graph, pinned_ids, target_token), time.time() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="ia_pc", choices=["ia_pc", "c_only", "ia"])
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--prompts", default="all", help="all or comma-separated IDs")
    parser.add_argument("--skip-steer", action="store_true", help="Skip causal validation (circuit build only)")
    parser.add_argument("--feature-budget", type=int, default=450, help="Adaptive target count for kept feature nodes before scoring")
    parser.add_argument("--min-keep-ratio", type=float, default=0.2, help="Minimum pruning keep ratio")
    parser.add_argument("--max-keep-ratio", type=float, default=0.4, help="Maximum pruning keep ratio")
    parser.add_argument(
        "--device",
        default=os.environ.get("EVAL_SCORER_DEVICE", "auto"),
        choices=["auto", "cuda", "mps", "cpu"],
        help="Scorer device. auto prefers CUDA, then MPS, then CPU.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    scorer_device = resolve_scorer_device(args.device)

    if args.prompts == "all":
        prompt_ids = list(PROMPTS.keys())
    else:
        prompt_ids = [p.strip() for p in args.prompts.split(",")]

    print(f"{'=' * 70}")
    print(f"  NATIVE PYTHON EVAL RUNNER")
    print(f"  Method: {args.method} | Alpha: {args.alpha} | Prompts: {len(prompt_ids)}")
    print(f"  Steer URL: {STEER_URL} | Model: {MODEL_ID}")
    print(f"  Scorer device: {scorer_device} (requested: {args.device})")
    print(f"{'=' * 70}\n")

    # Import scorer in-process (with pruning, no subprocess overhead)
    from neuronpedia_graph.scorer import BatchGraphScorer, prune_graph_for_scoring

    suite_start = time.time()
    results = []
    pending_validations = []
    total_load_time = 0.0
    total_build_time = 0.0
    total_steer_time = 0.0

    def flush_completed(force: bool = False):
        nonlocal total_steer_time
        while pending_validations and (force or pending_validations[0]["future"].done()):
            pending = pending_validations.pop(0)
            try:
                causal, t_steer = pending["future"].result()
                total_steer_time += t_steer
                pending["result"].update({
                    "baseline": causal["baseline"],
                    "necessity": causal["necessity"],
                    "sufficiency": causal["sufficiency"],
                    "steer_time": t_steer,
                    "prompt_elapsed_time": time.time() - pending["prompt_start"],
                })
                print(
                    f"  Steer [{pending['id']}]: {t_steer:.1f}s | "
                    f"Baseline: {causal['baseline']*100:.1f}% | "
                    f"Nec: {causal['necessity']*100:.1f}pp | "
                    f"Suf: {causal['sufficiency']*100:.1f}% | "
                    f"Total: {pending['result']['prompt_elapsed_time']:.1f}s"
                )
            except Exception as exc:
                pending["result"]["steer_error"] = str(exc)
                pending["result"]["prompt_elapsed_time"] = time.time() - pending["prompt_start"]
                print(f"  Steer [{pending['id']}] FAILED: {exc}")
            results.append(pending["result"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as validation_pool:
        for i, pid in enumerate(prompt_ids):
            p = PROMPTS.get(pid)
            if not p:
                print(f"[{pid}] Unknown prompt, skipping")
                continue

            print(f"[{i+1}/{len(prompt_ids)}] {pid}: \"{p['prompt'][:50]}\" → \"{p['target']}\"")

            try:
                prompt_start = time.time()
                t0 = time.time()
                raw_graph = load_graph(pid, p["graph_url"])
                t_load = time.time() - t0
                total_load_time += t_load
                graph_stats = graph_size_stats(raw_graph)

                build = build_circuit_for_prompt(
                    raw_graph=raw_graph,
                    target_token=p["target"],
                    method=args.method,
                    alpha=args.alpha,
                    feature_budget=args.feature_budget,
                    min_keep_ratio=args.min_keep_ratio,
                    max_keep_ratio=args.max_keep_ratio,
                    device_str=scorer_device,
                    scorer_cls=BatchGraphScorer,
                    prune_fn=prune_graph_for_scoring,
                )
                circuit = build["circuit"]
                t_build = build["build_time"]
                total_build_time += t_build

                features = circuit["totalFeatures"]
                pinned_pct_of_graph_nodes = (
                    features / graph_stats["graph_nodes"] * 100
                    if graph_stats["graph_nodes"] else 0.0
                )
                pinned_pct_of_feature_nodes = (
                    features / graph_stats["graph_feature_nodes"] * 100
                    if graph_stats["graph_feature_nodes"] else 0.0
                )
                print(
                    f"  Build: {t_build:.2f}s | {features} features "
                    f"({circuit['iaFeatures']} IA + {circuit['pcFeatures']} PC) | "
                    f"Graph: {graph_stats['graph_nodes']} nodes / "
                    f"{graph_stats['graph_feature_nodes']} feature nodes / "
                    f"{graph_stats['graph_links']} links | "
                    f"Pinned: {pinned_pct_of_graph_nodes:.2f}% of nodes, "
                    f"{pinned_pct_of_feature_nodes:.2f}% of features | "
                    f"Load: {t_load:.2f}s | Keep: {build['keep_ratio']:.2f} "
                    f"({build['pruned_nodes']} nodes / {build['pruned_links']} links)"
                )

                result = {
                    "order": i,
                    "id": pid,
                    "category": p["category"],
                    "prompt": p["prompt"],
                    "target": p["target"],
                    "features": features,
                    "ia_features": circuit["iaFeatures"],
                    "pc_features": circuit["pcFeatures"],
                    "graph_nodes": graph_stats["graph_nodes"],
                    "graph_links": graph_stats["graph_links"],
                    "graph_feature_nodes": graph_stats["graph_feature_nodes"],
                    "pinned_feature_pct_of_graph_nodes": pinned_pct_of_graph_nodes,
                    "pinned_feature_pct_of_feature_nodes": pinned_pct_of_feature_nodes,
                    "pinned_ids": circuit["pinnedIds"],
                    "R": circuit["replacementScore"],
                    "C": circuit["completenessScore"],
                    "load_graph_time": t_load,
                    "build_time": t_build,
                    "device": build["device"],
                    "keep_ratio": build["keep_ratio"],
                    "pruned_nodes": build["pruned_nodes"],
                    "pruned_links": build["pruned_links"],
                }

                if args.skip_steer:
                    result["prompt_elapsed_time"] = time.time() - prompt_start
                    print("  (steer skipped)")
                    results.append(result)
                else:
                    pending_validations.append({
                        "id": pid,
                        "prompt_start": prompt_start,
                        "result": result,
                        "future": validation_pool.submit(run_validation_timed, raw_graph, circuit["pinnedIds"], p["target"]),
                    })
                    flush_completed(force=False)

            except Exception as e:
                print(f"  FAILED: {e}")

        flush_completed(force=True)

    results.sort(key=lambda r: r.get("order", 0))
    for r in results:
        r.pop("order", None)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY ({len(results)} prompts)")
    print(f"{'=' * 70}")
    print(f"  Total load time:  {total_load_time:.1f}s (avg {total_load_time/max(len(results),1):.2f}s/graph)")
    print(f"  Total build time: {total_build_time:.1f}s (avg {total_build_time/max(len(results),1):.2f}s/circuit)")
    print(f"  Total steer time: {total_steer_time:.1f}s (avg {total_steer_time/max(len(results),1):.1f}s/circuit)")
    wall_time = time.time() - suite_start
    print(f"  Serial estimate:  {total_load_time + total_build_time + total_steer_time:.1f}s")
    print(f"  Total wall time:  {wall_time:.1f}s")

    if results and "necessity" in results[0]:
        necs = [abs(r["necessity"]) for r in results if "necessity" in r]
        sufs = [r["sufficiency"] for r in results if "sufficiency" in r]
        print(f"\n  Avg necessity:   {sum(necs)/len(necs)*100:.1f}pp")
        print(f"  Med necessity:   {sorted(necs)[len(necs)//2]*100:.1f}pp")
        print(f"  Avg sufficiency: {sum(sufs)/len(sufs)*100:.1f}%")
        print(f"  Med sufficiency: {sorted(sufs)[len(sufs)//2]*100:.1f}%")

        # Per-category
        categories = sorted(set(r["category"] for r in results))
        print(f"\n  {'Category':<28} {'N':>3} {'Avg Nec':>8} {'Avg Suf':>8}")
        print(f"  {'-'*50}")
        for cat in categories:
            cr = [r for r in results if r["category"] == cat and "necessity" in r]
            if not cr:
                continue
            avg_n = sum(abs(r["necessity"]) for r in cr) / len(cr)
            avg_s = sum(r["sufficiency"] for r in cr) / len(cr)
            print(f"  {cat:<28} {len(cr):>3} {avg_n*100:>7.1f}pp {avg_s*100:>7.1f}%")

    # Save
    out_dir = get_model_eval_results_dir()
    os.makedirs(out_dir, exist_ok=True)
    out_file = args.output or os.path.join(out_dir, f"native-{args.method}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out_file, "w") as f:
        json.dump({"method": args.method, "model": MODEL_ID, "device": scorer_device, "results": results,
                    "total_load_time": total_load_time,
                    "total_build_time": total_build_time, "total_steer_time": total_steer_time,
                    "wall_time": wall_time}, f, indent=2)
    print(f"\n  Results saved to: {out_file}")


if __name__ == "__main__":
    main()
