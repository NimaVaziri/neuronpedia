# Qwen3-4B IA+PC Full Eval Parameters

## Run Target

- Eval runner: `apps/graph/eval/eval_runner.py`
- Model-local eval launcher: `apps/graph/eval/models/qwen3-4b/scripts/run-iapc-eval.sh`
- Graph server URL: `http://127.0.0.1:5004`
- Graph model ID: `Qwen/Qwen3-4B`
- Local graph directory: `apps/graph/eval/models/qwen3-4b/graphs`
- Transcoder set loaded by graph server: `mwhanna/qwen3-4b-transcoders`
- Neuronpedia source set mapping: `qwen3-4b/transcoder-hp`

## Graph Generation Parameters

- `batch_size`: `1`
- `max_n_logits`: `10`
- `desired_logit_prob`: `0.95`
- `node_threshold`: `0.8`
- `edge_threshold`: `0.98`
- `max_feature_nodes`: `10000`

## IA Search Parameters

- Method: Influence-Aware circuit construction
- `alpha`: `0.15`
- Initial pins: endpoints plus top 5 positive direct target-influence features
- Search budget: `200` iterations
- Candidate pool: score-suggested candidates plus top 10 positive target-influence candidates
- Per-iteration reranking budget: top 5 candidates
- Stop condition: `stagnant >= 8`
- Accept rule: combined score must be positive and trial completeness must be at least `bestC * 0.995`

## Pathway Completion Parameters

- Method: Pathway Completion
- `maxPasses`: `3`
- Candidate budget per pass: top 30 pathway candidates
- Accept rule: trial completeness must remain at least `currentC * 0.98`

## Causal Validation Parameters

- Endpoint: `/steer`
- `n_tokens`: `1`
- `top_k`: `50`
- `temperature`: `0`
- `freq_penalty`: `0`
- `seed`: `42`
- `freeze_attention`: `false`
- Request timeout: `120000ms`
- Delay before causal validation: `1500ms`

## Eval Scope

- Prompt set: all prompts in `ALL_PROMPTS`
- Prompt count at launch: `62`
- Counterfactual pairs: up to 5 pairs per category