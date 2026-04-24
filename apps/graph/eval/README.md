# Graph Eval

This directory contains model-scoped input graphs, result artifacts, and notes for attribution-graph circuit discovery evals.

The supported eval entrypoint is `apps/graph/eval/eval_runner.py`. Older TypeScript eval runners and helper libraries were removed so this folder does not maintain a second eval stack.

## Directory Structure

- `README.md`: this guide.
- `models/README.md`: notes for model-scoped eval artifacts.
- `models/<model>/graphs`: local graph JSON inputs, when available.
- `models/<model>/results`: generated eval reports and intermediate outputs.
- `models/<model>/*.md`: model-specific run notes and parameter records.
- `models/<model>/scripts`: optional model-local launchers that directly run the supported Python eval runner.
- `eval_runner.py`: native Python eval runner.
- `verified-circuits/verified-circuits.json`: structured researcher-verified circuit fixtures.
- `verified-circuits/verified-circuits.schema.json`: JSON Schema for `verified-circuits.json`.

## Eval Flow

The Python eval runner follows this pipeline:

1. Select prompts from `eval_runner.py`.
2. Load graph JSON from `EVAL_GRAPH_DIR`, `models/<model>/graphs`, a configured graph URL prefix, or the prompt's `graphUrl`.
3. Focus the graph on the target logit by zeroing non-target logit probabilities.
    - This makes scoring target-specific instead of letting other plausible output logits influence the selected circuit.
4. Build candidate circuits with the Python scorer and the selected method.
5. Score circuits with `neuronpedia_graph.scorer`.
6. Optionally call the graph server's steer endpoints to measure necessity and sufficiency.
7. Write outputs to `models/<model>/results`.

## Native Python Eval Runner

`apps/graph/eval/eval_runner.py` builds circuits with the Python scorer directly and only uses HTTP for causal validation steer calls. This is useful when you want the eval loop to avoid model contention with graph construction.

Run it from `apps/graph/eval`, the directory containing this README:

```bash
cd apps/graph/eval
PYTHONPATH=.. ../.venv/bin/python eval_runner.py --method ia_pc --prompts all
```

Useful options:

- `--method ia_pc`: influence-aware build plus pathway completion.
- `--method ia`: influence-aware build without pathway completion.
- `--method c_only`: completeness-only build.
- `--prompts all`: run all configured prompts.
- `--prompts capital-france,dallas-austin`: run selected prompt IDs.
- `--skip-steer`: build circuits without causal validation.
- `--feature-budget`: adaptive target count for feature pruning before scoring.
- `--device auto`: scorer device selection. `auto` prefers CUDA, then MPS, then CPU; use `--device cpu`, `--device mps`, or `--device cuda` to force a device.
- `--output`: explicit output JSON path.

Python runner environment variables:

- `STEER_URL`: graph server URL for `/steer` and `/steer-batch`, default `http://127.0.0.1:5005`.
- `STEER_SECRET`: secret sent as `x-secret-key`, default `SECRET`.
- `MODEL_ID`: model id for result paths and steer requests.
- `GRAPH_BASE_URL`: base URL used for relative graph URLs, default `http://localhost:3000`.
- `EVAL_GRAPH_DIR`: local graph JSON override directory.
- `EVAL_GRAPH_URL_PREFIX`: remote graph JSON prefix override.

Example:

```bash
cd apps/graph/eval
STEER_URL=http://127.0.0.1:5005 \
STEER_SECRET=SECRET \
MODEL_ID=google/gemma-2-2b \
PYTHONPATH=.. \
../.venv/bin/python eval_runner.py --method ia_pc --prompts capital-france --skip-steer
```

## Result Artifacts

Results are grouped by sanitized model id:

```text
apps/graph/eval/models/<model-id>/graphs
apps/graph/eval/models/<model-id>/results
```

For example, `google/gemma-2-2b` maps to `models/gemma-2-2b`.

Large result files and generated graph JSONs should be treated as eval artifacts. Commit them only when they are intentionally part of the benchmark record.

## Verified Circuits

`verified-circuits/verified-circuits.json` is the canonical structured form of the researcher-verified circuit list. Each entry records the source Neuronpedia graph URL, prompt, target token, decoded pinned nodes, clicked node, supernodes, CLERP labels, and view thresholds.

Use `verified-circuits/verified-circuits.schema.json` when validating or transforming the fixture outside Python. The schema intentionally treats node ids as opaque strings because current verified examples include several historical id formats.

Unit tests under `apps/graph/tests` check the JSON fixture's shape, decoded source URL state, schema metadata, and representative circuit values.

## Notes

- Unit coverage for the supported eval runner lives under `apps/graph/tests`.
- Standalone TypeScript research runners and helper modules were intentionally removed from this folder. Use `apps/graph/eval/eval_runner.py` as the supported eval entrypoint.
