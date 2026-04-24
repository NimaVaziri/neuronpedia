# Gemma 2 2B IA+PC Circuit Build And Validate Eval

Runner: `apps/graph/eval/eval_runner.py`

Scorer device: `cpu`

Causal validation device: `mps`

Validation graph server: `http://127.0.0.1:5004`

Command:

Run this from the repo root. The `STEER_SECRET` value was read from the local webapp runtime env and is intentionally not shown here.

```bash
cd apps/graph/eval

PYTHONUNBUFFERED=1 \
PYTHONPATH=.. \
MODEL_ID=google/gemma-2-2b \
STEER_URL=http://127.0.0.1:5004 \
EVAL_SCORER_DEVICE=cpu \
STEER_SECRET=<local-graph-secret> \
../.venv/bin/python eval_runner.py \
  --method ia_pc \
  --prompts all \
  --device cpu \
  --output models/gemma-2-2b/results/native-ia_pc.json
```

## Summary

This run measured both circuit construction and causal validation. Causal validation used the graph server's steer path to compute baseline target probability, necessity, and sufficiency for each discovered circuit.

Validation was overlapped with subsequent circuit builds by the eval runner, so total wall time is lower than the serial load + build + steer estimate.

| Metric | Value |
| --- | ---: |
| Prompts / circuits | 62 |
| Method | `ia_pc` |
| Alpha | 0.15 |
| Total wall time | 368.7s |
| Total wall time (human) | 6m 8.7s |
| Total graph load time | 7.1s |
| Avg graph load time | 0.11s / graph |
| Total circuit build time | 359.6s |
| Avg circuit build time | 5.80s / circuit |
| Total steer time | 91.0s |
| Avg steer time | 1.47s / circuit |
| Serial estimate | 457.7s |
| Serial estimate (human) | 7m 37.7s |

## Validation Summary

Necessity values in this summary use absolute deltas, matching the eval runner's aggregate reporting.

| Metric | Value |
| --- | ---: |
| Avg necessity | 35.7pp |
| Median necessity | 33.7pp |
| Avg sufficiency | 84.2% |
| Median sufficiency | 87.7% |

## Category Breakdown

| Category | N | Avg build time | Avg steer time | Avg features | Avg necessity | Avg sufficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| antonym | 5 | 3.14s | 1.30s | 261.6 | 16.0pp | 103.6% |
| conditional-reasoning | 5 | 5.12s | 1.47s | 238.8 | 33.8pp | 77.1% |
| cross-lingual | 5 | 3.20s | 1.34s | 285.2 | 64.6pp | 97.8% |
| factual-recall | 5 | 4.24s | 1.69s | 274.6 | 56.4pp | 77.5% |
| irregular-morphology | 6 | 3.52s | 1.33s | 216.0 | 59.8pp | 81.6% |
| multi-step | 16 | 8.00s | 1.53s | 251.9 | 28.7pp | 66.7% |
| negation | 7 | 6.72s | 1.52s | 246.1 | 35.9pp | 95.8% |
| syntactic-agreement | 8 | 4.88s | 1.36s | 283.8 | 26.7pp | 98.5% |
| transitive-reasoning | 5 | 9.20s | 1.62s | 172.2 | 15.5pp | 85.1% |

## Fastest Circuit Builds

| Prompt | Category | Build time | Steer time | Prompt elapsed | Features | Necessity | Sufficiency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `opposite-up-down` | antonym | 2.36s | 1.33s | 8.37s | 234 | 46.7pp | 85.0% |
| `ciao-italian` | cross-lingual | 2.61s | 1.20s | 6.46s | 292 | 60.7pp | 103.5% |
| `plural-mice` | irregular-morphology | 2.89s | 1.36s | 8.17s | 171 | 59.5pp | 67.3% |
| `plural-children` | irregular-morphology | 2.93s | 1.35s | 6.28s | 210 | 53.0pp | 98.9% |
| `hola-spanish` | cross-lingual | 3.05s | 1.22s | 5.73s | 255 | 79.3pp | 91.7% |

## Slowest Circuit Builds

| Prompt | Category | Build time | Steer time | Prompt elapsed | Features | Necessity | Sufficiency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bmw-euro` | multi-step | 12.62s | 1.53s | 20.37s | 295 | 35.5pp | 82.3% |
| `tajmahal-hindi` | multi-step | 11.00s | 1.47s | 21.52s | 252 | 5.0pp | 135.5% |
| `transitive-distance` | transitive-reasoning | 10.92s | 1.59s | 12.77s | 195 | 50.8pp | 87.7% |
| `colosseum-italian` | multi-step | 10.77s | 1.46s | 22.27s | 294 | 26.6pp | 64.5% |
| `transitive-people` | transitive-reasoning | 10.47s | 1.63s | 18.92s | 175 | 10.5pp | 96.3% |

## Slowest End-To-End Prompts

`Prompt elapsed` measures the time from the beginning of graph load for that prompt until its validation result completed. Because validation ran asynchronously while later prompts were built, this is not a purely serial per-prompt runtime.

| Prompt | Category | Build time | Steer time | Prompt elapsed | Features | Necessity | Sufficiency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `toyota-yen` | multi-step | 9.96s | 1.60s | 23.26s | 275 | 15.9pp | 79.8% |
| `colosseum-italian` | multi-step | 10.77s | 1.46s | 22.27s | 294 | 26.6pp | 64.5% |
| `tajmahal-hindi` | multi-step | 11.00s | 1.47s | 21.52s | 252 | 5.0pp | 135.5% |
| `london-english` | multi-step | 9.59s | 1.56s | 20.83s | 252 | 8.1pp | 162.2% |
| `bmw-euro` | multi-step | 12.62s | 1.53s | 20.37s | 295 | 35.5pp | 82.3% |

## Slowest Causal Validation Calls

| Prompt | Category | Build time | Steer time | Prompt elapsed | Features | Necessity | Sufficiency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `capital-france` | factual-recall | 4.29s | 3.10s | 8.77s | 293 | 56.2pp | 57.5% |
| `dallas-austin` | multi-step | 4.89s | 1.81s | 10.33s | 186 | 33.6pp | 9.0% |
| `transitive-abc` | transitive-reasoning | 9.51s | 1.78s | 17.15s | 106 | 2.0pp | 98.3% |
| `samsung-won` | multi-step | 8.38s | 1.70s | 17.69s | 266 | 26.8pp | 143.1% |
| `sequence-not-finite` | negation | 8.81s | 1.65s | 12.70s | 267 | 37.4pp | 53.4% |
