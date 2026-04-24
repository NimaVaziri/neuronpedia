# Gemma 2 2B IA+PC Circuit Build-Only Eval

Runner: `apps/graph/eval/eval_runner.py`

Scorer device: `cpu`

Command:

Run this from `apps/graph/eval`, the directory containing the eval README.

```bash
env PYTHONPATH=.. \
  MODEL_ID=google/gemma-2-2b \
  ../.venv/bin/python \
  eval_runner.py \
  --method ia_pc \
  --prompts all \
  --skip-steer \
  --output /tmp/gemma-eval-suite-skip-steer.json
```

## Summary

This run measured circuit construction time only. Causal validation was not a goal for this run.

| Metric | Value |
| --- | ---: |
| Prompts / circuits | 62 |
| Method | `ia_pc` |
| Alpha | 0.15 |
| Total wall time | 385.3s |
| Total wall time (human) | 6m 25.3s |
| Total graph load time | 10.1s |
| Avg graph load time | 0.16s / graph |
| Total circuit build time | 374.8s |
| Avg circuit build time | 6.05s / circuit |
| Total steer time | 0.0s |

## MPS Smoke Test

After adding explicit scorer device selection to the runner, this smoke command was run from `apps/graph/eval`:

```bash
env PYTHONPATH=.. \
  MODEL_ID=google/gemma-2-2b \
  ../.venv/bin/python \
  eval_runner.py \
  --method ia_pc \
  --prompts capital-france \
  --skip-steer \
  --device mps \
  --output /tmp/gemma-eval-mps-smoke.json
```

| Metric | CPU suite value for `capital-france` | MPS smoke value |
| --- | ---: | ---: |
| Build time | 4.30s | 104.37s |
| Graph load time | 0.09s | 0.09s |
| Features | 253 | 253 |
| IA features | 165 | 165 |
| PC features | 88 | 88 |
| Wall time | N/A | 104.46s |

The full 62-prompt MPS suite was not run because the smoke result implies a multi-hour run and is already substantially slower than CPU for the same circuit.

The likely reason is workload shape relative to tensor size. For `capital-france`, the pruned scorer graph had hundreds of nodes, but the greedy search mostly scored tiny candidate batches: profiling showed `score_batch` was called 163 times, with 160 calls scoring only 5 candidate circuits and 3 calls scoring 30. That creates many small MPS kernel launches and frequent Python/device synchronization, which can dominate the actual math. CPU handles this small-batch, Python-driven loop more efficiently.

MPS could become more competitive if the circuit-building path were changed to present larger tensor workloads to the accelerator. For example, a larger circuit or graph would not automatically help if the loop still scores only 5 candidates per dispatch. But if a larger search were batched into hundreds or thousands of candidate masks per scoring pass, kept tensors resident on MPS, and avoided repeated `.item()` synchronization, MPS could amortize launch overhead and use more parallelism. In that shape, larger tensors may help; in the current implementation, larger graphs are more likely to make both CPU and MPS slower, with MPS still paying the same small-batch overhead.

The same workload-shape caveat applies to CUDA GPUs too. CUDA may handle this pattern better than MPS because PyTorch's CUDA kernels and synchronization paths are generally more mature, but the current scorer would still be limited by many small launches, Python control flow, and host synchronization. A GPU is most likely to help after the scorer is redesigned around larger batched candidate scoring.

## Category Breakdown

| Category | N | Avg build time | Avg features |
| --- | ---: | ---: | ---: |
| antonym | 5 | 3.19s | 264.2 |
| conditional-reasoning | 5 | 4.80s | 238.4 |
| cross-lingual | 5 | 3.13s | 274.0 |
| factual-recall | 5 | 4.60s | 274.6 |
| irregular-morphology | 6 | 3.21s | 215.8 |
| multi-step | 16 | 8.74s | 256.6 |
| negation | 7 | 7.61s | 246.0 |
| syntactic-agreement | 8 | 4.62s | 275.4 |
| transitive-reasoning | 5 | 9.38s | 197.4 |

## Fastest Circuits

| Prompt | Category | Build time | Features |
| --- | --- | ---: | ---: |
| `plural-mice` | irregular-morphology | 1.76s | 150 |
| `bonjour-french` | cross-lingual | 2.02s | 214 |
| `opposite-hot-cold` | antonym | 2.08s | 214 |
| `dallas-austin` | multi-step | 2.25s | 278 |
| `ciao-italian` | cross-lingual | 2.83s | 293 |

## Slowest Circuits

| Prompt | Category | Build time | Features |
| --- | --- | ---: | ---: |
| `transitive-distance` | transitive-reasoning | 14.26s | 215 |
| `bmw-euro` | multi-step | 13.29s | 295 |
| `tajmahal-hindi` | multi-step | 13.25s | 252 |
| `colosseum-italian` | multi-step | 12.99s | 294 |
| `transitive-people` | transitive-reasoning | 12.43s | 175 |
