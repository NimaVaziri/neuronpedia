# Verified Circuits vs CircuitExplorer

- Generated at: `2026-04-24T16:47:42.107331Z`
- Model: `google/gemma-2-2b`
- Steer URL: `http://127.0.0.1:5004`
- Scorer device: `cuda`
- CircuitExplorer IA steps: `40`
- CircuitExplorer PC passes: `1`
- Circuits evaluated: `15`
- Errors: `0`
- Canonical run selection: this is run 3 from a 5-run batch. It was selected as the median run because it has the smallest absolute distance to the batch median summary metrics: verified necessity `42.5pp`, verified sufficiency `50.8%`, CircuitExplorer necessity `46.1pp`, and CircuitExplorer sufficiency `62.3%`.

## Median Scores

| Circuit family | Median necessity | Median sufficiency |
| --- | ---: | ---: |
| Verified circuits | 42.5pp | 51.2% |
| CircuitExplorer circuits | 46.1pp | 62.3% |

## Per-Circuit Results

| ID | Target | Baseline Target Prob | Verified Necessity | CircuitExplorer Necessity | Verified Sufficiency | CircuitExplorer Sufficiency | Verified Features | CircuitExplorer Features |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| girls-are | `are` | 16.8% | 5.4pp | 4.9pp | 42.4% | 56.7% | 21 | 75 |
| G | `G` | 93.8% | 78.1pp | 90.9pp | 54.0% | 79.1% | 78 | 75 |
| gp-nps | `was` | 14.5% | 5.5pp | 4.6pp | 212.2% | 112.2% | 24 | 58 |
| euro | `euro` | 25.2% | 17.3pp | 17.7pp | 176.0% | 229.5% | 129 | 72 |
| keys-cabinet | `are` | 29.9% | 8.7pp | 10.4pp | 47.4% | 72.9% | 15 | 75 |
| michael-jordan | `basketball` | 82.8% | 57.2pp | 46.1pp | 69.8% | 84.4% | 166 | 67 |
| saison | `été` | 62.9% | 52.6pp | 60.9pp | 0.5% | 1.1% | 61 | 50 |
| verano | `verano` | 51.6% | 42.5pp | 49.2pp | 0.0% | 0.0% | 198 | 48 |
| girl-is | `is` | 16.9% | 2.3pp | 1.9pp | 51.2% | 60.2% | 25 | 75 |
| basket | `basket` | 56.2% | 52.7pp | 46.4pp | 34.4% | 1.3% | 34 | 46 |
| michael-jordan-es | `baloncesto` | 60.5% | 61.0pp | 52.3pp | 10.9% | 0.0% | 30 | 75 |
| addition2 | `3` | 77.3% | 46.7pp | 59.6pp | 54.0% | 85.9% | 27 | 75 |
| addition | `8` | 70.7% | 45.9pp | 61.2pp | 51.9% | 62.3% | 27 | 75 |
| english | `English` | 15.2% | 8.1pp | 9.5pp | 57.7% | 108.4% | 49 | 75 |
| dollar | `dollar` | 36.7% | 26.2pp | 19.5pp | 42.0% | 39.6% | 38 | 75 |

## Canonical Run Selection

This report is run 3 from a controlled 5-run batch. The canonical run was selected by comparing each run's `Median Scores` table to the batch median summary metrics and choosing the run with the smallest total absolute distance.

| Summary metric | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Batch median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Verified necessity | 42.5pp | 42.5pp | 42.5pp | 42.5pp | 42.5pp | 42.5pp |
| Verified sufficiency | 51.4% | 50.8% | 51.2% | 50.8% | 50.5% | 50.8% |
| CircuitExplorer necessity | 46.1pp | 46.1pp | 46.1pp | 46.1pp | 46.1pp | 46.1pp |
| CircuitExplorer sufficiency | 62.3% | 64.8% | 62.3% | 62.8% | 61.7% | 62.3% |

Median calculations:

- Verified necessity values were all `42.5pp`, so the batch median is `42.5pp`.
- Verified sufficiency sorted values were `50.5%`, `50.8%`, `50.8%`, `51.2%`, `51.4%`, so the batch median is `50.8%`.
- CircuitExplorer necessity values were all `46.1pp`, so the batch median is `46.1pp`.
- CircuitExplorer sufficiency sorted values were `61.7%`, `62.3%`, `62.3%`, `62.8%`, `64.8%`, so the batch median is `62.3%`.

Run 3 distance to the batch medians:

| Summary metric | Run 3 value | Batch median | Absolute distance |
| --- | ---: | ---: | ---: |
| Verified necessity | 42.5pp | 42.5pp | 0.0 |
| Verified sufficiency | 51.2% | 50.8% | 0.4 |
| CircuitExplorer necessity | 46.1pp | 46.1pp | 0.0 |
| CircuitExplorer sufficiency | 62.3% | 62.3% | 0.0 |
| Total absolute distance |  |  | 0.4 |

Run 3 had the smallest total absolute distance across the five runs, so it is the canonical comparison run for reporting.
