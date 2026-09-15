# HumanEval graph/eager 5-repeat determinism matrix

- environment control clean: True
- non-condition environment fingerprints: 1

| condition | stable samples | pairwise divergence | first step min/med/max | finish unstable | pass unstable | cap hits |
|---|---:|---:|---|---:|---:|---:|
| fp16_graph | 3/3 | 0/30 | None/None/None | 0 | 0 | 0 |
| fp16_eager | 3/3 | 0/30 | None/None/None | 0 | 0 | 0 |
| kvarn_graph | 3/3 | 0/30 | None/None/None | 0 | 0 | 0 |
| kvarn_eager | 3/3 | 0/30 | None/None/None | 0 | 0 | 0 |

## Interpretation

- The matrix does not isolate graph execution as the dominant source of KVarN repeat instability.

## Decision rule

- If FP16 graph/eager are stable, KVarN graph is unstable, KVarN eager is stable, and environment control is clean: use eager for the causal trajectory-mechanism experiment.
- Keep normal graph-mode vLLM for production/system-effect validation, but aggregate across seeds/repeats.
- If KVarN eager is also unstable, stop before the 164-task causal run and isolate KVarN kernel/store/decode determinism.
