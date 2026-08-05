# Supplemental aligned-replay diagnostics

- Reference generations: `/data/wanglin/KVarN/results/trajectory_pilot_qwen3_4b_n20_65k_v2/kvarn_run1/generations.jsonl`
- Reference source run: **kvarn_run1**
- Reference source mode: **kvarn**
- Aligned rows: **20480**

## Natural adherence to the forced reference

| Run | Matching raw top-1 | Adherence | Mismatch steps | Samples with mismatch | First mismatch median | P90 |
|---|---:|---:|---:|---:|---:|---:|
| fp16_run1 | 20217/20480 | 0.987158 | 263 | 20 | 227.5 | 371.0 |
| fp16_run2 | 20217/20480 | 0.987158 | 263 | 20 | 227.5 | 371.0 |
| kvarn_run1 | 20405/20480 | 0.996338 | 75 | 19 | 392.0 | 592.0 |
| kvarn_run2 | 20405/20480 | 0.996338 | 75 | 19 | 392.0 | 592.0 |

## Cross-mode flip probability by FP16 decision margin

| Margin bucket | run1 flips | run1 rate | run2 flips | run2 rate |
|---|---:|---:|---:|---:|
| [0,0.05) | 57/120 | 0.475000 | 57/120 | 0.475000 |
| [0.05,0.1) | 54/155 | 0.348387 | 54/155 | 0.348387 |
| [0.1,0.2) | 55/210 | 0.261905 | 55/210 | 0.261905 |
| [0.2,0.5) | 63/631 | 0.099842 | 63/631 | 0.099842 |
| [0.5,1) | 29/1052 | 0.027567 | 29/1052 | 0.027567 |
| [1,2) | 4/1721 | 0.002324 | 4/1721 | 0.002324 |
| [2,5) | 1/3136 | 0.000319 | 1/3136 | 0.000319 |
| [5,+inf) | 0/13455 | 0.000000 | 0/13455 | 0.000000 |

## Warnings

- The aligned kvarn_run1 engine does not naturally reproduce the source trajectory at 75 steps; the forced path remains valid, but it is not an exact natural replay under the aligned runner.

`reference_mismatch_steps.csv` contains every step where at least one aligned engine would not naturally choose the forced source token.
`cross_disagreement_steps.csv` contains every FP16/KVarN raw-top1 flip.
