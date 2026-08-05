# Supplemental aligned-replay diagnostics

- Reference generations: `/data/wanglin/KVarN/results/trajectory_pilot_qwen3_4b_n20_65k_v2/fp16_run1/generations.jsonl`
- Reference source run: **fp16_run1**
- Reference source mode: **fp16**
- Aligned rows: **20480**

## Natural adherence to the forced reference

| Run | Matching raw top-1 | Adherence | Mismatch steps | Samples with mismatch | First mismatch median | P90 |
|---|---:|---:|---:|---:|---:|---:|
| fp16_run1 | 20457/20480 | 0.998877 | 23 | 15 | 421.0 | 744.2 |
| fp16_run2 | 20457/20480 | 0.998877 | 23 | 15 | 421.0 | 744.2 |
| kvarn_run1 | 20229/20480 | 0.987744 | 251 | 20 | 227.5 | 357.5 |
| kvarn_run2 | 20229/20480 | 0.987744 | 251 | 20 | 227.5 | 357.5 |

## Cross-mode flip probability by FP16 decision margin

| Margin bucket | run1 flips | run1 rate | run2 flips | run2 rate |
|---|---:|---:|---:|---:|
| [0,0.05) | 59/138 | 0.427536 | 59/138 | 0.427536 |
| [0.05,0.1) | 48/128 | 0.375000 | 48/128 | 0.375000 |
| [0.1,0.2) | 44/203 | 0.216749 | 44/203 | 0.216749 |
| [0.2,0.5) | 60/630 | 0.095238 | 60/630 | 0.095238 |
| [0.5,1) | 27/1007 | 0.026812 | 27/1007 | 0.026812 |
| [1,2) | 5/1607 | 0.003111 | 5/1607 | 0.003111 |
| [2,5) | 1/3093 | 0.000323 | 1/3093 | 0.000323 |
| [5,+inf) | 0/13674 | 0.000000 | 0/13674 | 0.000000 |

## Warnings

- The aligned fp16_run1 engine does not naturally reproduce the source trajectory at 23 steps; the forced path remains valid, but it is not an exact natural replay under the aligned runner.

`reference_mismatch_steps.csv` contains every step where at least one aligned engine would not naturally choose the forced source token.
`cross_disagreement_steps.csv` contains every FP16/KVarN raw-top1 flip.
