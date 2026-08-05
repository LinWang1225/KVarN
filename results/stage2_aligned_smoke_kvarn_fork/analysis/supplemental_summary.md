# Supplemental aligned-replay diagnostics

- Reference generations: `/data/wanglin/KVarN/results/trajectory_pilot_qwen3_4b_n20_65k_v2/fp16_run1/generations.jsonl`
- Reference source run: **fp16_run1**
- Reference source mode: **fp16**
- Aligned rows: **3072**

## Natural adherence to the forced reference

| Run | Matching raw top-1 | Adherence | Mismatch steps | Samples with mismatch | First mismatch median | P90 |
|---|---:|---:|---:|---:|---:|---:|
| fp16_run1 | 3066/3072 | 0.998047 | 6 | 3 | 421.0 | 469.8 |
| fp16_run2 | 3066/3072 | 0.998047 | 6 | 3 | 421.0 | 469.8 |
| kvarn_run1 | 3036/3072 | 0.988281 | 36 | 3 | 247.0 | 251.8 |
| kvarn_run2 | 3036/3072 | 0.988281 | 36 | 3 | 247.0 | 251.8 |

## Cross-mode flip probability by FP16 decision margin

| Margin bucket | run1 flips | run1 rate | run2 flips | run2 rate |
|---|---:|---:|---:|---:|
| [0,0.05) | 5/20 | 0.250000 | 5/20 | 0.250000 |
| [0.05,0.1) | 8/21 | 0.380952 | 8/21 | 0.380952 |
| [0.1,0.2) | 6/31 | 0.193548 | 6/31 | 0.193548 |
| [0.2,0.5) | 9/81 | 0.111111 | 9/81 | 0.111111 |
| [0.5,1) | 1/148 | 0.006757 | 1/148 | 0.006757 |
| [1,2) | 0/248 | 0.000000 | 0/248 | 0.000000 |
| [2,5) | 1/447 | 0.002237 | 1/447 | 0.002237 |
| [5,+inf) | 0/2076 | 0.000000 | 0/2076 | 0.000000 |

## Warnings

- The aligned fp16_run1 engine does not naturally reproduce the source trajectory at 6 steps; the forced path remains valid, but it is not an exact natural replay under the aligned runner.

`reference_mismatch_steps.csv` contains every step where at least one aligned engine would not naturally choose the forced source token.
`cross_disagreement_steps.csv` contains every FP16/KVarN raw-top1 flip.
