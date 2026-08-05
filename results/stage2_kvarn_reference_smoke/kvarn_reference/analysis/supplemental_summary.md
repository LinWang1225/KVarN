# Supplemental aligned-replay diagnostics

- Reference generations: `/data/wanglin/KVarN/results/trajectory_pilot_qwen3_4b_n20_65k_v2/kvarn_run1/generations.jsonl`
- Reference source run: **kvarn_run1**
- Reference source mode: **kvarn**
- Aligned rows: **3072**

## Natural adherence to the forced reference

| Run | Matching raw top-1 | Adherence | Mismatch steps | Samples with mismatch | First mismatch median | P90 |
|---|---:|---:|---:|---:|---:|---:|
| fp16_run1 | 3033/3072 | 0.987305 | 39 | 3 | 247.0 | 251.8 |
| fp16_run2 | 3033/3072 | 0.987305 | 39 | 3 | 247.0 | 251.8 |
| kvarn_run1 | 3060/3072 | 0.996094 | 12 | 3 | 331.0 | 450.2 |
| kvarn_run2 | 3060/3072 | 0.996094 | 12 | 3 | 331.0 | 450.2 |

## Cross-mode flip probability by FP16 decision margin

| Margin bucket | run1 flips | run1 rate | run2 flips | run2 rate |
|---|---:|---:|---:|---:|
| [0,0.05) | 9/19 | 0.473684 | 9/19 | 0.473684 |
| [0.05,0.1) | 9/30 | 0.300000 | 9/30 | 0.300000 |
| [0.1,0.2) | 9/32 | 0.281250 | 9/32 | 0.281250 |
| [0.2,0.5) | 12/113 | 0.106195 | 12/113 | 0.106195 |
| [0.5,1) | 2/184 | 0.010870 | 2/184 | 0.010870 |
| [1,2) | 0/291 | 0.000000 | 0/291 | 0.000000 |
| [2,5) | 0/535 | 0.000000 | 0/535 | 0.000000 |
| [5,+inf) | 0/1868 | 0.000000 | 0/1868 | 0.000000 |

## Warnings

- The aligned kvarn_run1 engine does not naturally reproduce the source trajectory at 12 steps; the forced path remains valid, but it is not an exact natural replay under the aligned runner.

`reference_mismatch_steps.csv` contains every step where at least one aligned engine would not naturally choose the forced source token.
`cross_disagreement_steps.csv` contains every FP16/KVarN raw-top1 flip.
