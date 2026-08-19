# HumanEval TurboQuant precision sweep

Methods are ordered from higher to lower nominal KV precision. Signed TIR and absolute TIR are both shown because longer/shorter trajectories can cancel in the mean.

| Method | K/V | Mean output Δ | Median TIR | P95 TIR | Mean |TIR| | P95 |TIR| | TIR>50% | |TIR|>50% | Acc drop | Mean TPOT Δ | Stable >50% (majority seeds) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| turboquant_k8v4 | 8/4 | 49.96% | NA | NA | NA | NA | NA | NA | 66.67 pp | NA | 0/0 |
| turboquant_4bit_nc | 4/4 | 17.21% | 23.97% | 88.80% | 48.35% | 88.80% | 50.00% | 50.00% | -11.11 pp | 8.73% | 1/1 |
| turboquant_k3v4_nc | 3/4 | -6.00% | -19.90% | 29.76% | 37.90% | 57.81% | 0.00% | 33.33% | 0.00 pp | 7.60% | 0/2 |
| turboquant_3bit_nc | 3/3 | 7.18% | -9.10% | 64.79% | 34.00% | 76.58% | 20.00% | 40.00% | 11.11 pp | 8.55% | 0/1 |

## Severe-inflation source (TIR > 50%)

| Method | Severe pairs | Pass→pass severe | Thinking share of extra tokens |
|---|---:|---:|---:|
| turboquant_k8v4 | 0 | 0 | NA |
| turboquant_4bit_nc | 2 | 2 | 104.42% |
| turboquant_k3v4_nc | 0 | 0 | NA |
| turboquant_3bit_nc | 1 | 1 | 106.15% |

## Does lower precision produce a stronger effect?

Positive Spearman rho means the metric tends to increase as precision becomes more aggressive (rank 1 → 4). With only four presets, treat this as a descriptive trend, not a significance test.

| Metric | Spearman ρ vs severity | Strictly non-decreasing? |
|---|---:|---:|
| mean_abs_TIR | NA | None |
| P95_abs_TIR | NA | None |
| TIR_gt_50_rate | NA | None |
| abs_TIR_gt_50_rate | NA | None |
| accuracy_drop_points | -0.200 | False |
| mean_TPOT_overhead | NA | None |

## Task-level monotonicity

- Tasks uncensored for all seeds and all methods: **0**
- Strictly non-decreasing mean |TIR| from high→low precision: **0**
- Non-decreasing within 5 percentage-point tolerance: **0**
- Mean per-task Spearman(severity, mean |TIR|): **NA**

Interpretation priority: first inspect mean/P95 |TIR| and |TIR|>50%; then positive-only inflation, correctness, and TPOT. If absolute disturbance grows monotonically while signed mean TIR stays near zero, lower precision is increasing trajectory instability rather than only making outputs longer.
