# HumanEval system-consequence summary

## Overall

| Metric | Value |
|---|---:|
| Valid paired requests | 164 |
| Uncensored pairs | 139 |
| Divergence rate | 1.0000 |
| FP16 pass@1 | 0.8415 |
| KVarN pass@1 | 0.8963 |
| Mean TIR | 0.0556 |
| Median TIR | 0.0066 |
| P95 TIR | 0.7390 |
| Fraction TIR > 10% | 0.3381 |
| Fraction TIR > 50% | 0.1151 |
| Mean E2E LIR | 0.1468 |
| Mean decode LIR | 0.1463 |
| pass→pass uncensored pairs | 136 |

## Correlations (Spearman)

| Relation | n | rho |
|---|---:|---:|
| Normalized first divergence vs TIR | 139 | 0.1630 |
| TIR vs E2E LIR | 139 | 0.9991 |
| TIR vs decode LIR | 139 | 0.9991 |

## By divergence group

| Group | Samples | Uncensored | Mean TIR | Mean decode LIR |
|---|---:|---:|---:|---:|
| early_[0,0.25) | 156 | 131 | 0.0414 | 0.1301 |
| late_[0.5,1+] | 2 | 2 | 0.5975 | 0.7534 |
| middle_[0.25,0.5) | 6 | 6 | 0.1849 | 0.2974 |
