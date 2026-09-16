# HumanEval system-consequence summary

## Overall

| Metric | Value |
|---|---:|
| Valid paired requests | 164 |
| Uncensored pairs | 138 |
| Divergence rate | 1.0000 |
| FP16 pass@1 | 0.8415 |
| KVarN pass@1 | 0.8841 |
| Mean TIR | 0.0388 |
| Median TIR | 0.0038 |
| P95 TIR | 0.7110 |
| Fraction TIR > 10% | 0.3116 |
| Fraction TIR > 50% | 0.1014 |
| Mean E2E LIR | 0.1418 |
| Mean decode LIR | 0.1412 |
| pass→pass uncensored pairs | 135 |

## Correlations (Spearman)

| Relation | n | rho |
|---|---:|---:|
| Normalized first divergence vs TIR | 138 | 0.2111 |
| TIR vs E2E LIR | 138 | 0.9982 |
| TIR vs decode LIR | 138 | 0.9984 |

## By divergence group

| Group | Samples | Uncensored | Mean TIR | Mean decode LIR |
|---|---:|---:|---:|---:|
| early_[0,0.25) | 156 | 130 | 0.0235 | 0.1232 |
| late_[0.5,1+] | 2 | 2 | 0.5975 | 0.7836 |
| middle_[0.25,0.5) | 6 | 6 | 0.1849 | 0.3164 |
