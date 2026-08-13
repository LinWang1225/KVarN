# HumanEval system-consequence summary

## Overall

| Metric | Value |
|---|---:|
| Valid paired requests | 3 |
| Uncensored pairs | 2 |
| Divergence rate | 1.0000 |
| FP16 pass@1 | 0.6667 |
| KVarN pass@1 | 0.6667 |
| Mean TIR | 1.1539 |
| Median TIR | 1.1539 |
| P95 TIR | 1.5160 |
| Fraction TIR > 10% | 1.0000 |
| Fraction TIR > 50% | 1.0000 |
| Mean E2E LIR | 1.3626 |
| Mean decode LIR | 1.3658 |
| pass→pass uncensored pairs | 2 |

## Correlations (Spearman)

| Relation | n | rho |
|---|---:|---:|
| Normalized first divergence vs TIR | 2 | 1.0000 |
| TIR vs E2E LIR | 2 | 1.0000 |
| TIR vs decode LIR | 2 | 1.0000 |

## By divergence group

| Group | Samples | Uncensored | Mean TIR | Mean decode LIR |
|---|---:|---:|---:|---:|
| early_[0,0.25) | 3 | 2 | 1.1539 | 1.3658 |
