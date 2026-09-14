# HumanEval system-consequence summary

## Overall

| Metric | Value |
|---|---:|
| Valid paired requests | 3 |
| Uncensored pairs | 2 |
| Divergence rate | 1.0000 |
| FP16 pass@1 | 1.0000 |
| KVarN pass@1 | 0.6667 |
| Mean TIR | 1.2182 |
| Median TIR | 1.2182 |
| P95 TIR | 1.7142 |
| Fraction TIR > 10% | 1.0000 |
| Fraction TIR > 50% | 1.0000 |
| Mean E2E LIR | 1.4605 |
| Mean decode LIR | 1.4639 |
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
| early_[0,0.25) | 3 | 2 | 1.2182 | 1.4639 |
