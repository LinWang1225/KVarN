# HumanEval sampling validation — thinking_on

Primary interpretation: request-level system effects under Qwen3-recommended stochastic sampling. 
Token-level first-divergence is intentionally not used in this arm because sampling is stochastic.

## Paper-style run aggregate

| Metric | FP16 | KVarN K4V2 | Change |
|---|---:|---:|---:|
| Mean output tokens across runs | 3411.6 ± 107.2 | 3526.5 ± 66.1 | 3.37% |
| Mean pass@1 across runs | 94.72% ± 0.35% | 94.11% ± 1.41% | -0.61% points |

## Pooled paired request distribution (uncensored only)

| Metric | Median | Mean | P90 | P95 |
|---|---:|---:|---:|---:|
| Output-token change | -1.18% | 7.28% | 53.43% | 74.30% |
| E2E latency change | 7.32% | 16.80% | 68.23% | 90.96% |
| Decode latency change | 7.20% | 16.71% | 68.29% | 90.79% |
| TPOT change | 8.63% | 8.42% | 9.80% | 10.09% |

### Tail rates

| Request category | Fraction |
|---|---:|
| Output >10% longer | 35.92% |
| Output >25% longer | 23.11% |
| Output >50% longer | 11.97% |
| Output >100% longer | 2.31% |
| Output >10% shorter | 34.24% |
| Output within ±10% | 29.83% |

## Severe-inflation token source

Severe means TIR > 50%.

| Metric | Value |
|---|---:|
| Severe paired requests | 57 |
| Thinking split available | 57 |
| Aggregate extra tokens | 115990 |
| Extra thinking tokens | 108427 |
| Extra final/code tokens | 7563 |
| Thinking share of net extra tokens | 93.48% |
| pass→pass thinking share | 93.03% |

## Cross-seed robustness by task

Only tasks uncensored in all 3 seeds are counted below.

| Stable task-level pattern | Tasks |
|---|---:|
| All seeds uncensored | 154 |
| >10% longer in majority of seeds | 42 |
| >25% longer in majority of seeds | 18 |
| >50% longer in majority of seeds | 4 |
| >10% shorter in majority of seeds | 40 |
| >10% longer in every seed | 9 |
| >50% longer in every seed | 2 |

## Per-seed sanity table

| Seed | FP16 tokens | KVarN tokens | Δ tokens | FP16 pass@1 | KVarN pass@1 | Uncensored | TIR P95 | Decode LIR P95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 3418.7 | 3451.5 | 0.96% | 95.12% | 95.73% | 160 | 60.54% | 76.56% |
| 2027 | 3515.0 | 3576.4 | 1.74% | 94.51% | 93.29% | 158 | 69.29% | 84.94% |
| 2028 | 3301.0 | 3551.5 | 7.59% | 94.51% | 93.29% | 158 | 83.70% | 101.94% |

## Correlation diagnostic

- TIR vs E2E LIR: n=476, rho=0.9993
- TIR vs decode LIR: n=476, rho=0.9993

The per-seed and per-task CSV files are the preferred source for follow-up analysis.
