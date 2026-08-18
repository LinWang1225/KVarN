# HumanEval sampling validation — thinking_on

Primary interpretation: request-level system effects under Qwen3-recommended stochastic sampling. 
Token-level first-divergence is intentionally not used in this arm because sampling is stochastic.

## Paper-style run aggregate

| Metric | FP16 | KVarN K4V2 | Change |
|---|---:|---:|---:|
| Mean output tokens across runs | 2726.4 ± 125.3 | 3105.2 ± 530.4 | 13.89% |
| Mean pass@1 across runs | 66.67% ± 0.00% | 55.56% ± 19.25% | -11.11% points |

## Pooled paired request distribution (uncensored only)

| Metric | Median | Mean | P90 | P95 |
|---|---:|---:|---:|---:|
| Output-token change | 15.44% | 3.91% | 59.06% | 69.67% |
| E2E latency change | 25.36% | 13.06% | 73.41% | 85.16% |
| Decode latency change | 25.39% | 13.07% | 73.55% | 85.33% |
| TPOT change | 8.61% | 8.67% | 9.03% | 9.17% |

### Tail rates

| Request category | Fraction |
|---|---:|
| Output >10% longer | 60.00% |
| Output >25% longer | 40.00% |
| Output >50% longer | 20.00% |
| Output >100% longer | 0.00% |
| Output >10% shorter | 40.00% |
| Output within ±10% | 0.00% |

## Severe-inflation token source

Severe means TIR > 50%.

| Metric | Value |
|---|---:|
| Severe paired requests | 1 |
| Thinking split available | 1 |
| Aggregate extra tokens | 1607 |
| Extra thinking tokens | 1758 |
| Extra final/code tokens | -151 |
| Thinking share of net extra tokens | 109.40% |
| pass→pass thinking share | 109.40% |

## Cross-seed robustness by task

Only tasks uncensored in all 3 seeds are counted below.

| Stable task-level pattern | Tasks |
|---|---:|
| All seeds uncensored | 1 |
| >10% longer in majority of seeds | 1 |
| >25% longer in majority of seeds | 1 |
| >50% longer in majority of seeds | 0 |
| >10% shorter in majority of seeds | 0 |
| >10% longer in every seed | 1 |
| >50% longer in every seed | 0 |

## Per-seed sanity table

| Seed | FP16 tokens | KVarN tokens | Δ tokens | FP16 pass@1 | KVarN pass@1 | Uncensored | TIR P95 | Decode LIR P95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 2716.0 | 3717.3 | 36.87% | 66.67% | 33.33% | 1 | 15.44% | 25.39% |
| 2027 | 2606.7 | 2816.7 | 8.06% | 66.67% | 66.67% | 2 | 73.42% | 89.59% |
| 2028 | 2856.7 | 2781.7 | -2.63% | 66.67% | 66.67% | 2 | 23.55% | 34.19% |

## Correlation diagnostic

- TIR vs E2E LIR: n=5, rho=1.0000
- TIR vs decode LIR: n=5, rho=1.0000

The per-seed and per-task CSV files are the preferred source for follow-up analysis.
