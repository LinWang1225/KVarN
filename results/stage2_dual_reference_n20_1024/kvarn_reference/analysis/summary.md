# Teacher-forced aligned replay summary

- Samples complete in all four runs: **20**
- Aligned step records: **20480**

## Pairwise raw-logit decisions

| Pair | Samples with top1 disagreement | Rate | Median first step | P90 | Step disagreement rate | Mean abs ref-logprob difference |
|---|---:|---:|---:|---:|---:|---:|
| fp16_self | 0/20 | 0.0000 | N/A | N/A | 0.000000 | 0.000000 |
| kvarn_self | 0/20 | 0.0000 | N/A | N/A | 0.000000 | 0.000000 |
| cross_run1 | 20/20 | 1.0000 | 227.5 | 357.5 | 0.012842 | 0.015389 |
| cross_run2 | 20/20 | 1.0000 | 227.5 | 357.5 | 0.012842 | 0.015389 |

A raw-top1 disagreement is measured before the logits processor masks the vocabulary to the FP16 reference token.
All four runs therefore consume the same reference history even after their unmodified argmax decisions differ.
