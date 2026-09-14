# Repeat determinism: model40k_16k_graph

## Configuration

- max_tokens: `16384`
- max_model_len: `40960`
- enforce_eager: `False`
- decoding: greedy (`temperature=0`, `top_p=1`, `top_k=-1`)

## Summary

| Metric | Value |
|---|---:|
| Valid repeat pairs | 3 |
| Self-diverged | 3 |
| Self-divergence rate | 1.0000 |
| Diverged before step 256 | 2 |
| Diverged before step 1024 | 3 |
| Diverged before step 16384 | 3 |
| Median first divergence step | 207.0 |
| Either repeat hit max_tokens | 1 |
| Finish-reason mismatches | 0 |
| Pass/fail mismatches | 0 |

## Per sample

| Sample | Diverged | First diff | Run1 tokens | Run2 tokens | Run1 finish | Run2 finish | Pass match | Same code |
|---|---:|---:|---:|---:|---|---|---:|---:|
| HumanEval/0 | True | 136 | 2488 | 2549 | stop | stop | True | False |
| HumanEval/1 | True | 281 | 16384 | 16384 | length | length | True | False |
| HumanEval/2 | True | 207 | 2971 | 4381 | stop | stop | True | False |
