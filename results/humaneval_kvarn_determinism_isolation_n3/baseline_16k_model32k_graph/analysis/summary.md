# Repeat determinism: baseline_16k_model32k_graph

## Configuration

- max_tokens: `16384`
- max_model_len: `32768`
- enforce_eager: `False`
- decoding: greedy (`temperature=0`, `top_p=1`, `top_k=-1`)

## Summary

| Metric | Value |
|---|---:|
| Valid repeat pairs | 3 |
| Self-diverged | 3 |
| Self-divergence rate | 1.0000 |
| Diverged before step 256 | 3 |
| Diverged before step 1024 | 3 |
| Diverged before step 16384 | 3 |
| Median first divergence step | 136.0 |
| Either repeat hit max_tokens | 1 |
| Finish-reason mismatches | 1 |
| Pass/fail mismatches | 1 |

## Per sample

| Sample | Diverged | First diff | Run1 tokens | Run2 tokens | Run1 finish | Run2 finish | Pass match | Same code |
|---|---:|---:|---:|---:|---|---|---:|---:|
| HumanEval/0 | True | 136 | 2549 | 2678 | stop | stop | True | False |
| HumanEval/1 | True | 123 | 16384 | 13753 | length | stop | False | False |
| HumanEval/2 | True | 250 | 4380 | 4044 | stop | stop | True | False |
