# Repeat determinism: extended_32k_model40k_eager

## Configuration

- max_tokens: `32768`
- max_model_len: `40960`
- enforce_eager: `True`
- decoding: greedy (`temperature=0`, `top_p=1`, `top_k=-1`)

## Summary

| Metric | Value |
|---|---:|
| Valid repeat pairs | 3 |
| Self-diverged | 0 |
| Self-divergence rate | 0.0000 |
| Diverged before step 256 | 0 |
| Diverged before step 1024 | 0 |
| Diverged before step 16384 | 0 |
| Median first divergence step | None |
| Either repeat hit max_tokens | 0 |
| Finish-reason mismatches | 0 |
| Pass/fail mismatches | 0 |

## Per sample

| Sample | Diverged | First diff | Run1 tokens | Run2 tokens | Run1 finish | Run2 finish | Pass match | Same code |
|---|---:|---:|---:|---:|---|---|---:|---:|
| HumanEval/0 | False | None | 2072 | 2072 | stop | stop | True | True |
| HumanEval/1 | False | None | 10198 | 10198 | stop | stop | True | True |
| HumanEval/2 | False | None | 1021 | 1021 | stop | stop | True | True |
