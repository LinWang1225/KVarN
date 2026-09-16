# Trajectory divergence summary

## Overall

| Metric | Value |
|---|---:|
| Total sample IDs | 164 |
| Valid FP16/KVarN pairs | 164 |
| Diverged | 164 |
| Divergence rate | 1.0000 |
| Identical | 0 |
| Token mismatch | 164 |
| Length-only mismatch | 0 |
| Median first divergence step | 116.5 |
| P90 first divergence step | 194.1 |
| Mean FP16 output tokens | 6868.8 |
| Mean KVarN output tokens | 5779.8 |
| Mean length ratio | 1.2519 |
| Natural-completion pairs | 138 |
| Mean length ratio (uncensored) | 1.0388 |
| Either run hit max_tokens | 26 |
| Correct→wrong | 3 |
| Prompt hash mismatches | 0 |
| Input-token count mismatches | 0 |

## Divergence by FP16 output length

| FP16 output tokens | Samples | Diverged | Rate | Median first step | Mean length ratio |
|---|---:|---:|---:|---:|---:|
| [0,512) | 3 | 3 | 1.0000 | 181 | 1.1838 |
| [512,1024) | 18 | 18 | 1.0000 | 150.0 | 1.1899 |
| [1024,2048) | 42 | 42 | 1.0000 | 123.0 | 1.7129 |
| [2048,4096) | 54 | 54 | 1.0000 | 112.0 | 1.2140 |
| [4096,+inf) | 47 | 47 | 1.0000 | 96 | 0.9114 |

## Determinism controls

- FP16: 0/164 self-diverged (rate=0.0000).
- KVarN: 0/164 self-diverged (rate=0.0000).
