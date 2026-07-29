# Trajectory divergence summary

## Overall

| Metric | Value |
|---|---:|
| Total sample IDs | 20 |
| Valid FP16/KVarN pairs | 20 |
| Diverged | 20 |
| Divergence rate | 1.0000 |
| Identical | 0 |
| Token mismatch | 20 |
| Length-only mismatch | 0 |
| Median first divergence step | 237.5 |
| P90 first divergence step | 371.0 |
| Mean FP16 output tokens | 6248.6 |
| Mean KVarN output tokens | 7186.6 |
| Mean length ratio | 1.1405 |
| Natural-completion pairs | 19 |
| Mean length ratio (uncensored) | 1.0201 |
| Either run hit max_tokens | 1 |
| Correct→wrong | 0 |
| Prompt hash mismatches | 0 |
| Input-token count mismatches | 0 |

## Divergence by FP16 output length

| FP16 output tokens | Samples | Diverged | Rate | Median first step | Mean length ratio |
|---|---:|---:|---:|---:|---:|
| [0,512) | 0 | 0 | N/A | N/A | N/A |
| [512,1024) | 0 | 0 | N/A | N/A | N/A |
| [1024,2048) | 3 | 3 | 1.0000 | 247 | 1.2464 |
| [2048,4096) | 7 | 7 | 1.0000 | 280 | 1.0122 |
| [4096,+inf) | 10 | 10 | 1.0000 | 222.5 | 1.1986 |

## Determinism controls

- FP16: 0/20 self-diverged (rate=0.0000).
- KVarN: 0/20 self-diverged (rate=0.0000).
