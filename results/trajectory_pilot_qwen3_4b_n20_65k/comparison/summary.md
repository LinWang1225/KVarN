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
| Median first divergence step | 229.0 |
| P90 first divergence step | 357.5 |
| Mean FP16 output tokens | 6248.6 |
| Mean KVarN output tokens | 7227.5 |
| Mean length ratio | 1.0894 |
| Natural-completion pairs | 19 |
| Mean length ratio (uncensored) | 1.0523 |
| Either run hit max_tokens | 1 |
| Correct→wrong | 1 |
| Prompt hash mismatches | 0 |
| Input-token count mismatches | 0 |

## Divergence by FP16 output length

| FP16 output tokens | Samples | Diverged | Rate | Median first step | Mean length ratio |
|---|---:|---:|---:|---:|---:|
| [0,512) | 0 | 0 | N/A | N/A | N/A |
| [512,1024) | 0 | 0 | N/A | N/A | N/A |
| [1024,2048) | 3 | 3 | 1.0000 | 228 | 1.1166 |
| [2048,4096) | 7 | 7 | 1.0000 | 227 | 1.0113 |
| [4096,+inf) | 10 | 10 | 1.0000 | 241.5 | 1.1358 |

## Determinism controls

- FP16: 0/20 self-diverged (rate=0.0000).
- KVarN: 20/20 self-diverged (rate=1.0000).
