# Trajectory divergence summary

## Overall

| Metric | Value |
|---|---:|
| Total sample IDs | 3 |
| Valid FP16/KVarN pairs | 3 |
| Diverged | 3 |
| Divergence rate | 1.0000 |
| Identical | 0 |
| Token mismatch | 3 |
| Length-only mismatch | 0 |
| Median first divergence step | 112 |
| P90 first divergence step | 176.0 |
| Mean FP16 output tokens | 3910.7 |
| Mean KVarN output tokens | 13232.7 |
| Mean length ratio | 2.7458 |
| Natural-completion pairs | 2 |
| Mean length ratio (uncensored) | 2.2182 |
| Either run hit max_tokens | 1 |
| Correct→wrong | 1 |
| Prompt hash mismatches | 0 |
| Input-token count mismatches | 0 |

## Divergence by FP16 output length

| FP16 output tokens | Samples | Diverged | Rate | Median first step | Mean length ratio |
|---|---:|---:|---:|---:|---:|
| [0,512) | 0 | 0 | N/A | N/A | N/A |
| [512,1024) | 0 | 0 | N/A | N/A | N/A |
| [1024,2048) | 2 | 2 | 1.0000 | 142.0 | 2.2182 |
| [2048,4096) | 0 | 0 | N/A | N/A | N/A |
| [4096,+inf) | 1 | 1 | 1.0000 | 112 | 3.8010 |

## Determinism controls

- FP16: 0/3 self-diverged (rate=0.0000).
- KVarN: 3/3 self-diverged (rate=1.0000).
