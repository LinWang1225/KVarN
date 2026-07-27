# Trajectory divergence summary

## Overall

| Metric | Value |
|---|---:|
| Total sample IDs | 3 |
| Valid FP16/KVarN pairs | 3 |
| Diverged | 1 |
| Divergence rate | 0.3333 |
| Identical | 2 |
| Token mismatch | 1 |
| Length-only mismatch | 0 |
| Median first divergence step | 141 |
| P90 first divergence step | 141.0 |
| Mean FP16 output tokens | 256.0 |
| Mean KVarN output tokens | 256.0 |
| Mean length ratio | 1.0000 |
| Natural-completion pairs | 0 |
| Mean length ratio (uncensored) | N/A |
| Either run hit max_tokens | 3 |
| Correct→wrong | 0 |
| Prompt hash mismatches | 0 |
| Input-token count mismatches | 0 |

## Divergence by FP16 output length

| FP16 output tokens | Samples | Diverged | Rate | Median first step | Mean length ratio |
|---|---:|---:|---:|---:|---:|
| [0,512) | 3 | 1 | 0.3333 | 141 | 1.0000 |
| [512,1024) | 0 | 0 | N/A | N/A | N/A |
| [1024,2048) | 0 | 0 | N/A | N/A | N/A |
| [2048,4096) | 0 | 0 | N/A | N/A | N/A |
| [4096,+inf) | 0 | 0 | N/A | N/A | N/A |

## Determinism controls

- FP16: 1/3 self-diverged (rate=0.3333).
- KVarN: 0/3 self-diverged (rate=0.0000).
