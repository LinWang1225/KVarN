# Trajectory divergence summary

## Overall

| Metric | Value |
|---|---:|
| Total sample IDs | 3 |
| Valid FP16/KVarN pairs | 3 |
| Diverged | 2 |
| Divergence rate | 0.6667 |
| Identical | 1 |
| Token mismatch | 2 |
| Length-only mismatch | 0 |
| Median first divergence step | 217.5 |
| P90 first divergence step | 229.9 |
| Mean FP16 output tokens | 256.0 |
| Mean KVarN output tokens | 256.0 |
| Mean length ratio | 1.0000 |
| Correct→wrong | 0 |
| Prompt hash mismatches | 0 |

## Divergence by FP16 output length

| FP16 output tokens | Samples | Diverged | Rate | Median first step | Mean length ratio |
|---|---:|---:|---:|---:|---:|
| [0,512) | 3 | 2 | 0.6667 | 217.5 | 1.0000 |
| [512,1024) | 0 | 0 | N/A | N/A | N/A |
| [1024,2048) | 0 | 0 | N/A | N/A | N/A |
| [2048,4096) | 0 | 0 | N/A | N/A | N/A |
| [4096,+inf) | 0 | 0 | N/A | N/A | N/A |

## Determinism controls

- FP16: 0/3 self-diverged (rate=0.0000).
- KVarN: 0/3 self-diverged (rate=0.0000).
