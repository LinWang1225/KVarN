# HumanEval KVarN greedy determinism isolation

This diagnostic isolates the `3/3` KVarN repeat self-divergence observed after
raising the HumanEval greedy experiment from `16K/32K` to `32K/40K`.

It intentionally runs **KVarN only**, twice per condition, because FP16 was
already repeat-deterministic in the smoke and the immediate question is which
configuration change triggers KVarN self-divergence.

## Conditions

| Condition | max_tokens | max_model_len | CUDA graph | Isolates |
|---|---:|---:|---|---|
| `baseline_16k_model32k_graph` | 16384 | 32768 | on | old HumanEval geometry on current code |
| `model40k_16k_graph` | 16384 | 40960 | on | `max_model_len` / cache geometry only |
| `cap30k_model32k_graph` | 30000 | 32768 | on | decode horizon while preserving old model length |
| `extended_32k_model40k_graph` | 32768 | 40960 | on | current problematic point; reused from existing smoke by default |
| `extended_32k_model40k_eager` | 32768 | 40960 | off | graph-vs-eager execution path |

All conditions keep greedy decoding (`temperature=0`, `top_p=1`, `top_k=-1`),
seed 2026, KVarN K4/V2 G128, block size 128, TP=1, and prefix caching disabled.

## Reading the matrix

The most useful columns in `matrix_summary.md` are self-divergence rate and the
first-divergence location.

- Baseline stable, `model40k_16k_graph` unstable: suspect `max_model_len`-driven
  KV allocation / CUDA graph shape changes.
- Baseline stable, `cap30k_model32k_graph` unstable: decode-horizon behavior can
  trigger the issue even with the old model length.
- Graph 32K/40K unstable, eager 32K/40K stable: CUDA graph execution is strongly
  implicated.
- Both graph and eager unstable at early steps: investigate numerical
  determinism in the KVarN store/Sinkhorn/decode path before treating one
  greedy FP16-vs-KVarN trajectory as a clean causal sample.

The script is resume-safe: generation calls use `--resume`. It also saves
human-readable per-sample outputs for every newly generated condition.
