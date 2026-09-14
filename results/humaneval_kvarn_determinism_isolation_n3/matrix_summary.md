# HumanEval KVarN determinism isolation matrix

| Condition | max_tokens | max_model_len | eager | Diverged | Rate | <256 | <1024 | Median first diff | Cap hits | Finish mismatch | Pass mismatch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_16k_model32k_graph | 16384 | 32768 | False | 3/3 | 1.0000 | 3 | 3 | 136.0 | 1 | 1 | 1 |
| cap30k_model32k_graph | 30000 | 32768 | False | 0/3 | 0.0000 | 0 | 0 | None | 0 | 0 | 0 |
| extended_32k_model40k_eager | 32768 | 40960 | True | 0/3 | 0.0000 | 0 | 0 | None | 0 | 0 | 0 |
| extended_32k_model40k_graph | 32768 | 40960 | False | 3/3 | 1.0000 | 3 | 3 | 136.0 | 1 | 1 | 1 |
| model40k_16k_graph | 16384 | 40960 | False | 3/3 | 1.0000 | 2 | 3 | 207.0 | 1 | 0 | 0 |

## Interpretation guide

- `baseline_16k_model32k_graph` stable but `model40k_16k_graph` unstable: changing `max_model_len`/cache allocation is the primary trigger.
- `baseline_16k_model32k_graph` stable but `cap30k_model32k_graph` unstable: output-cap/decode-horizon effects matter even with the old model length.
- `extended_32k_model40k_graph` unstable but `extended_32k_model40k_eager` stable: CUDA graph/execution-path effects are strongly implicated.
- Graph and eager both unstable at early steps: investigate KVarN kernel/reduction numerical determinism before using greedy cross-mode divergence as a clean causal result.
