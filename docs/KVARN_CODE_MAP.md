# KVarN Code Map

This document maps the current KVarN implementation surface in the repository.

## Repository-level version notes

- `README.md` says the fork is built on **vLLM v0.23.0**.
- The local git HEAD is **`7586257f1c632e63187bfacbbe21ccb51540f7b3`**.
- This is a working fork state, not a clean upstream vLLM checkout.

## KVarN-related files

### Configuration and presets

#### `vllm/model_executor/layers/quantization/kvarn/config.py`

Responsibilities:

- Defines the KVarN preset table.
- Provides the `KVarNConfig` dataclass.
- Encodes storage layout math and tile byte offsets.
- Computes fp16 tail-pool sizing and capacity limits.
- Exposes helper methods for boundary-layer skipping and weight-size estimation.

Key preset entries:

- `kvarn_k4v2_g128`
- `kvarn_k4v4_g128`
- `kvarn_k4v2_g64`
- `kvarn_k4v4_g64`

Important invariants:

- `group` is the KVarN tile size.
- `group` must equal vLLM `block_size`.
- `head_dim` is expected to be 128 for the validated path, but the backend advertises support for 128 / 256 / 512.

### Quantization and layout helpers

#### `vllm/model_executor/layers/quantization/kvarn/sinkhorn.py`

Responsibilities:

- Pure PyTorch reference Sinkhorn / variance normalization.
- Single-tile and batched-tile normalization.
- Returns balanced tile plus `s_col` / `s_row` scale factors.

This is the canonical reference for the normalization step used by the packers and the Triton port.

#### `vllm/model_executor/layers/quantization/kvarn/mla_quant.py`

Responsibilities:

- Local MLA latent packing experiment.
- Per-token Hadamard + asymmetric RTN packing.
- Not the main KVarN KV-cache path.

#### `vllm/model_executor/layers/quantization/kvarn/mla_probe.py`

Responsibilities:

- Local accuracy probe for MLA latent round-trip.
- Uses the KVarN-style rotation + Sinkhorn + RTN recipe.
- Intended for offline analysis, not the runtime backend.

#### `vllm/model_executor/layers/quantization/kvarn/__init__.py`

Responsibilities:

- Re-exports `KVarNConfig`.
- Documents the KVarN recipe at package level.

### Attention backend

#### `vllm/v1/attention/backends/kvarn_attn.py`

Responsibilities:

- Implements the vLLM v1 KVarN attention backend.
- Defines backend capability checks.
- Defines KVarN attention metadata and metadata builder.
- Manages fp16 tail pool, physical block tracking, and flush scheduling.
- Contains the runtime KV write path and the decode / prefill / mixed-batch forward paths.

This is the most important file for runtime audit.

Key methods to inspect:

- `KVarNAttentionBackend.get_supported_kernel_block_sizes()`
- `KVarNAttentionBackend.get_preferred_block_size()`
- `KVarNAttentionBackend.get_kv_cache_shape()`
- `KVarNAttentionImpl.do_kv_cache_update()`
- `KVarNAttentionImpl.forward()`
- `KVarNAttentionImpl._flush_tail()`
- `KVarNAttentionImpl._read_block_dequantized()`
- `KVarNAttentionImpl._decode_path()`

### Runtime cache ops

#### `vllm/v1/attention/ops/kvarn_store.py`

Responsibilities:

- Pure PyTorch reference packer.
- Quantizes rotated K and V tiles.
- Produces packed bytes and absorbed scale / zero-point metadata.

Main functions:

- `kvarn_store_tile_k()`
- `kvarn_store_tile_v()`
- `kvarn_store_tile_k_batch_from_sinkhorn()`
- `kvarn_store_tile_v_batch_from_sinkhorn()`

#### `vllm/v1/attention/ops/kvarn_decode.py`

Responsibilities:

- Pure PyTorch reference inverse of the packer.
- Dequantizes to rotated-space K/V tiles.
- Leaves inverse Hadamard and attention math to the caller.

Main functions:

- `kvarn_dequant_tile_k()`
- `kvarn_dequant_tile_v()`

#### `vllm/v1/attention/ops/triton_kvarn_sinkhorn.py`

Responsibilities:

- Triton implementation of the Sinkhorn / variance-normalization step.
- Used by the runtime backend for fast packing.

#### `vllm/v1/attention/ops/triton_kvarn_decode.py`

Responsibilities:

- Triton decode path for KVarN.
- Dequantizes cached tiles, gathers fp16 tail buffers, and calls FlashAttention.
- Also includes a prior / fallback dequant kernel and split-K logic.

#### `vllm/v1/attention/ops/kvarn_mla_paged_proto.py`
#### `vllm/v1/attention/ops/kvarn_mla_attn_proto.py`
#### `vllm/v1/attention/ops/kvarn_store.py`

These files are KVarN-adjacent prototyping surfaces for MLA / paged attention experiments.

## Data-flow map

### 1. User configuration entry

User-facing preset selection flows through the standard vLLM cache-dtype configuration path:

1. CLI / serving flags parse `--kv-cache-dtype` in `vllm/engine/arg_utils.py`.
2. `resolve_kv_cache_dtype_string(...)` resolves `auto` into the final string value.
3. The resolved value is stored in `CacheConfig.cache_dtype`.
4. `vllm/v1/attention/selector.py` passes that dtype into `current_platform.get_attn_backend_cls(...)`.
5. `vllm/platforms/cuda.py` inspects `cache_dtype.startswith("kvarn_")` and constructs `KVarNConfig`.
6. `AttentionBackendEnum.KVARN` resolves to `vllm.v1.attention.backends.kvarn_attn.KVarNAttentionBackend`.

The current KVarN preset strings are:

- `kvarn_k4v2_g128`
- `kvarn_k4v4_g128`
- `kvarn_k4v2_g64`
- `kvarn_k4v4_g64`

### 2. Runtime write path

`do_kv_cache_update()` rotates incoming fp16 K/V, then writes the rotated data into a per-block fp16 tail pool indexed by physical block id.

### 3. Flush / quantization path

When a block becomes full, `_flush_tail()` or the batched flush path:

- collects the rotated tile,
- applies Sinkhorn normalization,
- applies asymmetric RTN,
- packs bytes,
- writes the packed record into the KV cache.

### 4. Read / attention path

`forward()` routes to one of the following:

- prefill first chunk
- cached multi-query path
- decode path
- verify decode path
- mixed batch path

The decode path eventually materializes a dequantized block and runs attention.

### 5. Dispatch / fallback behavior

Relevant runtime checks in the current tree:

- `vllm/platforms/cuda.py` adds `AttentionBackendEnum.KVARN` to the CUDA backend priority list.
- `vllm/platforms/cuda.py` validates that KVarN only runs on `head_dim in (128, 256, 512)`.
- `vllm/platforms/cuda.py` appends `sliding_window` to `kv_cache_dtype_skip_layers` unless `KVARN_QUANT_SLIDING=1`.
- `KVarNConfig.from_cache_dtype(...)` throws if the preset string is unknown.
- `KVarNAttentionBackend.get_kv_cache_shape(...)` asserts `block_size == cfg.group`.
- `KVarNAttentionBackend.supports_kv_cache_dtype(...)` accepts only `kvarn_*` dtypes that are not MLA presets.

Observed fallback / rejection behavior:

- There is no silent fallback from a validated `kvarn_*` preset to FullKV / FP8 in the KVarN path.
- Invalid KVarN configuration raises `ValueError` or assertion failures during backend selection or cache-shape construction.
- If the platform selector chooses a different backend, it is because the KVarN configuration was rejected, not because KVarN silently downgraded.

## Practical audit notes

- The implementation already carries the core runtime plumbing needed for reproduction.
- The current tree does **not** show a separate dynamic precision controller.
- The KVarN backend is designed around block-level tile sharing, not token-level per-row scales.
- The current backend appears focused on native vLLM execution rather than offline experimentation.

## Minimal files likely needed for audit-driven follow-up work

If we later add instrumentation or ablation flags, the likely touch points are:

- `vllm/v1/attention/backends/kvarn_attn.py`
- `vllm/model_executor/layers/quantization/kvarn/config.py`
- `vllm/v1/attention/ops/kvarn_store.py`
- `vllm/v1/attention/ops/kvarn_decode.py`
- `vllm/v1/attention/ops/triton_kvarn_decode.py`
- `vllm/v1/attention/ops/triton_kvarn_sinkhorn.py`
- potentially the generic cache-dtype dispatch path in the main vLLM config layer

## Current limitation summary

The code map above is sufficient to start a reproduction plan, but not yet enough for a controlled experimental suite. The main missing pieces are tracing, selective ablation controls, and an explicit policy surface for comparing FullKV / FP8 / KVarN under identical traces.
