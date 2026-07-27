# KVarN Runtime Smoke Test

This document records the runtime smoke-verification pass only. It does **not** include the full benchmark suite, trajectory analysis, Teacher-Forced Replay, or any dynamic controller work.

## What was verified in code

### 1) KVarN backend enablement

The runtime path is wired through the normal vLLM attention backend selector:

- `vllm/engine/arg_utils.py` resolves `--kv-cache-dtype` into `CacheConfig.cache_dtype`.
- `vllm/v1/attention/selector.py` passes that dtype into the backend selector.
- `vllm/platforms/cuda.py` includes `AttentionBackendEnum.KVARN` in CUDA backend priorities.
- `vllm/platforms/cuda.py` special-cases `cache_dtype.startswith("kvarn_")` and constructs `KVarNConfig`.
- `vllm/v1/attention/backends/kvarn_attn.py` implements `KVarNAttentionBackend`.

### 2) Presets exposed by the code

The current preset surface in `vllm/model_executor/layers/quantization/kvarn/config.py` is:

- `kvarn_k4v2_g128`
- `kvarn_k4v4_g128`
- `kvarn_k4v2_g64`
- `kvarn_k4v4_g64`

The current audit only confirms K4V2 / K4V4 presets. It does **not** assume any K2V2 support.

### 3) Dispatch path for `kvarn_k4v2_g128`

The observed code path is:

1. CLI / config input via `--kv-cache-dtype kvarn_k4v2_g128`
2. `vllm.engine.arg_utils.resolve_kv_cache_dtype_string(...)`
3. `CacheConfig.cache_dtype`
4. `vllm.v1.attention.selector.get_attn_backend(...)`
5. `current_platform.get_attn_backend_cls(...)`
6. `vllm.platforms.cuda.CudaPlatform.get_attn_backend_cls(...)`
7. `KVarNConfig.from_cache_dtype(...)`
8. `KVarNAttentionBackend`

### 4) How to confirm the real backend

The reliable runtime checks are:

- log output from `vllm.platforms.cuda` saying `Using KVARN backend.`
- backend class name resolution to `vllm.v1.attention.backends.kvarn_attn.KVarNAttentionBackend`
- optional debug prints if `KVARN_DBG_LAYERS=1`

### 5) Fallback behavior

The code does **not** implement an automatic silent fallback from KVarN to another backend once `cache_dtype` is a `kvarn_*` preset and the CUDA platform validation passes.

Observed failure / rejection paths:

- unsupported `head_dim` raises `ValueError`
- invalid `block_size != group` raises an assertion in `get_kv_cache_shape`
- unsupported backend selection raises `ValueError` in the platform selector
- missing `kvarn_*` preset raises a `ValueError` in `KVarNConfig.from_cache_dtype`

### 6) FullKV / FP8 / KVarN switching

The current repository supports switching between runtime families through `--kv-cache-dtype`, but they are not all equivalent in backend behavior:

- FullKV uses `float16` or `bfloat16`
- FP8 uses `fp8` / `fp8_e4m3` / `fp8_e5m2` where supported
- KVarN uses `kvarn_*` presets and the dedicated KVarN backend path

## Smoke-test conclusion status

A full live inference run still needs to be executed to mark these rows as runtime-confirmed:

- KVarNAttentionBackend initialized at runtime
- `do_kv_cache_update()` invoked during actual generation
- flush / packing triggered at a full block boundary
- decode path reading quantized blocks
- tail pool read for incomplete blocks
- no silent fallback during the run

## Recommended Phase A launch target

The best current candidate from code inspection is:

- model: `Qwen/Qwen3-0.6B`
- preset: `kvarn_k4v4_g128` first, then `kvarn_k4v2_g128`
- block size: `128`
- prompt: a short fixed prompt that forces at least one full block boundary during decode if you want to observe flush behavior

This recommendation is based on the model being present in local Hugging Face cache and matching the validated head dimension.

## Remaining blockers

- This repository session did not yet execute the live vLLM launch and token-generation loop.
- The exact serving command still needs to be run in the `vllm-V1` environment.
- Runtime logs for the full smoke matrix still need to be captured before this can be called a completed verification.
