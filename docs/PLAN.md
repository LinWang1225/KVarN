# KVarN Reproduction and Trace-Motivation Audit Plan

## Scope

This repository is a vLLM fork that already contains a KVarN attention backend, KVarN tile store/dequant helpers, a Sinkhorn normalization reference, and MLA-side local experiments. The immediate goal is audit and reproduction planning only.

Do **not** start broad refactors or add a dynamic precision controller yet.

## What we need to validate

1. Exact upstream vLLM base version / commit.
2. Whether the KVarN backend is fully wired into the runtime.
3. End-to-end data flow:
   - config entry
   - KV write path
   - quantization path
   - scale / metadata layout
   - attention read / decode path
4. Which presets are exposed and how they map to block size and bit width.
5. Which parts are already suitable for vLLM-native instrumentation vs HF/PyTorch fake-quant analysis.
6. Gaps that prevent controlled reproduction and trace-motivated experiments.

## Findings summary

- The repo is based on vLLM `v0.23.0` according to `README.md`, but the local tree is a later forked state with commit `7586257f1c632e63187bfacbbe21ccb51540f7b3`.
- KVarN backend code exists under `vllm/v1/attention/backends/kvarn_attn.py` and the associated ops/config live under `vllm/model_executor/layers/quantization/kvarn/` and `vllm/v1/attention/ops/`.
- The backend is implemented for vLLM's new `v1` attention stack.
- The current backend is mostly a full KVarN path, not a dynamic controller or hybrid policy engine.

## Audit conclusions

### 1. Config and preset surface

`vllm/model_executor/layers/quantization/kvarn/config.py` defines the user-facing KVarN presets:

- `kvarn_k4v2_g128`
- `kvarn_k4v4_g128`
- `kvarn_k4v2_g64`
- `kvarn_k4v4_g64`

These presets select key/value bit width and the tile size (`group`). The code treats `group == block_size` as a hard invariant.

### 2. Data flow

The KVarN flow in the backend is:

1. **Config resolve** from cache dtype preset.
2. **Incoming K/V rotate** by Hadamard in `do_kv_cache_update()`.
3. **Stage into fp16 tail pool** keyed by physical block id.
4. **Flush when a full tile is ready**.
5. **Apply Sinkhorn-style variance normalization**.
6. **Per-row asymmetric RTN**.
7. **Pack into the cache record** with absorbed scale / zero-point metadata.
8. **Decode path** dequantizes cached tiles and combines them with the fp16 tail pool during attention.

### 3. What is still missing for the experiment agenda

- No dynamic precision controller.
- No obvious built-in trace plumbing for per-step logits / queries / hidden states.
- No explicit layer/block targeting CLI or config for selective quantization beyond the preset-level knobs.
- No mature support for FullKV / FP8 / KVarN switching in the KVarN backend itself; that likely remains a higher-level cache dtype decision handled elsewhere.
- No separate K-only / V-only user-facing mode visible from this audit.

## Recommended implementation phases

### Phase 0 — Repro baseline, no code changes

Collect and freeze:

- local commit hash
- vLLM upstream base indicated by README
- supported presets
- current default block size assumptions
- whether `kvarn` backend is reachable from `kv_cache_dtype`

Deliverable: audit note and code map.

### Phase 1 — Trace instrumentation in vLLM

Add only minimal tracing hooks if needed for experiments:

- per-step token ids and logits
- hidden states / query tensors at attention entry
- layer id / attention branch / decode-vs-prefill tag
- physical block id and block age (from block_table and builder metadata)

Keep the trace code optional and low overhead.

### Phase 2 — Selective ablation hooks

If the experiments require it, add narrowly-scoped switches for:

- layer-wise quantization skipping
- block-wise quantization skipping
- K-only / V-only ablations
- controlled FullKV vs FP8 vs KVarN comparisons

This should remain declarative and avoid controller logic.

### Phase 3 — Offline analysis / fake-quant comparison

Use HF/PyTorch replays for analyses that do not need full scheduler realism:

- sensitivity by layer
- sensitivity by block age
- K vs V error decomposition
- causal comparisons under controlled traces
- repeated replay / counterfactual analysis

### Phase 4 — Controlled runtime experiments in vLLM

Use the real backend for experiments that depend on scheduler / cache behavior:

- block lifecycle and flush timing
- prefix cache reuse behavior
- decode throughput / latency
- tile alignment effects (`g64` vs `g128`)
- real attention readback error accumulation

## Risks

- `block_size == group` is a hard invariant in the current backend; invalid combinations will fail early.
- Hybrid models with non-attention layers can distort pool sizing if layer counting is wrong.
- Prefill / decode mixed batches need careful handling because the backend already branches across several paths.
- Anything that inspects internal block age must account for physical block reuse / recycling.

## Alternative plan if direct runtime tracing is too invasive

If vLLM-native tracing is too costly for the first iteration:

1. Reproduce with offline traces only.
2. Use fake-quant on saved queries / K/V / logits.
3. Move to a minimal vLLM patch only after you know which signals matter.

## Immediate next steps

1. Confirm base commit and upstream delta.
2. Verify all KVarN entry points and dispatch points.
3. Decide the smallest tracing hook set needed for the first reproduction run.
4. Only then design any selective ablation or controller logic.
