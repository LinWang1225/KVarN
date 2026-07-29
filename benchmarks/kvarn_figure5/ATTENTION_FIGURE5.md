# Figure 5 protocol with native vLLM kernels

This patch keeps the deployment presets requested for this repository:

| Method | Native vLLM `kv_cache_dtype` |
|---|---|
| KVarN | `kvarn_k4v2_g128` |
| TurboQuant | `turboquant_3bit_nc` |

There is no KIVI implementation and no benchmark-local quantizer. Cache storage,
packing, dequantization and attention all remain in the repository's native
KVarN/TurboQuant backends and Triton kernels.

## What changed from the old benchmark

The existing `run_figure5.py` measures final-model log-probability drift,
top-1 agreement and prompt-scoring throughput. Those are useful endpoint
metrics, but they are not the quantities in paper Figure 5.

`run_attention_figure5.py` adds the paper's measurement protocol:

1. **Panel (a)**: element-weighted mean absolute reconstruction error over
   every 128-token pseudo-decode chunk and every attention-layer output, for
   both **static** and **accumulated** runs.
2. **Panel (b)**: `MAE_KVarN - MAE_TurboQuant` under each regime.
3. **Panel (c)**: `(KVarN-TurboQuant)_static -
   (KVarN-TurboQuant)_accumulated`.

The native backend output (the `softmax(QK^T)V` result before the model's
output projection) is compared with a shadow FP16 KV history:

- **Accumulated** returns the native quantized-cache attention output, so its
  error changes later hidden states and later cache entries.
- **Static** still executes the same native quantized cache path and records
  its local error, but returns the shadow FP16 attention output to the model.
  The local error therefore cannot propagate into later layers or chunks.

The benchmark records every chunk rather than only the final 128 tokens, then
uses several deterministic WikiText-2 windows and reports paired sample means
and 95% confidence intervals. This removes the earlier single-window/final-block
content confound and makes panels (b) and (c) paired at the sample level.

## Scope and naming

This is a **Figure-5 protocol-aligned adaptation**, not a numerical recreation
of the paper's K2/V2 comparison. The paper compares KVarN K2/V2 with KIVI K2/V2.
This patch deliberately retains the released native presets:

- KVarN K4/V2 G128
- TurboQuant K3/V3 with norm correction

Accordingly, panel (b) uses TurboQuant instead of KIVI. It tests the same error
accumulation hypothesis but cannot be expected to reproduce the paper's exact
values or method gap.

## Apply and install

From the repository root:

```bash
git apply --check /path/to/kvarn_figure5_native_attention.patch
git apply /path/to/kvarn_figure5_native_attention.patch
VLLM_USE_PRECOMPILED=1 pip install -e .
pip install -r benchmarks/kvarn_figure5/requirements.txt
```

The backend hook is dormant unless the benchmark sets
`VLLM_FIGURE5_PROBE_OUTPUT`, so normal serving behavior is unchanged.

## Smoke test

```bash
python benchmarks/kvarn_figure5/run_attention_figure5.py \
  --model /data/wanglin/models/Qwen3-4B \
  --methods kvarn turboquant \
  --context-lengths 1024 2048 4096 \
  --num-samples 1 \
  --text-file /data/corpora/sample.txt \
  --repeat-short-text \
  --output-dir runs/figure5_attention_smoke \
  --overwrite
```

## Full run

```bash
MODEL=/data/wanglin/models/Qwen3-4B \
NUM_SAMPLES=4 \
GPU_MEMORY_UTILIZATION=0.90 \
SHADOW_MEMORY_SAFETY_GIB=1.0 \
OUTPUT_DIR=runs/figure5_native_attention_qwen3_4b \
bash benchmarks/kvarn_figure5/run_attention_figure5.sh
```

For a smoother context curve, use more observation points:

```bash
python benchmarks/kvarn_figure5/run_attention_figure5.py \
  --model /data/wanglin/models/Qwen3-4B \
  --methods kvarn turboquant \
  --context-lengths 2048 4096 6144 8192 12288 16384 24576 32768 \
  --num-samples 4 \
  --output-dir runs/figure5_native_attention_dense \
  --overwrite
```

## Outputs

```text
config_attention_figure5.json
input_samples.json
worker_kvarn_static.jsonl
worker_kvarn_accumulated.jsonl
worker_turboquant_static.jsonl
worker_turboquant_accumulated.jsonl
figure5_attention_layer_raw.csv
figure5_attention_sample.csv
figure5_attention_summary.csv
figure5_native_attention_reconstruction.png
figure5_native_attention_reconstruction.pdf
figure5_native_attention_reconstruction.svg
```

`figure5_attention_layer_raw.csv` contains one row per method, regime, sample,
context, 128-token chunk and attention layer. Layers intentionally kept in FP16
by a preset are included as zero-error layers when averaging over all model
attention layers. `figure5_attention_sample.csv` contains the full-sequence,
all-layer MAE for each paired sample.

## Resource note

Both regimes keep a shadow FP16 KV history on GPU and evaluate an FP16 attention
reference for every 128-token chunk. Before creating vLLM, the worker estimates
that shadow allocation from the model configuration and automatically subtracts
it—plus `--shadow-memory-safety-gib`—from the requested vLLM GPU-memory
utilization. For Qwen3-4B at 32K, the shadow cache is roughly several GiB; this
automatic reservation avoids the OOM that would occur if vLLM first consumed
90% of the GPU. The worker prints the requested and effective utilization and
records them in `config_attention_figure5.json`.

The probe currently requires `--tensor-parallel-size 1`. `enforce_eager=True` is
set internally because Python-side probe state and JSONL recording must not be
captured by CUDA graphs. This benchmark is deliberately much more expensive
than normal serving, but the KVarN and TurboQuant store/dequant/attention Triton
kernels themselves remain the repository's native implementations.
