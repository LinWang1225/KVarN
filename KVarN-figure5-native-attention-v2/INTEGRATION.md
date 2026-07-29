# Integration

Apply from the root of the current `LinWang1225/KVarN` checkout:

```bash
git apply --check /path/to/kvarn_figure5_native_attention.patch
git apply /path/to/kvarn_figure5_native_attention.patch
```

Then reinstall the editable fork because two attention backends are hooked and
one benchmark-only vLLM probe module is added:

```bash
VLLM_USE_PRECOMPILED=1 pip install -e .
pip install -r benchmarks/kvarn_figure5/requirements.txt
```

Run the full native-kernel, Figure-5-protocol experiment:

```bash
MODEL=/data/wanglin/models/Qwen3-4B \
NUM_SAMPLES=4 \
GPU_MEMORY_UTILIZATION=0.90 \
SHADOW_MEMORY_SAFETY_GIB=1.0 \
OUTPUT_DIR=runs/figure5_native_attention_qwen3_4b \
bash benchmarks/kvarn_figure5/run_attention_figure5.sh
```

The worker automatically reserves GPU memory for its FP16 shadow K/V history.
The benchmark requires one GPU (`tensor_parallel_size=1`) and is intentionally
slower than serving because it computes a full-precision attention reference
for every 128-token chunk and every layer.

This patch does not delete or modify existing result directories. It leaves the
previous endpoint-metric benchmark available as `run_figure5.py`; the new entry
point is `run_attention_figure5.py`.
