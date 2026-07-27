# Native-vLLM KVarN vs TurboQuant comparison

This benchmark replaces the earlier paper-derived KIVI/KVarN/TurboQuant proxy.
It contains **no local quantizer implementation** and does not import the old
`quantizers.py` code.

The three runs are created through the public `vllm.LLM` entrypoint:

| Run | `kv_cache_dtype` | Implementation used |
|---|---|---|
| FP16 reference | `auto` | normal vLLM attention backend |
| KVarN | `kvarn_k4v2_g128` | in-tree KVarN store/Sinkhorn/decode Triton kernels |
| TurboQuant | `turboquant_3bit_nc` | in-tree TurboQuant store/decode Triton kernels |

`block_size=128`, chunked prefill, and `max_num_batched_tokens=128` are pinned.
After the first prefill block, continuation chunks therefore consume the
quantized cache through each backend's native vLLM path.

## What the new plot measures

This is intentionally called a **Figure-5-style backend comparison**, not an
exact reproduction of the paper's Figure 5. The paper plots per-layer attention
output error for K2/V2 KIVI and KVarN. The released KVarN backend instead exposes
K4/V2, and vLLM does not expose those internal per-layer reference tensors.

The script scores an identical fixed token trajectory with prompt logprobs and
reports, over the final 128 tokens at each context length:

1. Mean absolute target-token log-probability drift relative to FP16.
2. Top-1 prediction agreement with FP16.
3. End-to-end prompt-scoring throughput.

This preserves a fair, directly executable comparison while ensuring all
quantized cache writes and reads use the repository's production kernels.

## Installation

Run from the KVarN repository root, in the environment where the fork is
installed editable:

```bash
VLLM_USE_PRECOMPILED=1 pip install -e .
pip install -r benchmarks/kvarn_figure5/requirements.txt
```

Verify that Python imports this checkout rather than another vLLM install:

```bash
python -c "import vllm; print(vllm.__file__)"
```

## Smoke test

Use a local text file to avoid downloading a dataset:

```bash
python benchmarks/kvarn_figure5/run_figure5.py \
  --model /data/models/Qwen3-4B \
  --text-file /data/corpora/sample.txt \
  --repeat-short-text \
  --context-lengths 1024 2048 4096 \
  --methods kvarn turboquant \
  --output-dir runs/figure5_vllm_smoke \
  --overwrite
```

The block/chunk size remains 128 even in the smoke test.

## Default 32K run

When `--text-file` is omitted, the script uses WikiText-2 test text:

```bash
MODEL=/data/models/Qwen3-4B \
OUTPUT_DIR=runs/figure5_vllm_qwen3_4b \
bash benchmarks/kvarn_figure5/run_figure5.sh
```

Equivalent direct command:

```bash
python benchmarks/kvarn_figure5/run_figure5.py \
  --model /data/models/Qwen3-4B \
  --methods kvarn turboquant \
  --context-lengths 4096 8192 16384 32768 \
  --block-size 128 \
  --chunk-size 128 \
  --eval-window 128 \
  --output-dir runs/figure5_vllm_qwen3_4b \
  --overwrite
```

For a 128K experiment, provide a corpus with enough tokens and use:

```bash
python benchmarks/kvarn_figure5/run_figure5.py \
  --model /data/models/Qwen3-4B \
  --text-file /data/corpora/long.txt \
  --context-lengths 16384 32768 65536 98304 131072 \
  --methods kvarn turboquant \
  --output-dir runs/figure5_vllm_128k \
  --overwrite
```

The model checkpoint must itself support the requested maximum context length.

## Output

```text
runs/figure5_vllm_qwen3_4b/
├── config.json
├── input_tokens.json
├── worker_fp16.json
├── worker_kvarn.json
├── worker_turboquant.json
├── figure5_vllm_raw.csv
├── figure5_vllm.csv
├── figure5_vllm_backend_comparison.png
├── figure5_vllm_backend_comparison.pdf
└── figure5_vllm_backend_comparison.svg
```

Each backend runs in a separate subprocess so CUDA memory and vLLM engine state
are released before the next configuration is loaded. Use `--enforce-eager`
only for debugging; omit it for the normal production-style comparison.
