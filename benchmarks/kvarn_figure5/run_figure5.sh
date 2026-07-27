#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-4B}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/figure5_vllm_qwen3_4b}"
TP="${TP:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

python benchmarks/kvarn_figure5/run_figure5.py \
  --model "${MODEL}" \
  --methods kvarn turboquant \
  --context-lengths 4096 8192 16384 32768 \
  --block-size 128 \
  --chunk-size 128 \
  --eval-window 128 \
  --tensor-parallel-size "${TP}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --output-dir "${OUTPUT_DIR}" \
  --overwrite
