#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-4B}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/figure5_native_attention_qwen3_4b}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
SHADOW_MEMORY_SAFETY_GIB="${SHADOW_MEMORY_SAFETY_GIB:-1.0}"

python benchmarks/kvarn_figure5/run_attention_figure5.py \
  --model "$MODEL" \
  --methods kvarn turboquant \
  --context-lengths 4096 8192 16384 32768 \
  --num-samples "$NUM_SAMPLES" \
  --block-size 128 \
  --chunk-size 128 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --shadow-memory-safety-gib "$SHADOW_MEMORY_SAFETY_GIB" \
  --output-dir "$OUTPUT_DIR" \
  --overwrite
