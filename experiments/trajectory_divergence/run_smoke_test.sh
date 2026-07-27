#!/usr/bin/env bash
# End-to-end 3-sample determinism and trajectory-divergence smoke test.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
#PYTHON="${PYTHON:-python}"
PYTHON="/home/wanglin/miniconda3/envs/kvarn-smoke/bin/python"
#MODEL="${MODEL:-Qwen/Qwen3-4B}"
MODEL="/home/wanglin/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/results/smoke}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
KVARN_DTYPE="${KVARN_DTYPE:-kvarn_k4v2_g128}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
SEED="${SEED:-2026}"

# The repository's own validation scripts disable this sampler when testing
# exact greedy replay. Preserve an explicit user override.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

mkdir -p "${OUTPUT_ROOT}"
SAMPLES_FILE="${OUTPUT_ROOT}/selected_samples.json"

cd "${REPO_ROOT}"

"${PYTHON}" "${SCRIPT_DIR}/prepare_samples.py" \
  --dataset-name "${DATASET_NAME:-HuggingFaceH4/MATH-500}" \
  --dataset-split "${DATASET_SPLIT:-test}" \
  --num-samples 3 \
  --seed "${SEED}" \
  --output "${SAMPLES_FILE}"

run_generation() {
  local mode="$1"
  local run_name="$2"
  local out_dir="${OUTPUT_ROOT}/${run_name}"
  "${PYTHON}" "${SCRIPT_DIR}/run_generation.py" \
    --mode "${mode}" \
    --run-name "${run_name}" \
    --samples-file "${SAMPLES_FILE}" \
    --output-dir "${out_dir}" \
    --model "${MODEL}" \
    --kvarn-kv-cache-dtype "${KVARN_DTYPE}" \
    --block-size "${BLOCK_SIZE}" \
    --max-tokens 256 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --seed "${SEED}"
}

# Separate Python processes ensure that no two vLLM engines coexist in GPU memory.
run_generation fp16 fp16_run1
run_generation fp16 fp16_run2
run_generation kvarn kvarn_run1
run_generation kvarn kvarn_run2

COMPARISON_DIR="${OUTPUT_ROOT}/comparison"
"${PYTHON}" "${SCRIPT_DIR}/compare_trajectories.py" \
  --reference "${OUTPUT_ROOT}/fp16_run1/generations.jsonl" \
  --candidate "${OUTPUT_ROOT}/kvarn_run1/generations.jsonl" \
  --reference-repeat "${OUTPUT_ROOT}/fp16_run2/generations.jsonl" \
  --candidate-repeat "${OUTPUT_ROOT}/kvarn_run2/generations.jsonl" \
  --output-dir "${COMPARISON_DIR}" \
  --tokenizer "${MODEL}" \
  --trust-remote-code \
  --block-size "${BLOCK_SIZE}"

PLOT_DIR="${COMPARISON_DIR}/plots"
"${PYTHON}" "${SCRIPT_DIR}/plot_results.py" \
  --comparison-csv "${COMPARISON_DIR}/per_sample_comparison.csv" \
  --summary-json "${COMPARISON_DIR}/summary.json" \
  --output-dir "${PLOT_DIR}"

required_files=(
  "${OUTPUT_ROOT}/fp16_run1/generations.jsonl"
  "${OUTPUT_ROOT}/kvarn_run1/generations.jsonl"
  "${COMPARISON_DIR}/per_sample_comparison.csv"
  "${COMPARISON_DIR}/summary.json"
)
for path in "${required_files[@]}"; do
  if [[ ! -s "${path}" ]]; then
    echo "Missing or empty required file: ${path}" >&2
    exit 1
  fi
done

if ! find "${PLOT_DIR}" -maxdepth 1 -name '*.png' -type f -size +0c | grep -q .; then
  echo "No non-empty PNG plots found in ${PLOT_DIR}" >&2
  exit 1
fi

echo "Smoke test completed: ${OUTPUT_ROOT}"
