#!/usr/bin/env bash
# Full FP16-vs-KVarN trajectory-divergence experiment on a fixed MATH-500 subset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
NUM_REPEATS="${NUM_REPEATS:-1}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
DATASET_NAME="${DATASET_NAME:-HuggingFaceH4/MATH-500}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/results/math500_n${NUM_SAMPLES}}"
KVARN_DTYPE="${KVARN_DTYPE:-kvarn_k4v2_g128}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
SEED="${SEED:-2026}"
TP_SIZE="${TP_SIZE:-1}"

if (( NUM_REPEATS < 1 )); then
  echo "NUM_REPEATS must be at least 1" >&2
  exit 2
fi

export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
mkdir -p "${OUTPUT_ROOT}"
SAMPLES_FILE="${OUTPUT_ROOT}/selected_samples.json"

cd "${REPO_ROOT}"

"${PYTHON}" "${SCRIPT_DIR}/prepare_samples.py" \
  --dataset-name "${DATASET_NAME}" \
  --dataset-split "${DATASET_SPLIT}" \
  --num-samples "${NUM_SAMPLES}" \
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
    --max-tokens "${MAX_TOKENS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --seed "${SEED}"
}

for repeat in $(seq 1 "${NUM_REPEATS}"); do
  run_generation fp16 "fp16_run${repeat}"
done
for repeat in $(seq 1 "${NUM_REPEATS}"); do
  run_generation kvarn "kvarn_run${repeat}"
done

COMPARISON_DIR="${OUTPUT_ROOT}/comparison"
compare_args=(
  --reference "${OUTPUT_ROOT}/fp16_run1/generations.jsonl"
  --candidate "${OUTPUT_ROOT}/kvarn_run1/generations.jsonl"
  --output-dir "${COMPARISON_DIR}"
  --tokenizer "${MODEL}"
  --trust-remote-code
  --block-size "${BLOCK_SIZE}"
)
if (( NUM_REPEATS >= 2 )); then
  compare_args+=(
    --reference-repeat "${OUTPUT_ROOT}/fp16_run2/generations.jsonl"
    --candidate-repeat "${OUTPUT_ROOT}/kvarn_run2/generations.jsonl"
  )
fi
"${PYTHON}" "${SCRIPT_DIR}/compare_trajectories.py" "${compare_args[@]}"

"${PYTHON}" "${SCRIPT_DIR}/plot_results.py" \
  --comparison-csv "${COMPARISON_DIR}/per_sample_comparison.csv" \
  --summary-json "${COMPARISON_DIR}/summary.json" \
  --output-dir "${COMPARISON_DIR}/plots"

echo "Full experiment completed: ${OUTPUT_ROOT}"
