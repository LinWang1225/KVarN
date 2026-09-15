#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
DATASET_NAME="${DATASET_NAME:-openai/openai_humaneval}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
NUM_SAMPLES="${NUM_SAMPLES:-3}"
NUM_REPEATS="${NUM_REPEATS:-5}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
KVARN_DTYPE="${KVARN_DTYPE:-kvarn_k4v2_g128}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TP_SIZE="${TP_SIZE:-1}"
SEED="${SEED:-2026}"
EXECUTION_TIMEOUT="${EXECUTION_TIMEOUT:-30}"
EXECUTION_MEMORY_MB="${EXECUTION_MEMORY_MB:-1024}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/humaneval_graph_eager_repeat_n${NUM_SAMPLES}_r${NUM_REPEATS}}"

if (( NUM_REPEATS < 2 )); then
  echo "NUM_REPEATS must be >= 2" >&2
  exit 2
fi
if (( MAX_TOKENS >= MAX_MODEL_LEN )); then
  echo "MAX_TOKENS must be smaller than MAX_MODEL_LEN" >&2
  exit 2
fi

export VLLM_USE_FLASHINFER_SAMPLER=0

mkdir -p "${OUTPUT_ROOT}"
SAMPLES_FILE="${OUTPUT_ROOT}/selected_samples.json"
cd "${REPO_ROOT}"

"${PYTHON}" "${SCRIPT_DIR}/prepare_humaneval_samples.py"   --dataset-name "${DATASET_NAME}"   --dataset-split "${DATASET_SPLIT}"   --num-samples "${NUM_SAMPLES}"   --output "${SAMPLES_FILE}"

run_one() {
  local mode="$1"
  local exec_mode="$2"
  local repeat="$3"
  local condition="${mode}_${exec_mode}"
  local run_name="${condition}_run${repeat}"
  local run_dir="${OUTPUT_ROOT}/${condition}/run${repeat}"
  local readable_dir="${OUTPUT_ROOT}/${condition}/per_sample_outputs/run${repeat}"

  mkdir -p "${run_dir}" "${readable_dir}"

  local args=(
    --mode "${mode}"
    --run-name "${run_name}"
    --samples-file "${SAMPLES_FILE}"
    --output-dir "${run_dir}"
    --model "${MODEL}"
    --fp16-kv-cache-dtype auto
    --kvarn-kv-cache-dtype "${KVARN_DTYPE}"
    --block-size "${BLOCK_SIZE}"
    --max-tokens "${MAX_TOKENS}"
    --max-model-len "${MAX_MODEL_LEN}"
    --temperature 0.0
    --top-p 1.0
    --top-k -1
    --min-p 0.0
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --tensor-parallel-size "${TP_SIZE}"
    --seed "${SEED}"
    --execution-timeout "${EXECUTION_TIMEOUT}"
    --execution-memory-mb "${EXECUTION_MEMORY_MB}"
    --per-sample-output-dir "${readable_dir}"
    --require-backend-verification
    --resume
    --execute-tests
  )

  if [[ "${exec_mode}" == "eager" ]]; then
    args+=(--enforce-eager)
  fi

  echo "=== ${condition} repeat ${repeat}/${NUM_REPEATS} ==="
  "${PYTHON}" "${SCRIPT_DIR}/run_humaneval_generation.py" "${args[@]}"
}

for mode in fp16 kvarn; do
  for exec_mode in graph eager; do
    for repeat in $(seq 1 "${NUM_REPEATS}"); do
      run_one "${mode}" "${exec_mode}" "${repeat}"
    done
  done
done

"${PYTHON}" "${SCRIPT_DIR}/analyze_humaneval_graph_eager_repeats.py"   --root "${OUTPUT_ROOT}"   --num-repeats "${NUM_REPEATS}"

echo "Done. Read: ${OUTPUT_ROOT}/analysis/matrix_summary.md"
