#!/usr/bin/env bash
# HumanEval Stage 1: free-generation trajectory + system consequence experiment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
DATASET_NAME="${DATASET_NAME:-openai/openai_humaneval}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
NUM_SAMPLES="${NUM_SAMPLES:-164}"
NUM_REPEATS="${NUM_REPEATS:-2}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
ROPE_SCALING_JSON="${ROPE_SCALING_JSON:-}"
KVARN_DTYPE="${KVARN_DTYPE:-kvarn_k4v2_g128}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TP_SIZE="${TP_SIZE:-1}"
SEED="${SEED:-2026}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
EXECUTE_TESTS="${EXECUTE_TESTS:-1}"
EXECUTION_TIMEOUT="${EXECUTION_TIMEOUT:-30}"
EXECUTION_MEMORY_MB="${EXECUTION_MEMORY_MB:-1024}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-5000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/humaneval_stage1_qwen3_4b_n${NUM_SAMPLES}}"
REQUIRE_CLEAN_GIT="${REQUIRE_CLEAN_GIT:-0}"

if (( NUM_REPEATS < 2 )); then
  echo "NUM_REPEATS must be >= 2 so FP16/KVarN repeat determinism is audited." >&2
  exit 2
fi
if (( MAX_TOKENS >= MAX_MODEL_LEN )); then
  echo "MAX_TOKENS must be smaller than MAX_MODEL_LEN." >&2
  exit 2
fi

export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
mkdir -p "${OUTPUT_ROOT}"
SAMPLES_FILE="${OUTPUT_ROOT}/selected_samples.json"

cd "${REPO_ROOT}"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  if [[ "${REQUIRE_CLEAN_GIT}" == "1" ]]; then
    echo "Repository is dirty; commit/stash changes or set REQUIRE_CLEAN_GIT=0." >&2
    exit 2
  fi
  echo "WARNING: repository has tracked uncommitted changes; configs will record git state." >&2
fi

"${PYTHON}" "${SCRIPT_DIR}/prepare_humaneval_samples.py" \
  --dataset-name "${DATASET_NAME}" \
  --dataset-split "${DATASET_SPLIT}" \
  --num-samples "${NUM_SAMPLES}" \
  --output "${SAMPLES_FILE}"

run_generation() {
  local mode="$1"
  local run_name="$2"
  local output_dir="${OUTPUT_ROOT}/${run_name}"
  local args=(
    --mode "${mode}"
    --run-name "${run_name}"
    --samples-file "${SAMPLES_FILE}"
    --output-dir "${output_dir}"
    --model "${MODEL}"
    --kvarn-kv-cache-dtype "${KVARN_DTYPE}"
    --block-size "${BLOCK_SIZE}"
    --max-tokens "${MAX_TOKENS}"
    --max-model-len "${MAX_MODEL_LEN}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --top-k "${TOP_K}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --tensor-parallel-size "${TP_SIZE}"
    --seed "${SEED}"
    --execution-timeout "${EXECUTION_TIMEOUT}"
    --execution-memory-mb "${EXECUTION_MEMORY_MB}"
  )
  if [[ -n "${ROPE_SCALING_JSON}" ]]; then
    args+=(--rope-scaling-json "${ROPE_SCALING_JSON}")
  fi
  if [[ "${EXECUTE_TESTS}" == "1" ]]; then
    args+=(--execute-tests)
  else
    args+=(--no-execute-tests)
  fi
  "${PYTHON}" "${SCRIPT_DIR}/run_humaneval_generation.py" "${args[@]}"
}

for repeat in $(seq 1 "${NUM_REPEATS}"); do
  run_generation fp16 "fp16_run${repeat}"
done
for repeat in $(seq 1 "${NUM_REPEATS}"); do
  run_generation kvarn "kvarn_run${repeat}"
done

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

"${PYTHON}" "${SCRIPT_DIR}/plot_results.py" \
  --comparison-csv "${COMPARISON_DIR}/per_sample_comparison.csv" \
  --summary-json "${COMPARISON_DIR}/summary.json" \
  --output-dir "${COMPARISON_DIR}/plots" \
  --boundary-step "${BLOCK_SIZE}"

SYSTEM_DIR="${OUTPUT_ROOT}/system_effects"
"${PYTHON}" "${SCRIPT_DIR}/analyze_humaneval_system_effects.py" \
  --fp16 "${OUTPUT_ROOT}/fp16_run1/generations.jsonl" \
  --kvarn "${OUTPUT_ROOT}/kvarn_run1/generations.jsonl" \
  --comparison-csv "${COMPARISON_DIR}/per_sample_comparison.csv" \
  --output-dir "${SYSTEM_DIR}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --bootstrap-seed "${SEED}"

"${PYTHON}" "${SCRIPT_DIR}/plot_humaneval_system_effects.py" \
  --input-csv "${SYSTEM_DIR}/per_sample_system_effects.csv" \
  --output-dir "${SYSTEM_DIR}/plots"

cat <<EOF
HumanEval Stage 1 completed.
Output: ${OUTPUT_ROOT}
Main summaries:
  ${COMPARISON_DIR}/summary.md
  ${SYSTEM_DIR}/summary.md
EOF
