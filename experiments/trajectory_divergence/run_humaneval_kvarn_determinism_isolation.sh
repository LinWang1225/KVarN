#!/usr/bin/env bash
# Isolate the trigger for KVarN greedy repeat self-divergence on HumanEval.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
NUM_SAMPLES="${NUM_SAMPLES:-3}"
KVARN_DTYPE="${KVARN_DTYPE:-kvarn_k4v2_g128}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TP_SIZE="${TP_SIZE:-1}"
SEED="${SEED:-2026}"
EXECUTE_TESTS="${EXECUTE_TESTS:-1}"
EXECUTION_TIMEOUT="${EXECUTION_TIMEOUT:-30}"
EXECUTION_MEMORY_MB="${EXECUTION_MEMORY_MB:-1024}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/humaneval_kvarn_determinism_isolation_n${NUM_SAMPLES}}"
EXISTING_32K_ROOT="${EXISTING_32K_ROOT:-${REPO_ROOT}/results/humaneval_stage1_greedy32k_smoke}"
SOURCE_SAMPLES_FILE="${SOURCE_SAMPLES_FILE:-${EXISTING_32K_ROOT}/selected_samples.json}"

# Set to 0 if you deliberately want to regenerate the graph-mode 32K/40K point.
REUSE_EXISTING_32K="${REUSE_EXISTING_32K:-1}"

mkdir -p "${OUTPUT_ROOT}"
SAMPLES_FILE="${OUTPUT_ROOT}/selected_samples.json"

if [[ "${NUM_SAMPLES}" == "3" && -f "${SOURCE_SAMPLES_FILE}" ]]; then
  cp "${SOURCE_SAMPLES_FILE}" "${SAMPLES_FILE}"
else
  "${PYTHON}" "${SCRIPT_DIR}/prepare_humaneval_samples.py" \
    --dataset-name openai/openai_humaneval \
    --dataset-split test \
    --num-samples "${NUM_SAMPLES}" \
    --output "${SAMPLES_FILE}"
fi

run_repeat() {
  local condition="$1"
  local repeat="$2"
  local max_tokens="$3"
  local max_model_len="$4"
  local eager="$5"

  local condition_dir="${OUTPUT_ROOT}/${condition}"
  local run_name="kvarn_run${repeat}"
  local run_dir="${condition_dir}/${run_name}"
  local per_sample_dir="${condition_dir}/per_sample_outputs/${run_name}"
  mkdir -p "${condition_dir}" "${per_sample_dir}"

  local args=(
    --mode kvarn
    --run-name "${run_name}"
    --samples-file "${SAMPLES_FILE}"
    --output-dir "${run_dir}"
    --model "${MODEL}"
    --kvarn-kv-cache-dtype "${KVARN_DTYPE}"
    --block-size "${BLOCK_SIZE}"
    --max-tokens "${max_tokens}"
    --max-model-len "${max_model_len}"
    --temperature 0.0
    --top-p 1.0
    --top-k -1
    --min-p 0.0
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --tensor-parallel-size "${TP_SIZE}"
    --seed "${SEED}"
    --execution-timeout "${EXECUTION_TIMEOUT}"
    --execution-memory-mb "${EXECUTION_MEMORY_MB}"
    --per-sample-output-dir "${per_sample_dir}"
    --require-backend-verification
    --resume
  )
  if [[ "${eager}" == "1" ]]; then
    args+=(--enforce-eager)
  fi
  if [[ "${EXECUTE_TESTS}" == "1" ]]; then
    args+=(--execute-tests)
  else
    args+=(--no-execute-tests)
  fi

  echo "=== ${condition} / ${run_name} ==="
  echo "max_tokens=${max_tokens} max_model_len=${max_model_len} eager=${eager}"
  "${PYTHON}" "${SCRIPT_DIR}/run_humaneval_generation.py" "${args[@]}"
}

analyze_condition() {
  local condition="$1"
  local max_tokens="$2"
  local max_model_len="$3"
  local eager="$4"
  local run1="$5"
  local run2="$6"
  local analysis_dir="${OUTPUT_ROOT}/${condition}/analysis"
  local args=(
    --run1 "${run1}"
    --run2 "${run2}"
    --output-dir "${analysis_dir}"
    --condition-label "${condition}"
    --max-tokens "${max_tokens}"
    --max-model-len "${max_model_len}"
  )
  if [[ "${eager}" == "1" ]]; then
    args+=(--enforce-eager)
  fi
  "${PYTHON}" "${SCRIPT_DIR}/analyze_humaneval_repeat_determinism.py" "${args[@]}"
}

run_and_analyze() {
  local condition="$1"
  local max_tokens="$2"
  local max_model_len="$3"
  local eager="$4"
  run_repeat "${condition}" 1 "${max_tokens}" "${max_model_len}" "${eager}"
  run_repeat "${condition}" 2 "${max_tokens}" "${max_model_len}" "${eager}"
  analyze_condition \
    "${condition}" "${max_tokens}" "${max_model_len}" "${eager}" \
    "${OUTPUT_ROOT}/${condition}/kvarn_run1/generations.jsonl" \
    "${OUTPUT_ROOT}/${condition}/kvarn_run2/generations.jsonl"
}

# A. Reproduce the old deterministic HumanEval geometry on the current code.
run_and_analyze baseline_16k_model32k_graph 16384 32768 0

# B. Change only max_model_len/cache geometry.
run_and_analyze model40k_16k_graph 16384 40960 0

# C. Keep the old max_model_len, but greatly extend the decode horizon.
run_and_analyze cap30k_model32k_graph 30000 32768 0

# D. Current 32K/40K graph point. Reuse the smoke that already exists by default.
if [[ "${REUSE_EXISTING_32K}" == "1" \
      && -f "${EXISTING_32K_ROOT}/kvarn_run1/generations.jsonl" \
      && -f "${EXISTING_32K_ROOT}/kvarn_run2/generations.jsonl" ]]; then
  echo "=== extended_32k_model40k_graph / reusing ${EXISTING_32K_ROOT} ==="
  analyze_condition \
    extended_32k_model40k_graph 32768 40960 0 \
    "${EXISTING_32K_ROOT}/kvarn_run1/generations.jsonl" \
    "${EXISTING_32K_ROOT}/kvarn_run2/generations.jsonl"
else
  run_and_analyze extended_32k_model40k_graph 32768 40960 0
fi

# E. Same current geometry, but disable CUDA graph capture/replay.
run_and_analyze extended_32k_model40k_eager 32768 40960 1

"${PYTHON}" "${SCRIPT_DIR}/summarize_humaneval_determinism_matrix.py" \
  --root "${OUTPUT_ROOT}"

echo
cat "${OUTPUT_ROOT}/matrix_summary.md"
echo
echo "Isolation experiment complete: ${OUTPUT_ROOT}"
