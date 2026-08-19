#!/usr/bin/env bash
# HumanEval K4V2 production-validity experiment under Qwen3-recommended sampling.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
DATASET_NAME="${DATASET_NAME:-openai/openai_humaneval}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
NUM_SAMPLES="${NUM_SAMPLES:-164}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
ROPE_SCALING_JSON="${ROPE_SCALING_JSON:-}"
KVARN_DTYPE="${KVARN_DTYPE:-kvarn_k4v2_g128}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TP_SIZE="${TP_SIZE:-1}"
SEEDS="${SEEDS:-2026,2027,2028}"
EXECUTE_TESTS="${EXECUTE_TESTS:-1}"
EXECUTION_TIMEOUT="${EXECUTION_TIMEOUT:-30}"
EXECUTION_MEMORY_MB="${EXECUTION_MEMORY_MB:-1024}"
RESUME="${RESUME:-0}"
RUN_THINKING_OFF="${RUN_THINKING_OFF:-0}"
REQUIRE_CLEAN_GIT="${REQUIRE_CLEAN_GIT:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/humaneval_sampling_k4v2_qwen3_4b_n${NUM_SAMPLES}}"

# Qwen3 official/recommended sampling parameters.
THINKING_TEMPERATURE="${THINKING_TEMPERATURE:-0.6}"
THINKING_TOP_P="${THINKING_TOP_P:-0.95}"
THINKING_TOP_K="${THINKING_TOP_K:-20}"
# min_p=0 is the vLLM SamplingParams default in this repository.

# Official Qwen3 non-thinking recommendation. This arm is optional and is a
# mechanism/control experiment rather than the primary benchmark arm.
NONTHINKING_TEMPERATURE="${NONTHINKING_TEMPERATURE:-0.7}"
NONTHINKING_TOP_P="${NONTHINKING_TOP_P:-0.8}"
NONTHINKING_TOP_K="${NONTHINKING_TOP_K:-20}"

if [[ "${KVARN_DTYPE}" != "kvarn_k4v2_g128" ]]; then
  echo "This validation intentionally targets the shipped production K4V2 preset only." >&2
  echo "Expected KVARN_DTYPE=kvarn_k4v2_g128, got ${KVARN_DTYPE}." >&2
  exit 2
fi
if (( MAX_TOKENS >= MAX_MODEL_LEN )); then
  echo "MAX_TOKENS must be smaller than MAX_MODEL_LEN." >&2
  exit 2
fi

IFS=',' read -r -a SEED_ARRAY <<< "${SEEDS}"
if (( ${#SEED_ARRAY[@]} < 3 )); then
  echo "Use at least 3 independent seeds for the sampling validation." >&2
  exit 2
fi
for raw_seed in "${SEED_ARRAY[@]}"; do
  if ! [[ "${raw_seed}" =~ ^[0-9]+$ ]]; then
    echo "Invalid seed in SEEDS=${SEEDS}: ${raw_seed}" >&2
    exit 2
  fi
done

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

if [[ ! -f "${SAMPLES_FILE}" ]]; then
  "${PYTHON}" "${SCRIPT_DIR}/prepare_humaneval_samples.py" \
    --dataset-name "${DATASET_NAME}" \
    --dataset-split "${DATASET_SPLIT}" \
    --num-samples "${NUM_SAMPLES}" \
    --output "${SAMPLES_FILE}"
fi

cat > "${OUTPUT_ROOT}/sampling_validation_config.json" <<EOF
{
  "experiment": "humaneval_k4v2_sampling_validation",
  "model": "${MODEL}",
  "dataset": "${DATASET_NAME}",
  "num_samples": ${NUM_SAMPLES},
  "kvarn_dtype": "${KVARN_DTYPE}",
  "seeds": "${SEEDS}",
  "max_tokens": ${MAX_TOKENS},
  "max_model_len": ${MAX_MODEL_LEN},
  "thinking_on": {
    "enable_thinking": true,
    "temperature": ${THINKING_TEMPERATURE},
    "top_p": ${THINKING_TOP_P},
    "top_k": ${THINKING_TOP_K},
    "min_p": 0.0
  },
  "thinking_off": {
    "enabled": ${RUN_THINKING_OFF},
    "enable_thinking": false,
    "temperature": ${NONTHINKING_TEMPERATURE},
    "top_p": ${NONTHINKING_TOP_P},
    "top_k": ${NONTHINKING_TOP_K},
    "min_p": 0.0
  }
}
EOF

run_one() {
  local condition="$1"
  local enable_thinking="$2"
  local temperature="$3"
  local top_p="$4"
  local top_k="$5"
  local seed="$6"
  local mode="$7"
  local output_dir="${OUTPUT_ROOT}/${condition}/seed_${seed}/${mode}"
  local run_name="${condition}_${mode}_seed${seed}"

  mkdir -p "${output_dir}"
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
    --temperature "${temperature}"
    --top-p "${top_p}"
    --top-k "${top_k}"
    --min-p "0.0"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --tensor-parallel-size "${TP_SIZE}"
    --seed "${seed}"
    --execution-timeout "${EXECUTION_TIMEOUT}"
    --execution-memory-mb "${EXECUTION_MEMORY_MB}"
    --require-backend-verification
  )

  if [[ -n "${ROPE_SCALING_JSON}" ]]; then
    args+=(--rope-scaling-json "${ROPE_SCALING_JSON}")
  fi
  if [[ "${enable_thinking}" == "1" ]]; then
    args+=(--enable-thinking)
  else
    args+=(--no-enable-thinking)
  fi
  if [[ "${EXECUTE_TESTS}" == "1" ]]; then
    args+=(--execute-tests)
  else
    args+=(--no-execute-tests)
  fi
  if [[ "${RESUME}" == "1" ]]; then
    args+=(--resume)
  fi

  "${PYTHON}" "${SCRIPT_DIR}/run_humaneval_generation.py" "${args[@]}"
}

run_condition() {
  local condition="$1"
  local enable_thinking="$2"
  local temperature="$3"
  local top_p="$4"
  local top_k="$5"

  echo "=== ${condition}: temperature=${temperature}, top_p=${top_p}, top_k=${top_k}, seeds=${SEEDS} ==="
  for seed in "${SEED_ARRAY[@]}"; do
    # Same seed on FP16 and KVarN makes the request pair maximally controlled,
    # while independent seeds quantify sampling variance across runs.
    run_one "${condition}" "${enable_thinking}" "${temperature}" "${top_p}" "${top_k}" "${seed}" fp16
    run_one "${condition}" "${enable_thinking}" "${temperature}" "${top_p}" "${top_k}" "${seed}" kvarn
  done

  "${PYTHON}" "${SCRIPT_DIR}/analyze_humaneval_sampling_validation.py" \
    --condition-dir "${OUTPUT_ROOT}/${condition}" \
    --seeds "${SEEDS}" \
    --condition-label "${condition}" \
    --output-dir "${OUTPUT_ROOT}/${condition}/aggregate"
}

# Primary production-validity arm: Qwen3 official thinking sampling.
run_condition thinking_on 1 \
  "${THINKING_TEMPERATURE}" "${THINKING_TOP_P}" "${THINKING_TOP_K}"

# Optional mechanism control: hard-disable thinking and use Qwen3's recommended
# non-thinking sampling. Run only after the primary arm unless compute is cheap.
if [[ "${RUN_THINKING_OFF}" == "1" ]]; then
  run_condition thinking_off 0 \
    "${NONTHINKING_TEMPERATURE}" "${NONTHINKING_TOP_P}" "${NONTHINKING_TOP_K}"

  "${PYTHON}" "${SCRIPT_DIR}/compare_humaneval_sampling_conditions.py" \
    --thinking-on "${OUTPUT_ROOT}/thinking_on/aggregate/summary.json" \
    --thinking-off "${OUTPUT_ROOT}/thinking_off/aggregate/summary.json" \
    --output-dir "${OUTPUT_ROOT}/condition_comparison"
fi

cat <<EOF
HumanEval K4V2 sampling validation completed.
Output root: ${OUTPUT_ROOT}
Primary summary:
  ${OUTPUT_ROOT}/thinking_on/aggregate/summary.md
Primary per-request data:
  ${OUTPUT_ROOT}/thinking_on/aggregate/per_seed_request_metrics.csv
Primary per-task cross-seed data:
  ${OUTPUT_ROOT}/thinking_on/aggregate/per_task_across_seeds.csv
EOF
if [[ "${RUN_THINKING_OFF}" == "1" ]]; then
  cat <<EOF
Thinking-off summary:
  ${OUTPUT_ROOT}/thinking_off/aggregate/summary.md
Thinking-mode comparison:
  ${OUTPUT_ROOT}/condition_comparison/thinking_mode_comparison.md
EOF
fi
