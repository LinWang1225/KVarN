#!/usr/bin/env bash
# HumanEval TurboQuant precision sweep under Qwen3-recommended thinking sampling.
#
# FP16 is generated once per seed and reused across all TurboQuant presets.
# The existing HumanEval runner is intentionally reused without changing vLLM:
# --mode kvarn is only the runner's legacy "quantized candidate" slot; the real
# backend is selected by --kvarn-kv-cache-dtype=turboquant_*.  The generated
# experiment_config.json records the requested/resolved dtype and backend check.
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
BLOCK_SIZE="${BLOCK_SIZE:-128}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TP_SIZE="${TP_SIZE:-1}"
SEEDS="${SEEDS:-2026,2027,2028}"
TQ_PRESETS="${TQ_PRESETS:-turboquant_k8v4,turboquant_4bit_nc,turboquant_k3v4_nc,turboquant_3bit_nc}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
EXECUTE_TESTS="${EXECUTE_TESTS:-1}"
EXECUTION_TIMEOUT="${EXECUTION_TIMEOUT:-30}"
EXECUTION_MEMORY_MB="${EXECUTION_MEMORY_MB:-1024}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/humaneval_turboquant_sweep_qwen3_4b_n${NUM_SAMPLES}}"
RESUME="${RESUME:-0}"

# Optional: reuse already-generated FP16 sampling runs instead of recomputing.
# Pattern must contain literal {seed}, for example:
#   /data/wanglin/KVarN/results/humaneval_sampling_k4v2_qwen3_4b_n164/thinking_on/seed_{seed}/fp16/generations.jsonl
FP16_SOURCE_PATTERN="${FP16_SOURCE_PATTERN:-}"

VALID_PRESETS=(
  turboquant_k8v4
  turboquant_4bit_nc
  turboquant_k3v4_nc
  turboquant_3bit_nc
)

is_valid_preset() {
  local value="$1"
  local item
  for item in "${VALID_PRESETS[@]}"; do
    [[ "${item}" == "${value}" ]] && return 0
  done
  return 1
}

IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
IFS=',' read -r -a PRESET_LIST <<< "${TQ_PRESETS}"

if (( ${#SEED_LIST[@]} < 1 )); then
  echo "SEEDS must contain at least one integer seed." >&2
  exit 2
fi
if (( ${#PRESET_LIST[@]} < 1 )); then
  echo "TQ_PRESETS must contain at least one TurboQuant preset." >&2
  exit 2
fi
for preset in "${PRESET_LIST[@]}"; do
  if ! is_valid_preset "${preset}"; then
    echo "Unsupported TurboQuant preset: ${preset}" >&2
    echo "Valid: ${VALID_PRESETS[*]}" >&2
    exit 2
  fi
done
if (( MAX_TOKENS >= MAX_MODEL_LEN )); then
  echo "MAX_TOKENS must be smaller than MAX_MODEL_LEN." >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
SAMPLES_FILE="${OUTPUT_ROOT}/selected_samples.json"
cd "${REPO_ROOT}"

if [[ ! -f "${SAMPLES_FILE}" ]]; then
  "${PYTHON}" "${SCRIPT_DIR}/prepare_humaneval_samples.py" \
    --dataset-name "${DATASET_NAME}" \
    --dataset-split "${DATASET_SPLIT}" \
    --num-samples "${NUM_SAMPLES}" \
    --output "${SAMPLES_FILE}"
fi

run_generation() {
  local mode="$1"
  local run_name="$2"
  local output_dir="$3"
  local cache_dtype="$4"
  local seed="$5"
  local args=(
    --mode "${mode}"
    --run-name "${run_name}"
    --samples-file "${SAMPLES_FILE}"
    --output-dir "${output_dir}"
    --model "${MODEL}"
    --block-size "${BLOCK_SIZE}"
    --max-tokens "${MAX_TOKENS}"
    --max-model-len "${MAX_MODEL_LEN}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --top-k "${TOP_K}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --tensor-parallel-size "${TP_SIZE}"
    --seed "${seed}"
    --execution-timeout "${EXECUTION_TIMEOUT}"
    --execution-memory-mb "${EXECUTION_MEMORY_MB}"
    --enable-thinking
  )

  if [[ "${mode}" == "kvarn" ]]; then
    args+=(--kvarn-kv-cache-dtype "${cache_dtype}" --require-backend-verification)
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

# Record the exact sweep contract. min_p is intentionally omitted from the
# legacy runner because vLLM SamplingParams defaults min_p=0.0; this is the
# Qwen3 recommended value used by this experiment.
"${PYTHON}" - "${OUTPUT_ROOT}/sweep_config.json" <<PY
import json, sys
path = sys.argv[1]
obj = {
    "experiment": "humaneval_turboquant_precision_sweep",
    "model": "${MODEL}",
    "dataset": "${DATASET_NAME}",
    "num_samples": int("${NUM_SAMPLES}"),
    "seeds": [int(x) for x in "${SEEDS}".split(",") if x],
    "turboquant_presets_high_to_low_precision": "${TQ_PRESETS}".split(','),
    "sampling": {
        "enable_thinking": True,
        "temperature": float("${TEMPERATURE}"),
        "top_p": float("${TOP_P}"),
        "top_k": int("${TOP_K}"),
        "min_p": 0.0,
    },
    "max_tokens": int("${MAX_TOKENS}"),
    "max_model_len": int("${MAX_MODEL_LEN}"),
    "block_size": int("${BLOCK_SIZE}"),
    "fp16_source_pattern": "${FP16_SOURCE_PATTERN}" or None,
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(obj, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY

# Baseline: one FP16 run per seed. If FP16_SOURCE_PATTERN is set, the analyzer
# consumes those existing files and this section is skipped.
if [[ -z "${FP16_SOURCE_PATTERN}" ]]; then
  for seed in "${SEED_LIST[@]}"; do
    output_dir="${OUTPUT_ROOT}/seed_${seed}/fp16"
    if [[ -f "${output_dir}/generations.jsonl" && "${RESUME}" != "1" ]]; then
      echo "Refusing to overwrite ${output_dir}/generations.jsonl; set RESUME=1 or choose a new OUTPUT_ROOT." >&2
      exit 2
    fi
    run_generation fp16 "fp16_seed${seed}" "${output_dir}" auto "${seed}"
  done
fi

# Candidates: precision order is intentionally high -> low.
for preset in "${PRESET_LIST[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    output_dir="${OUTPUT_ROOT}/seed_${seed}/${preset}"
    if [[ -f "${output_dir}/generations.jsonl" && "${RESUME}" != "1" ]]; then
      echo "Refusing to overwrite ${output_dir}/generations.jsonl; set RESUME=1 or choose a new OUTPUT_ROOT." >&2
      exit 2
    fi
    run_generation kvarn "${preset}_seed${seed}" "${output_dir}" "${preset}" "${seed}"
  done
done

ANALYSIS_ARGS=(
  --experiment-root "${OUTPUT_ROOT}"
  --seeds "${SEEDS}"
  --methods "${TQ_PRESETS}"
  --output-dir "${OUTPUT_ROOT}/aggregate"
)
if [[ -n "${FP16_SOURCE_PATTERN}" ]]; then
  ANALYSIS_ARGS+=(--fp16-source-pattern "${FP16_SOURCE_PATTERN}")
fi
"${PYTHON}" "${SCRIPT_DIR}/analyze_humaneval_turboquant_sweep.py" "${ANALYSIS_ARGS[@]}"

cat <<EOF2
TurboQuant HumanEval precision sweep completed.

Main table:
  ${OUTPUT_ROOT}/aggregate/precision_sweep.md
Machine-readable:
  ${OUTPUT_ROOT}/aggregate/precision_sweep.csv
  ${OUTPUT_ROOT}/aggregate/summary.json
Pair-level data:
  ${OUTPUT_ROOT}/aggregate/per_pair_metrics.csv
Task-level cross-seed data:
  ${OUTPUT_ROOT}/aggregate/per_task_method.csv
EOF2
