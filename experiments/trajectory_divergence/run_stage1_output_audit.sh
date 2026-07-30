#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_TOKENS="${MAX_TOKENS:-38912}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
BOUNDARY_STEP="${BOUNDARY_STEP:-${BLOCK_SIZE}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TP_SIZE="${TP_SIZE:-1}"
SEED="${SEED:-2026}"
DATASET_NAME="${DATASET_NAME:-HuggingFaceH4/MATH-500}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
ROPE_SCALING_JSON="${ROPE_SCALING_JSON:-{\"rope_type\":\"yarn\",\"factor\":2.0,\"original_max_position_embeddings\":32768}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/trajectory_stage1_qwen3_4b_n${NUM_SAMPLES}_65k}"
AUDIT_OUTPUT_DIR="${AUDIT_OUTPUT_DIR:-${OUTPUT_ROOT}/output_audit}"
KVARN_DTYPE="${KVARN_DTYPE:-kvarn_k4v2_g128}"
REQUIRE_CLEAN_GIT="${REQUIRE_CLEAN_GIT:-0}"
AUDIT_OVERWRITE="${AUDIT_OVERWRITE:-0}"

# Stage 1 requires exactly two independent processes per method so that every
# FP16 and KVarN token trajectory has a same-mode repeat for determinism checks.
NUM_REPEATS=2
export NUM_REPEATS

# Keep the deterministic controls used by the successful pilot. These variables
# are inherited by both repeats and are also captured in experiment_config.json
# by the existing generation script where supported.
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"

if [[ "${MAX_TOKENS}" -ge "${MAX_MODEL_LEN}" ]]; then
  echo "MAX_TOKENS must be smaller than MAX_MODEL_LEN" >&2
  exit 2
fi

if [[ "${REQUIRE_CLEAN_GIT}" == "1" ]]; then
  if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
    echo "Repository is not clean and REQUIRE_CLEAN_GIT=1" >&2
    git -C "${REPO_ROOT}" status --short >&2
    exit 2
  fi
fi

mkdir -p "${OUTPUT_ROOT}"

# Reuse the existing full experiment, but force two runs for both methods.
PYTHON="${PYTHON}" \
MODEL="${MODEL}" \
NUM_SAMPLES="${NUM_SAMPLES}" \
NUM_REPEATS="${NUM_REPEATS}" \
MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
MAX_TOKENS="${MAX_TOKENS}" \
BLOCK_SIZE="${BLOCK_SIZE}" \
BOUNDARY_STEP="${BOUNDARY_STEP}" \
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
TP_SIZE="${TP_SIZE}" \
SEED="${SEED}" \
DATASET_NAME="${DATASET_NAME}" \
DATASET_SPLIT="${DATASET_SPLIT}" \
ROPE_SCALING_JSON="${ROPE_SCALING_JSON}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
KVARN_DTYPE="${KVARN_DTYPE}" \
REQUIRE_CLEAN_GIT="${REQUIRE_CLEAN_GIT}" \
bash "${SCRIPT_DIR}/run_full_experiment.sh"

AUDIT_ARGS=(
  "${SCRIPT_DIR}/audit_repeat_outputs.py"
  --output-root "${OUTPUT_ROOT}"
  --output-dir "${AUDIT_OUTPUT_DIR}"
  --tokenizer "${MODEL}"
  --block-size "${BLOCK_SIZE}"
  --window-radius 24
  --trust-remote-code
)
if [[ "${AUDIT_OVERWRITE}" == "1" ]]; then
  AUDIT_ARGS+=(--overwrite)
fi

"${PYTHON}" "${AUDIT_ARGS[@]}"

echo "Stage-1 audit complete"
echo "  experiment root: ${OUTPUT_ROOT}"
echo "  audit summary:   ${AUDIT_OUTPUT_DIR}/summary.md"
echo "  exact outputs:   ${AUDIT_OUTPUT_DIR}/raw_outputs/"
echo "  exact token IDs: ${AUDIT_OUTPUT_DIR}/raw_tokens/"
echo "  sample reports:  ${AUDIT_OUTPUT_DIR}/samples/"
