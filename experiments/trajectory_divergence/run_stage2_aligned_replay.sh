#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-/home/wanglin/miniconda3/envs/kvarn-smoke/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:?Set SOURCE_ROOT to the existing four-run trajectory result directory}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOURCE_ROOT}/aligned_replay_1024}"
REFERENCE_GENERATIONS="${REFERENCE_GENERATIONS:-${SOURCE_ROOT}/fp16_run1/generations.jsonl}"
REPLAY_TOKENS="${REPLAY_TOKENS:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
ROPE_SCALING_JSON="${ROPE_SCALING_JSON:-}"
if [[ -z "${ROPE_SCALING_JSON}" ]]; then
  ROPE_SCALING_JSON='{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}'
fi
BLOCK_SIZE="${BLOCK_SIZE:-128}"
TOP_K_LOGITS="${TOP_K_LOGITS:-20}"
FLUSH_EVERY="${FLUSH_EVERY:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TP_SIZE="${TP_SIZE:-1}"
SEED="${SEED:-2026}"
LIMIT="${LIMIT:-}"
KVARN_DTYPE="${KVARN_DTYPE:-kvarn_k4v2_g128}"
REQUIRE_CLEAN_GIT="${REQUIRE_CLEAN_GIT:-0}"

if [[ ! -f "${REFERENCE_GENERATIONS}" ]]; then
  echo "Missing FP16 reference generations: ${REFERENCE_GENERATIONS}" >&2
  exit 2
fi
if [[ "${REQUIRE_CLEAN_GIT}" == "1" ]] && [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  echo "Repository is not clean and REQUIRE_CLEAN_GIT=1" >&2
  exit 2
fi

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable not found: ${PYTHON}" >&2
  exit 2
fi
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
mkdir -p "${OUTPUT_ROOT}"

run_replay() {
  local mode="$1"
  local run_name="$2"
  local out_dir="${OUTPUT_ROOT}/${run_name}"
  local extra=()
  if [[ -n "${LIMIT}" ]]; then
    extra+=(--limit "${LIMIT}")
  fi
  (
    cd /tmp
    "${PYTHON}" "${SCRIPT_DIR}/run_aligned_replay.py" \
      --mode "${mode}" \
      --replay-run-name "${run_name}" \
      --reference-generations "${REFERENCE_GENERATIONS}" \
      --output-dir "${out_dir}" \
      --model "${MODEL}" \
      --tokenizer "${MODEL}" \
      --max-replay-tokens "${REPLAY_TOKENS}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --rope-scaling-json "${ROPE_SCALING_JSON}" \
      --block-size "${BLOCK_SIZE}" \
      --top-k-logits "${TOP_K_LOGITS}" \
      --flush-every "${FLUSH_EVERY}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --tensor-parallel-size "${TP_SIZE}" \
      --seed "${SEED}" \
      --kvarn-kv-cache-dtype "${KVARN_DTYPE}" \
      --require-backend-verification \
      "${extra[@]}"
  )
}

run_replay fp16 fp16_run1
run_replay fp16 fp16_run2
run_replay kvarn kvarn_run1
run_replay kvarn kvarn_run2

"${PYTHON}" "${SCRIPT_DIR}/analyze_aligned_replay.py" \
  --aligned-root "${OUTPUT_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/analysis" \
  --tokenizer "${MODEL}" \
  --block-size "${BLOCK_SIZE}"

echo "Stage-2 aligned replay complete: ${OUTPUT_ROOT}"
