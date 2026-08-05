#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-/home/wanglin/miniconda3/envs/kvarn-smoke/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:?Set SOURCE_ROOT to the existing four-run free-generation result directory}"
REPLAY_TOKENS="${REPLAY_TOKENS:-1024}"
DUAL_OUTPUT_ROOT="${DUAL_OUTPUT_ROOT:-${SOURCE_ROOT}/aligned_replay_dual_${REPLAY_TOKENS}}"
RUN_FP16_REFERENCE="${RUN_FP16_REFERENCE:-1}"
RUN_KVARN_REFERENCE="${RUN_KVARN_REFERENCE:-1}"
SUPPLEMENT_OVERWRITE="${SUPPLEMENT_OVERWRITE:-0}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable not found: ${PYTHON}" >&2
  exit 2
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${DUAL_OUTPUT_ROOT}"

run_reference() {
  local source_run="$1"
  local output_name="$2"
  local reference_file="${SOURCE_ROOT}/${source_run}/generations.jsonl"
  local output_root="${DUAL_OUTPUT_ROOT}/${output_name}"

  if [[ ! -f "${reference_file}" ]]; then
    echo "Missing reference generations: ${reference_file}" >&2
    exit 2
  fi

  echo "=== Stage 2 reference: ${source_run} ==="
  SOURCE_ROOT="${SOURCE_ROOT}" \
  REFERENCE_GENERATIONS="${reference_file}" \
  OUTPUT_ROOT="${output_root}" \
  REPLAY_TOKENS="${REPLAY_TOKENS}" \
  PYTHON="${PYTHON}" \
  bash "${SCRIPT_DIR}/run_stage2_aligned_replay.sh"

  local overwrite=()
  if [[ "${SUPPLEMENT_OVERWRITE}" == "1" ]]; then
    overwrite+=(--overwrite)
  fi
  "${PYTHON}" "${SCRIPT_DIR}/analyze_aligned_replay_supplement.py" \
    --aligned-root "${output_root}" \
    --analysis-dir "${output_root}/analysis" \
    "${overwrite[@]}"
}

if [[ "${RUN_FP16_REFERENCE}" == "1" ]]; then
  run_reference fp16_run1 fp16_reference
fi
if [[ "${RUN_KVARN_REFERENCE}" == "1" ]]; then
  run_reference kvarn_run1 kvarn_reference
fi

echo "Dual-reference aligned replay complete: ${DUAL_OUTPUT_ROOT}"
