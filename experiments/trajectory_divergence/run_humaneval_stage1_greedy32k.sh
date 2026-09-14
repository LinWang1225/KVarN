#!/usr/bin/env bash
# HumanEval Stage-1 extended-cap diagnostic.
#
# Keeps the controlled greedy-decoding mechanism experiment, raises the output
# cap from 16K to 32K, and saves one human-readable artifact per task/run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

NUM_SAMPLES="${NUM_SAMPLES:-164}"
export NUM_SAMPLES
export MAX_TOKENS="${MAX_TOKENS:-32768}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/humaneval_stage1_greedy32k_qwen3_4b_n${NUM_SAMPLES}}"
export PER_SAMPLE_OUTPUT_ROOT="${PER_SAMPLE_OUTPUT_ROOT:-${OUTPUT_ROOT}/per_sample_outputs}"

# This diagnostic intentionally preserves deterministic greedy decoding so that
# FP16-vs-KVarN token divergence remains attributable to the KV-cache mode.
export TEMPERATURE=0.0
export TOP_P=1.0
export TOP_K=-1

# Qwen3-4B's current config has max_position_embeddings=40960, which is enough
# for a 32768-token output plus a short HumanEval prompt. Keep YaRN disabled by
# default; callers may still override this variable if their local model config
# differs and they deliberately want a RoPE-scaling experiment.
export ROPE_SCALING_JSON="${ROPE_SCALING_JSON:-}"

echo "=== HumanEval greedy extended-cap diagnostic ==="
echo "MAX_TOKENS=${MAX_TOKENS}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "TEMPERATURE=${TEMPERATURE} TOP_P=${TOP_P} TOP_K=${TOP_K}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "PER_SAMPLE_OUTPUT_ROOT=${PER_SAMPLE_OUTPUT_ROOT}"

exec bash "${SCRIPT_DIR}/run_humaneval_stage1.sh"
