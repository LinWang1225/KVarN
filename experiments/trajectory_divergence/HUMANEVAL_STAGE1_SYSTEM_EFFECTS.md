# HumanEval Stage 1: trajectory divergence → generation inflation → latency

This patch adds a HumanEval-specific Stage 1 experiment without changing KVarN
attention kernels, cache managers, or the existing MATH-500 experiment scripts.

The goal is to test a stronger system-level hypothesis than "KV quantization can
change a token": under a fixed greedy decoding configuration, sparse FP16/KVarN
trajectory divergence may cause extra reasoning/generation, which in turn may
increase decode work and request latency even when both outputs still pass the
HumanEval tests.

## Files added

```text
experiments/trajectory_divergence/
├── humaneval_utils.py
├── prepare_humaneval_samples.py
├── run_humaneval_generation.py
├── analyze_humaneval_system_effects.py
├── plot_humaneval_system_effects.py
├── run_humaneval_stage1.sh
└── HUMANEVAL_STAGE1_SYSTEM_EFFECTS.md
```

Existing generic components are reused unchanged:

- `compare_trajectories.py` for exact output-token first divergence/LCP;
- `plot_results.py` for the existing trajectory/length figures.

## Default experiment

- dataset: `openai/openai_humaneval`, full 164-task test set;
- model: `Qwen/Qwen3-4B`;
- thinking mode: enabled through the Qwen chat template;
- decoding: greedy (`temperature=0`, `top_p=1`, `top_k=-1`);
- FP16 KV: `kv_cache_dtype=auto`;
- KVarN KV: `kvarn_k4v2_g128`;
- block size: 128;
- prefix caching: disabled;
- output budget: 16384 tokens;
- model context ceiling: 32768 tokens;
- repeats: FP16 x2 and KVarN x2;
- one request at a time (`max_num_seqs=1`);
- HumanEval execution timeout: 30 seconds;
- bootstrap confidence intervals: 5000 paired resamples.

Greedy decoding is intentional: this arm is for trajectory attribution. A
separate sampling arm can later reproduce the paper-style HumanEval setup, but
sampling randomness should not be mixed into the first causal analysis.

## HumanEval prompt and correctness

`prepare_humaneval_samples.py` preserves the HumanEval-native fields:

```text
task_id
prompt
canonical_solution
test
entry_point
```

It also writes the generic `problem`/`sample_id` fields expected by the existing
trajectory analysis scripts.

`run_humaneval_generation.py` wraps the raw HumanEval prompt in a short coding
instruction and the Qwen chat template. It records the exact prompt hash, input
and output token IDs, output text, finish reason, and KVarN backend metadata.

For correctness, the runner:

1. removes the visible reasoning prefix up to the last `</think>`;
2. extracts the final Python code block when present;
3. supports both a full returned function and a completion appended to the
   original HumanEval prompt;
4. appends the HumanEval test and `check(entry_point)` call;
5. executes the candidate in a short-lived Python subprocess;
6. records `passed`, `failed`, `syntax_error`, `memory_error`, or `timeout`.

### Security warning

Generated code is untrusted. The helper applies a wall-clock timeout, isolated
Python mode (`-I`), a temporary working directory, a minimal environment, and
best-effort POSIX CPU/memory/file/process limits. This is **not** a hardened
security sandbox and does not prove network isolation. Run the experiment in a
dedicated container/VM with no credentials, secrets, or sensitive host mounts.

## Per-request metrics

### Quality

```text
fp16_pass
kvarn_pass
correctness_transition
execution_result
```

The most important subgroup is `pass_to_pass`: both modes produce functionally
correct code, so any additional KVarN tokens/latency cannot be dismissed as a
simple correctness failure.

### Output/generation cost

For FP16 output length `N_fp` and KVarN output length `N_q`:

```text
delta_tokens = N_q - N_fp
TIR = (N_q - N_fp) / N_fp
```

The analysis reports mean/median/P90/P95, plus fractions with:

```text
TIR > 10%
TIR > 25%
TIR > 50%
TIR > 100%
TIR < -10%
```

Pairs where either run terminates because of `max_tokens` are marked censored
and excluded from the primary inflation distribution.

### Thinking versus final/code region

When `</think>` is found in the generated token sequence, the runner records:

```text
thinking_tokens
final_tokens
delta_thinking_tokens
delta_final_tokens
reasoning_inflation_ratio
final_inflation_ratio
```

This separates "more internal reasoning" from merely returning more final code.

### System timing

The generation wall-clock time is always recorded:

```text
latency_seconds
tokens_per_second
```

When the vLLM `RequestOutput.metrics` object exposes timestamps, the runner also
records:

```text
ttft_seconds
queue_time_seconds
prefill_time_seconds
decode_time_seconds
tpot_seconds
engine_inference_span_seconds
```

For paired analysis:

```text
E2E LIR    = (T_q - T_fp) / T_fp
Decode LIR = (D_q - D_fp) / D_fp
```

If per-request vLLM timestamps are unavailable in the local build, E2E latency
remains available and `timing_coverage` in `summary.json` makes the missing
coverage explicit.

### Trajectory linkage

The generic `compare_trajectories.py` still provides:

```text
first_divergence_step
lcp_ratio
diverged
```

This patch adds:

```text
normalized_first_divergence = first_divergence_step / FP16_output_tokens
```

and the request groups:

```text
no_divergence
early_[0,0.25)
middle_[0.25,0.5)
late_[0.5,1+]
```

The main correlations are Spearman rank correlations between:

```text
normalized first divergence ↔ TIR
TIR ↔ E2E LIR
TIR ↔ decode LIR
```

These are correlation diagnostics, not yet a causal intervention result.

## Output layout

```text
results/humaneval_stage1_qwen3_4b_n164/
├── selected_samples.json
├── fp16_run1/
│   ├── generations.jsonl
│   └── experiment_config.json
├── fp16_run2/
├── kvarn_run1/
├── kvarn_run2/
├── comparison/
│   ├── per_sample_comparison.csv
│   ├── summary.json
│   ├── summary.md
│   └── plots/
└── system_effects/
    ├── per_sample_system_effects.csv
    ├── per_sample_system_effects.jsonl
    ├── summary.json
    ├── summary.md
    └── plots/
        ├── figure_humaneval_tir_cdf.{png,pdf}
        ├── figure_humaneval_tir_vs_lir.{png,pdf}
        ├── figure_humaneval_divergence_vs_tir.{png,pdf}
        ├── figure_humaneval_tir_by_divergence_group.{png,pdf}
        └── figure_humaneval_thinking_vs_final_delta.{png,pdf}
```

## Smoke test first

Run 3 tasks before the full set:

```bash
cd /data/wanglin/KVarN

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHON=/home/wanglin/miniconda3/envs/kvarn-smoke/bin/python \
MODEL=/data/wanglin/models/Qwen3-4B \
NUM_SAMPLES=3 \
MAX_TOKENS=4096 \
MAX_MODEL_LEN=32768 \
OUTPUT_ROOT=$PWD/results/humaneval_stage1_smoke \
bash experiments/trajectory_divergence/run_humaneval_stage1.sh
```

Inspect at minimum:

```text
results/humaneval_stage1_smoke/fp16_run1/generations.jsonl
results/humaneval_stage1_smoke/kvarn_run1/generations.jsonl
results/humaneval_stage1_smoke/comparison/summary.md
results/humaneval_stage1_smoke/system_effects/summary.md
```

Check that:

- `backend_verified=true` for KVarN;
- the prompt hashes match across modes;
- `execution_result` is populated;
- `candidate_code` is valid-looking Python for representative tasks;
- `timing_coverage.decode_pairs` is non-zero if the local vLLM exposes request
  timestamps;
- FP16/KVarN repeat self-divergence is not unexpectedly large.

## Full 164-task run

```bash
cd /data/wanglin/KVarN

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHON=/home/wanglin/miniconda3/envs/kvarn-smoke/bin/python \
MODEL=/data/wanglin/models/Qwen3-4B \
NUM_SAMPLES=164 \
NUM_REPEATS=2 \
MAX_TOKENS=16384 \
MAX_MODEL_LEN=32768 \
BLOCK_SIZE=128 \
GPU_MEMORY_UTILIZATION=0.90 \
TP_SIZE=1 \
SEED=2026 \
OUTPUT_ROOT=$PWD/results/humaneval_stage1_qwen3_4b_n164 \
bash experiments/trajectory_divergence/run_humaneval_stage1.sh
```

If the HumanEval dataset is not already in the Hugging Face cache, temporarily
run without `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` once to populate the
cache, or point the dataset loader at an already available local copy.

## Primary result checks

A useful system-level signal would have all or most of the following:

1. KVarN `pass@1` remains close to FP16;
2. the `pass_to_pass` subset still has positive TIR;
3. TIR has a non-trivial positive right tail rather than only a tiny mean shift;
4. TIR positively tracks E2E/decode LIR;
5. early divergence has larger TIR/LIR than late/no divergence;
6. extra tokens are concentrated in the thinking region rather than final code.

If these hold, the next experiment should select representative high-inflation,
low-inflation, contraction, and quality-change tasks and run bounded aligned
replay. The replay horizon should be selected from the HumanEval first-divergence
CDF rather than fixed to the previous MATH-500 1024-token horizon.
