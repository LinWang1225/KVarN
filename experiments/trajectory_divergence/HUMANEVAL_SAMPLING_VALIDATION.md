# HumanEval K4V2 sampling validation

This experiment is the production-validity follow-up to the deterministic HumanEval
trajectory-divergence arm. It keeps the shipped KVarN production preset
`kvarn_k4v2_g128` and switches Qwen3 to its recommended stochastic decoding.

The goal is not to reproduce the paper's 2/2-bit KVarN row. The goal is to answer:

> Under realistic Qwen3 sampling, does the production K4V2 preset still create a
> meaningful generation-length / decode-latency tail, and is that tail stable
> across independent sampling seeds?

## Why this is a separate arm

The existing `run_humaneval_stage1.sh` intentionally uses greedy decoding to make
first token divergence attributable to the KV-cache mode. Greedy is useful for
mechanism study, but Qwen3 recommends stochastic sampling in thinking mode.

Under sampling, exact FP16-vs-KVarN token divergence is no longer a clean causal
metric: even two full-precision stochastic runs can choose different tokens.
Therefore this arm does **not** use `compare_trajectories.py` as its primary
analysis. Instead it uses:

1. the same seed for each FP16/KVarN request pair;
2. three independent seeds by default;
3. per-seed benchmark-style aggregate statistics;
4. pooled paired request distributions;
5. per-task consistency across seeds.

The original greedy trajectory experiment remains unchanged and should still be
used for first-divergence / aligned-replay mechanism analysis.

## Primary configuration

The default `thinking_on` arm uses the Qwen3 recommended thinking configuration:

```text
enable_thinking = true
temperature = 0.6
top_p = 0.95
top_k = 20
min_p = 0
```

`min_p=0` is passed explicitly by this experiment. The patch adds a backward-compatible
`--min-p` option to `run_humaneval_generation.py`; its default remains `0.0`, so the
existing greedy experiment is unchanged.

Other defaults:

```text
model = Qwen3-4B
dataset = HumanEval 164
KV baseline = FP16 / auto
KV quantization = kvarn_k4v2_g128 only
max_tokens = 16384
max_model_len = 32768
block_size = 128
prefix caching = off
max_num_seqs = 1
seeds = 2026,2027,2028
HumanEval test timeout = 30 s
```

The script rejects non-K4V2 KVarN dtypes intentionally. This experiment is about
the shipped production configuration, not the paper's 2/2-bit research setup.

## Optional thinking-off control

Set `RUN_THINKING_OFF=1` after the main arm is validated. It hard-disables Qwen3
thinking and uses the Qwen3 recommended non-thinking core sampling values:

```text
enable_thinking = false
temperature = 0.7
top_p = 0.8
top_k = 20
min_p = 0
```

This is a mechanism control. If K4V2 has a strong positive tail with thinking on
but the tail weakens substantially with thinking off, that supports a
reasoning-trajectory-specific effect rather than a generic generation effect.

## Patch scope

```text
experiments/trajectory_divergence/
├── run_humaneval_generation.py            # minimal: add --min-p passthrough/recording
├── run_humaneval_sampling_validation.sh    # new
├── analyze_humaneval_sampling_validation.py# new
├── compare_humaneval_sampling_conditions.py# new
└── HUMANEVAL_SAMPLING_VALIDATION.md         # new
```

The existing greedy defaults remain unchanged. No aligned-replay, KVarN kernel,
attention, or cache-manager code is modified.

## Smoke test

Use a small sample count and shorter output cap only to validate the pipeline:

```bash
cd /data/wanglin/KVarN

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHON=/home/wanglin/miniconda3/envs/kvarn-smoke/bin/python \
MODEL=/data/wanglin/models/Qwen3-4B \
NUM_SAMPLES=3 \
MAX_TOKENS=4096 \
MAX_MODEL_LEN=32768 \
SEEDS=2026,2027,2028 \
OUTPUT_ROOT=$PWD/results/humaneval_sampling_k4v2_smoke \
bash experiments/trajectory_divergence/run_humaneval_sampling_validation.sh
```

Check:

```text
thinking_on/seed_2026/fp16/experiment_config.json
thinking_on/seed_2026/kvarn/experiment_config.json
thinking_on/aggregate/summary.md
thinking_on/aggregate/per_seed_request_metrics.csv
thinking_on/aggregate/per_task_across_seeds.csv
```

For every KVarN run, `backend_verified` should be true and the recorded dtype
should be `kvarn_k4v2_g128`.

## Full primary run

```bash
cd /data/wanglin/KVarN

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHON=/home/wanglin/miniconda3/envs/kvarn-smoke/bin/python \
MODEL=/data/wanglin/models/Qwen3-4B \
NUM_SAMPLES=164 \
MAX_TOKENS=16384 \
MAX_MODEL_LEN=32768 \
BLOCK_SIZE=128 \
GPU_MEMORY_UTILIZATION=0.90 \
TP_SIZE=1 \
SEEDS=2026,2027,2028 \
OUTPUT_ROOT=$PWD/results/humaneval_sampling_k4v2_qwen3_4b_n164 \
bash experiments/trajectory_divergence/run_humaneval_sampling_validation.sh
```

Resume an interrupted run with:

```bash
RESUME=1 \
OUTPUT_ROOT=$PWD/results/humaneval_sampling_k4v2_qwen3_4b_n164 \
bash experiments/trajectory_divergence/run_humaneval_sampling_validation.sh
```

## Optional thinking-off control

After the primary result is inspected:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHON=/home/wanglin/miniconda3/envs/kvarn-smoke/bin/python \
MODEL=/data/wanglin/models/Qwen3-4B \
NUM_SAMPLES=164 \
MAX_TOKENS=16384 \
MAX_MODEL_LEN=32768 \
SEEDS=2026,2027,2028 \
RUN_THINKING_OFF=1 \
RESUME=1 \
OUTPUT_ROOT=$PWD/results/humaneval_sampling_k4v2_qwen3_4b_n164 \
bash experiments/trajectory_divergence/run_humaneval_sampling_validation.sh
```

Because `RESUME=1`, already completed thinking-on runs are reused and only the
missing thinking-off arm is generated.

## Output layout

```text
results/humaneval_sampling_k4v2_qwen3_4b_n164/
├── selected_samples.json
├── thinking_on/
│   ├── seed_2026/
│   │   ├── fp16/{generations.jsonl,experiment_config.json}
│   │   └── kvarn/{generations.jsonl,experiment_config.json}
│   ├── seed_2027/
│   ├── seed_2028/
│   └── aggregate/
│       ├── per_seed_request_metrics.csv
│       ├── per_seed_request_metrics.jsonl
│       ├── per_task_across_seeds.csv
│       ├── per_task_across_seeds.jsonl
│       ├── per_seed_summary.csv
│       ├── summary.json
│       └── summary.md
├── thinking_off/                    # optional
└── condition_comparison/            # optional
    ├── thinking_mode_comparison.csv
    └── thinking_mode_comparison.md
```

## What the aggregate summary means

### 1. Paper-style three-run aggregate

For each seed, compute the dataset-wide mean output tokens and pass@1 for FP16 and
K4V2. Then report mean ± standard deviation across seeds.

This is the closest number in this experiment to the `# Tokens (mean ± std)`
style used in benchmark tables. Capped outputs are included, so the summary also
reports censoring separately.

### 2. Pooled paired request distribution

For each seed and HumanEval task, pair FP16 and K4V2 using the same seed and
compute:

```text
TIR = (KVarN output tokens - FP16 output tokens) / FP16 output tokens
E2E LIR
Decode LIR
TPOT change
TTFT change
```

Pairs where either side hits `max_tokens` are excluded from the primary TIR/LIR
distribution.

The summary directly reports median / mean / P90 / P95 and the fraction of
requests with:

```text
TIR > 10%
TIR > 25%
TIR > 50%
TIR > 100%
TIR < -10%
|TIR| <= 10%
```

### 3. Severe-inflation thinking contribution

For requests with `TIR > 50%`, compute:

```text
sum(delta_thinking_tokens) / sum(delta_total_tokens)
```

and repeat for pass→pass requests. This directly answers whether a severe
positive tail is mainly extra reasoning rather than extra final code.

### 4. Cross-seed robustness

Sampling can create large one-off differences. To avoid treating a lucky/unlucky
seed as a K4V2 effect, `per_task_across_seeds.csv` counts how many seeds show a
positive tail for each task.

The main robust counts are tasks that are uncensored in all seeds and are:

```text
>10% longer in >=2/3 seeds
>25% longer in >=2/3 seeds
>50% longer in >=2/3 seeds
>10% shorter in >=2/3 seeds
>10% longer in all 3 seeds
>50% longer in all 3 seeds
```

A tail that appears only in pooled request-seed pairs but is not repeatable on
the same tasks across seeds is much weaker evidence for a controller.

## Decision rule for the research direction

Do not judge the experiment by mean TIR alone. Use the following as research
decision heuristics, not universal production SLO thresholds:

### Continue toward serving / consequential-divergence analysis if

- P95 TIR remains roughly 30-50% or larger under recommended sampling;
- P95 decode-latency inflation is similarly substantial;
- a non-trivial set of tasks are >50% longer in >=2/3 seeds;
- the positive tail remains visible in pass→pass requests;
- severe extra tokens remain concentrated in thinking.

Then the next experiment should be a serving-load test to determine whether
these request-level stragglers amplify P95/P99 latency and reduce SLO goodput.

### De-prioritize HumanEval generation inflation if

- the positive tail largely disappears under recommended sampling;
- severe inflation is mostly seed-specific and not task-stable;
- K4V2's per-token overhead dominates while trajectory-induced work is small;
- thinking-off and thinking-on behave similarly and neither produces a
  meaningful production tail.

In that case, move to a longer-horizon workload (e.g. harder coding or agentic
coding) or change the research target from generic divergence to consequential
divergence / cache-risk prediction.
