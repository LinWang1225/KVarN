# Stage 2 follow-up: dual-reference aligned replay

This follow-up keeps the existing FP16-reference aligned replay and adds a
second replay whose forced token history comes from `kvarn_run1` free
generation.

The distinction matters:

- **FP16 reference** measures the systematic FP16/KVarN logit shift along the
  FP16 natural path.
- **KVarN reference** tests whether two independent KVarN engines remain
  repeatable along a path that KVarN actually generated during Stage 1.

The original Stage-2 files are not replaced.  This patch adds:

```text
experiments/trajectory_divergence/
├── analyze_aligned_replay_supplement.py
├── run_stage2_dual_reference_replay.sh
└── STAGE2_DUAL_REFERENCE_REPLAY.md
```

## Supplemental diagnostics

For either reference path, the supplemental analyzer reads the existing
`analysis/per_step_aligned.csv` and adds:

```text
analysis/
├── supplemental_summary.json
├── supplemental_summary.md
├── reference_mismatch_steps.csv
├── cross_disagreement_steps.csv
├── margin_flip_probability.csv
└── plots/
    ├── figure_aligned_reference_adherence_by_step.{png,pdf}
    ├── figure_aligned_reference_logprob_abs_by_step.{png,pdf}
    ├── figure_aligned_reference_logprob_at_disagreement.{png,pdf}
    └── figure_aligned_margin_flip_probability.{png,pdf}
```

`reference adherence` means that the unmodified raw top-1 equals the forced
source token before masking.  Teacher forcing remains valid when adherence is
below 1.0, but the forced source trajectory is then not an exact natural replay
under the aligned V1 runner.

The new log-probability plots avoid the main weakness of the original median
plot: sparse but large shifts are no longer hidden by bins containing mostly
near-zero steps.

## Add supplemental analysis to an existing Stage-2 result

```bash
cd /data/wanglin/KVarN

PYTHONPATH=$PWD \
/home/wanglin/miniconda3/envs/kvarn-smoke/bin/python \
  experiments/trajectory_divergence/analyze_aligned_replay_supplement.py \
  --aligned-root results/stage2_aligned_smoke_kvarn_fork \
  --analysis-dir results/stage2_aligned_smoke_kvarn_fork/analysis
```

Use `--overwrite` to regenerate supplemental files.

## Run both FP16-reference and KVarN-reference replay

The wrapper invokes the existing `run_stage2_aligned_replay.sh` twice.  Each
reference runs four fresh engines (`FP16 x2`, `KVarN x2`).

```bash
cd /data/wanglin/KVarN

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHON=/home/wanglin/miniconda3/envs/kvarn-smoke/bin/python \
MODEL=/data/wanglin/models/Qwen3-4B \
SOURCE_ROOT=$PWD/results/trajectory_pilot_qwen3_4b_n20_65k_v2 \
DUAL_OUTPUT_ROOT=$PWD/results/stage2_dual_reference_n20_1024 \
LIMIT=20 \
REPLAY_TOKENS=1024 \
MAX_MODEL_LEN=65536 \
ROPE_SCALING_JSON='{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' \
BLOCK_SIZE=128 \
TOP_K_LOGITS=20 \
GPU_MEMORY_UTILIZATION=0.90 \
TP_SIZE=1 \
bash experiments/trajectory_divergence/run_stage2_dual_reference_replay.sh
```

Outputs:

```text
stage2_dual_reference_n20_1024/
├── fp16_reference/
│   ├── fp16_run1/
│   ├── fp16_run2/
│   ├── kvarn_run1/
│   ├── kvarn_run2/
│   └── analysis/
└── kvarn_reference/
    ├── fp16_run1/
    ├── fp16_run2/
    ├── kvarn_run1/
    ├── kvarn_run2/
    └── analysis/
```

Set `RUN_FP16_REFERENCE=0` or `RUN_KVARN_REFERENCE=0` to run only one side.

## Key interpretation

For the **FP16-reference** directory:

- cross-mode low-margin flips quantify the systematic quantization shift;
- FP16 reference adherence reports whether the V1 aligned runner naturally
  reproduces the original FP16 free path.

For the **KVarN-reference** directory:

- `kvarn_self` raw-top1 disagreements are direct evidence of repeatability
  failure under identical KVarN-generated history;
- if `kvarn_self` remains zero while Stage 1 free runs self-diverge, compare
  Stage-1 engine configuration, runner version, autotune state, and the exact
  KVarN reference run used for replay;
- KVarN source adherence below 1.0 means the aligned runner does not naturally
  reproduce every Stage-1 KVarN token, so those mismatch rows should be checked
  before attributing later behavior to trajectory feedback.
