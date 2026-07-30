# Stage 2: teacher-forced aligned logits replay

This stage follows the four-run output audit. It does **not** replace the free-generation files. It uses `fp16_run1/generations.jsonl` as one fixed token reference and replays the same prefix through four fresh vLLM engines:

- FP16 aligned run 1;
- FP16 aligned run 2;
- KVarN aligned run 1;
- KVarN aligned run 2.

A custom vLLM V1 logits processor records raw next-token statistics **before** changing the logits, then masks every token except the next FP16 reference token. Consequently all four engines receive the same generated-token history at every replay step. This makes per-step comparisons meaningful after the point where free-running outputs would have diverged.

## Why the default replay length is 1024

The existing output audit reports:

- cross FP16/KVarN first-divergence P90 around 358–371 tokens;
- KVarN self-divergence median 401.5 and P90 about 737 tokens.

Replaying 1024 tokens covers the observed P90 while keeping four-run cost well below full 6K–39K generation. Increase `REPLAY_TOKENS` only after inspecting the Stage-2 CDF.

## Files added

```text
experiments/trajectory_divergence/
├── teacher_forced_logits_processor.py
├── run_aligned_replay.py
├── analyze_aligned_replay.py
├── run_stage2_aligned_replay.sh
└── STAGE2_ALIGNED_REPLAY.md
```

No KVarN attention kernel or cache-manager file is modified.

## Run on the existing 20-sample result

```bash
cd /data/wanglin/KVarN

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHON=/home/wanglin/miniconda3/envs/kvarn-smoke/bin/python \
MODEL=/data/wanglin/models/Qwen3-4B \
SOURCE_ROOT=$PWD/results/trajectory_pilot_qwen3_4b_n20_65k_v2 \
OUTPUT_ROOT=$PWD/results/trajectory_pilot_qwen3_4b_n20_65k_v2/aligned_replay_1024 \
REPLAY_TOKENS=1024 \
MAX_MODEL_LEN=65536 \
ROPE_SCALING_JSON='{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' \
BLOCK_SIZE=128 \
TOP_K_LOGITS=20 \
GPU_MEMORY_UTILIZATION=0.90 \
TP_SIZE=1 \
bash experiments/trajectory_divergence/run_stage2_aligned_replay.sh
```

The wrapper always runs two independent engines per mode. Set `LIMIT=3` for a functional smoke test without changing the fixed sample manifest.

## Raw metrics per step

Each run writes one file per sample:

```text
aligned_replay_1024/<run>/metrics/<sample_id>.jsonl
```

Every line includes:

- exact forced FP16 reference token;
- raw top-1 and top-2 token IDs before masking;
- raw top-1/top-2 log probabilities and margin;
- FP16 reference-token log probability and exact rank;
- raw top-k token IDs/logits/log probabilities;
- output-relative replay step;
- absolute token position, 128-token block index and block offset;
- thinking/boundary/final-answer region when `</think>` is detectable.

Each run also writes `replay_records.jsonl`, which stores the full forced token prefix and the full token prefix actually returned by vLLM. `replay_verified=true` and `metrics_complete=true` are required before analysis.

## Analysis outputs

```text
aligned_replay_1024/analysis/
├── summary.json
├── summary.md
├── per_step_aligned.csv
├── per_step_aligned.jsonl
└── plots/
    ├── figure_aligned_first_disagreement_cdf.{png,pdf}
    ├── figure_aligned_disagreement_by_step.{png,pdf}
    ├── figure_aligned_reference_logprob_drop.{png,pdf}
    ├── figure_aligned_block_offset_disagreement.{png,pdf}
    └── figure_aligned_margin_at_cross_disagreement.{png,pdf}
```

## How to interpret the result

- `kvarn_self` raw-top1 disagreement under the same forced history is direct evidence that two KVarN engine runs produce different next-token decisions even before trajectory feedback is allowed.
- A positive `cross_run*_reference_logprob_drop` means the FP16 reference token is assigned lower probability by KVarN at the identical history.
- Cross-mode raw-top1 disagreements concentrated at low FP16 top1 margins support the hypothesis that small KV-induced logit changes flip fragile decisions.
- Disagreement peaks at particular `absolute_offset_in_block` values motivate the next stage: layer/K/V/block instrumentation and cache intervention.

This stage records top-k and targeted scalar statistics rather than the full vocabulary logits. It therefore does not claim an exact full-vocabulary KL divergence. Exact KL would require retaining or comparing the complete logits tensor and would substantially increase storage and transfer cost.
