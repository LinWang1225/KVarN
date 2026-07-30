# Stage 1: exact output and repeat-determinism audit

This patch is the first step of the trajectory-divergence study. It deliberately does **not** add teacher-forced logits hooks or modify KVarN kernels yet. Before internal-state analysis, it makes the four free-generation trajectories fully auditable:

- FP16 run 1;
- FP16 run 2;
- KVarN run 1;
- KVarN run 2.

The existing `run_generation.py` already writes the exact `output_token_ids` returned by vLLM and the exact `candidate.text` string to each run's `generations.jsonl`. This patch keeps those files unchanged and adds a consolidated audit layer that is easier to inspect manually.

## Files added

```text
experiments/trajectory_divergence/
├── audit_repeat_outputs.py
├── run_stage1_output_audit.sh
└── STAGE1_OUTPUT_AUDIT.md
```

No KVarN attention/kernel source is changed.

## What is recorded

For every sample and every one of the four runs, the audit writes:

- exact prompt text and prompt hash;
- exact input token IDs;
- exact output token IDs;
- exact vLLM output text;
- a second decode from `output_token_ids` with special tokens retained;
- output/token SHA-256 checksums;
- finish and stop reasons;
- extracted `\boxed{}` answer and approximate correctness;
- whether `enable_thinking` was requested and accepted by the chat template;
- whether `<think>` is in the prompt or generated output;
- whether `</think>` is generated;
- reasoning/final-answer token boundaries when detectable;
- the visible reasoning text and final-answer text as separate files.

The untouched `.output.txt` file is the source of truth. The reasoning/final split is only an audit aid based on `<think>`/`</think>` markers in the prompt, output text, and token sequence.

## Repeat and cross-mode comparisons

The audit compares:

```text
FP16 run1  vs FP16 run2
KVarN run1 vs KVarN run2
FP16 run1  vs KVarN run1
FP16 run2  vs KVarN run2
```

For each pair it records:

- exact/length-only/token-mismatch relation;
- first divergence step;
- longest common prefix;
- output and absolute cache-block positions;
- whether the first divergence is in thinking, the `</think>` boundary, the final answer, or unknown;
- exact token-ID and decoded-text windows around the first divergence.

It also reports prefix divergence rates at 128, 256, 512, 1K, 2K, 4K, 8K, and 16K tokens. This directly distinguishes:

- a three-sample smoke test that happens to stay identical for 512 tokens;
- a longer run that self-diverges only after later reasoning steps;
- a fixed kernel/block threshold where divergence begins to concentrate.

## Run the 20-sample 65K stage

Use the same local model path and deterministic controls as the successful pilot:

```bash
cd /data/wanglin/KVarN

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHON=/home/wanglin/miniconda3/envs/kvarn-smoke/bin/python \
MODEL=/data/wanglin/models/Qwen3-4B \
NUM_SAMPLES=20 \
MAX_MODEL_LEN=65536 \
MAX_TOKENS=38912 \
ROPE_SCALING_JSON='{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' \
BLOCK_SIZE=128 \
BOUNDARY_STEP=128 \
GPU_MEMORY_UTILIZATION=0.90 \
TP_SIZE=1 \
SEED=2026 \
OUTPUT_ROOT=$PWD/results/trajectory_stage1_qwen3_4b_n20_65k \
bash experiments/trajectory_divergence/run_stage1_output_audit.sh
```

`run_stage1_output_audit.sh` always sets `NUM_REPEATS=2`. Each method therefore runs in two independent vLLM processes and all four full token sequences are retained.

## Audit an existing four-run result without rerunning

```bash
python experiments/trajectory_divergence/audit_repeat_outputs.py \
  --output-root results/trajectory_pilot_qwen3_4b_n20_65k_v2 \
  --tokenizer /data/wanglin/models/Qwen3-4B \
  --block-size 128
```

If the output directory already exists, use a new `--output-dir` or pass `--overwrite`.

## Output layout

```text
OUTPUT_ROOT/
├── fp16_run1/generations.jsonl
├── fp16_run2/generations.jsonl
├── kvarn_run1/generations.jsonl
├── kvarn_run2/generations.jsonl
└── output_audit/
    ├── summary.json
    ├── summary.md
    ├── per_sample_audit.csv
    ├── per_sample_audit.jsonl
    ├── prompts/<sample>.prompt.txt
    ├── raw_outputs/
    │   ├── fp16_run1/<sample>.output.txt
    │   ├── fp16_run2/<sample>.output.txt
    │   ├── kvarn_run1/<sample>.output.txt
    │   └── kvarn_run2/<sample>.output.txt
    ├── raw_tokens/<run>/<sample>.tokens.json
    └── samples/<sample>.md
```

Each `samples/<sample>.md` report links the four exact outputs/tokens and shows all first-divergence windows in one place.

## What this stage can and cannot explain

This stage can determine:

- whether KVarN self-divergence happens before or after 512 tokens;
- whether it occurs in visible thinking or final-answer generation;
- whether the same sample, answer, or stopping behavior changes between repeats;
- whether divergence clusters near absolute 128-token cache-block boundaries;
- the exact generated text and tokens on both sides of every split.

It cannot yet identify the numerical cause inside a layer, attention output, K/V branch, or quantization block. After this audit identifies representative samples and the first-divergence range, the next patch should add a bounded teacher-forced aligned replay for those selected samples, recording logits, margins, ranks, and KL only up to the relevant horizon.
