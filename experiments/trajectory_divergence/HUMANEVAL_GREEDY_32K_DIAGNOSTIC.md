# HumanEval greedy 32K diagnostic

This experiment is a follow-up to the deterministic HumanEval Stage-1 mechanism
experiment. It deliberately keeps greedy decoding and changes only the generation
budget plus output observability. The later sampling experiment remains separate.

## Configuration

Default settings:

```text
enable_thinking = true
temperature = 0
top_p = 1
top_k = -1
max_tokens = 32768
max_model_len = 40960
rope_scaling = disabled
max_num_seqs = 1
prefix_caching = off
```

The original `run_humaneval_stage1.sh` defaults remain unchanged at the previous
16K/32K settings. Use the new wrapper for this diagnostic so existing results are
not overwritten or silently reinterpreted.

## Full run

```bash
cd /data/wanglin/KVarN

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHON=/home/wanglin/miniconda3/envs/kvarn-smoke/bin/python \
MODEL=/data/wanglin/models/Qwen3-4B \
NUM_SAMPLES=164 \
NUM_REPEATS=2 \
BLOCK_SIZE=128 \
GPU_MEMORY_UTILIZATION=0.90 \
TP_SIZE=1 \
OUTPUT_ROOT=$PWD/results/humaneval_stage1_greedy32k_qwen3_4b_n164 \
bash experiments/trajectory_divergence/run_humaneval_stage1_greedy32k.sh
```

Before the full run, a 3-task smoke test is recommended by setting
`NUM_SAMPLES=3` and a fresh `OUTPUT_ROOT`.

## Per-task inspection files

The wrapper creates a dedicated tree:

```text
results/humaneval_stage1_greedy32k_qwen3_4b_n164/
├── fp16_run1/
│   ├── generations.jsonl
│   └── experiment_config.json
├── fp16_run2/
├── kvarn_run1/
├── kvarn_run2/
└── per_sample_outputs/
    ├── fp16_run1/
    │   ├── HumanEval_000.txt
    │   ├── HumanEval_001.txt
    │   └── ...
    ├── fp16_run2/
    ├── kvarn_run1/
    └── kvarn_run2/
```

Each `.txt` file includes token counts, cap utilization, finish reason, think-boundary
status, latency, HumanEval execution status, simple inspection flags, the full chat
prompt, the complete raw model output, the visible final answer, and the extracted
Python candidate. `generations.jsonl` remains the machine-readable source of truth.

Useful inspection flags are:

- `HIT_LENGTH_CAP`: `finish_reason=length`;
- `NEAR_LENGTH_CAP`: output used at least 80% of the configured generation cap;
- `NO_THINK_END`: non-empty output never emitted `</think>`;
- `EXECUTION_FAILED`: extracted HumanEval candidate failed execution/tests;
- `GENERATION_ERROR`: generation/evaluation raised an exception.

To find suspicious tasks quickly:

```bash
grep -R "inspection_flags: .*HIT_LENGTH_CAP\|inspection_flags: .*NO_THINK_END\|inspection_flags: .*GENERATION_ERROR" \
  results/humaneval_stage1_greedy32k_qwen3_4b_n164/per_sample_outputs
```

## Interpretation

The key comparison is whether tasks that previously stopped at 16,384 tokens now
terminate naturally before 32,768. If a task still reaches 32,768 with
`finish_reason=length`, treat it as a non-termination / pathological-long-generation
diagnostic case rather than simply increasing the cap again. Compare FP16 and KVarN
for the same task to determine whether the termination behavior changes with KV-cache
quantization.
