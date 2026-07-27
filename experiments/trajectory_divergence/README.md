# FP16–KVarN generation trajectory divergence experiment

This directory adds an **independent motivation experiment** to the KVarN vLLM fork. It does not modify attention kernels, cache managers, or the KVarN backend.

The experiment runs the same fixed MATH-500 prompts in separate processes with:

- FP16 model compute and an unquantized KV cache (`kv_cache_dtype=auto`);
- FP16 model compute and the released KVarN preset (`kvarn_k4v2_g128`, `block_size=128`).

It then compares the exact `output_token_ids` returned by vLLM and measures the first token at which the two greedy generation trajectories differ.

## Research question

Under the same model, prompt, chat template, seed, and greedy decoding configuration:

1. Does KVarN change the generated token trajectory relative to FP16 KV cache?
2. Where does the first divergence occur?
3. Is the divergence rate higher for longer FP16 outputs?
4. Does divergence coincide with output-length inflation, a changed stopping point, or a changed extracted answer?
5. Are repeated FP16 or repeated KVarN runs themselves deterministic?

This is an end-to-end phenomenon experiment. It establishes trajectory changes and correlations; it does **not** identify a causal layer, KV branch, head, or cache block. Those require later logits/KV hooks and cache intervention.

## File placement

Add the complete directory at the KVarN repository root:

```text
KVarN/
├── experiments/
│   └── trajectory_divergence/
│       ├── README.md
│       ├── requirements.txt
│       ├── prepare_samples.py
│       ├── run_generation.py
│       ├── compare_trajectories.py
│       ├── plot_results.py
│       ├── run_smoke_test.sh
│       └── run_full_experiment.sh
├── scripts_kvarn_dense/
├── vllm/
└── ...
```

No existing KVarN source file needs to be edited.

## Files

| File | Responsibility |
|---|---|
| `prepare_samples.py` | Selects one deterministic subset from `HuggingFaceH4/MATH-500` and stores the prompt/reference manifest shared by all modes. |
| `run_generation.py` | Creates exactly one vLLM engine, runs either FP16 or KVarN, and writes token IDs and metadata to an append-only JSONL file. |
| `compare_trajectories.py` | Aligns records by `sample_id`, computes the longest common prefix and first divergence, summarizes length/correctness changes, and optionally checks same-mode repeat determinism. |
| `plot_results.py` | Produces six PNG/PDF comparison plots without rerunning inference. |
| `run_smoke_test.sh` | Runs 3 samples with 256 output tokens for FP16×2 and KVarN×2, compares results, and validates output files. |
| `run_full_experiment.sh` | Runs the default 100-sample experiment with a 131072-token total context ceiling and a 38912-token output budget. |
| `requirements.txt` | Adds only the experiment dependencies: Hugging Face Datasets and Matplotlib. |

## Installation

Run from the KVarN repository and the environment in which the local fork is installed:

```bash
VLLM_USE_PRECOMPILED=1 pip install -e .
pip install -r experiments/trajectory_divergence/requirements.txt
```

The scripts must import `vllm` from this KVarN checkout, not another installed vLLM package. Verify before running:

```bash
python -c "import vllm; print(vllm.__file__)"
```

The path should point into the current KVarN repository.

## Smoke test

```bash
MODEL=Qwen/Qwen3-4B \
PYTHON=/path/to/kvarn-env/bin/python \
BOUNDARY_STEP=128 \
bash experiments/trajectory_divergence/run_smoke_test.sh
```

Default smoke output:

```text
experiments/trajectory_divergence/results/smoke/
```

The smoke test launches four separate Python processes so only one vLLM engine occupies the GPU at a time.

## Full experiment

Default configuration:

- dataset: `HuggingFaceH4/MATH-500`, split `test`;
- samples: 100;
- model: `Qwen/Qwen3-4B`;
- thinking mode: enabled through the tokenizer chat template;
- decoding: greedy, `temperature=0`, intentionally used for deterministic
  trajectory attribution rather than model-quality benchmarking;
- total context capacity (`prompt + output`): 131072 tokens;
- YaRN: factor 4 with `original_max_position_embeddings=32768`;
- maximum generated output: 38912 tokens;
- prefix caching: disabled;
- FP16 mode: `kv_cache_dtype=auto`;
- KVarN mode: `kvarn_k4v2_g128`;
- block size: 128;
- plot boundary spacing: 128;
- batch size / `max_num_seqs`: 1;
- seed: 2026.

`MAX_MODEL_LEN=131072` is the 128K total sequence ceiling. It is not an
output-only limit. Qwen recommends 38912 output tokens for difficult math and
programming benchmarks. Static YaRN is enabled only by the full-run shell
script; the smoke test stays in the native window unless `ROPE_SCALING_JSON`
is explicitly supplied.

The FP16 baseline has a much larger KV cache than KVarN. A 128K FP16 run may
require multiple GPUs or more VRAM. Increase `TP_SIZE` or lower
`MAX_MODEL_LEN` if the FP16 engine cannot allocate its cache.

Run:

```bash
MODEL=Qwen/Qwen3-4B \
PYTHON=/path/to/kvarn-env/bin/python \
NUM_SAMPLES=100 \
MAX_TOKENS=38912 \
MAX_MODEL_LEN=131072 \
BOUNDARY_STEP=128 \
TP_SIZE=2 \
OUTPUT_ROOT=/path/to/results/math500_n100 \
bash experiments/trajectory_divergence/run_full_experiment.sh
```

Optional two-run determinism controls on the full subset:

```bash
NUM_REPEATS=2 bash experiments/trajectory_divergence/run_full_experiment.sh
```

Existing generation files are never overwritten. Choose a new `OUTPUT_ROOT`, or run `run_generation.py --resume` manually after an interrupted job.

## Manual execution

### 1. Fix the sample set

```bash
python experiments/trajectory_divergence/prepare_samples.py \
  --dataset-name HuggingFaceH4/MATH-500 \
  --dataset-split test \
  --num-samples 100 \
  --seed 2026 \
  --output results/trajectory/selected_samples.json
```

### 2. FP16 KV cache

```bash
python experiments/trajectory_divergence/run_generation.py \
  --mode fp16 \
  --run-name fp16_run1 \
  --samples-file results/trajectory/selected_samples.json \
  --output-dir results/trajectory/fp16_run1 \
  --model Qwen/Qwen3-4B \
  --max-tokens 38912 \
  --max-model-len 131072 \
  --rope-scaling-json '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --block-size 128
```

### 3. KVarN KV cache

```bash
python experiments/trajectory_divergence/run_generation.py \
  --mode kvarn \
  --run-name kvarn_run1 \
  --samples-file results/trajectory/selected_samples.json \
  --output-dir results/trajectory/kvarn_run1 \
  --model Qwen/Qwen3-4B \
  --kvarn-kv-cache-dtype kvarn_k4v2_g128 \
  --max-tokens 38912 \
  --max-model-len 131072 \
  --rope-scaling-json '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --block-size 128
```

### 4. Compare exact token trajectories

```bash
python experiments/trajectory_divergence/compare_trajectories.py \
  --reference results/trajectory/fp16_run1/generations.jsonl \
  --candidate results/trajectory/kvarn_run1/generations.jsonl \
  --output-dir results/trajectory/comparison \
  --tokenizer Qwen/Qwen3-4B \
  --trust-remote-code \
  --block-size 128
```

### 5. Plot

```bash
python experiments/trajectory_divergence/plot_results.py \
  --comparison-csv results/trajectory/comparison/per_sample_comparison.csv \
  --summary-json results/trajectory/comparison/summary.json \
  --output-dir results/trajectory/comparison/plots \
  --boundary-step 128
```

## Main definitions

For FP16 output token IDs \(y^{fp}\) and KVarN output token IDs \(y^{q}\), the longest common prefix is the number of equal tokens from output position zero.

The first divergence is classified as:

- `token_mismatch`: both sequences have a token at position \(t\), but the token IDs differ;
- `length_only`: the shorter sequence is a complete prefix of the longer sequence;
- `identical`: token IDs and sequence lengths are equal.

For a diverged sample:

```text
first_divergence_step = longest_common_prefix_length
# Output-relative block index:
first_divergence_block = first_divergence_step // block_size
offset_in_block = first_divergence_step % block_size

# Absolute sequence position including the prompt:
absolute_divergence_position = fp16_input_tokens + first_divergence_step
absolute_divergence_block = absolute_divergence_position // block_size
absolute_offset_in_block = absolute_divergence_position % block_size

length_ratio = kvarn_output_tokens / max(1, fp16_output_tokens)
```

When either run ends with `finish_reason=length`, the pair is marked censored. Censored pairs remain visible in the output-length scatter, but are excluded from length-ratio distribution and divergence-vs-length-ratio analyses.

The scripts use vLLM's returned `output_token_ids`; they do not reconstruct output IDs by retokenizing text.

## Outputs

```text
OUTPUT_ROOT/
├── selected_samples.json
├── fp16_run1/
│   ├── experiment_config.json
│   └── generations.jsonl
├── kvarn_run1/
│   ├── experiment_config.json
│   └── generations.jsonl
└── comparison/
    ├── per_sample_comparison.csv
    ├── per_sample_comparison.jsonl
    ├── summary.json
    ├── summary.md
    └── plots/
        ├── figure_length_scatter.{png,pdf}
        ├── figure_first_divergence_hist.{png,pdf}
        ├── figure_first_divergence_cdf.{png,pdf}
        ├── figure_divergence_by_length.{png,pdf}
        ├── figure_length_ratio_hist.{png,pdf}
        └── figure_divergence_vs_length_ratio.{png,pdf}
```

Each generation record stores:

- dataset index, problem, and reference answer;
- exact rendered prompt and SHA-256 hash;
- input and output token IDs;
- output text, length, finish reason, and stop reason;
- elapsed generation time and token throughput;
- requested and introspected KV cache dtype;
- last `\\boxed{...}` answer and a deliberately limited normalized exact-match field named `approx_correct`;
- errors without terminating the remaining long experiment.

`approx_correct` is not the official MATH-500 symbolic-equivalence score.

## Interpreting the first result

The minimum useful deliverables are:

1. `divergence_rate` and the same-mode self-divergence rates;
2. first-divergence histogram/CDF;
3. divergence rate by FP16 output-length bucket;
4. FP16-vs-KVarN output-length scatter;
5. correct-to-wrong cases and their local token windows.

Do not attribute cross-mode divergence to KVarN when:

- prompt hashes differ;
- FP16 repeat runs self-diverge materially;
- KVarN repeat runs self-diverge materially;
- the KVarN engine configuration cannot be confirmed in initialization logs/config.

The public KVarN repository ships the production-oriented K4V2 preset. Therefore, these results are an experiment on `kvarn_k4v2_g128`, not a strict reproduction of the paper's 2-bit K/2-bit V analysis configuration.
