# Repeat/output audit summary

- Samples present in all four runs: **20**
- Prompt hash mismatches across four runs: **0**
- Input token mismatches across four runs: **0**
- Runs with detected complete thinking boundary: **78/80**

## Pairwise divergence

| Pair | Diverged | Rate | Median first step | P90 | Thinking | Boundary | Final answer | Unknown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp16_self_run1_vs_run2 | 0/20 | 0.0000 | N/A | N/A | 0 | 0 | 0 | 20 |
| kvarn_self_run1_vs_run2 | 0/20 | 0.0000 | N/A | N/A | 0 | 0 | 0 | 20 |
| cross_fp16_run1_vs_kvarn_run1 | 20/20 | 1.0000 | 237.5 | 371.0 | 20 | 0 | 0 | 0 |
| cross_fp16_run2_vs_kvarn_run2 | 20/20 | 1.0000 | 237.5 | 371.0 | 20 | 0 | 0 | 0 |

## Prefix divergence rates

This table explains why a short 3-sample/512-token smoke test can be stable while longer generation later self-diverges.

| Pair | 128 | 256 | 512 | 1K | 2K | 4K | 8K | 16K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp16_self_run1_vs_run2 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| kvarn_self_run1_vs_run2 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| cross_fp16_run1_vs_kvarn_run1 | 2/20 | 13/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 |
| cross_fp16_run2_vs_kvarn_run2 | 2/20 | 13/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 |

## Manual inspection

- Exact vLLM output text: `raw_outputs/<run>/<sample>.output.txt`
- Extracted visible reasoning: `raw_outputs/<run>/<sample>.reasoning.txt`
- Extracted final-answer section: `raw_outputs/<run>/<sample>.final_answer.txt`
- Exact input/output token IDs: `raw_tokens/<run>/<sample>.tokens.json`
- Four-run comparison report: `samples/<sample>.md`

The reasoning/final split is an audit aid based on `<think>`/`</think>` markers in the prompt, output text, and token sequence. The untouched raw output remains the source of truth.
