# HumanEval graph/eager 5-repeat determinism study

Freshly run, same checkout/runtime:

- FP16 graph × 5
- FP16 eager × 5
- KVarN graph × 5
- KVarN eager × 5

Defaults:

- HumanEval first 3 tasks
- Qwen3-4B
- thinking enabled
- greedy: temperature=0, top_p=1, top_k=-1, min_p=0
- seed=2026
- max_tokens=32768
- max_model_len=40960
- block_size=128
- TP=1
- prefix caching disabled
- VLLM_USE_FLASHINFER_SAMPLER=0

No old smoke directory is reused.

Run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHON=/home/wanglin/miniconda3/envs/kvarn-smoke/bin/python MODEL=/data/wanglin/models/Qwen3-4B NUM_SAMPLES=3 NUM_REPEATS=5 OUTPUT_ROOT=$PWD/results/humaneval_graph_eager_repeat_n3_r5 bash experiments/trajectory_divergence/run_humaneval_graph_eager_repeat_study.sh
```

Primary result:

```text
results/humaneval_graph_eager_repeat_n3_r5/analysis/matrix_summary.md
```

The analyzer clusters repeats by exact token IDs and reports all C(5,2)=10
pairwise comparisons per task, first divergence step distribution, finish/pass
stability, thinking-boundary stability, cap hits, and runtime-environment
fingerprints.

Clean KVarN × graph evidence requires:

```text
FP16 graph: stable
FP16 eager: stable
KVarN graph: unstable
KVarN eager: stable
environment_control_clean: true
```

If this holds, use `--enforce-eager` for the full greedy causal-mechanism run,
and keep normal graph-mode vLLM for production/system validation with
multi-seed/multi-repeat aggregation.
