# HumanEval TurboQuant precision sweep

This experiment tests whether more aggressive TurboQuant KV-cache precision
causes larger trajectory/system effects under the same Qwen3 thinking sampling
used in the production-oriented HumanEval validation.

No TurboQuant kernel or vLLM attention code is modified. The current repository
already registers all four TurboQuant cache dtypes and selects the TurboQuant
attention backend from `kv_cache_dtype`.

## Presets, high to low precision

| Preset | K | V | Norm correction | Repository note |
|---|---:|---:|---:|---|
| `turboquant_k8v4` | FP8 | 4 bit | off | 2.6x, +1.17% PPL |
| `turboquant_4bit_nc` | 4 bit | 4 bit | on | 3.8x, +2.71% PPL |
| `turboquant_k3v4_nc` | 3 bit | 4 bit | on | ~3.5x, +10.63% PPL |
| `turboquant_3bit_nc` | 3 bit | 3 bit | on | 4.9x, +20.59% PPL |

The order above is a *nominal precision severity* order based on K/V bit-width.
Storage compression is not perfectly monotonic because metadata/packing overhead
differs between presets.

## Fixed experiment controls

- Qwen3-4B
- HumanEval 164
- thinking enabled
- temperature = 0.6
- top-p = 0.95
- top-k = 20
- min-p = 0.0 (vLLM default; the legacy HumanEval runner does not expose it)
- seeds = 2026, 2027, 2028
- max output tokens = 16384 for the full run
- prefix caching disabled by the existing HumanEval runner
- block size = 128
- one request at a time

FP16 is generated only once per seed and reused across every TurboQuant preset.
The sweep therefore requires 3 FP16 runs + 12 TurboQuant runs for the full
four-preset, three-seed experiment.

### Why the wrapper passes `--mode kvarn`

`run_humaneval_generation.py` currently names its quantized-candidate slot
`kvarn`. Internally, however, it passes the supplied
`--kvarn-kv-cache-dtype` directly to `LLM(kv_cache_dtype=...)`. Supplying a
`turboquant_*` dtype therefore selects the TurboQuant backend. The runner also
records `requested_kv_cache_dtype`, `resolved_kv_cache_dtype`, and
`backend_verified`; the sweep analysis keys methods by the actual dtype/path,
not by the legacy `mode` string.

This keeps the patch add-only and avoids changing already validated greedy/KVarN
experiments. If the generic candidate runner is later refactored, this wrapper
can switch to the new argument without changing the analysis contract.

## Primary scientific question

Do lower-precision TurboQuant presets cause a stronger effect?

Do **not** judge this only by signed mean output inflation. Quantization can push
some trajectories longer and others shorter, so positive and negative changes
can cancel. The main precision-severity metrics are:

1. mean `|TIR|`;
2. P95 `|TIR|`;
3. fraction `|TIR| > 50%`;
4. fraction `TIR > 50%`;
5. stable severe tasks across >=2/3 seeds;
6. Pass@1 degradation;
7. mean TPOT overhead;
8. thinking share among severe positive-inflation requests.

`TIR = (N_quant - N_fp16) / N_fp16`.

The analyzer reports Spearman correlation between nominal severity rank
(high precision=1, low precision=4) and the key metrics. With only four presets,
this is descriptive rather than a significance test.

## Smoke test

Run every preset on 3 tasks first so all four backends/configurations are
validated:

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
OUTPUT_ROOT=$PWD/results/humaneval_turboquant_sweep_smoke \
bash experiments/trajectory_divergence/run_humaneval_turboquant_sweep.sh
```

Check each candidate `experiment_config.json` for:

```text
requested_kv_cache_dtype = turboquant_...
resolved_kv_cache_dtype  = turboquant_...
backend_verified          = true
```

The 4096-token smoke cap is only for engineering validation; do not interpret
its output-length distribution as a benchmark result.

## Full sweep

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
OUTPUT_ROOT=$PWD/results/humaneval_turboquant_sweep_qwen3_4b_n164 \
bash experiments/trajectory_divergence/run_humaneval_turboquant_sweep.sh
```

To run a subset:

```bash
TQ_PRESETS=turboquant_k8v4,turboquant_4bit_nc ...
```

## Optional: reuse the existing FP16 sampling baseline

If the previous HumanEval K4V2 sampling experiment used exactly the same model,
prompts, decoding parameters, seeds, and max-token budget, its FP16 files can be
reused:

```bash
FP16_SOURCE_PATTERN="$PWD/results/humaneval_sampling_k4v2_qwen3_4b_n164/thinking_on/seed_{seed}/fp16/generations.jsonl" \
OUTPUT_ROOT=$PWD/results/humaneval_turboquant_sweep_qwen3_4b_n164 \
bash experiments/trajectory_divergence/run_humaneval_turboquant_sweep.sh
```

Use reuse only when those controls are identical. The sweep still prepares its
own HumanEval manifest, so verify the same 164-task order before publication.

## Outputs

```text
results/humaneval_turboquant_sweep_qwen3_4b_n164/
├── selected_samples.json
├── sweep_config.json
├── seed_2026/
│   ├── fp16/
│   ├── turboquant_k8v4/
│   ├── turboquant_4bit_nc/
│   ├── turboquant_k3v4_nc/
│   └── turboquant_3bit_nc/
├── seed_2027/
├── seed_2028/
└── aggregate/
    ├── precision_sweep.md
    ├── precision_sweep.csv
    ├── summary.json
    ├── per_pair_metrics.csv
    └── per_task_method.csv
```

The first file to read is `aggregate/precision_sweep.md`.

## How to interpret the sweep

A strong "precision causes larger disturbance" result would look like:

- mean/P95 `|TIR|` rises from k8v4 -> 4/4 -> 3/4 -> 3/3;
- `|TIR| > 50%` and positive `TIR > 50%` become more frequent;
- more tasks show severe effects in >=2/3 seeds;
- accuracy degradation increases at the aggressive end;
- severe positive inflation continues to come mostly from thinking.

If signed mean TIR remains near zero but `|TIR|` grows strongly, the correct
conclusion is **trajectory instability increases with quantization severity**,
not "lower precision always makes outputs longer."

If none of the disturbance metrics grows with lower precision, the HumanEval
length/tail phenomenon is unlikely to be a simple quantization-strength effect,
and the next research step should shift toward method-specific mechanisms or a
longer-horizon workload rather than making the quantizer even more aggressive.
