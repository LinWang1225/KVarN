# KVarN Runtime Environment

This file records the local environment used for the runtime smoke verification pass.

## Hardware

- GPU model: NVIDIA GeForce RTX 3090
- GPU count: 1
- GPU memory capacity: 24,576 MiB (24 GiB)

## CUDA / driver

- NVIDIA driver: 560.35.05
- CUDA version reported by `nvidia-smi`: 12.6

## Software

- Python: 3.12 (in `vllm-V1`)
- PyTorch: 2.11.0+cu126
- Triton: 3.6.0
- vLLM: 0.23.1.dev0+g0fc695fc6.d20260626.cu126
- vLLM import path observed in the serving env: `/data/wanglin/KVarN/vllm/__init__.py`
- Editable install target reported by `pip`: `/data/wanglin/vllm-0.23.0-cu126`
- Current git commit: `7586257f1c632e63187bfacbbe21ccb51540f7b3`

## Available local model paths

The following cached model directories were found on this machine:

- `/home/wanglin/data/KVarN` (workspace, not a model)
- `/home/wanglin/miniconda3/envs/vllm-V1`
- `/home/wanglin/data/KVarN` contains no local Hugging Face model snapshot directories under the usual cache locations that were discoverable from the workspace session.
- Known datasets cached locally include `aime_2024`, `MATH-500`, and `aime_2025`.

The workspace session did identify these model cache names under `/home/wanglin/data`:

- `models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B`
- `models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B`
- `models--mistralai--Ministral-3-8B-Instruct-2512`
- `models--mistralai--Mistral-7B-Instruct-v0.3`
- `models--Qwen--Qwen2.5-7B-Instruct`
- `models--Qwen--Qwen3-0.6B`
- `models--Qwen--Qwen3-4B-Instruct-2507`
- `models--Qwen--Qwen3-8B`

If a full runtime run is required, these should be resolved to actual snapshot paths before launch.

## Notes

- The base shell environment does not have `torch` installed, but the `vllm-V1` Conda environment does.
- The active repo HEAD at the time of capture was `7586257f1c632e63187bfacbbe21ccb51540f7b3`.
