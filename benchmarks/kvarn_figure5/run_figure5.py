# SPDX-License-Identifier: Apache-2.0
"""Compare KVarN and TurboQuant using their in-tree vLLM kernels.

This is a backend-native replacement for the earlier paper-derived proxy. It
runs three isolated vLLM engines (FP16, KVarN K4/V2 G128, TurboQuant 3-bit NC),
forces 128-token chunked prefill, scores the same fixed token trajectory, and
compares model-output log probabilities against FP16.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from metrics import METHOD_SPECS, compare_worker_results, extract_prompt_window
    from plot_figure5 import plot_figure
else:
    from .metrics import METHOD_SPECS, compare_worker_results, extract_prompt_window
    from .plot_figure5 import plot_figure


def parse_int_list(values: list[str]) -> list[int]:
    result = sorted({int(value.replace("k", "000").replace("K", "000")) for value in values})
    if not result or result[0] < 2:
        raise ValueError("context lengths must contain values >= 2")
    return result


def load_source_text(args: argparse.Namespace) -> str:
    if args.text_file is not None:
        return args.text_file.read_text(encoding="utf-8")
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Install datasets or pass --text-file with a local corpus"
        ) from error
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.dataset_split,
        cache_dir=str(args.dataset_cache_dir) if args.dataset_cache_dir else None,
    )
    texts = [str(row[args.dataset_text_column]) for row in dataset]
    return "\n\n".join(text for text in texts if text.strip())


def prepare_tokens(args: argparse.Namespace, output_path: Path) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        trust_remote_code=args.trust_remote_code,
        revision=args.tokenizer_revision or args.revision,
    )
    text = load_source_text(args)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    bos_id = tokenizer.bos_token_id
    if bos_id is not None:
        token_ids = [int(bos_id), *map(int, token_ids)]
    else:
        token_ids = list(map(int, token_ids))
    required = max(args.context_lengths)
    if len(token_ids) < required:
        if not args.repeat_short_text or not token_ids:
            raise ValueError(
                f"Corpus has {len(token_ids)} tokens but {required} are required. "
                "Use a larger --text-file or pass --repeat-short-text."
            )
        repeats = (required + len(token_ids) - 1) // len(token_ids)
        token_ids = (token_ids * repeats)[:required]
    else:
        token_ids = token_ids[:required]
    output_path.write_text(json.dumps(token_ids), encoding="utf-8")
    return token_ids


def run_worker(args: argparse.Namespace) -> None:
    method = args.worker_method
    spec = METHOD_SPECS[method]
    token_ids = json.loads(args.token_file.read_text(encoding="utf-8"))

    # Delayed imports keep aggregation/tests usable without a GPU vLLM install.
    from vllm import LLM, SamplingParams

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tokenizer": args.tokenizer,
        "skip_tokenizer_init": True,
        "dtype": "float16",
        "kv_cache_dtype": spec["kv_cache_dtype"],
        "block_size": args.block_size,
        "max_model_len": max(args.context_lengths) + 1,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "revision": args.revision,
        "tokenizer_revision": args.tokenizer_revision,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": args.chunk_size,
        "max_num_seqs": 1,
        "enable_prefix_caching": False,
        "enforce_eager": args.enforce_eager,
        "seed": args.seed,
        "disable_log_stats": True,
    }
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        ignore_eos=True,
        prompt_logprobs=1,
        detokenize=False,
    )

    warmup_length = min(args.warmup_length, min(args.context_lengths))
    if warmup_length >= 2:
        llm.generate(
            [{"prompt_token_ids": token_ids[:warmup_length]}],
            sampling,
            use_tqdm=False,
        )

    points: list[dict[str, Any]] = []
    for context_tokens in args.context_lengths:
        prompt_ids = token_ids[:context_tokens]
        elapsed_samples: list[float] = []
        final_output = None
        for _ in range(args.repeats):
            started = time.perf_counter()
            output = llm.generate(
                [{"prompt_token_ids": prompt_ids}],
                sampling,
                use_tqdm=False,
            )[0]
            elapsed_samples.append(time.perf_counter() - started)
            final_output = output
        assert final_output is not None
        elapsed = sorted(elapsed_samples)[len(elapsed_samples) // 2]
        if final_output.prompt_logprobs is None:
            raise RuntimeError("vLLM did not return prompt_logprobs")
        window_start = max(1, context_tokens - args.eval_window)
        window = extract_prompt_window(
            final_output.prompt_token_ids or prompt_ids,
            final_output.prompt_logprobs,
            start=window_start,
            end=context_tokens,
        )
        points.append(
            {
                "context_tokens": context_tokens,
                "elapsed_seconds": elapsed,
                "prompt_tokens_per_second": context_tokens / elapsed,
                "window_start": window_start,
                "window_end": context_tokens,
                "window": window,
            }
        )
        print(
            f"[{method}] context={context_tokens} elapsed={elapsed:.3f}s "
            f"prompt_tps={context_tokens / elapsed:.1f}",
            flush=True,
        )

    payload = {
        "method": method,
        "label": spec["label"],
        "kv_cache_dtype": spec["kv_cache_dtype"],
        "model": args.model,
        "block_size": args.block_size,
        "chunk_size": args.chunk_size,
        "eval_window": args.eval_window,
        "points": points,
    }
    args.worker_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_worker_args(command: list[str], args: argparse.Namespace) -> None:
    command.extend(
        [
            "--model",
            args.model,
            "--token-file",
            str(args.token_file),
            "--worker-output",
            str(args.worker_output),
            "--block-size",
            str(args.block_size),
            "--chunk-size",
            str(args.chunk_size),
            "--eval-window",
            str(args.eval_window),
            "--warmup-length",
            str(args.warmup_length),
            "--repeats",
            str(args.repeats),
            "--tensor-parallel-size",
            str(args.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--seed",
            str(args.seed),
            "--context-lengths",
            *map(str, args.context_lengths),
        ]
    )
    if args.tokenizer:
        command.extend(["--tokenizer", args.tokenizer])
    if args.revision:
        command.extend(["--revision", args.revision])
    if args.tokenizer_revision:
        command.extend(["--tokenizer-revision", args.tokenizer_revision])
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if args.enforce_eager:
        command.append("--enforce-eager")


def launch_worker(method: str, args: argparse.Namespace, output: Path) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-method",
        method,
    ]
    worker_args = argparse.Namespace(**vars(args))
    worker_args.token_file = args.output_dir / "input_tokens.json"
    worker_args.worker_output = output
    append_worker_args(command, worker_args)
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    subprocess.run(command, check=True, env=env)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_controller(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    token_path = args.output_dir / "input_tokens.json"
    prepare_tokens(args, token_path)

    worker_paths: dict[str, Path] = {}
    for method in ["fp16", *args.methods]:
        path = args.output_dir / f"worker_{method}.json"
        if path.exists() and not args.overwrite:
            print(f"Reusing {path}")
        else:
            launch_worker(method, args, path)
        worker_paths[method] = path

    baseline = json.loads(worker_paths["fp16"].read_text(encoding="utf-8"))
    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for method in args.methods:
        candidate = json.loads(worker_paths[method].read_text(encoding="utf-8"))
        method_raw, method_summary = compare_worker_results(baseline, candidate)
        raw_rows.extend(method_raw)
        summary_rows.extend(method_summary)

    raw_csv = args.output_dir / "figure5_vllm_raw.csv"
    summary_csv = args.output_dir / "figure5_vllm.csv"
    write_csv(raw_csv, raw_rows)
    write_csv(summary_csv, summary_rows)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if not key.startswith("worker_")
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    plot_figure(
        summary_csv,
        args.output_dir / "figure5_vllm_backend_comparison",
        title=f"{args.model}: native vLLM KV-cache backends",
    )
    print(f"Wrote results to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--methods", nargs="+", choices=["kvarn", "turboquant"], default=["kvarn", "turboquant"])
    parser.add_argument("--context-lengths", nargs="+", default=["4096", "8192", "16384", "32768"])
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--eval-window", type=int, default=128)
    parser.add_argument("--warmup-length", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/figure5_vllm"))
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--text-file", type=Path, default=None)
    parser.add_argument("--repeat-short-text", action="store_true")
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--dataset-text-column", default="text")
    parser.add_argument("--dataset-cache-dir", type=Path, default=None)

    parser.add_argument("--worker-method", choices=list(METHOD_SPECS), default=None)
    parser.add_argument("--token-file", type=Path, default=None)
    parser.add_argument("--worker-output", type=Path, default=None)
    args = parser.parse_args()
    args.context_lengths = parse_int_list(args.context_lengths)
    if args.block_size != 128:
        raise ValueError("This comparison pins block_size=128 for kvarn_k4v2_g128")
    if args.chunk_size != 128:
        raise ValueError("Use chunk_size=128 so continuation prefill exercises both native kernels")
    if args.eval_window < 1:
        raise ValueError("eval_window must be positive")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.worker_method is not None:
        if parsed.token_file is None or parsed.worker_output is None:
            raise ValueError("Worker mode requires --token-file and --worker-output")
        run_worker(parsed)
    else:
        run_controller(parsed)
