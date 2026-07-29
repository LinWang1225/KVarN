# SPDX-License-Identifier: Apache-2.0
"""Paper-protocol Figure 5 adaptation using native vLLM cache kernels.

The released backends are intentionally retained:
  * KVarN: ``kvarn_k4v2_g128``
  * TurboQuant: ``turboquant_3bit_nc``

KIVI is not introduced.  The script measures attention-output MAE in static
and accumulated regimes and renders the same three statistical panels as the
paper.  Since the compared methods/bit allocations differ from the paper, the
result is a protocol-aligned adaptation rather than a numerical reproduction.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from attention_metrics import aggregate_figure5, read_jsonl, write_csv
    from plot_attention_figure5 import plot_figure5
else:
    from .attention_metrics import aggregate_figure5, read_jsonl, write_csv
    from .plot_attention_figure5 import plot_figure5

METHOD_SPECS = {
    "kvarn": {"label": "KVarN K4/V2 G128", "kv_cache_dtype": "kvarn_k4v2_g128"},
    "turboquant": {"label": "TurboQuant 3-bit NC", "kv_cache_dtype": "turboquant_3bit_nc"},
}
REGIMES = ("static", "accumulated")


def estimate_shadow_cache_bytes(model_config: Any, max_context: int) -> int:
    """Estimate the FP16 K+V shadow history allocated by the probe."""
    num_layers = int(model_config.num_hidden_layers)
    num_attention_heads = int(model_config.num_attention_heads)
    num_kv_heads = int(
        getattr(model_config, "num_key_value_heads", num_attention_heads)
    )
    head_dim = int(
        getattr(
            model_config,
            "head_dim",
            int(model_config.hidden_size) // num_attention_heads,
        )
    )
    fp16_bytes = 2
    # Two tensors (K and V), one FP16 element per token/head/channel/layer.
    return 2 * num_layers * max_context * num_kv_heads * head_dim * fp16_bytes


def reserve_probe_memory(
    *,
    requested_utilization: float,
    shadow_bytes: int,
    total_gpu_bytes: int,
    safety_bytes: int,
) -> float:
    """Lower vLLM's reservation so the probe's shadow cache cannot OOM it."""
    if total_gpu_bytes <= 0:
        raise ValueError("total_gpu_bytes must be positive")
    effective = requested_utilization - (shadow_bytes + safety_bytes) / total_gpu_bytes
    if effective < 0.35:
        required_gib = (shadow_bytes + safety_bytes) / (1024**3)
        total_gib = total_gpu_bytes / (1024**3)
        raise RuntimeError(
            "Insufficient headroom for the Figure 5 FP16 shadow cache: "
            f"reserve={required_gib:.2f} GiB, total={total_gib:.2f} GiB, "
            f"requested utilization={requested_utilization:.3f}. "
            "Use a shorter maximum context or a GPU with more memory."
        )
    return effective


def parse_int_list(values: list[str]) -> list[int]:
    parsed = sorted(
        {int(value.replace("k", "000").replace("K", "000")) for value in values}
    )
    if not parsed or parsed[0] < 256:
        raise ValueError("context lengths must contain values >= 256")
    return parsed


def load_documents(args: argparse.Namespace) -> list[str]:
    if args.text_file is not None:
        return [args.text_file.read_text(encoding="utf-8")]
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("Install datasets or pass --text-file") from error
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.dataset_split,
        cache_dir=str(args.dataset_cache_dir) if args.dataset_cache_dir else None,
    )
    return [
        str(row[args.dataset_text_column])
        for row in dataset
        if str(row[args.dataset_text_column]).strip()
    ]


def prepare_samples(args: argparse.Namespace, output_path: Path) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        trust_remote_code=args.trust_remote_code,
        revision=args.tokenizer_revision or args.revision,
    )
    # Tokenize document-by-document to avoid the misleading model_max_length
    # warning produced by encoding the entire WikiText corpus as one sequence.
    token_ids: list[int] = []
    for document in load_documents(args):
        token_ids.extend(
            int(token) for token in tokenizer.encode(document, add_special_tokens=False)
        )
    bos_id = tokenizer.bos_token_id
    max_context = max(args.context_lengths)
    if bos_id is not None:
        segment_length = max_context - 1
    else:
        segment_length = max_context
    if segment_length <= 0:
        raise ValueError("Maximum context is too short for the BOS token")

    if len(token_ids) < segment_length:
        if not args.repeat_short_text or not token_ids:
            raise ValueError(
                f"Corpus has {len(token_ids)} tokens but {segment_length} are required"
            )
        repeats = (segment_length + len(token_ids) - 1) // len(token_ids)
        token_ids = token_ids * repeats

    max_start = len(token_ids) - segment_length
    if args.sample_offsets:
        offsets = [int(value) for value in args.sample_offsets]
        if len(offsets) != args.num_samples:
            raise ValueError("--sample-offsets count must equal --num-samples")
    elif args.num_samples == 1:
        offsets = [0]
    else:
        offsets = [round(index * max_start / (args.num_samples - 1)) for index in range(args.num_samples)]

    samples: list[dict[str, Any]] = []
    for sample_id, offset in enumerate(offsets):
        if offset < 0 or offset + segment_length > len(token_ids):
            raise ValueError(f"Sample offset {offset} is outside the tokenized corpus")
        segment = token_ids[offset : offset + segment_length]
        if bos_id is not None:
            segment = [int(bos_id), *segment]
        samples.append(
            {
                "sample_id": sample_id,
                "token_offset": offset,
                "token_ids": segment[:max_context],
            }
        )
    output_path.write_text(json.dumps(samples), encoding="utf-8")
    return samples


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, path)


def run_worker(args: argparse.Namespace) -> None:
    method = args.worker_method
    regime = args.worker_regime
    if method is None or regime is None:
        raise ValueError("Worker mode requires method and regime")
    spec = METHOD_SPECS[method]
    samples = json.loads(args.sample_file.read_text(encoding="utf-8"))

    trace_path = args.worker_trace.resolve()
    control_path = args.worker_control.resolve()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    if trace_path.exists():
        trace_path.unlink()

    # Environment is inherited by EngineCore. The control file changes between
    # synchronous generate calls and tells each layer which full context it is
    # currently observing.
    os.environ["VLLM_FIGURE5_PROBE_OUTPUT"] = str(trace_path)
    os.environ["VLLM_FIGURE5_PROBE_CONTROL"] = str(control_path)
    os.environ["VLLM_FIGURE5_PROBE_METHOD"] = method
    os.environ["VLLM_FIGURE5_PROBE_REGIME"] = regime
    os.environ["VLLM_FIGURE5_PROBE_MAX_CONTEXT"] = str(max(args.context_lengths))

    # The probe retains an unquantized FP16 K/V history for every attention
    # layer. vLLM must not reserve that memory for its native KV pool first.
    import torch
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
    )
    shadow_bytes = estimate_shadow_cache_bytes(
        model_config, max(args.context_lengths)
    )
    total_gpu_bytes = int(torch.cuda.get_device_properties(0).total_memory)
    safety_bytes = int(args.shadow_memory_safety_gib * 1024**3)
    effective_gpu_memory_utilization = reserve_probe_memory(
        requested_utilization=args.gpu_memory_utilization,
        shadow_bytes=shadow_bytes,
        total_gpu_bytes=total_gpu_bytes,
        safety_bytes=safety_bytes,
    )
    print(
        f"[{method}/{regime}] reserving "
        f"{shadow_bytes / (1024**3):.2f} GiB shadow + "
        f"{args.shadow_memory_safety_gib:.2f} GiB safety; "
        f"vLLM gpu_memory_utilization={effective_gpu_memory_utilization:.3f}",
        flush=True,
    )

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
        "gpu_memory_utilization": effective_gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "revision": args.revision,
        "tokenizer_revision": args.tokenizer_revision,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": args.chunk_size,
        "max_num_seqs": 1,
        "enable_prefix_caching": False,
        # Python-side shadow-cache state and JSONL recording are deliberately
        # benchmark-only and must not be captured by torch.compile/CUDA graphs.
        # Native KVarN/TurboQuant Triton store/decode kernels remain unchanged.
        "enforce_eager": True,
        "seed": args.seed,
        "disable_log_stats": True,
    }
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        ignore_eos=True,
        detokenize=False,
    )

    request_index = 0
    for sample in samples:
        sample_id = int(sample["sample_id"])
        tokens = [int(token) for token in sample["token_ids"]]
        for context_tokens in args.context_lengths:
            _atomic_json(
                control_path,
                {
                    "request_index": request_index,
                    "sample_id": sample_id,
                    "target_context": context_tokens,
                },
            )
            llm.generate(
                [{"prompt_token_ids": tokens[:context_tokens]}],
                sampling,
                use_tqdm=False,
            )
            print(
                f"[{method}/{regime}] sample={sample_id} context={context_tokens}",
                flush=True,
            )
            request_index += 1

    if not trace_path.exists() or trace_path.stat().st_size == 0:
        raise RuntimeError(
            f"No probe output was written to {trace_path}. Confirm that the backend hook patch is applied."
        )
    args.worker_output.write_text(
        json.dumps(
            {
                "method": method,
                "regime": regime,
                "kv_cache_dtype": spec["kv_cache_dtype"],
                "num_requests": request_index,
                "trace": str(trace_path),
                "shadow_cache_bytes": shadow_bytes,
                "shadow_memory_safety_bytes": safety_bytes,
                "requested_gpu_memory_utilization": args.gpu_memory_utilization,
                "effective_gpu_memory_utilization": effective_gpu_memory_utilization,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def append_worker_args(command: list[str], args: argparse.Namespace) -> None:
    command.extend(
        [
            "--model",
            args.model,
            "--sample-file",
            str(args.sample_file),
            "--worker-output",
            str(args.worker_output),
            "--worker-trace",
            str(args.worker_trace),
            "--worker-control",
            str(args.worker_control),
            "--block-size",
            str(args.block_size),
            "--chunk-size",
            str(args.chunk_size),
            "--tensor-parallel-size",
            str(args.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--shadow-memory-safety-gib",
            str(args.shadow_memory_safety_gib),
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


def launch_worker(method: str, regime: str, args: argparse.Namespace) -> tuple[Path, Path]:
    stem = f"worker_{method}_{regime}"
    worker_output = args.output_dir / f"{stem}.json"
    worker_trace = args.output_dir / f"{stem}.jsonl"
    worker_control = args.output_dir / f"{stem}_control.json"
    if worker_output.exists() and worker_trace.exists() and not args.overwrite:
        print(f"Reusing {worker_trace}")
        return worker_output, worker_trace

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-method",
        method,
        "--worker-regime",
        regime,
    ]
    worker_args = argparse.Namespace(**vars(args))
    worker_args.sample_file = args.output_dir / "input_samples.json"
    worker_args.worker_output = worker_output
    worker_args.worker_trace = worker_trace
    worker_args.worker_control = worker_control
    append_worker_args(command, worker_args)
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    subprocess.run(command, check=True, env=env)
    return worker_output, worker_trace


def run_controller(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / "input_samples.json"
    prepare_samples(args, sample_path)

    trace_paths: list[Path] = []
    worker_metadata: list[dict[str, Any]] = []
    for method in args.methods:
        for regime in REGIMES:
            worker_output, trace = launch_worker(method, regime, args)
            trace_paths.append(trace)
            worker_metadata.append(json.loads(worker_output.read_text(encoding="utf-8")))

    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
    )
    num_hidden_layers = int(model_config.num_hidden_layers)
    raw_rows = read_jsonl(trace_paths)
    completed, sample_rows, combined_summary = aggregate_figure5(
        raw_rows,
        num_hidden_layers=num_hidden_layers,
    )

    raw_csv = args.output_dir / "figure5_attention_layer_raw.csv"
    sample_csv = args.output_dir / "figure5_attention_sample.csv"
    summary_csv = args.output_dir / "figure5_attention_summary.csv"
    write_csv(raw_csv, completed)
    write_csv(sample_csv, sample_rows)
    write_csv(summary_csv, combined_summary)

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if not key.startswith("worker_")
    }
    config.update(
        {
            "num_hidden_layers": num_hidden_layers,
            "protocol": "paper-Figure-5-aligned native-vLLM adaptation",
            "panel_a": (
                "element-weighted mean attention-output MAE over every "
                "128-token chunk and all attention layers"
            ),
            "panel_b": "MAE_KVarN - MAE_TurboQuant, static and accumulated",
            "panel_c": "panel-b static - accumulated",
            "important_difference_from_paper": (
                "Released KVarN K4/V2 and TurboQuant K3/V3 replace the paper's "
                "KVarN K2/V2 and KIVI K2/V2 comparison."
            ),
            "worker_metadata": worker_metadata,
        }
    )
    (args.output_dir / "config_attention_figure5.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    plot_figure5(
        summary_csv,
        args.output_dir / "figure5_native_attention_reconstruction",
        title=f"{args.model}: Figure 5 protocol with native vLLM kernels",
    )
    print(f"Wrote Figure 5 attention results to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHOD_SPECS),
        default=list(METHOD_SPECS),
    )
    parser.add_argument(
        "--context-lengths",
        nargs="+",
        default=["4096", "8192", "16384", "32768"],
    )
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--sample-offsets", nargs="+", default=None)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--shadow-memory-safety-gib",
        type=float,
        default=1.0,
        help="Extra memory left outside vLLM in addition to the estimated FP16 shadow cache",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/figure5_native_attention")
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--text-file", type=Path, default=None)
    parser.add_argument("--repeat-short-text", action="store_true")
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--dataset-text-column", default="text")
    parser.add_argument("--dataset-cache-dir", type=Path, default=None)

    parser.add_argument("--worker-method", choices=list(METHOD_SPECS), default=None)
    parser.add_argument("--worker-regime", choices=REGIMES, default=None)
    parser.add_argument("--sample-file", type=Path, default=None)
    parser.add_argument("--worker-output", type=Path, default=None)
    parser.add_argument("--worker-trace", type=Path, default=None)
    parser.add_argument("--worker-control", type=Path, default=None)

    args = parser.parse_args()
    args.context_lengths = parse_int_list(args.context_lengths)
    if args.block_size != 128:
        raise ValueError("kvarn_k4v2_g128 requires block_size=128")
    if args.chunk_size != 128:
        raise ValueError("Use chunk_size=128 to match the paper pseudo-decode block size")
    if args.num_samples < 1:
        raise ValueError("num_samples must be positive")
    if args.tensor_parallel_size != 1:
        raise ValueError(
            "The Python-side Figure 5 probe currently supports tensor_parallel_size=1 only"
        )
    if not 0.0 <= args.shadow_memory_safety_gib:
        raise ValueError("shadow_memory_safety_gib must be non-negative")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    if any(length % args.chunk_size for length in args.context_lengths):
        raise ValueError("Every context length must be divisible by chunk_size=128")
    if len(args.methods) != len(METHOD_SPECS) or set(args.methods) != set(METHOD_SPECS):
        raise ValueError(
            "All three Figure 5 panels require paired KVarN and TurboQuant runs; "
            "keep --methods kvarn turboquant"
        )
    # Normalize order so paired worker outputs and legends are deterministic.
    args.methods = list(METHOD_SPECS)
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.worker_method is not None:
        required = (
            parsed.worker_regime,
            parsed.sample_file,
            parsed.worker_output,
            parsed.worker_trace,
            parsed.worker_control,
        )
        if any(value is None for value in required):
            raise ValueError("Worker mode is missing one or more internal arguments")
        run_worker(parsed)
    else:
        run_controller(parsed)
