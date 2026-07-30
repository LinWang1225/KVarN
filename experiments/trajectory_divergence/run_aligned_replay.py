#!/usr/bin/env python3
"""Replay an FP16 reference trajectory with identical tokens under one KV mode.

Raw next-token logits statistics are captured before a custom logits processor
forces the corresponding FP16 reference token. Run this script in four separate
processes (FP16 x2, KVarN x2) so same-mode numerical stability can be measured.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from teacher_forced_logits_processor import TeacherForcedReplayLogitsProcessor

LOGGER = logging.getLogger("aligned_replay")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teacher-force an FP16 token trajectory through FP16 or KVarN."
    )
    parser.add_argument("--mode", choices=("fp16", "kvarn"), required=True)
    parser.add_argument("--replay-run-name", required=True)
    parser.add_argument("--reference-generations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--fp16-kv-cache-dtype", default="auto")
    parser.add_argument("--kvarn-kv-cache-dtype", default="kvarn_k4v2_g128")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-replay-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--rope-scaling-json", default="")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--top-k-logits", type=int, default=20)
    parser.add_argument("--flush-every", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--require-backend-verification", action="store_true")
    return parser.parse_args()


def parse_json_object(value: str) -> dict[str, Any] | None:
    if not value.strip():
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--rope-scaling-json must decode to an object")
    return parsed


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_command(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def collect_environment(repo_root: Path) -> dict[str, Any]:
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        gpu_names = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ] if cuda_available else []
        torch_version = torch.__version__
        cuda_version = torch.version.cuda
    except Exception:
        cuda_available = False
        gpu_names = []
        torch_version = package_version("torch")
        cuda_version = None
    tracked_env = (
        "CUDA_VISIBLE_DEVICES",
        "VLLM_ENABLE_V1_MULTIPROCESSING",
        "VLLM_USE_FLASHINFER_SAMPLER",
        "VLLM_USE_DEEP_GEMM",
        "VLLM_MOE_USE_DEEP_GEMM",
        "KVARN_SPLIT_K",
        "KVARN_FUSED_DECODE",
        "CUDA_LAUNCH_BLOCKING",
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "torch": torch_version,
            "vllm": package_version("vllm"),
            "transformers": package_version("transformers"),
        },
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "gpu_names": gpu_names,
        "environment_variables": {name: os.environ.get(name) for name in tracked_env},
        "git_commit": run_command(["git", "rev-parse", "HEAD"], cwd=repo_root),
        "git_status_porcelain": run_command(
            ["git", "status", "--porcelain"], cwd=repo_root
        ),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL line {line_number} in {path}") from exc
            if record.get("error") is None:
                records.append(record)
    return records


def completed_sample_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(record["sample_id"])
        for record in load_jsonl(path)
        if record.get("replay_verified") is True
    }


def append_jsonl(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def get_nested_attr(obj: Any, path: str) -> Any:
    current = obj
    for part in path.split("."):
        current = getattr(current, part)
    return current


def resolve_engine_cache_dtype(llm: Any) -> tuple[str | None, str | None]:
    for path in (
        "llm_engine.vllm_config.cache_config.cache_dtype",
        "llm_engine.cache_config.cache_dtype",
        "llm_engine.engine_core.vllm_config.cache_config.cache_dtype",
    ):
        try:
            return str(get_nested_attr(llm, path)), path
        except (AttributeError, TypeError):
            continue
    return None, None


def find_subsequence(sequence: list[int], pattern: list[int]) -> int | None:
    if not pattern or len(pattern) > len(sequence):
        return None
    width = len(pattern)
    for start in range(len(sequence) - width + 1):
        if sequence[start : start + width] == pattern:
            return start
    return None


def thinking_boundary(
    tokenizer: Any,
    full_output_ids: list[int],
) -> tuple[int | None, int | None]:
    close_ids = list(tokenizer.encode("</think>", add_special_tokens=False))
    start = find_subsequence(full_output_ids, close_ids)
    if start is None:
        return None, None
    return start, start + len(close_ids)


def sha256_ids(token_ids: list[int]) -> str:
    payload = ",".join(str(token_id) for token_id in token_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def create_llm(args: argparse.Namespace, rope_scaling: dict[str, Any] | None) -> tuple[Any, str]:
    from vllm import LLM

    requested = (
        args.kvarn_kv_cache_dtype if args.mode == "kvarn" else args.fp16_kv_cache_dtype
    )
    kwargs: dict[str, Any] = {
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "enable_prefix_caching": False,
        "block_size": args.block_size,
        "seed": args.seed,
        "logits_processors": [TeacherForcedReplayLogitsProcessor],
    }
    if requested != "auto":
        kwargs["kv_cache_dtype"] = requested
    if rope_scaling is not None:
        kwargs["hf_overrides"] = {"rope_scaling": rope_scaling}
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    LOGGER.info("Initializing %s aligned-replay engine (%s)", args.mode, requested)
    return LLM(**kwargs), requested


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if args.max_replay_tokens <= 0:
        raise ValueError("--max-replay-tokens must be positive")
    if args.top_k_logits <= 1:
        raise ValueError("--top-k-logits must be at least 2")
    if args.flush_every <= 0:
        raise ValueError("--flush-every must be positive")

    rope_scaling = parse_json_object(args.rope_scaling_json)
    reference_records = load_jsonl(args.reference_generations)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        reference_records = reference_records[: args.limit]
    if not reference_records:
        raise ValueError("No successful reference records were found")

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = args.output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    record_path = args.output_dir / "replay_records.jsonl"
    config_path = args.output_dir / "experiment_config.json"
    completed = completed_sample_ids(record_path) if args.resume else set()
    if record_path.exists() and not args.resume:
        raise FileExistsError(f"{record_path} exists; use a new directory or --resume")

    llm, requested_dtype = create_llm(args, rope_scaling)
    tokenizer = llm.get_tokenizer()
    resolved_dtype, resolved_path = resolve_engine_cache_dtype(llm)
    backend_verified = (
        args.mode == "fp16" and requested_dtype == "auto"
    ) or (
        resolved_dtype is not None and requested_dtype.lower() in resolved_dtype.lower()
    )
    if args.mode == "kvarn" and args.require_backend_verification and not backend_verified:
        raise RuntimeError("Could not verify the requested KVarN cache dtype")

    config = {
        "schema_version": 1,
        "mode": args.mode,
        "replay_run_name": args.replay_run_name,
        "reference_generations": str(args.reference_generations.resolve()),
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "dtype": args.dtype,
        "requested_kv_cache_dtype": requested_dtype,
        "resolved_kv_cache_dtype": resolved_dtype,
        "resolved_kv_cache_dtype_path": resolved_path,
        "backend_verified": backend_verified,
        "block_size": args.block_size,
        "max_replay_tokens": args.max_replay_tokens,
        "max_model_len": args.max_model_len,
        "rope_scaling": rope_scaling,
        "top_k_logits": args.top_k_logits,
        "flush_every": args.flush_every,
        "seed": args.seed,
        "num_reference_samples": len(reference_records),
        "environment": collect_environment(repo_root),
    }
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    from vllm import SamplingParams

    mode = "a" if args.resume else "w"
    with record_path.open(mode, encoding="utf-8") as output_handle:
        for index, reference in enumerate(reference_records, start=1):
            sample_id = str(reference["sample_id"])
            if sample_id in completed:
                LOGGER.info("[%d/%d] skip %s", index, len(reference_records), sample_id)
                continue
            base_record = {
                "schema_version": 1,
                "sample_id": sample_id,
                "mode": args.mode,
                "replay_run_name": args.replay_run_name,
                "reference_run_name": reference.get("run_name"),
                "reference_prompt_sha256": reference.get("prompt_sha256"),
                "requested_kv_cache_dtype": requested_dtype,
                "resolved_kv_cache_dtype": resolved_dtype,
                "backend_verified": backend_verified,
            }
            try:
                prompt = str(reference["prompt_text"])
                full_reference_ids = [int(value) for value in reference["output_token_ids"]]
                if not full_reference_ids:
                    raise ValueError("Reference output token list is empty")
                target_ids = full_reference_ids[: args.max_replay_tokens]
                input_ids = [int(value) for value in reference.get("input_token_ids", [])]
                boundary_start, boundary_end = thinking_boundary(tokenizer, full_reference_ids)
                metrics_path = (metrics_dir / f"{sample_id}.jsonl").resolve()
                if metrics_path.exists() and not args.resume:
                    raise FileExistsError(f"Metrics path already exists: {metrics_path}")
                if metrics_path.exists() and args.resume:
                    metrics_path.unlink()

                sampling_params = SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    top_k=-1,
                    max_tokens=len(target_ids),
                    min_tokens=len(target_ids),
                    ignore_eos=True,
                    seed=args.seed,
                    skip_special_tokens=False,
                    extra_args={
                        "forced_token_ids": target_ids,
                        "metrics_path": str(metrics_path),
                        "sample_id": sample_id,
                        "mode": args.mode,
                        "replay_run_name": args.replay_run_name,
                        "prompt_token_count": len(input_ids),
                        "block_size": args.block_size,
                        "top_k": args.top_k_logits,
                        "flush_every": args.flush_every,
                        "thinking_boundary_start": boundary_start,
                        "thinking_boundary_end": boundary_end,
                    },
                )
                LOGGER.info(
                    "[%d/%d] replay sample=%s tokens=%d",
                    index,
                    len(reference_records),
                    sample_id,
                    len(target_ids),
                )
                started = time.perf_counter()
                request_output = llm.generate([prompt], sampling_params, use_tqdm=False)[0]
                elapsed = time.perf_counter() - started
                if not request_output.outputs:
                    raise RuntimeError("vLLM returned no output")
                candidate = request_output.outputs[0]
                replayed_ids = [int(value) for value in candidate.token_ids]
                replay_prompt_ids = [
                    int(value) for value in (request_output.prompt_token_ids or [])
                ]
                replay_verified = replayed_ids == target_ids
                prompt_ids_match = not input_ids or replay_prompt_ids == input_ids
                metric_lines = 0
                if metrics_path.exists():
                    with metrics_path.open("r", encoding="utf-8") as handle:
                        metric_lines = sum(1 for line in handle if line.strip())
                record = {
                    **base_record,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "prompt_text": prompt,
                    "prompt_token_ids_match_reference": prompt_ids_match,
                    "reference_input_token_ids": input_ids,
                    "replay_input_token_ids": replay_prompt_ids,
                    "full_reference_output_tokens": len(full_reference_ids),
                    "target_replay_tokens": len(target_ids),
                    "forced_reference_token_ids": target_ids,
                    "forced_reference_token_ids_sha256": sha256_ids(target_ids),
                    "replayed_token_ids": replayed_ids,
                    "replayed_token_ids_sha256": sha256_ids(replayed_ids),
                    "replayed_text": tokenizer.decode(
                        replayed_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                    "replay_verified": replay_verified,
                    "metrics_path": str(metrics_path),
                    "metrics_lines": metric_lines,
                    "metrics_complete": metric_lines == len(target_ids),
                    "thinking_boundary_start": boundary_start,
                    "thinking_boundary_end": boundary_end,
                    "latency_seconds": elapsed,
                    "tokens_per_second": len(replayed_ids) / elapsed if elapsed > 0 else None,
                    "finish_reason": getattr(candidate, "finish_reason", None),
                    "stop_reason": getattr(candidate, "stop_reason", None),
                    "error": None,
                }
                if not replay_verified:
                    raise RuntimeError(
                        f"Teacher forcing verification failed for {sample_id}: "
                        f"expected {len(target_ids)} exact tokens, got {len(replayed_ids)}"
                    )
                if metric_lines != len(target_ids):
                    raise RuntimeError(
                        f"Metrics incomplete for {sample_id}: {metric_lines}/{len(target_ids)}"
                    )
            except Exception as exc:
                LOGGER.exception("Aligned replay failed for %s", sample_id)
                record = {
                    **base_record,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "replay_verified": False,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                }
            append_jsonl(output_handle, record)

    LOGGER.info("Aligned replay complete: %s", record_path)


if __name__ == "__main__":
    main()
