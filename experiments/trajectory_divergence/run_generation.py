#!/usr/bin/env python3
"""Run one trajectory-divergence generation mode: FP16 KV or KVarN KV.

The script intentionally creates only one vLLM engine. Run FP16 and KVarN in
separate processes so their memory use and runtime state cannot interfere.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

LOGGER = logging.getLogger("trajectory_generation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MATH-500 trajectories with either FP16 or KVarN KV cache."
    )
    parser.add_argument("--mode", choices=("fp16", "kvarn"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--samples-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--fp16-kv-cache-dtype", default="auto")
    parser.add_argument("--kvarn-kv-cache-dtype", default="kvarn_k4v2_g128")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--swap-space", type=float, default=4.0)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass enable_thinking to the Qwen chat template.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        default=None,
        help="Optional text file prepended as a system message.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip sample IDs already completed successfully in generations.jsonl.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit applied to the beginning of the fixed sample manifest.",
    )
    parser.add_argument(
        "--require-backend-verification",
        action="store_true",
        help=(
            "Fail if the initialized vLLM engine cannot be introspected to confirm "
            "the requested KVarN cache dtype."
        ),
    )
    return parser.parse_args()


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
        gpu_names = (
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if cuda_available
            else []
        )
        torch_version = torch.__version__
        torch_cuda = torch.version.cuda
    except Exception:  # pragma: no cover - environment dependent
        cuda_available = False
        gpu_names = []
        torch_version = package_version("torch")
        torch_cuda = None

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "working_directory": str(Path.cwd()),
        "repo_root": str(repo_root),
        "git_commit": run_command(["git", "rev-parse", "HEAD"], cwd=repo_root),
        "git_status_porcelain": run_command(
            ["git", "status", "--porcelain"], cwd=repo_root
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "torch": torch_version,
            "vllm": package_version("vllm"),
            "transformers": package_version("transformers"),
            "datasets": package_version("datasets"),
        },
        "cuda_available": cuda_available,
        "cuda_version": torch_cuda,
        "gpu_names": gpu_names,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "vllm_use_flashinfer_sampler": os.environ.get(
            "VLLM_USE_FLASHINFER_SAMPLER"
        ),
    }


def load_manifest(path: Path, limit: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    samples = list(manifest.get("samples", []))
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        samples = samples[:limit]
    if not samples:
        raise ValueError(f"No samples found in {path}")
    return manifest, samples


def read_completed_sample_ids(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("Ignoring malformed JSONL line %d in %s", line_number, path)
                continue
            if record.get("error") is None and record.get("sample_id"):
                completed.add(str(record["sample_id"]))
    return completed


def get_nested_attr(obj: Any, path: str) -> Any:
    current = obj
    for part in path.split("."):
        current = getattr(current, part)
    return current


def resolve_engine_cache_dtype(llm: Any) -> tuple[str | None, str | None]:
    candidate_paths = (
        "llm_engine.vllm_config.cache_config.cache_dtype",
        "llm_engine.cache_config.cache_dtype",
        "llm_engine.engine_core.vllm_config.cache_config.cache_dtype",
    )
    for path in candidate_paths:
        try:
            value = get_nested_attr(llm, path)
        except (AttributeError, TypeError):
            continue
        return str(value), path
    return None, None


def extract_last_boxed(text: str) -> str | None:
    """Extract the last \boxed{...} payload, supporting nested braces."""
    marker = r"\boxed{"
    starts = [match.start() for match in re.finditer(re.escape(marker), text)]
    for start in reversed(starts):
        cursor = start + len(marker)
        depth = 1
        while cursor < len(text):
            char = text[cursor]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start + len(marker) : cursor]
            cursor += 1
    return None


def normalize_answer(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    boxed = extract_last_boxed(text)
    if boxed is not None:
        text = boxed
    text = text.strip().strip("$")
    text = re.sub(r"\s+", "", text)
    text = text.replace(r"\,", "")
    text = text.replace("−", "-")
    return text or None


def append_jsonl(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def build_prompt(
    tokenizer: Any,
    problem: str,
    enable_thinking: bool,
    system_prompt: str | None,
) -> tuple[str, bool]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": problem})

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return str(prompt), True
    except TypeError:
        LOGGER.warning(
            "Tokenizer chat template rejected enable_thinking; falling back without it."
        )
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return str(prompt), False


def create_llm(args: argparse.Namespace) -> tuple[Any, str]:
    try:
        from vllm import LLM
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Could not import the local KVarN vLLM package. Run this script from "
            "the installed KVarN environment."
        ) from exc

    requested_kv_dtype = (
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
        "swap_space": args.swap_space,
        "trust_remote_code": args.trust_remote_code,
        "enable_prefix_caching": False,
        "disable_log_stats": False,
        "block_size": args.block_size,
        "seed": args.seed,
    }
    if args.revision:
        kwargs["revision"] = args.revision
    if args.tokenizer_revision:
        kwargs["tokenizer_revision"] = args.tokenizer_revision
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    if requested_kv_dtype != "auto":
        kwargs["kv_cache_dtype"] = requested_kv_dtype

    LOGGER.info("Initializing %s engine with kv_cache_dtype=%s", args.mode, requested_kv_dtype)
    return LLM(**kwargs), requested_kv_dtype


def create_sampling_params(args: argparse.Namespace) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )


def safe_list(value: Iterable[Any] | None) -> list[Any]:
    return list(value) if value is not None else []


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "generations.jsonl"
    config_path = args.output_dir / "experiment_config.json"

    manifest, samples = load_manifest(args.samples_file, args.limit)
    completed = read_completed_sample_ids(output_path) if args.resume else set()
    if output_path.exists() and not args.resume:
        raise FileExistsError(
            f"{output_path} already exists. Use a new output directory or pass --resume."
        )

    system_prompt = None
    if args.system_prompt_file:
        system_prompt = args.system_prompt_file.read_text(encoding="utf-8").strip()

    llm, requested_kv_dtype = create_llm(args)
    tokenizer = llm.get_tokenizer()
    sampling_params = create_sampling_params(args)
    resolved_kv_dtype, resolved_path = resolve_engine_cache_dtype(llm)
    backend_verified = (
        args.mode == "fp16" and requested_kv_dtype == "auto"
    ) or (
        resolved_kv_dtype is not None
        and requested_kv_dtype.lower() in resolved_kv_dtype.lower()
    )

    LOGGER.info(
        "Requested KV dtype=%s; resolved=%s via %s; verified=%s",
        requested_kv_dtype,
        resolved_kv_dtype,
        resolved_path,
        backend_verified,
    )
    if args.mode == "kvarn" and not backend_verified:
        message = (
            "Could not verify from engine internals that the requested KVarN dtype "
            "is active. Inspect vLLM initialization logs and experiment_config.json."
        )
        if args.require_backend_verification:
            raise RuntimeError(message)
        LOGGER.warning(message)

    config = {
        "schema_version": 1,
        "mode": args.mode,
        "run_name": args.run_name,
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "revision": args.revision,
        "tokenizer_revision": args.tokenizer_revision,
        "dtype": args.dtype,
        "requested_kv_cache_dtype": requested_kv_dtype,
        "resolved_kv_cache_dtype": resolved_kv_dtype,
        "resolved_kv_cache_dtype_path": resolved_path,
        "backend_verified": backend_verified,
        "block_size": args.block_size,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "swap_space": args.swap_space,
        "max_num_seqs": args.max_num_seqs,
        "enable_thinking_requested": args.enable_thinking,
        "prefix_caching": False,
        "samples_file": str(args.samples_file.resolve()),
        "num_manifest_samples": len(samples),
        "dataset": {
            key: manifest.get(key)
            for key in (
                "dataset_name",
                "dataset_config",
                "dataset_split",
                "dataset_size",
                "seed",
            )
        },
        "environment": collect_environment(repo_root),
    }
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")

    LOGGER.info(
        "Starting %d samples (%d already complete)", len(samples), len(completed)
    )
    mode_open = "a" if args.resume else "w"
    with output_path.open(mode_open, encoding="utf-8") as output_handle:
        for position, sample in enumerate(samples, start=1):
            sample_id = str(sample["sample_id"])
            if sample_id in completed:
                LOGGER.info("[%d/%d] skip completed %s", position, len(samples), sample_id)
                continue

            problem = str(sample["problem"])
            reference_answer = sample.get("reference_answer")
            base_record: dict[str, Any] = {
                "schema_version": 1,
                "sample_id": sample_id,
                "source_id": sample.get("source_id"),
                "dataset_index": sample.get("dataset_index"),
                "selection_order": sample.get("selection_order"),
                "problem": problem,
                "reference_answer": reference_answer,
                "subject": sample.get("subject"),
                "level": sample.get("level"),
                "mode": args.mode,
                "run_name": args.run_name,
                "seed": args.seed,
                "model": args.model,
                "dtype": args.dtype,
                "requested_kv_cache_dtype": requested_kv_dtype,
                "resolved_kv_cache_dtype": resolved_kv_dtype,
                "backend_verified": backend_verified,
                "block_size": args.block_size,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_tokens": args.max_tokens,
            }

            try:
                prompt, thinking_arg_applied = build_prompt(
                    tokenizer=tokenizer,
                    problem=problem,
                    enable_thinking=args.enable_thinking,
                    system_prompt=system_prompt,
                )
                prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                LOGGER.info(
                    "[%d/%d] generating sample=%s prompt_sha256=%s",
                    position,
                    len(samples),
                    sample_id,
                    prompt_sha256[:12],
                )

                started = time.perf_counter()
                request_output = llm.generate(
                    [prompt],
                    sampling_params,
                    use_tqdm=False,
                )[0]
                elapsed = time.perf_counter() - started
                if not request_output.outputs:
                    raise RuntimeError("vLLM returned no candidate outputs")
                candidate = request_output.outputs[0]

                input_token_ids = safe_list(
                    getattr(request_output, "prompt_token_ids", None)
                )
                if not input_token_ids:
                    input_token_ids = safe_list(tokenizer.encode(prompt))
                output_token_ids = safe_list(getattr(candidate, "token_ids", None))
                output_text = str(getattr(candidate, "text", ""))
                extracted_answer = extract_last_boxed(output_text)
                normalized_prediction = normalize_answer(extracted_answer)
                normalized_reference = normalize_answer(reference_answer)
                approx_correct = (
                    normalized_prediction is not None
                    and normalized_reference is not None
                    and normalized_prediction == normalized_reference
                )

                record = {
                    **base_record,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "prompt_text": prompt,
                    "prompt_sha256": prompt_sha256,
                    "enable_thinking_requested": args.enable_thinking,
                    "enable_thinking_argument_applied": thinking_arg_applied,
                    "input_token_ids": input_token_ids,
                    "input_tokens": len(input_token_ids),
                    "output_token_ids": output_token_ids,
                    "output_tokens": len(output_token_ids),
                    "output_text": output_text,
                    "finish_reason": getattr(candidate, "finish_reason", None),
                    "stop_reason": getattr(candidate, "stop_reason", None),
                    "latency_seconds": elapsed,
                    "tokens_per_second": (
                        len(output_token_ids) / elapsed if elapsed > 0 else None
                    ),
                    "extracted_answer": extracted_answer,
                    "normalized_extracted_answer": normalized_prediction,
                    "normalized_reference_answer": normalized_reference,
                    "approx_correct": approx_correct,
                    "error": None,
                }
            except Exception as exc:  # keep the long experiment alive
                LOGGER.exception("Sample %s failed", sample_id)
                record = {
                    **base_record,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "prompt_text": None,
                    "prompt_sha256": None,
                    "input_token_ids": [],
                    "input_tokens": None,
                    "output_token_ids": [],
                    "output_tokens": None,
                    "output_text": None,
                    "finish_reason": None,
                    "stop_reason": None,
                    "latency_seconds": None,
                    "tokens_per_second": None,
                    "extracted_answer": None,
                    "normalized_extracted_answer": None,
                    "normalized_reference_answer": normalize_answer(reference_answer),
                    "approx_correct": None,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                }
            append_jsonl(output_handle, record)

    LOGGER.info("Generation complete: %s", output_path)


if __name__ == "__main__":
    main()
