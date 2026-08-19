#!/usr/bin/env python3
"""Run HumanEval free generation with FP16 KV or KVarN KV cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from humaneval_utils import (
    evaluate_humaneval_candidate,
    extract_python_candidate,
    split_thinking_tokens,
)
from run_generation import (
    append_jsonl,
    collect_environment,
    get_nested_attr,
    parse_json_object,
    read_completed_sample_ids,
    resolve_engine_cache_dtype,
    safe_list,
)

LOGGER = logging.getLogger("humaneval_generation")

DEFAULT_TASK_INSTRUCTION = (
    "Complete the following Python function so that it satisfies its docstring and tests. "
    "Return a valid Python implementation. You may reason before the final answer, but the "
    "final answer should contain executable Python code.\n\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate HumanEval trajectories with either FP16 or KVarN KV cache."
    )
    parser.add_argument("--mode", choices=("fp16", "kvarn"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--samples-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--fp16-kv-cache-dtype", default="auto")
    parser.add_argument("--kvarn-kv-cache-dtype", default="kvarn_k4v2_g128")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--rope-scaling-json", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--swap-space", type=float, default=4.0)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--execute-tests",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Execute generated HumanEval code in a resource-limited subprocess.",
    )
    parser.add_argument("--execution-timeout", type=float, default=30.0)
    parser.add_argument("--execution-memory-mb", type=int, default=1024)
    parser.add_argument(
        "--task-instruction-file",
        type=Path,
        default=None,
        help="Optional replacement for the built-in HumanEval user instruction.",
    )
    parser.add_argument("--system-prompt-file", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--require-backend-verification", action="store_true")
    return parser.parse_args()


def load_manifest(path: Path, limit: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    samples = list(manifest.get("samples", []))
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        samples = samples[:limit]
    if not samples:
        raise ValueError(f"No samples found in {path}")
    return manifest, samples


def build_humaneval_prompt(
    tokenizer: Any,
    *,
    raw_prompt: str,
    task_instruction: str,
    enable_thinking: bool,
    system_prompt: str | None,
) -> tuple[str, bool]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": task_instruction + raw_prompt})
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return str(prompt), True
    except TypeError:
        LOGGER.warning("Tokenizer chat template rejected enable_thinking; retrying without it")
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return str(prompt), False


def create_llm(args: argparse.Namespace, rope_scaling: dict[str, Any] | None) -> tuple[Any, str]:
    try:
        from vllm import LLM
    except ImportError as exc:
        raise SystemExit("Could not import the local KVarN vLLM package") from exc

    requested_dtype = (
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
        "trust_remote_code": True,
        "enable_prefix_caching": False,
        "disable_log_stats": False,
        "block_size": args.block_size,
        "seed": args.seed,
    }
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    if rope_scaling is not None:
        kwargs["hf_overrides"] = {"rope_scaling": rope_scaling}
    if requested_dtype != "auto":
        kwargs["kv_cache_dtype"] = requested_dtype
    LOGGER.info("Initializing %s engine with kv_cache_dtype=%s", args.mode, requested_dtype)
    return LLM(**kwargs), requested_dtype


def extract_request_timing(request_output: Any, output_tokens: int) -> dict[str, Any]:
    metrics = getattr(request_output, "metrics", None)
    if metrics is None:
        return {
            "ttft_seconds": None,
            "queue_time_seconds": None,
            "prefill_time_seconds": None,
            "decode_time_seconds": None,
            "tpot_seconds": None,
            "engine_inference_span_seconds": None,
        }

    def number(name: str) -> float | None:
        value = getattr(metrics, name, None)
        return float(value) if isinstance(value, (int, float)) else None

    first_token_latency = number("first_token_latency")
    queued_ts = number("queued_ts")
    scheduled_ts = number("scheduled_ts")
    first_token_ts = number("first_token_ts")
    last_token_ts = number("last_token_ts")

    queue_time = None
    if queued_ts is not None and scheduled_ts is not None and scheduled_ts >= queued_ts:
        queue_time = scheduled_ts - queued_ts

    prefill_time = None
    if scheduled_ts is not None and first_token_ts is not None and first_token_ts >= scheduled_ts:
        prefill_time = first_token_ts - scheduled_ts

    decode_time = None
    if first_token_ts is not None and last_token_ts is not None and last_token_ts >= first_token_ts:
        decode_time = last_token_ts - first_token_ts

    tpot = None
    if decode_time is not None and output_tokens > 1:
        tpot = decode_time / (output_tokens - 1)

    inference_span = None
    if scheduled_ts is not None and last_token_ts is not None and last_token_ts >= scheduled_ts:
        inference_span = last_token_ts - scheduled_ts

    return {
        "ttft_seconds": first_token_latency,
        "queue_time_seconds": queue_time,
        "prefill_time_seconds": prefill_time,
        "decode_time_seconds": decode_time,
        "tpot_seconds": tpot,
        "engine_inference_span_seconds": inference_span,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if args.max_tokens <= 0 or args.max_model_len <= 0 or args.max_tokens >= args.max_model_len:
        raise ValueError("Require 0 < max_tokens < max_model_len")
    if args.execution_timeout <= 0:
        raise ValueError("--execution-timeout must be positive")

    rope_scaling = parse_json_object(args.rope_scaling_json, "--rope-scaling-json")
    manifest, samples = load_manifest(args.samples_file, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "generations.jsonl"
    config_path = args.output_dir / "experiment_config.json"
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"{output_path} already exists; use a new output dir or --resume")
    completed = read_completed_sample_ids(output_path) if args.resume else set()

    task_instruction = DEFAULT_TASK_INSTRUCTION
    if args.task_instruction_file:
        task_instruction = args.task_instruction_file.read_text(encoding="utf-8").rstrip() + "\n\n"
    system_prompt = None
    if args.system_prompt_file:
        system_prompt = args.system_prompt_file.read_text(encoding="utf-8").strip()

    llm, requested_dtype = create_llm(args, rope_scaling)
    tokenizer = llm.get_tokenizer()
    resolved_dtype, resolved_path = resolve_engine_cache_dtype(llm)
    backend_verified = (
        args.mode == "fp16" and requested_dtype == "auto"
    ) or (
        resolved_dtype is not None and requested_dtype.lower() in resolved_dtype.lower()
    )
    if args.mode == "kvarn" and not backend_verified:
        message = "Could not verify active KVarN cache dtype from engine internals"
        if args.require_backend_verification:
            raise RuntimeError(message)
        LOGGER.warning(message)

    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    config = {
        "schema_version": 1,
        "task_type": "humaneval",
        "mode": args.mode,
        "run_name": args.run_name,
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "dtype": args.dtype,
        "requested_kv_cache_dtype": requested_dtype,
        "resolved_kv_cache_dtype": resolved_dtype,
        "resolved_kv_cache_dtype_path": resolved_path,
        "backend_verified": backend_verified,
        "block_size": args.block_size,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "rope_scaling": rope_scaling,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "seed": args.seed,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
        "enable_thinking_requested": args.enable_thinking,
        "prefix_caching": False,
        "execute_tests": args.execute_tests,
        "execution_timeout": args.execution_timeout,
        "execution_memory_mb": args.execution_memory_mb,
        "task_instruction": task_instruction,
        "samples_file": str(args.samples_file.resolve()),
        "num_manifest_samples": len(samples),
        "dataset": {
            key: manifest.get(key)
            for key in ("dataset_name", "dataset_config", "dataset_split", "dataset_size")
        },
        "environment": collect_environment(repo_root),
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    LOGGER.info("Starting %d samples (%d already complete)", len(samples), len(completed))
    mode = "a" if args.resume else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for position, sample in enumerate(samples, start=1):
            sample_id = str(sample["sample_id"])
            if sample_id in completed:
                LOGGER.info("[%d/%d] skip completed %s", position, len(samples), sample_id)
                continue

            raw_prompt = str(sample["prompt"])
            entry_point = str(sample["entry_point"])
            base_record: dict[str, Any] = {
                "schema_version": 1,
                "task_type": "humaneval",
                "sample_id": sample_id,
                "source_id": sample.get("source_id"),
                "task_id": sample.get("task_id"),
                "dataset_index": sample.get("dataset_index"),
                "selection_order": sample.get("selection_order"),
                "problem": raw_prompt,
                "entry_point": entry_point,
                "subject": "code",
                "level": None,
                "mode": args.mode,
                "run_name": args.run_name,
                "seed": args.seed,
                "model": args.model,
                "dtype": args.dtype,
                "requested_kv_cache_dtype": requested_dtype,
                "resolved_kv_cache_dtype": resolved_dtype,
                "backend_verified": backend_verified,
                "block_size": args.block_size,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "min_p": args.min_p,
                "max_tokens": args.max_tokens,
                "max_model_len": args.max_model_len,
            }

            try:
                prompt_text, thinking_arg_applied = build_humaneval_prompt(
                    tokenizer,
                    raw_prompt=raw_prompt,
                    task_instruction=task_instruction,
                    enable_thinking=args.enable_thinking,
                    system_prompt=system_prompt,
                )
                prompt_sha256 = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
                LOGGER.info("[%d/%d] generating %s", position, len(samples), sample_id)
                started = time.perf_counter()
                request_output = llm.generate([prompt_text], sampling_params, use_tqdm=False)[0]
                generation_elapsed = time.perf_counter() - started
                if not request_output.outputs:
                    raise RuntimeError("vLLM returned no candidate outputs")
                candidate = request_output.outputs[0]
                input_token_ids = safe_list(getattr(request_output, "prompt_token_ids", None))
                if not input_token_ids:
                    input_token_ids = safe_list(tokenizer.encode(prompt_text))
                output_token_ids = [int(value) for value in safe_list(getattr(candidate, "token_ids", None))]
                output_text = str(getattr(candidate, "text", ""))

                split = split_thinking_tokens(tokenizer, output_token_ids)
                extraction = extract_python_candidate(output_text, entry_point)
                if args.execute_tests:
                    execution = evaluate_humaneval_candidate(
                        prompt=raw_prompt,
                        candidate_code=extraction["candidate_code"],
                        test=str(sample["test"]),
                        entry_point=entry_point,
                        timeout_seconds=args.execution_timeout,
                        memory_mb=args.execution_memory_mb,
                    )
                    approx_correct: bool | None = bool(execution["passed"])
                else:
                    execution = {
                        "passed": None,
                        "result": "not_executed",
                        "returncode": None,
                        "execution_seconds": None,
                        "assembly": None,
                        "program_sha256": None,
                        "stdout_tail": "",
                        "stderr_tail": "",
                        "timed_out": False,
                    }
                    approx_correct = None

                timing = extract_request_timing(request_output, len(output_token_ids))
                record = {
                    **base_record,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "prompt_text": prompt_text,
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
                    "latency_seconds": generation_elapsed,
                    "tokens_per_second": (
                        len(output_token_ids) / generation_elapsed if generation_elapsed > 0 else None
                    ),
                    **timing,
                    **split,
                    "visible_final_output": extraction["visible_output"],
                    "candidate_code": extraction["candidate_code"],
                    "candidate_source": extraction["candidate_source"],
                    "candidate_code_sha256": extraction["candidate_code_sha256"],
                    "contains_entry_point_definition": extraction["contains_entry_point_definition"],
                    "execution_result": execution["result"],
                    "execution_passed": execution["passed"],
                    "execution_seconds": execution["execution_seconds"],
                    "execution_returncode": execution["returncode"],
                    "execution_assembly": execution["assembly"],
                    "execution_program_sha256": execution["program_sha256"],
                    "execution_stdout_tail": execution["stdout_tail"],
                    "execution_stderr_tail": execution["stderr_tail"],
                    "execution_timed_out": execution["timed_out"],
                    # Compatibility fields consumed by compare_trajectories.py.
                    "extracted_answer": extraction["candidate_code"],
                    "normalized_extracted_answer": extraction["candidate_code_sha256"],
                    "normalized_reference_answer": None,
                    "approx_correct": approx_correct,
                    "error": None,
                }
            except Exception as exc:
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
                    "ttft_seconds": None,
                    "queue_time_seconds": None,
                    "prefill_time_seconds": None,
                    "decode_time_seconds": None,
                    "tpot_seconds": None,
                    "engine_inference_span_seconds": None,
                    "thinking_boundary_detected": False,
                    "thinking_end_step": None,
                    "thinking_tokens": None,
                    "final_tokens": None,
                    "candidate_code": None,
                    "candidate_code_sha256": None,
                    "execution_result": None,
                    "execution_passed": None,
                    "execution_seconds": None,
                    "extracted_answer": None,
                    "normalized_extracted_answer": None,
                    "normalized_reference_answer": None,
                    "approx_correct": None,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                }
            append_jsonl(handle, record)

    LOGGER.info("HumanEval generation complete: %s", output_path)


if __name__ == "__main__":
    main()
