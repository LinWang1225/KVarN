#!/usr/bin/env python3
"""Teacher-forced logits processor for aligned KVarN trajectory analysis.

The processor records raw, pre-mask next-token statistics and then masks every
vocabulary entry except the requested FP16 reference token. This keeps all replay
modes on exactly the same token history while preserving evidence about what the
unmodified model would have selected at each step.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from vllm import SamplingParams
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)


class TeacherForcedRequestLogitsProcessor:
    """Record raw logits statistics, then force one reference token per step."""

    def __init__(
        self,
        *,
        forced_token_ids: list[int],
        metrics_path: str,
        sample_id: str,
        mode: str,
        replay_run_name: str,
        prompt_token_count: int,
        block_size: int,
        top_k: int,
        flush_every: int,
        thinking_boundary_start: int | None,
        thinking_boundary_end: int | None,
    ) -> None:
        self.forced_token_ids = forced_token_ids
        self.metrics_path = Path(metrics_path)
        self.sample_id = sample_id
        self.mode = mode
        self.replay_run_name = replay_run_name
        self.prompt_token_count = prompt_token_count
        self.block_size = block_size
        self.top_k = top_k
        self.flush_every = flush_every
        self.thinking_boundary_start = thinking_boundary_start
        self.thinking_boundary_end = thinking_boundary_end
        self.buffer: list[dict[str, Any]] = []
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def _region(self, step: int) -> str:
        start = self.thinking_boundary_start
        end = self.thinking_boundary_end
        if start is None or end is None:
            return "unknown"
        if step < start:
            return "thinking"
        if step < end:
            return "boundary"
        return "final_answer"

    def _flush(self) -> None:
        if not self.buffer:
            return
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            for record in self.buffer:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
        self.buffer.clear()

    def __call__(self, output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        step = len(output_ids)
        if step >= len(self.forced_token_ids):
            raise RuntimeError(
                f"Teacher-forced replay requested step {step}, but only "
                f"{len(self.forced_token_ids)} reference tokens are available."
            )

        forced_token_id = int(self.forced_token_ids[step])
        if forced_token_id < 0 or forced_token_id >= logits.numel():
            raise RuntimeError(
                f"Reference token {forced_token_id} is outside logits vocabulary "
                f"of size {logits.numel()} at step {step}."
            )

        raw = logits.float()
        k = min(self.top_k, raw.numel())
        top_values, top_indices = torch.topk(raw, k=k, largest=True, sorted=True)
        logsumexp = torch.logsumexp(raw, dim=-1)
        forced_logit = raw[forced_token_id]
        forced_logprob = forced_logit - logsumexp
        forced_rank = 1 + torch.count_nonzero(raw > forced_logit)

        top_logprobs = top_values - logsumexp
        top1_token_id = int(top_indices[0].item())
        top2_token_id = int(top_indices[1].item()) if k >= 2 else None
        top1_margin = (
            float((top_values[0] - top_values[1]).item()) if k >= 2 else math.inf
        )
        absolute_position = self.prompt_token_count + step

        record = {
            "schema_version": 1,
            "sample_id": self.sample_id,
            "mode": self.mode,
            "replay_run_name": self.replay_run_name,
            "step": step,
            "absolute_position": absolute_position,
            "absolute_block": absolute_position // self.block_size,
            "absolute_offset_in_block": absolute_position % self.block_size,
            "region": self._region(step),
            "forced_token_id": forced_token_id,
            "raw_top1_token_id": top1_token_id,
            "raw_top2_token_id": top2_token_id,
            "raw_top1_is_forced": top1_token_id == forced_token_id,
            "raw_top1_logit": float(top_values[0].item()),
            "raw_top2_logit": float(top_values[1].item()) if k >= 2 else None,
            "raw_top1_logprob": float(top_logprobs[0].item()),
            "raw_top2_logprob": float(top_logprobs[1].item()) if k >= 2 else None,
            "raw_top1_margin": top1_margin,
            "forced_token_logit": float(forced_logit.item()),
            "forced_token_logprob": float(forced_logprob.item()),
            "forced_token_rank": int(forced_rank.item()),
            "raw_logsumexp": float(logsumexp.item()),
            "raw_topk_token_ids": [int(value) for value in top_indices.tolist()],
            "raw_topk_logits": [float(value) for value in top_values.tolist()],
            "raw_topk_logprobs": [float(value) for value in top_logprobs.tolist()],
        }
        self.buffer.append(record)
        if (step + 1) % self.flush_every == 0 or step + 1 == len(
            self.forced_token_ids
        ):
            self._flush()

        value_to_keep = logits[forced_token_id].clone()
        logits[:] = float("-inf")
        logits[forced_token_id] = value_to_keep
        return logits


class TeacherForcedReplayLogitsProcessor(AdapterLogitsProcessor):
    """vLLM batch adapter that creates one teacher-forcing processor/request."""

    @classmethod
    def validate_params(cls, params: SamplingParams) -> None:
        args = params.extra_args or {}
        forced = args.get("forced_token_ids")
        if forced is None:
            return
        if not isinstance(forced, list) or not forced:
            raise ValueError("forced_token_ids must be a non-empty list of integers")
        if not all(isinstance(token_id, int) for token_id in forced):
            raise ValueError("forced_token_ids must contain only integers")
        if not isinstance(args.get("metrics_path"), str):
            raise ValueError("metrics_path must be a string")
        if not isinstance(args.get("sample_id"), str):
            raise ValueError("sample_id must be a string")
        if not isinstance(args.get("mode"), str):
            raise ValueError("mode must be a string")
        if not isinstance(args.get("replay_run_name"), str):
            raise ValueError("replay_run_name must be a string")
        for key in ("prompt_token_count", "block_size", "top_k", "flush_every"):
            if not isinstance(args.get(key), int) or int(args[key]) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("thinking_boundary_start", "thinking_boundary_end"):
            value = args.get(key)
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError(f"{key} must be null or a non-negative integer")

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> RequestLogitsProcessor | None:
        args = params.extra_args or {}
        forced = args.get("forced_token_ids")
        if forced is None:
            return None
        self.validate_params(params)
        return TeacherForcedRequestLogitsProcessor(
            forced_token_ids=list(forced),
            metrics_path=str(args["metrics_path"]),
            sample_id=str(args["sample_id"]),
            mode=str(args["mode"]),
            replay_run_name=str(args["replay_run_name"]),
            prompt_token_count=int(args["prompt_token_count"]),
            block_size=int(args["block_size"]),
            top_k=int(args["top_k"]),
            flush_every=int(args["flush_every"]),
            thinking_boundary_start=args.get("thinking_boundary_start"),
            thinking_boundary_end=args.get("thinking_boundary_end"),
        )
