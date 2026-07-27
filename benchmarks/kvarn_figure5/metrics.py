# SPDX-License-Identifier: Apache-2.0
"""Pure-Python result extraction and comparison helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


METHOD_SPECS: dict[str, dict[str, str]] = {
    "fp16": {
        "label": "FP16",
        "kv_cache_dtype": "auto",
    },
    "kvarn": {
        "label": "KVarN K4/V2 G128",
        "kv_cache_dtype": "kvarn_k4v2_g128",
    },
    "turboquant": {
        "label": "TurboQuant 3-bit NC",
        "kv_cache_dtype": "turboquant_3bit_nc",
    },
}


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def extract_prompt_window(
    prompt_token_ids: Sequence[int],
    prompt_logprobs: Sequence[Mapping[int, Any] | None],
    *,
    start: int,
    end: int,
) -> list[dict[str, int | float | None]]:
    """Materialize target-token and top-1 information for ``[start, end)``.

    vLLM always includes the prompt token itself in each requested prompt-
    logprob dictionary, even when it is outside the returned top-k. The first
    prompt position has no log probability and is skipped.
    """
    if len(prompt_token_ids) != len(prompt_logprobs):
        raise ValueError(
            "prompt token/logprob length mismatch: "
            f"{len(prompt_token_ids)} != {len(prompt_logprobs)}"
        )
    start = max(1, start)
    end = min(end, len(prompt_token_ids))
    rows: list[dict[str, int | float | None]] = []
    for position in range(start, end):
        entry = prompt_logprobs[position]
        if not entry:
            raise ValueError(f"Missing prompt logprobs at position {position}")
        target_token_id = int(prompt_token_ids[position])
        target = entry.get(target_token_id)
        if target is None:
            raise ValueError(
                f"Target token {target_token_id} missing at position {position}"
            )

        ranked = []
        for token_id, info in entry.items():
            rank = _field(info, "rank")
            logprob = float(_field(info, "logprob"))
            ranked.append((rank, -logprob, int(token_id), logprob))
        ranked.sort(key=lambda item: (item[0] is None, item[0] or 10**9, item[1]))
        _, _, top1_token_id, top1_logprob = ranked[0]

        rows.append(
            {
                "position": position,
                "token_id": target_token_id,
                "target_logprob": float(_field(target, "logprob")),
                "target_rank": _field(target, "rank"),
                "top1_token_id": top1_token_id,
                "top1_logprob": top1_logprob,
            }
        )
    return rows


def compare_worker_results(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return per-token and per-context comparisons against FP16."""
    base_points = {int(p["context_tokens"]): p for p in baseline["points"]}
    candidate_points = {int(p["context_tokens"]): p for p in candidate["points"]}
    if set(base_points) != set(candidate_points):
        raise ValueError("Baseline and candidate context lengths differ")

    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for context_tokens in sorted(base_points):
        base_point = base_points[context_tokens]
        cand_point = candidate_points[context_tokens]
        base_window = {int(row["position"]): row for row in base_point["window"]}
        cand_window = {int(row["position"]): row for row in cand_point["window"]}
        positions = sorted(set(base_window) & set(cand_window))
        if not positions:
            raise ValueError(f"No overlapping evaluation window at {context_tokens}")

        abs_errors: list[float] = []
        signed_errors: list[float] = []
        agreements: list[float] = []
        target_rank_deltas: list[float] = []
        for position in positions:
            base_row = base_window[position]
            cand_row = cand_window[position]
            if int(base_row["token_id"]) != int(cand_row["token_id"]):
                raise ValueError(f"Token mismatch at position {position}")
            signed = float(cand_row["target_logprob"]) - float(
                base_row["target_logprob"]
            )
            agreement = float(
                int(cand_row["top1_token_id"]) == int(base_row["top1_token_id"])
            )
            base_rank = base_row.get("target_rank")
            cand_rank = cand_row.get("target_rank")
            rank_delta = (
                abs(float(cand_rank) - float(base_rank))
                if base_rank is not None and cand_rank is not None
                else float("nan")
            )
            raw_rows.append(
                {
                    "method": candidate["method"],
                    "label": candidate["label"],
                    "context_tokens": context_tokens,
                    "position": position,
                    "token_id": int(base_row["token_id"]),
                    "fp16_target_logprob": float(base_row["target_logprob"]),
                    "candidate_target_logprob": float(cand_row["target_logprob"]),
                    "signed_logprob_error": signed,
                    "abs_logprob_error": abs(signed),
                    "fp16_top1_token_id": int(base_row["top1_token_id"]),
                    "candidate_top1_token_id": int(cand_row["top1_token_id"]),
                    "top1_agreement": agreement,
                    "target_rank_delta": rank_delta,
                }
            )
            signed_errors.append(signed)
            abs_errors.append(abs(signed))
            agreements.append(agreement)
            target_rank_deltas.append(rank_delta)

        sorted_abs = sorted(abs_errors)
        p95_index = min(len(sorted_abs) - 1, int(0.95 * len(sorted_abs)))
        base_elapsed = float(base_point["elapsed_seconds"])
        cand_elapsed = float(cand_point["elapsed_seconds"])
        base_tps = float(base_point["prompt_tokens_per_second"])
        cand_tps = float(cand_point["prompt_tokens_per_second"])
        finite_rank_deltas = [x for x in target_rank_deltas if x == x]
        summary_rows.append(
            {
                "method": candidate["method"],
                "label": candidate["label"],
                "kv_cache_dtype": candidate["kv_cache_dtype"],
                "context_tokens": context_tokens,
                "eval_tokens": len(positions),
                "mean_abs_logprob_error": sum(abs_errors) / len(abs_errors),
                "p95_abs_logprob_error": sorted_abs[p95_index],
                "mean_signed_logprob_error": sum(signed_errors) / len(signed_errors),
                "top1_agreement": sum(agreements) / len(agreements),
                "mean_target_rank_delta": (
                    sum(finite_rank_deltas) / len(finite_rank_deltas)
                    if finite_rank_deltas
                    else float("nan")
                ),
                "elapsed_seconds": cand_elapsed,
                "prompt_tokens_per_second": cand_tps,
                "fp16_elapsed_seconds": base_elapsed,
                "fp16_prompt_tokens_per_second": base_tps,
                "throughput_ratio_vs_fp16": cand_tps / base_tps,
            }
        )
    return raw_rows, summary_rows
