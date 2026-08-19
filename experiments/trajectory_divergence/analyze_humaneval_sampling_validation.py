#!/usr/bin/env python3
"""Aggregate HumanEval FP16/KVarN sampling experiments across multiple seeds.

This analysis intentionally avoids token-level FP16-vs-KVarN trajectory alignment:
under stochastic sampling, different sampled tokens are not uniquely attributable
to KV quantization. Instead, the primary unit is a paired request under the same
sampling seed, plus a per-task aggregate across independent seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-dir", type=Path, required=True)
    parser.add_argument("--seeds", required=True, help="Comma-separated integer seeds")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition-label", default="thinking_on")
    parser.add_argument("--severe-tir", type=float, default=0.50)
    return parser.parse_args()


def parse_seeds(text: str) -> list[int]:
    seeds = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique")
    return seeds


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = record.get("sample_id")
            if sample_id is None:
                continue
            previous = records.get(str(sample_id))
            if previous is None or previous.get("error") is not None or record.get("error") is None:
                records[str(sample_id)] = record
    return records


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def as_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def as_int(value: Any) -> int | None:
    number = as_float(value)
    return int(number) if number is not None else None


def ratio_delta(candidate: float | int | None, reference: float | int | None) -> float | None:
    if candidate is None or reference is None or float(reference) == 0.0:
        return None
    return (float(candidate) - float(reference)) / float(reference)


def percentile(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def summarize(values: Iterable[float | int | None]) -> dict[str, Any]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "p90": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "median": statistics.median(clean),
        "p90": percentile(clean, 0.90),
        "p95": percentile(clean, 0.95),
        "min": min(clean),
        "max": max(clean),
    }


def accuracy(values: Iterable[bool | None]) -> dict[str, Any]:
    clean = [value for value in values if isinstance(value, bool)]
    correct = sum(bool(value) for value in clean)
    return {
        "correct": correct,
        "evaluated": len(clean),
        "accuracy": correct / len(clean) if clean else None,
    }


def average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for offset in range(cursor, end):
            ranks[indexed[offset][0]] = rank
        cursor = end
    return ranks


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(x) != len(y):
        return None
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    dx = [value - mean_x for value in x]
    dy = [value - mean_y for value in y]
    denominator = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    if denominator == 0:
        return None
    return sum(left * right for left, right in zip(dx, dy)) / denominator


def spearman(pairs: Iterable[tuple[float | None, float | None]]) -> dict[str, Any]:
    clean = [
        (float(x), float(y))
        for x, y in pairs
        if x is not None
        and y is not None
        and math.isfinite(float(x))
        and math.isfinite(float(y))
    ]
    if len(clean) < 2:
        return {"n": len(clean), "rho": None}
    x = [item[0] for item in clean]
    y = [item[1] for item in clean]
    return {"n": len(clean), "rho": pearson(average_ranks(x), average_ranks(y))}


def is_censored(record: dict[str, Any]) -> bool:
    return str(record.get("finish_reason") or "").lower() == "length"


def build_row(seed: int, fp16: dict[str, Any], kvarn: dict[str, Any]) -> dict[str, Any]:
    fp_tokens = as_int(fp16.get("output_tokens"))
    kv_tokens = as_int(kvarn.get("output_tokens"))
    fp_thinking = as_int(fp16.get("thinking_tokens"))
    kv_thinking = as_int(kvarn.get("thinking_tokens"))
    fp_final = as_int(fp16.get("final_tokens"))
    kv_final = as_int(kvarn.get("final_tokens"))
    fp_latency = as_float(fp16.get("latency_seconds"))
    kv_latency = as_float(kvarn.get("latency_seconds"))
    fp_decode = as_float(fp16.get("decode_time_seconds"))
    kv_decode = as_float(kvarn.get("decode_time_seconds"))
    fp_tpot = as_float(fp16.get("tpot_seconds"))
    kv_tpot = as_float(kvarn.get("tpot_seconds"))
    fp_ttft = as_float(fp16.get("ttft_seconds"))
    kv_ttft = as_float(kvarn.get("ttft_seconds"))

    delta_tokens = (kv_tokens - fp_tokens) if fp_tokens is not None and kv_tokens is not None else None
    delta_thinking = (
        kv_thinking - fp_thinking
        if fp_thinking is not None and kv_thinking is not None
        else None
    )
    delta_final = (
        kv_final - fp_final if fp_final is not None and kv_final is not None else None
    )
    thinking_contribution = None
    if delta_tokens is not None and delta_tokens > 0 and delta_thinking is not None:
        thinking_contribution = delta_thinking / delta_tokens

    fp_pass = as_bool(fp16.get("execution_passed"))
    kv_pass = as_bool(kvarn.get("execution_passed"))
    fp_censored = is_censored(fp16)
    kv_censored = is_censored(kvarn)

    return {
        "seed": seed,
        "sample_id": str(fp16.get("sample_id")),
        "task_id": str(fp16.get("task_id") or fp16.get("sample_id")),
        "fp16_output_tokens": fp_tokens,
        "kvarn_output_tokens": kv_tokens,
        "delta_tokens": delta_tokens,
        "token_inflation_ratio": ratio_delta(kv_tokens, fp_tokens),
        "fp16_censored": fp_censored,
        "kvarn_censored": kv_censored,
        "either_censored": fp_censored or kv_censored,
        "fp16_pass": fp_pass,
        "kvarn_pass": kv_pass,
        "correctness_transition": (
            "unavailable"
            if fp_pass is None or kv_pass is None
            else f"{'pass' if fp_pass else 'fail'}_to_{'pass' if kv_pass else 'fail'}"
        ),
        "fp16_thinking_tokens": fp_thinking,
        "kvarn_thinking_tokens": kv_thinking,
        "delta_thinking_tokens": delta_thinking,
        "fp16_final_tokens": fp_final,
        "kvarn_final_tokens": kv_final,
        "delta_final_tokens": delta_final,
        "thinking_contribution_to_extra_tokens": thinking_contribution,
        "fp16_latency_seconds": fp_latency,
        "kvarn_latency_seconds": kv_latency,
        "latency_inflation_ratio": ratio_delta(kv_latency, fp_latency),
        "fp16_decode_time_seconds": fp_decode,
        "kvarn_decode_time_seconds": kv_decode,
        "decode_latency_inflation_ratio": ratio_delta(kv_decode, fp_decode),
        "fp16_tpot_seconds": fp_tpot,
        "kvarn_tpot_seconds": kv_tpot,
        "tpot_inflation_ratio": ratio_delta(kv_tpot, fp_tpot),
        "fp16_ttft_seconds": fp_ttft,
        "kvarn_ttft_seconds": kv_ttft,
        "ttft_inflation_ratio": ratio_delta(kv_ttft, fp_ttft),
        "fp16_tokens_per_second": as_float(fp16.get("tokens_per_second")),
        "kvarn_tokens_per_second": as_float(kvarn.get("tokens_per_second")),
    }


def tail_fractions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        float(row["token_inflation_ratio"])
        for row in rows
        if not row["either_censored"] and row["token_inflation_ratio"] is not None
    ]
    if not values:
        return {}
    n = len(values)
    return {
        "tir_gt_0_10": sum(value > 0.10 for value in values) / n,
        "tir_gt_0_25": sum(value > 0.25 for value in values) / n,
        "tir_gt_0_50": sum(value > 0.50 for value in values) / n,
        "tir_gt_1_00": sum(value > 1.00 for value in values) / n,
        "tir_lt_minus_0_10": sum(value < -0.10 for value in values) / n,
        "tir_between_minus_0_10_and_0_10": sum(-0.10 <= value <= 0.10 for value in values) / n,
    }


def severe_thinking_summary(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    severe = [
        row
        for row in rows
        if not row["either_censored"]
        and row["token_inflation_ratio"] is not None
        and row["token_inflation_ratio"] > threshold
    ]
    available = [
        row
        for row in severe
        if row["delta_tokens"] is not None
        and row["delta_tokens"] > 0
        and row["delta_thinking_tokens"] is not None
        and row["thinking_contribution_to_extra_tokens"] is not None
    ]
    total_extra = sum(int(row["delta_tokens"]) for row in available)
    total_thinking = sum(int(row["delta_thinking_tokens"]) for row in available)
    total_final = sum(
        int(row["delta_final_tokens"])
        for row in available
        if row["delta_final_tokens"] is not None
    )
    contribution = total_thinking / total_extra if total_extra > 0 else None
    pass_pass = [row for row in available if row["correctness_transition"] == "pass_to_pass"]
    pp_total_extra = sum(int(row["delta_tokens"]) for row in pass_pass)
    pp_total_thinking = sum(int(row["delta_thinking_tokens"]) for row in pass_pass)
    return {
        "threshold": threshold,
        "severe_pair_count": len(severe),
        "thinking_split_available_count": len(available),
        "total_extra_tokens": total_extra,
        "total_extra_thinking_tokens": total_thinking,
        "total_extra_final_tokens": total_final,
        "aggregate_thinking_contribution": contribution,
        "per_pair_thinking_contribution": summarize(
            row["thinking_contribution_to_extra_tokens"] for row in available
        ),
        "pass_to_pass_pair_count": len(pass_pass),
        "pass_to_pass_aggregate_thinking_contribution": (
            pp_total_thinking / pp_total_extra if pp_total_extra > 0 else None
        ),
    }


def summarize_seed(seed: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    uncensored = [row for row in rows if not row["either_censored"]]
    fp_tokens_all = [row["fp16_output_tokens"] for row in rows]
    kv_tokens_all = [row["kvarn_output_tokens"] for row in rows]
    fp_mean = statistics.fmean(float(value) for value in fp_tokens_all if value is not None)
    kv_mean = statistics.fmean(float(value) for value in kv_tokens_all if value is not None)
    return {
        "seed": seed,
        "valid_pairs": len(rows),
        "uncensored_pairs": len(uncensored),
        "fp16_censored": sum(row["fp16_censored"] for row in rows),
        "kvarn_censored": sum(row["kvarn_censored"] for row in rows),
        "fp16_pass_at_1": accuracy(row["fp16_pass"] for row in rows)["accuracy"],
        "kvarn_pass_at_1": accuracy(row["kvarn_pass"] for row in rows)["accuracy"],
        "fp16_mean_output_tokens_all": fp_mean,
        "kvarn_mean_output_tokens_all": kv_mean,
        "paper_style_output_token_change": ratio_delta(kv_mean, fp_mean),
        "tir_mean_uncensored": summarize(row["token_inflation_ratio"] for row in uncensored)["mean"],
        "tir_median_uncensored": summarize(row["token_inflation_ratio"] for row in uncensored)["median"],
        "tir_p95_uncensored": summarize(row["token_inflation_ratio"] for row in uncensored)["p95"],
        "decode_lir_mean_uncensored": summarize(
            row["decode_latency_inflation_ratio"] for row in uncensored
        )["mean"],
        "decode_lir_p95_uncensored": summarize(
            row["decode_latency_inflation_ratio"] for row in uncensored
        )["p95"],
        "tpot_lir_mean_uncensored": summarize(row["tpot_inflation_ratio"] for row in uncensored)["mean"],
        **tail_fractions(rows),
    }


def build_task_rows(rows: list[dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_id"])].append(row)

    result: list[dict[str, Any]] = []
    for task_id, task_rows in sorted(grouped.items()):
        task_rows.sort(key=lambda row: int(row["seed"]))
        uncensored = [row for row in task_rows if not row["either_censored"]]
        tir = [
            float(row["token_inflation_ratio"])
            for row in uncensored
            if row["token_inflation_ratio"] is not None
        ]
        fp_all = [float(row["fp16_output_tokens"]) for row in task_rows if row["fp16_output_tokens"] is not None]
        kv_all = [float(row["kvarn_output_tokens"]) for row in task_rows if row["kvarn_output_tokens"] is not None]
        fp_mean = statistics.fmean(fp_all) if fp_all else None
        kv_mean = statistics.fmean(kv_all) if kv_all else None
        result.append(
            {
                "task_id": task_id,
                "valid_seed_pairs": len(task_rows),
                "uncensored_seed_pairs": len(uncensored),
                "all_requested_seeds_uncensored": len(uncensored) == len(seeds),
                "fp16_mean_output_tokens_across_seeds": fp_mean,
                "kvarn_mean_output_tokens_across_seeds": kv_mean,
                "mean_output_token_change_across_seeds": ratio_delta(kv_mean, fp_mean),
                "median_paired_tir": statistics.median(tir) if tir else None,
                "kvarn_longer_seed_count": sum(value > 0 for value in tir),
                "tir_gt_10_seed_count": sum(value > 0.10 for value in tir),
                "tir_gt_25_seed_count": sum(value > 0.25 for value in tir),
                "tir_gt_50_seed_count": sum(value > 0.50 for value in tir),
                "tir_lt_minus_10_seed_count": sum(value < -0.10 for value in tir),
                "fp16_pass_rate_across_seeds": accuracy(row["fp16_pass"] for row in task_rows)["accuracy"],
                "kvarn_pass_rate_across_seeds": accuracy(row["kvarn_pass"] for row in task_rows)["accuracy"],
                "mean_decode_lir_uncensored": summarize(
                    row["decode_latency_inflation_ratio"] for row in uncensored
                )["mean"],
            }
        )
    return result


def robust_task_summary(task_rows: list[dict[str, Any]], seed_count: int) -> dict[str, Any]:
    full = [row for row in task_rows if row["all_requested_seeds_uncensored"]]
    majority = seed_count // 2 + 1
    return {
        "all_seeds_uncensored_task_count": len(full),
        "majority_seed_threshold": majority,
        "tasks_longer_gt_10_in_majority_seeds": sum(
            int(row["tir_gt_10_seed_count"]) >= majority for row in full
        ),
        "tasks_longer_gt_25_in_majority_seeds": sum(
            int(row["tir_gt_25_seed_count"]) >= majority for row in full
        ),
        "tasks_longer_gt_50_in_majority_seeds": sum(
            int(row["tir_gt_50_seed_count"]) >= majority for row in full
        ),
        "tasks_shorter_gt_10_in_majority_seeds": sum(
            int(row["tir_lt_minus_10_seed_count"]) >= majority for row in full
        ),
        "tasks_longer_gt_10_in_all_seeds": sum(
            int(row["tir_gt_10_seed_count"]) == seed_count for row in full
        ),
        "tasks_longer_gt_50_in_all_seeds": sum(
            int(row["tir_gt_50_seed_count"]) == seed_count for row in full
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def fmt_pct(value: Any) -> str:
    number = as_float(value)
    return "N/A" if number is None else f"{100.0 * number:.2f}%"


def fmt_num(value: Any, digits: int = 1) -> str:
    number = as_float(value)
    return "N/A" if number is None else f"{number:.{digits}f}"


def write_summary_md(path: Path, summary: dict[str, Any], per_seed: list[dict[str, Any]]) -> None:
    paper = summary["paper_style_avg3"]
    pooled = summary["pooled_uncensored"]
    severe = summary["severe_inflation_thinking"]
    robust = summary["robust_across_seeds"]

    lines = [
        f"# HumanEval sampling validation — {summary['condition_label']}",
        "",
        "Primary interpretation: request-level system effects under Qwen3-recommended stochastic sampling. ",
        "Token-level first-divergence is intentionally not used in this arm because sampling is stochastic.",
        "",
        "## Paper-style run aggregate",
        "",
        "| Metric | FP16 | KVarN K4V2 | Change |",
        "|---|---:|---:|---:|",
        (
            f"| Mean output tokens across runs | {fmt_num(paper['fp16_output_tokens']['mean'])} ± "
            f"{fmt_num(paper['fp16_output_tokens']['std'])} | "
            f"{fmt_num(paper['kvarn_output_tokens']['mean'])} ± {fmt_num(paper['kvarn_output_tokens']['std'])} | "
            f"{fmt_pct(paper['output_token_change'])} |"
        ),
        (
            f"| Mean pass@1 across runs | {fmt_pct(paper['fp16_pass_at_1']['mean'])} ± "
            f"{fmt_pct(paper['fp16_pass_at_1']['std'])} | "
            f"{fmt_pct(paper['kvarn_pass_at_1']['mean'])} ± {fmt_pct(paper['kvarn_pass_at_1']['std'])} | "
            f"{fmt_pct(paper['pass_at_1_change_points'])} points |"
        ),
        "",
        "## Pooled paired request distribution (uncensored only)",
        "",
        "| Metric | Median | Mean | P90 | P95 |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Output-token change | {fmt_pct(pooled['tir']['median'])} | {fmt_pct(pooled['tir']['mean'])} | "
            f"{fmt_pct(pooled['tir']['p90'])} | {fmt_pct(pooled['tir']['p95'])} |"
        ),
        (
            f"| E2E latency change | {fmt_pct(pooled['e2e_lir']['median'])} | {fmt_pct(pooled['e2e_lir']['mean'])} | "
            f"{fmt_pct(pooled['e2e_lir']['p90'])} | {fmt_pct(pooled['e2e_lir']['p95'])} |"
        ),
        (
            f"| Decode latency change | {fmt_pct(pooled['decode_lir']['median'])} | {fmt_pct(pooled['decode_lir']['mean'])} | "
            f"{fmt_pct(pooled['decode_lir']['p90'])} | {fmt_pct(pooled['decode_lir']['p95'])} |"
        ),
        (
            f"| TPOT change | {fmt_pct(pooled['tpot_lir']['median'])} | {fmt_pct(pooled['tpot_lir']['mean'])} | "
            f"{fmt_pct(pooled['tpot_lir']['p90'])} | {fmt_pct(pooled['tpot_lir']['p95'])} |"
        ),
        "",
        "### Tail rates",
        "",
        "| Request category | Fraction |",
        "|---|---:|",
        f"| Output >10% longer | {fmt_pct(pooled['tail_fractions'].get('tir_gt_0_10'))} |",
        f"| Output >25% longer | {fmt_pct(pooled['tail_fractions'].get('tir_gt_0_25'))} |",
        f"| Output >50% longer | {fmt_pct(pooled['tail_fractions'].get('tir_gt_0_50'))} |",
        f"| Output >100% longer | {fmt_pct(pooled['tail_fractions'].get('tir_gt_1_00'))} |",
        f"| Output >10% shorter | {fmt_pct(pooled['tail_fractions'].get('tir_lt_minus_0_10'))} |",
        f"| Output within ±10% | {fmt_pct(pooled['tail_fractions'].get('tir_between_minus_0_10_and_0_10'))} |",
        "",
        "## Severe-inflation token source",
        "",
        f"Severe means TIR > {100 * severe['threshold']:.0f}%.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Severe paired requests | {severe['severe_pair_count']} |",
        f"| Thinking split available | {severe['thinking_split_available_count']} |",
        f"| Aggregate extra tokens | {severe['total_extra_tokens']} |",
        f"| Extra thinking tokens | {severe['total_extra_thinking_tokens']} |",
        f"| Extra final/code tokens | {severe['total_extra_final_tokens']} |",
        f"| Thinking share of net extra tokens | {fmt_pct(severe['aggregate_thinking_contribution'])} |",
        f"| pass→pass thinking share | {fmt_pct(severe['pass_to_pass_aggregate_thinking_contribution'])} |",
        "",
        "## Cross-seed robustness by task",
        "",
        f"Only tasks uncensored in all {summary['seed_count']} seeds are counted below.",
        "",
        "| Stable task-level pattern | Tasks |",
        "|---|---:|",
        f"| All seeds uncensored | {robust['all_seeds_uncensored_task_count']} |",
        f"| >10% longer in majority of seeds | {robust['tasks_longer_gt_10_in_majority_seeds']} |",
        f"| >25% longer in majority of seeds | {robust['tasks_longer_gt_25_in_majority_seeds']} |",
        f"| >50% longer in majority of seeds | {robust['tasks_longer_gt_50_in_majority_seeds']} |",
        f"| >10% shorter in majority of seeds | {robust['tasks_shorter_gt_10_in_majority_seeds']} |",
        f"| >10% longer in every seed | {robust['tasks_longer_gt_10_in_all_seeds']} |",
        f"| >50% longer in every seed | {robust['tasks_longer_gt_50_in_all_seeds']} |",
        "",
        "## Per-seed sanity table",
        "",
        "| Seed | FP16 tokens | KVarN tokens | Δ tokens | FP16 pass@1 | KVarN pass@1 | Uncensored | TIR P95 | Decode LIR P95 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_seed:
        lines.append(
            f"| {row['seed']} | {fmt_num(row['fp16_mean_output_tokens_all'])} | "
            f"{fmt_num(row['kvarn_mean_output_tokens_all'])} | {fmt_pct(row['paper_style_output_token_change'])} | "
            f"{fmt_pct(row['fp16_pass_at_1'])} | {fmt_pct(row['kvarn_pass_at_1'])} | "
            f"{row['uncensored_pairs']} | {fmt_pct(row['tir_p95_uncensored'])} | "
            f"{fmt_pct(row['decode_lir_p95_uncensored'])} |"
        )
    lines.extend(
        [
            "",
            "## Correlation diagnostic",
            "",
            f"- TIR vs E2E LIR: n={pooled['tir_vs_e2e_lir']['n']}, rho={fmt_num(pooled['tir_vs_e2e_lir']['rho'], 4)}",
            f"- TIR vs decode LIR: n={pooled['tir_vs_decode_lir']['n']}, rho={fmt_num(pooled['tir_vs_decode_lir']['rho'], 4)}",
            "",
            "The per-seed and per-task CSV files are the preferred source for follow-up analysis.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    per_seed: list[dict[str, Any]] = []
    missing: list[str] = []

    for seed in seeds:
        seed_dir = args.condition_dir / f"seed_{seed}"
        fp_path = seed_dir / "fp16" / "generations.jsonl"
        kv_path = seed_dir / "kvarn" / "generations.jsonl"
        if not fp_path.exists() or not kv_path.exists():
            missing.append(str(seed_dir))
            continue
        fp16 = load_jsonl(fp_path)
        kvarn = load_jsonl(kv_path)
        common = sorted(set(fp16).intersection(kvarn))
        rows = [
            build_row(seed, fp16[sample_id], kvarn[sample_id])
            for sample_id in common
            if fp16[sample_id].get("error") is None and kvarn[sample_id].get("error") is None
        ]
        rows.sort(key=lambda row: str(row["task_id"]))
        all_rows.extend(rows)
        per_seed.append(summarize_seed(seed, rows))

    if missing:
        raise FileNotFoundError("Missing seed results: " + ", ".join(missing))
    if len(per_seed) != len(seeds):
        raise RuntimeError("Not all requested seeds were analyzed")

    uncensored = [row for row in all_rows if not row["either_censored"]]
    task_rows = build_task_rows(all_rows, seeds)

    fp_run_tokens = [float(row["fp16_mean_output_tokens_all"]) for row in per_seed]
    kv_run_tokens = [float(row["kvarn_mean_output_tokens_all"]) for row in per_seed]
    fp_run_pass = [float(row["fp16_pass_at_1"]) for row in per_seed]
    kv_run_pass = [float(row["kvarn_pass_at_1"]) for row in per_seed]
    fp_tokens_summary = summarize(fp_run_tokens)
    kv_tokens_summary = summarize(kv_run_tokens)
    fp_pass_summary = summarize(fp_run_pass)
    kv_pass_summary = summarize(kv_run_pass)

    pooled_tail = tail_fractions(all_rows)
    summary = {
        "schema_version": 1,
        "condition_label": args.condition_label,
        "seeds": seeds,
        "seed_count": len(seeds),
        "valid_request_seed_pairs": len(all_rows),
        "uncensored_request_seed_pairs": len(uncensored),
        "paper_style_avg3": {
            "fp16_output_tokens": fp_tokens_summary,
            "kvarn_output_tokens": kv_tokens_summary,
            "output_token_change": ratio_delta(
                kv_tokens_summary["mean"], fp_tokens_summary["mean"]
            ),
            "fp16_pass_at_1": fp_pass_summary,
            "kvarn_pass_at_1": kv_pass_summary,
            "pass_at_1_change_points": (
                kv_pass_summary["mean"] - fp_pass_summary["mean"]
                if fp_pass_summary["mean"] is not None and kv_pass_summary["mean"] is not None
                else None
            ),
            "note": "Mode means are computed within each seed over all valid requests, then averaged across seeds. Capped outputs are included, matching benchmark-style mean-token reporting more closely than the uncensored paired distribution.",
        },
        "pooled_uncensored": {
            "pair_count": len(uncensored),
            "tir": summarize(row["token_inflation_ratio"] for row in uncensored),
            "e2e_lir": summarize(row["latency_inflation_ratio"] for row in uncensored),
            "decode_lir": summarize(row["decode_latency_inflation_ratio"] for row in uncensored),
            "tpot_lir": summarize(row["tpot_inflation_ratio"] for row in uncensored),
            "ttft_lir": summarize(row["ttft_inflation_ratio"] for row in uncensored),
            "tail_fractions": pooled_tail,
            "tir_vs_e2e_lir": spearman(
                (row["token_inflation_ratio"], row["latency_inflation_ratio"])
                for row in uncensored
            ),
            "tir_vs_decode_lir": spearman(
                (row["token_inflation_ratio"], row["decode_latency_inflation_ratio"])
                for row in uncensored
            ),
        },
        "quality": {
            "fp16": accuracy(row["fp16_pass"] for row in all_rows),
            "kvarn": accuracy(row["kvarn_pass"] for row in all_rows),
            "pass_to_pass_pairs": sum(
                row["correctness_transition"] == "pass_to_pass" for row in all_rows
            ),
            "pass_to_fail_pairs": sum(
                row["correctness_transition"] == "pass_to_fail" for row in all_rows
            ),
            "fail_to_pass_pairs": sum(
                row["correctness_transition"] == "fail_to_pass" for row in all_rows
            ),
        },
        "censoring": {
            "fp16_pairs": sum(row["fp16_censored"] for row in all_rows),
            "kvarn_pairs": sum(row["kvarn_censored"] for row in all_rows),
            "either_pairs": sum(row["either_censored"] for row in all_rows),
        },
        "severe_inflation_thinking": severe_thinking_summary(all_rows, args.severe_tir),
        "robust_across_seeds": robust_task_summary(task_rows, len(seeds)),
        "per_seed": per_seed,
    }

    write_csv(args.output_dir / "per_seed_request_metrics.csv", all_rows)
    write_jsonl(args.output_dir / "per_seed_request_metrics.jsonl", all_rows)
    write_csv(args.output_dir / "per_task_across_seeds.csv", task_rows)
    write_jsonl(args.output_dir / "per_task_across_seeds.jsonl", task_rows)
    write_csv(args.output_dir / "per_seed_summary.csv", per_seed)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    write_summary_md(args.output_dir / "summary.md", summary, per_seed)


if __name__ == "__main__":
    main()
