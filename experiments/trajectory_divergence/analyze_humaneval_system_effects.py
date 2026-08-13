#!/usr/bin/env python3
"""Analyze HumanEval trajectory divergence, generation inflation, and latency."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join HumanEval free-generation outputs with trajectory comparisons."
    )
    parser.add_argument("--fp16", type=Path, required=True)
    parser.add_argument("--kvarn", type=Path, required=True)
    parser.add_argument("--comparison-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    return parser.parse_args()


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


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def parse_int(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(float(value))


def parse_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def load_comparison(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = str(row["sample_id"])
            rows[sample_id] = {
                **row,
                "diverged": bool(parse_bool(row.get("diverged"))),
                "first_divergence_step": parse_int(row.get("first_divergence_step")),
                "fp16_censored": bool(parse_bool(row.get("fp16_censored"))),
                "kvarn_censored": bool(parse_bool(row.get("kvarn_censored"))),
                "either_censored": bool(parse_bool(row.get("either_censored"))),
            }
    return rows


def ratio_delta(candidate: float | int | None, reference: float | int | None) -> float | None:
    if candidate is None or reference is None or float(reference) == 0.0:
        return None
    return (float(candidate) - float(reference)) / float(reference)


def accuracy(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [row[field] for row in records if isinstance(row.get(field), bool)]
    return {
        "correct": sum(bool(value) for value in values),
        "evaluated": len(values),
        "accuracy": sum(bool(value) for value in values) / len(values) if values else None,
    }


def percentile(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(float(value) for value in values)
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


def summarize_values(values: Iterable[float | int | None]) -> dict[str, Any]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "min": None, "max": None}
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "p90": percentile(clean, 0.90),
        "p95": percentile(clean, 0.95),
        "min": min(clean),
        "max": max(clean),
    }


def average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average = (cursor + 1 + end) / 2.0
        for offset in range(cursor, end):
            ranks[indexed[offset][0]] = average
        cursor = end
    return ranks


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(x) != len(y):
        return None
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    dx = [value - mean_x for value in x]
    dy = [value - mean_y for value in y]
    denom = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    if denom == 0:
        return None
    return sum(left * right for left, right in zip(dx, dy)) / denom


def spearman(pairs: Iterable[tuple[float | None, float | None]]) -> dict[str, Any]:
    clean = [(float(x), float(y)) for x, y in pairs if x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(clean) < 2:
        return {"n": len(clean), "rho": None}
    x = [item[0] for item in clean]
    y = [item[1] for item in clean]
    return {"n": len(clean), "rho": pearson(average_ranks(x), average_ranks(y))}


def bootstrap_ci(
    values: list[float],
    statistic: Callable[[list[float]], float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if not values:
        return {"estimate": None, "low": None, "high": None, "samples": 0}
    estimate = statistic(values)
    if len(values) == 1 or samples <= 0:
        return {"estimate": estimate, "low": estimate, "high": estimate, "samples": 0}
    rng = random.Random(seed)
    boot: list[float] = []
    n = len(values)
    for _ in range(samples):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        boot.append(statistic(draw))
    return {
        "estimate": estimate,
        "low": percentile(boot, 0.025),
        "high": percentile(boot, 0.975),
        "samples": samples,
    }


def divergence_group(diverged: bool, normalized_position: float | None) -> str:
    if not diverged:
        return "no_divergence"
    if normalized_position is None:
        return "diverged_unclassified"
    if normalized_position < 0.25:
        return "early_[0,0.25)"
    if normalized_position < 0.50:
        return "middle_[0.25,0.5)"
    return "late_[0.5,1+]"


def correctness_transition(fp16: bool | None, kvarn: bool | None) -> str:
    if fp16 is None or kvarn is None:
        return "unavailable"
    return f"{'pass' if fp16 else 'fail'}_to_{'pass' if kvarn else 'fail'}"


def build_row(sample_id: str, fp16: dict[str, Any], kvarn: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    fp_tokens = int(fp16.get("output_tokens") or 0)
    kv_tokens = int(kvarn.get("output_tokens") or 0)
    first_step = comparison.get("first_divergence_step")
    normalized_first = (float(first_step) / max(1, fp_tokens)) if first_step is not None else None
    fp_pass = parse_bool(fp16.get("execution_passed"))
    kv_pass = parse_bool(kvarn.get("execution_passed"))

    fp_latency = parse_float(fp16.get("latency_seconds"))
    kv_latency = parse_float(kvarn.get("latency_seconds"))
    fp_decode = parse_float(fp16.get("decode_time_seconds"))
    kv_decode = parse_float(kvarn.get("decode_time_seconds"))
    fp_ttft = parse_float(fp16.get("ttft_seconds"))
    kv_ttft = parse_float(kvarn.get("ttft_seconds"))
    fp_tpot = parse_float(fp16.get("tpot_seconds"))
    kv_tpot = parse_float(kvarn.get("tpot_seconds"))
    fp_thinking = parse_int(fp16.get("thinking_tokens"))
    kv_thinking = parse_int(kvarn.get("thinking_tokens"))
    fp_final = parse_int(fp16.get("final_tokens"))
    kv_final = parse_int(kvarn.get("final_tokens"))

    return {
        "sample_id": sample_id,
        "task_id": fp16.get("task_id") or sample_id,
        "diverged": bool(comparison.get("diverged")),
        "first_divergence_step": first_step,
        "normalized_first_divergence": normalized_first,
        "divergence_group": divergence_group(bool(comparison.get("diverged")), normalized_first),
        "lcp_ratio": parse_float(comparison.get("lcp_ratio")),
        "fp16_output_tokens": fp_tokens,
        "kvarn_output_tokens": kv_tokens,
        "delta_tokens": kv_tokens - fp_tokens,
        "token_inflation_ratio": ratio_delta(kv_tokens, fp_tokens),
        "abs_token_change_ratio": abs(kv_tokens - fp_tokens) / max(1, fp_tokens),
        "fp16_censored": bool(comparison.get("fp16_censored")),
        "kvarn_censored": bool(comparison.get("kvarn_censored")),
        "either_censored": bool(comparison.get("either_censored")),
        "fp16_finish_reason": fp16.get("finish_reason"),
        "kvarn_finish_reason": kvarn.get("finish_reason"),
        "fp16_pass": fp_pass,
        "kvarn_pass": kv_pass,
        "correctness_transition": correctness_transition(fp_pass, kv_pass),
        "fp16_execution_result": fp16.get("execution_result"),
        "kvarn_execution_result": kvarn.get("execution_result"),
        "same_candidate_code": fp16.get("candidate_code_sha256") == kvarn.get("candidate_code_sha256"),
        "fp16_thinking_tokens": fp_thinking,
        "kvarn_thinking_tokens": kv_thinking,
        "delta_thinking_tokens": (kv_thinking - fp_thinking) if fp_thinking is not None and kv_thinking is not None else None,
        "reasoning_inflation_ratio": ratio_delta(kv_thinking, fp_thinking),
        "fp16_final_tokens": fp_final,
        "kvarn_final_tokens": kv_final,
        "delta_final_tokens": (kv_final - fp_final) if fp_final is not None and kv_final is not None else None,
        "final_inflation_ratio": ratio_delta(kv_final, fp_final),
        "fp16_latency_seconds": fp_latency,
        "kvarn_latency_seconds": kv_latency,
        "delta_latency_seconds": (kv_latency - fp_latency) if fp_latency is not None and kv_latency is not None else None,
        "latency_inflation_ratio": ratio_delta(kv_latency, fp_latency),
        "fp16_decode_time_seconds": fp_decode,
        "kvarn_decode_time_seconds": kv_decode,
        "delta_decode_time_seconds": (kv_decode - fp_decode) if fp_decode is not None and kv_decode is not None else None,
        "decode_latency_inflation_ratio": ratio_delta(kv_decode, fp_decode),
        "fp16_ttft_seconds": fp_ttft,
        "kvarn_ttft_seconds": kv_ttft,
        "ttft_inflation_ratio": ratio_delta(kv_ttft, fp_ttft),
        "fp16_tpot_seconds": fp_tpot,
        "kvarn_tpot_seconds": kv_tpot,
        "tpot_inflation_ratio": ratio_delta(kv_tpot, fp_tpot),
        "fp16_tokens_per_second": parse_float(fp16.get("tokens_per_second")),
        "kvarn_tokens_per_second": parse_float(kvarn.get("tokens_per_second")),
    }


def subset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    uncensored = [row for row in rows if not row["either_censored"]]
    tir = [row["token_inflation_ratio"] for row in uncensored if row["token_inflation_ratio"] is not None]
    return {
        "sample_count": len(rows),
        "uncensored_count": len(uncensored),
        "token_inflation_ratio": summarize_values(tir),
        "latency_inflation_ratio": summarize_values(row["latency_inflation_ratio"] for row in uncensored),
        "decode_latency_inflation_ratio": summarize_values(row["decode_latency_inflation_ratio"] for row in uncensored),
        "pass_rate_fp16": accuracy(rows, "fp16_pass"),
        "pass_rate_kvarn": accuracy(rows, "kvarn_pass"),
    }


def main() -> None:
    args = parse_args()
    fp16 = load_jsonl(args.fp16)
    kvarn = load_jsonl(args.kvarn)
    comparison = load_comparison(args.comparison_csv)
    common = sorted(set(fp16).intersection(kvarn).intersection(comparison))
    rows = [
        build_row(sample_id, fp16[sample_id], kvarn[sample_id], comparison[sample_id])
        for sample_id in common
        if fp16[sample_id].get("error") is None and kvarn[sample_id].get("error") is None
    ]
    rows.sort(key=lambda row: str(row["task_id"]))
    uncensored = [row for row in rows if not row["either_censored"]]
    pass_pass = [row for row in uncensored if row["correctness_transition"] == "pass_to_pass"]

    tir_values = [float(row["token_inflation_ratio"]) for row in uncensored if row["token_inflation_ratio"] is not None]
    delta_tokens = [float(row["delta_tokens"]) for row in uncensored]
    lir_values = [float(row["latency_inflation_ratio"]) for row in uncensored if row["latency_inflation_ratio"] is not None]
    dlir_values = [float(row["decode_latency_inflation_ratio"]) for row in uncensored if row["decode_latency_inflation_ratio"] is not None]

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row["divergence_group"]].append(row)

    summary = {
        "schema_version": 1,
        "valid_pairs": len(rows),
        "uncensored_pairs": len(uncensored),
        "censored_pairs": len(rows) - len(uncensored),
        "diverged_count": sum(row["diverged"] for row in rows),
        "divergence_rate": sum(row["diverged"] for row in rows) / len(rows) if rows else None,
        "fp16_pass_at_1": accuracy(rows, "fp16_pass"),
        "kvarn_pass_at_1": accuracy(rows, "kvarn_pass"),
        "correctness_transition_counts": dict(Counter(row["correctness_transition"] for row in rows)),
        "pass_pass_uncensored_count": len(pass_pass),
        "fp16_output_tokens": summarize_values(row["fp16_output_tokens"] for row in uncensored),
        "kvarn_output_tokens": summarize_values(row["kvarn_output_tokens"] for row in uncensored),
        "delta_tokens": summarize_values(delta_tokens),
        "token_inflation_ratio": summarize_values(tir_values),
        "latency_inflation_ratio": summarize_values(lir_values),
        "decode_latency_inflation_ratio": summarize_values(dlir_values),
        "reasoning_inflation_ratio": summarize_values(row["reasoning_inflation_ratio"] for row in uncensored),
        "final_inflation_ratio": summarize_values(row["final_inflation_ratio"] for row in uncensored),
        "tail_fractions": {
            "tir_gt_0_10": sum(value > 0.10 for value in tir_values) / len(tir_values) if tir_values else None,
            "tir_gt_0_25": sum(value > 0.25 for value in tir_values) / len(tir_values) if tir_values else None,
            "tir_gt_0_50": sum(value > 0.50 for value in tir_values) / len(tir_values) if tir_values else None,
            "tir_gt_1_00": sum(value > 1.00 for value in tir_values) / len(tir_values) if tir_values else None,
            "tir_lt_minus_0_10": sum(value < -0.10 for value in tir_values) / len(tir_values) if tir_values else None,
        },
        "correlations": {
            "normalized_first_divergence_vs_tir": spearman(
                (row["normalized_first_divergence"], row["token_inflation_ratio"])
                for row in uncensored
                if row["diverged"]
            ),
            "tir_vs_e2e_lir": spearman(
                (row["token_inflation_ratio"], row["latency_inflation_ratio"])
                for row in uncensored
            ),
            "tir_vs_decode_lir": spearman(
                (row["token_inflation_ratio"], row["decode_latency_inflation_ratio"])
                for row in uncensored
            ),
        },
        "bootstrap_95ci": {
            "mean_delta_tokens": bootstrap_ci(
                delta_tokens,
                statistics.fmean,
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed,
            ),
            "mean_tir": bootstrap_ci(
                tir_values,
                statistics.fmean,
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed + 1,
            ),
            "mean_e2e_lir": bootstrap_ci(
                lir_values,
                statistics.fmean,
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed + 2,
            ),
            "mean_decode_lir": bootstrap_ci(
                dlir_values,
                statistics.fmean,
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed + 3,
            ),
        },
        "pass_pass_uncensored": subset_summary(pass_pass),
        "by_divergence_group": {
            key: subset_summary(group_rows) for key, group_rows in sorted(by_group.items())
        },
        "timing_coverage": {
            "e2e_pairs": sum(row["latency_inflation_ratio"] is not None for row in rows),
            "decode_pairs": sum(row["decode_latency_inflation_ratio"] is not None for row in rows),
            "ttft_pairs": sum(row["ttft_inflation_ratio"] is not None for row in rows),
            "tpot_pairs": sum(row["tpot_inflation_ratio"] is not None for row in rows),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "per_sample_system_effects.csv"
    jsonl_path = args.output_dir / "per_sample_system_effects.jsonl"
    fieldnames = list(rows[0].keys()) if rows else ["sample_id"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    def fmt(value: Any, digits: int = 4) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    tir = summary["token_inflation_ratio"]
    e2e = summary["latency_inflation_ratio"]
    decode = summary["decode_latency_inflation_ratio"]
    lines = [
        "# HumanEval system-consequence summary",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Valid paired requests | {summary['valid_pairs']} |",
        f"| Uncensored pairs | {summary['uncensored_pairs']} |",
        f"| Divergence rate | {fmt(summary['divergence_rate'])} |",
        f"| FP16 pass@1 | {fmt(summary['fp16_pass_at_1']['accuracy'])} |",
        f"| KVarN pass@1 | {fmt(summary['kvarn_pass_at_1']['accuracy'])} |",
        f"| Mean TIR | {fmt(tir['mean'])} |",
        f"| Median TIR | {fmt(tir['median'])} |",
        f"| P95 TIR | {fmt(tir['p95'])} |",
        f"| Fraction TIR > 10% | {fmt(summary['tail_fractions']['tir_gt_0_10'])} |",
        f"| Fraction TIR > 50% | {fmt(summary['tail_fractions']['tir_gt_0_50'])} |",
        f"| Mean E2E LIR | {fmt(e2e['mean'])} |",
        f"| Mean decode LIR | {fmt(decode['mean'])} |",
        f"| pass→pass uncensored pairs | {summary['pass_pass_uncensored_count']} |",
        "",
        "## Correlations (Spearman)",
        "",
        "| Relation | n | rho |",
        "|---|---:|---:|",
    ]
    for label, key in (
        ("Normalized first divergence vs TIR", "normalized_first_divergence_vs_tir"),
        ("TIR vs E2E LIR", "tir_vs_e2e_lir"),
        ("TIR vs decode LIR", "tir_vs_decode_lir"),
    ):
        value = summary["correlations"][key]
        lines.append(f"| {label} | {value['n']} | {fmt(value['rho'])} |")
    lines.extend(["", "## By divergence group", "", "| Group | Samples | Uncensored | Mean TIR | Mean decode LIR |", "|---|---:|---:|---:|---:|"])
    for group, value in summary["by_divergence_group"].items():
        lines.append(
            f"| {group} | {value['sample_count']} | {value['uncensored_count']} | "
            f"{fmt(value['token_inflation_ratio']['mean'])} | "
            f"{fmt(value['decode_latency_inflation_ratio']['mean'])} |"
        )
    lines.append("")
    (args.output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
