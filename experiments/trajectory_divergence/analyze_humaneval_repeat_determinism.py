#!/usr/bin/env python3
"""Analyze repeat determinism for two HumanEval generation runs.

This is intentionally mode-agnostic, but the isolation runner uses it for two
KVarN greedy repeats under the same configuration.  It reports the exact first
token at which the repeats disagree and keeps censoring/termination metadata so
small numerical nondeterminism can be separated from later autoregressive
amplification.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two repeated HumanEval runs.")
    parser.add_argument("--run1", type=Path, required=True)
    parser.add_argument("--run2", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition-label", required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row.get("sample_id")
            if sample_id is not None:
                rows[str(sample_id)] = row
    return rows


def first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def median_or_none(values: list[int]) -> float | None:
    return float(statistics.median(values)) if values else None


def mean_or_none(values: list[int | float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def row_for(sample_id: str, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_ids = [int(v) for v in left.get("output_token_ids", [])]
    right_ids = [int(v) for v in right.get("output_token_ids", [])]
    first = first_difference(left_ids, right_ids)
    common_prefix = first if first is not None else min(len(left_ids), len(right_ids))
    min_len = min(len(left_ids), len(right_ids))
    left_finish = str(left.get("finish_reason") or "")
    right_finish = str(right.get("finish_reason") or "")
    left_error = left.get("error")
    right_error = right.get("error")

    return {
        "sample_id": sample_id,
        "valid_pair": left_error is None and right_error is None,
        "diverged": first is not None,
        "first_divergence_step": first,
        "common_prefix_tokens": common_prefix,
        "lcp_ratio_vs_shorter": common_prefix / max(1, min_len),
        "first_divergence_before_256": first is not None and first < 256,
        "first_divergence_before_1024": first is not None and first < 1024,
        "first_divergence_before_16384": first is not None and first < 16384,
        "run1_output_tokens": len(left_ids),
        "run2_output_tokens": len(right_ids),
        "delta_output_tokens": len(right_ids) - len(left_ids),
        "run1_finish_reason": left.get("finish_reason"),
        "run2_finish_reason": right.get("finish_reason"),
        "run1_hit_length_cap": left_finish.lower() == "length",
        "run2_hit_length_cap": right_finish.lower() == "length",
        "finish_reason_match": left.get("finish_reason") == right.get("finish_reason"),
        "run1_thinking_boundary": bool(left.get("thinking_boundary_detected")),
        "run2_thinking_boundary": bool(right.get("thinking_boundary_detected")),
        "run1_thinking_tokens": left.get("thinking_tokens"),
        "run2_thinking_tokens": right.get("thinking_tokens"),
        "run1_execution_passed": left.get("execution_passed"),
        "run2_execution_passed": right.get("execution_passed"),
        "execution_pass_match": left.get("execution_passed") == right.get("execution_passed"),
        "same_candidate_code": (
            left.get("candidate_code_sha256") is not None
            and left.get("candidate_code_sha256") == right.get("candidate_code_sha256")
        ),
        "same_output_text": left.get("output_text") == right.get("output_text"),
        "run1_error": left_error,
        "run2_error": right_error,
    }


def write_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        f"# Repeat determinism: {summary['condition_label']}",
        "",
        "## Configuration",
        "",
        f"- max_tokens: `{summary['max_tokens']}`",
        f"- max_model_len: `{summary['max_model_len']}`",
        f"- enforce_eager: `{summary['enforce_eager']}`",
        "- decoding: greedy (`temperature=0`, `top_p=1`, `top_k=-1`)",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Valid repeat pairs | {summary['valid_pairs']} |",
        f"| Self-diverged | {summary['diverged_count']} |",
        f"| Self-divergence rate | {summary['self_divergence_rate']:.4f} |" if summary["self_divergence_rate"] is not None else "| Self-divergence rate | N/A |",
        f"| Diverged before step 256 | {summary['diverged_before_256']} |",
        f"| Diverged before step 1024 | {summary['diverged_before_1024']} |",
        f"| Diverged before step 16384 | {summary['diverged_before_16384']} |",
        f"| Median first divergence step | {summary['median_first_divergence_step']} |",
        f"| Either repeat hit max_tokens | {summary['either_hit_length_cap_count']} |",
        f"| Finish-reason mismatches | {summary['finish_reason_mismatch_count']} |",
        f"| Pass/fail mismatches | {summary['execution_pass_mismatch_count']} |",
        "",
        "## Per sample",
        "",
        "| Sample | Diverged | First diff | Run1 tokens | Run2 tokens | Run1 finish | Run2 finish | Pass match | Same code |",
        "|---|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {sample_id} | {diverged} | {first} | {l1} | {l2} | {f1} | {f2} | {pass_match} | {same_code} |".format(
                sample_id=row["sample_id"],
                diverged=row["diverged"],
                first=row["first_divergence_step"],
                l1=row["run1_output_tokens"],
                l2=row["run2_output_tokens"],
                f1=row["run1_finish_reason"],
                f2=row["run2_finish_reason"],
                pass_match=row["execution_pass_match"],
                same_code=row["same_candidate_code"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    left = load_jsonl(args.run1)
    right = load_jsonl(args.run2)
    common = sorted(set(left).intersection(right))
    rows = [row_for(sample_id, left[sample_id], right[sample_id]) for sample_id in common]
    valid = [row for row in rows if row["valid_pair"]]
    diverged = [row for row in valid if row["diverged"]]
    first_steps = [int(row["first_divergence_step"]) for row in diverged]

    summary = {
        "schema_version": 1,
        "condition_label": args.condition_label,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "enforce_eager": bool(args.enforce_eager),
        "common_sample_count": len(common),
        "valid_pairs": len(valid),
        "diverged_count": len(diverged),
        "self_divergence_rate": len(diverged) / len(valid) if valid else None,
        "diverged_before_256": sum(bool(row["first_divergence_before_256"]) for row in valid),
        "diverged_before_1024": sum(bool(row["first_divergence_before_1024"]) for row in valid),
        "diverged_before_16384": sum(bool(row["first_divergence_before_16384"]) for row in valid),
        "first_divergence_steps": first_steps,
        "mean_first_divergence_step": mean_or_none(first_steps),
        "median_first_divergence_step": median_or_none(first_steps),
        "min_first_divergence_step": min(first_steps) if first_steps else None,
        "max_first_divergence_step": max(first_steps) if first_steps else None,
        "either_hit_length_cap_count": sum(
            bool(row["run1_hit_length_cap"] or row["run2_hit_length_cap"]) for row in valid
        ),
        "finish_reason_mismatch_count": sum(not bool(row["finish_reason_match"]) for row in valid),
        "execution_pass_mismatch_count": sum(not bool(row["execution_pass_match"]) for row in valid),
        "same_candidate_code_count": sum(bool(row["same_candidate_code"]) for row in valid),
        "run1_finish_reason_counts": dict(Counter(str(row["run1_finish_reason"]) for row in valid)),
        "run2_finish_reason_counts": dict(Counter(str(row["run2_finish_reason"]) for row in valid)),
        "run1_mean_output_tokens": mean_or_none([int(row["run1_output_tokens"]) for row in valid]),
        "run2_mean_output_tokens": mean_or_none([int(row["run2_output_tokens"]) for row in valid]),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_sample_repeat_comparison.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    fieldnames = list(rows[0].keys()) if rows else ["sample_id"]
    with (args.output_dir / "per_sample_repeat_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(args.output_dir / "summary.md", summary, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
