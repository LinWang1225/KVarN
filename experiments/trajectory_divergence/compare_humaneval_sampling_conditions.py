#!/usr/bin/env python3
"""Compare thinking-on and thinking-off HumanEval sampling summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thinking-on", type=Path, required=True)
    parser.add_argument("--thinking-off", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_pct(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{100 * float(value):.2f}%"


def row(label: str, summary: dict[str, Any]) -> dict[str, Any]:
    paper = summary["paper_style_avg3"]
    pooled = summary["pooled_uncensored"]
    severe = summary["severe_inflation_thinking"]
    robust = summary["robust_across_seeds"]
    return {
        "condition": label,
        "paper_style_output_token_change": paper["output_token_change"],
        "fp16_pass_at_1": paper["fp16_pass_at_1"]["mean"],
        "kvarn_pass_at_1": paper["kvarn_pass_at_1"]["mean"],
        "median_tir": pooled["tir"]["median"],
        "p95_tir": pooled["tir"]["p95"],
        "tir_gt_50_fraction": pooled["tail_fractions"].get("tir_gt_0_50"),
        "median_decode_lir": pooled["decode_lir"]["median"],
        "p95_decode_lir": pooled["decode_lir"]["p95"],
        "mean_tpot_lir": pooled["tpot_lir"]["mean"],
        "severe_thinking_contribution": severe.get("aggregate_thinking_contribution"),
        "stable_gt_50_tasks_majority_seeds": robust.get(
            "tasks_longer_gt_50_in_majority_seeds"
        ),
        "all_seeds_uncensored_tasks": robust.get("all_seeds_uncensored_task_count"),
    }


def main() -> None:
    args = parse_args()
    on = load(args.thinking_on)
    off = load(args.thinking_off)
    rows = [row("thinking_on", on), row("thinking_off", off)]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "thinking_mode_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# HumanEval K4V2 sampling: thinking-on vs thinking-off",
        "",
        "| Metric | Thinking ON | Thinking OFF |",
        "|---|---:|---:|",
        f"| Paper-style mean output-token change | {fmt_pct(rows[0]['paper_style_output_token_change'])} | {fmt_pct(rows[1]['paper_style_output_token_change'])} |",
        f"| Median paired TIR | {fmt_pct(rows[0]['median_tir'])} | {fmt_pct(rows[1]['median_tir'])} |",
        f"| P95 paired TIR | {fmt_pct(rows[0]['p95_tir'])} | {fmt_pct(rows[1]['p95_tir'])} |",
        f"| Fraction TIR >50% | {fmt_pct(rows[0]['tir_gt_50_fraction'])} | {fmt_pct(rows[1]['tir_gt_50_fraction'])} |",
        f"| Median decode-latency change | {fmt_pct(rows[0]['median_decode_lir'])} | {fmt_pct(rows[1]['median_decode_lir'])} |",
        f"| P95 decode-latency change | {fmt_pct(rows[0]['p95_decode_lir'])} | {fmt_pct(rows[1]['p95_decode_lir'])} |",
        f"| Mean TPOT change | {fmt_pct(rows[0]['mean_tpot_lir'])} | {fmt_pct(rows[1]['mean_tpot_lir'])} |",
        f"| Severe-inflation thinking contribution | {fmt_pct(rows[0]['severe_thinking_contribution'])} | {fmt_pct(rows[1]['severe_thinking_contribution'])} |",
        f"| Tasks >50% longer in majority seeds | {rows[0]['stable_gt_50_tasks_majority_seeds']} | {rows[1]['stable_gt_50_tasks_majority_seeds']} |",
        "",
        "If the positive tail is strong only with thinking enabled, the evidence favors a reasoning-trajectory-specific effect rather than a generic decoding-length effect.",
    ]
    (args.output_dir / "thinking_mode_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
