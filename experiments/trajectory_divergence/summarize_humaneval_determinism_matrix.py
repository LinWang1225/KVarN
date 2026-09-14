#!/usr/bin/env python3
"""Aggregate repeat-determinism summaries across isolation conditions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize HumanEval determinism isolation matrix.")
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries: list[dict[str, Any]] = []
    for path in sorted(args.root.glob("*/analysis/summary.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["summary_path"] = str(path)
        summaries.append(row)
    if not summaries:
        raise SystemExit(f"No */analysis/summary.json files found under {args.root}")

    output_json = args.root / "matrix_summary.json"
    output_csv = args.root / "matrix_summary.csv"
    output_md = args.root / "matrix_summary.md"
    output_json.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fields = [
        "condition_label",
        "max_tokens",
        "max_model_len",
        "enforce_eager",
        "valid_pairs",
        "diverged_count",
        "self_divergence_rate",
        "diverged_before_256",
        "diverged_before_1024",
        "median_first_divergence_step",
        "either_hit_length_cap_count",
        "finish_reason_mismatch_count",
        "execution_pass_mismatch_count",
        "run1_mean_output_tokens",
        "run2_mean_output_tokens",
        "summary_path",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)

    lines = [
        "# HumanEval KVarN determinism isolation matrix",
        "",
        "| Condition | max_tokens | max_model_len | eager | Diverged | Rate | <256 | <1024 | Median first diff | Cap hits | Finish mismatch | Pass mismatch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        rate = row.get("self_divergence_rate")
        rate_text = "N/A" if rate is None else f"{float(rate):.4f}"
        lines.append(
            "| {label} | {mt} | {ml} | {eager} | {div}/{valid} | {rate} | {d256} | {d1024} | {median} | {cap} | {finish} | {passed} |".format(
                label=row["condition_label"],
                mt=row["max_tokens"],
                ml=row["max_model_len"],
                eager=row["enforce_eager"],
                div=row["diverged_count"],
                valid=row["valid_pairs"],
                rate=rate_text,
                d256=row["diverged_before_256"],
                d1024=row["diverged_before_1024"],
                median=row["median_first_divergence_step"],
                cap=row["either_hit_length_cap_count"],
                finish=row["finish_reason_mismatch_count"],
                passed=row["execution_pass_mismatch_count"],
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation guide",
            "",
            "- `baseline_16k_model32k_graph` stable but `model40k_16k_graph` unstable: changing `max_model_len`/cache allocation is the primary trigger.",
            "- `baseline_16k_model32k_graph` stable but `cap30k_model32k_graph` unstable: output-cap/decode-horizon effects matter even with the old model length.",
            "- `extended_32k_model40k_graph` unstable but `extended_32k_model40k_eager` stable: CUDA graph/execution-path effects are strongly implicated.",
            "- Graph and eager both unstable at early steps: investigate KVarN kernel/reduction numerical determinism before using greedy cross-mode divergence as a clean causal result.",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_md)


if __name__ == "__main__":
    main()
