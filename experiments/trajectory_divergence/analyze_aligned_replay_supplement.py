#!/usr/bin/env python3
"""Add reference-adherence and diagnostic plots to Stage-2 aligned replay.

This script consumes the existing ``analysis/per_step_aligned.csv`` produced by
``analyze_aligned_replay.py``.  It intentionally does not replace the base
analysis.  Its two goals are:

1. quantify whether each aligned engine would naturally choose the forced
   reference token before masking; and
2. expose sparse cross-mode changes that are hidden by per-bin medians.

It works with either an FP16-generated reference path or a KVarN-generated
reference path.  The reference source is inferred from each run's
``experiment_config.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

RUN_NAMES = ("fp16_run1", "fp16_run2", "kvarn_run1", "kvarn_run2")
CROSS_RUNS = ("cross_run1", "cross_run2")
MARGIN_EDGES = (0.0, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0, math.inf)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create supplemental diagnostics for aligned replay."
    )
    parser.add_argument("--aligned-root", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--step-bin-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def safe_mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reference_metadata(aligned_root: Path) -> dict[str, Any]:
    configs: dict[str, dict[str, Any]] = {}
    reference_paths: dict[str, str] = {}
    for run_name in RUN_NAMES:
        config_path = aligned_root / run_name / "experiment_config.json"
        config = read_json(config_path)
        configs[run_name] = config
        reference_paths[run_name] = str(config.get("reference_generations", ""))
    unique_paths = sorted(set(reference_paths.values()))
    if len(unique_paths) != 1:
        raise ValueError(
            "The four aligned runs do not use the same reference_generations: "
            + json.dumps(reference_paths, indent=2)
        )
    reference_path = Path(unique_paths[0])
    source_run = reference_path.parent.name or "unknown"
    source_mode = (
        "fp16" if source_run.startswith("fp16") else
        "kvarn" if source_run.startswith("kvarn") else
        "unknown"
    )
    return {
        "reference_generations": str(reference_path),
        "reference_source_run": source_run,
        "reference_source_mode": source_mode,
        "run_reference_paths": reference_paths,
    }


def adherence_summary(rows: list[dict[str, str]], run_name: str) -> dict[str, Any]:
    key = f"{run_name}_raw_top1_is_forced"
    observed = [row for row in rows if key in row]
    mismatches = [row for row in observed if not parse_bool(row[key])]
    first_by_sample: dict[str, int] = {}
    for row in mismatches:
        sample_id = str(row["sample_id"])
        step = int(row["step"])
        first_by_sample[sample_id] = min(first_by_sample.get(sample_id, step), step)
    first_values = [float(value) for value in first_by_sample.values()]
    return {
        "observed_steps": len(observed),
        "matching_steps": len(observed) - len(mismatches),
        "mismatch_steps": len(mismatches),
        "adherence_rate": (
            (len(observed) - len(mismatches)) / len(observed) if observed else None
        ),
        "mismatch_rate": len(mismatches) / len(observed) if observed else None,
        "samples_with_mismatch": len(first_by_sample),
        "first_mismatch_step_median": percentile(first_values, 0.5),
        "first_mismatch_step_p90": percentile(first_values, 0.9),
        "first_mismatch_by_sample": first_by_sample,
        "mismatch_region_counts": dict(Counter(str(row.get("region", "unknown")) for row in mismatches)),
    }


def margin_label(lower: float, upper: float) -> str:
    if math.isinf(upper):
        return f"[{lower:g},+inf)"
    return f"[{lower:g},{upper:g})"


def margin_bucket_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for lower, upper in zip(MARGIN_EDGES[:-1], MARGIN_EDGES[1:]):
        record: dict[str, Any] = {
            "margin_bucket": margin_label(lower, upper),
            "lower": lower,
            "upper": None if math.isinf(upper) else upper,
        }
        for index in (1, 2):
            margin_key = f"fp16_run{index}_raw_top1_margin"
            disagree_key = f"cross_run{index}_top1_disagree"
            selected: list[dict[str, str]] = []
            for row in rows:
                margin = safe_float(row.get(margin_key))
                if margin is None:
                    continue
                if lower <= margin < upper:
                    selected.append(row)
            disagreed = sum(parse_bool(row.get(disagree_key)) for row in selected)
            record[f"run{index}_steps"] = len(selected)
            record[f"run{index}_disagreements"] = disagreed
            record[f"run{index}_disagreement_rate"] = (
                disagreed / len(selected) if selected else None
            )
        result.append(record)
    return result


def selected_diagnostic_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    adherence_rows: list[dict[str, str]] = []
    cross_rows: list[dict[str, str]] = []
    for row in rows:
        if any(
            not parse_bool(row.get(f"{run_name}_raw_top1_is_forced", True))
            for run_name in RUN_NAMES
        ):
            adherence_rows.append(row)
        if any(parse_bool(row.get(f"{pair}_top1_disagree", False)) for pair in CROSS_RUNS):
            cross_rows.append(row)
    return adherence_rows, cross_rows


def create_plots(
    rows: list[dict[str, str]],
    analysis_dir: Path,
    step_bin_size: int,
    reference_source_run: str,
    margin_table: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    plot_dir = analysis_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    bins: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        start = (int(row["step"]) // step_bin_size) * step_bin_size
        bins[start].append(row)

    # Reference adherence by step.  This exposes V1-runner/reference mismatches
    # that are hidden when only cross-mode disagreements are plotted.
    fig, ax = plt.subplots(figsize=(9, 5))
    for run_name in RUN_NAMES:
        key = f"{run_name}_raw_top1_is_forced"
        xs: list[int] = []
        ys: list[float] = []
        for start in sorted(bins):
            bucket = bins[start]
            xs.append(start)
            ys.append(
                sum(not parse_bool(row[key]) for row in bucket) / len(bucket)
            )
        ax.plot(xs, ys, label=run_name)
    ax.set_xlabel(f"Replay step (binned every {step_bin_size})")
    ax.set_ylabel("Raw top-1 differs from forced reference")
    ax.set_title(f"Reference adherence along {reference_source_run} trajectory")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(plot_dir / f"figure_aligned_reference_adherence_by_step.{suffix}", dpi=180)
    plt.close(fig)

    # The old median/IQR plot can hide sparse but large changes.  Plot the
    # per-bin mean absolute difference instead.
    fig, ax = plt.subplots(figsize=(9, 5))
    for index in (1, 2):
        key = f"cross_run{index}_reference_logprob_drop"
        xs: list[int] = []
        ys: list[float] = []
        for start in sorted(bins):
            values = [abs(float(row[key])) for row in bins[start]]
            xs.append(start)
            ys.append(statistics.fmean(values) if values else 0.0)
        ax.plot(xs, ys, label=f"cross_run{index}")
    ax.set_xlabel(f"Replay step (binned every {step_bin_size})")
    ax.set_ylabel("Mean absolute reference-token logprob difference")
    ax.set_title("Magnitude of FP16/KVarN reference-token probability shift")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(plot_dir / f"figure_aligned_reference_logprob_abs_by_step.{suffix}", dpi=180)
    plt.close(fig)

    # Signed changes only at actual argmax disagreements.
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for index in (1, 2):
        disagree_key = f"cross_run{index}_top1_disagree"
        drop_key = f"cross_run{index}_reference_logprob_drop"
        selected = [row for row in rows if parse_bool(row[disagree_key])]
        if selected:
            plotted = True
            ax.scatter(
                [int(row["step"]) for row in selected],
                [float(row[drop_key]) for row in selected],
                alpha=0.75,
                label=f"cross_run{index}",
            )
    ax.axhline(0.0, linewidth=1, linestyle="--")
    ax.set_xlabel("Replay step")
    ax.set_ylabel("FP16 logprob - KVarN logprob for forced reference token")
    ax.set_title("Reference-token probability shift at cross-mode top-1 flips")
    ax.grid(alpha=0.25)
    if plotted:
        ax.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(plot_dir / f"figure_aligned_reference_logprob_at_disagreement.{suffix}", dpi=180)
    plt.close(fig)

    # Conditional flip probability is more interpretable than two density
    # histograms with very different sample counts.
    labels = [str(row["margin_bucket"]) for row in margin_table]
    positions = list(range(len(labels)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    for index, offset in ((1, -width / 2), (2, width / 2)):
        rates = [
            0.0 if row[f"run{index}_disagreement_rate"] is None else float(row[f"run{index}_disagreement_rate"])
            for row in margin_table
        ]
        bars = ax.bar(
            [position + offset for position in positions],
            rates,
            width=width,
            label=f"cross_run{index}",
        )
        for bar, row in zip(bars, margin_table):
            numerator = int(row[f"run{index}_disagreements"])
            denominator = int(row[f"run{index}_steps"])
            if denominator:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{numerator}/{denominator}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )
    ax.set_xticks(positions, labels, rotation=35, ha="right")
    ax.set_xlabel("FP16 raw top-1/top-2 margin bucket")
    ax.set_ylabel("P(cross-mode top-1 disagreement | margin bucket)")
    ax.set_title("Low-margin decisions are more likely to flip under KVarN")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(plot_dir / f"figure_aligned_margin_flip_probability.{suffix}", dpi=180)
    plt.close(fig)


def format_optional(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    args = parse_args()
    if args.step_bin_size <= 0:
        raise ValueError("--step-bin-size must be positive")
    analysis_dir = args.analysis_dir or (args.aligned_root / "analysis")
    per_step_path = analysis_dir / "per_step_aligned.csv"
    if not per_step_path.exists():
        raise FileNotFoundError(
            f"Missing {per_step_path}; run analyze_aligned_replay.py first"
        )
    summary_json = analysis_dir / "supplemental_summary.json"
    summary_md = analysis_dir / "supplemental_summary.md"
    if (summary_json.exists() or summary_md.exists()) and not args.overwrite:
        raise FileExistsError(
            f"Supplemental output already exists in {analysis_dir}; pass --overwrite"
        )

    rows = read_csv(per_step_path)
    if not rows:
        raise ValueError(f"No aligned rows in {per_step_path}")
    metadata = reference_metadata(args.aligned_root)
    adherence = {
        run_name: adherence_summary(rows, run_name) for run_name in RUN_NAMES
    }
    margin_table = margin_bucket_table(rows)
    mismatch_rows, cross_rows = selected_diagnostic_rows(rows)

    source_run = str(metadata["reference_source_run"])
    source_aligned_run = source_run if source_run in RUN_NAMES else None
    source_adherence = adherence.get(source_aligned_run) if source_aligned_run else None
    warnings: list[str] = []
    if source_adherence and source_adherence["mismatch_steps"]:
        warnings.append(
            f"The aligned {source_aligned_run} engine does not naturally reproduce "
            f"the source trajectory at {source_adherence['mismatch_steps']} steps; "
            "the forced path remains valid, but it is not an exact natural replay "
            "under the aligned runner."
        )

    summary = {
        "schema_version": 1,
        "reference": metadata,
        "observed_rows": len(rows),
        "reference_adherence": adherence,
        "source_reference_adherence": source_adherence,
        "margin_flip_probability": margin_table,
        "diagnostic_row_counts": {
            "any_reference_mismatch": len(mismatch_rows),
            "any_cross_mode_top1_disagreement": len(cross_rows),
        },
        "warnings": warnings,
        "interpretation": {
            "reference_adherence": (
                "Whether the unmodified aligned engine's raw top-1 equals the forced "
                "source token before masking."
            ),
            "positive_reference_logprob_drop": (
                "FP16 assigns the forced reference token a higher log-probability than KVarN."
            ),
            "kvarn_reference_use": (
                "When reference_source_mode is kvarn, kvarn_self disagreements directly "
                "test repeatability along a KVarN-generated natural path."
            ),
        },
    }
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    write_csv(analysis_dir / "reference_mismatch_steps.csv", mismatch_rows)
    write_csv(analysis_dir / "cross_disagreement_steps.csv", cross_rows)
    write_csv(analysis_dir / "margin_flip_probability.csv", margin_table)

    lines = [
        "# Supplemental aligned-replay diagnostics",
        "",
        f"- Reference generations: `{metadata['reference_generations']}`",
        f"- Reference source run: **{metadata['reference_source_run']}**",
        f"- Reference source mode: **{metadata['reference_source_mode']}**",
        f"- Aligned rows: **{len(rows)}**",
        "",
        "## Natural adherence to the forced reference",
        "",
        "| Run | Matching raw top-1 | Adherence | Mismatch steps | Samples with mismatch | First mismatch median | P90 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run_name in RUN_NAMES:
        data = adherence[run_name]
        lines.append(
            "| {run} | {matching}/{observed} | {rate} | {mismatch} | {samples} | {median} | {p90} |".format(
                run=run_name,
                matching=data["matching_steps"],
                observed=data["observed_steps"],
                rate=format_optional(data["adherence_rate"], 6),
                mismatch=data["mismatch_steps"],
                samples=data["samples_with_mismatch"],
                median=format_optional(data["first_mismatch_step_median"], 1),
                p90=format_optional(data["first_mismatch_step_p90"], 1),
            )
        )
    lines.extend(
        [
            "",
            "## Cross-mode flip probability by FP16 decision margin",
            "",
            "| Margin bucket | run1 flips | run1 rate | run2 flips | run2 rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in margin_table:
        lines.append(
            "| {bucket} | {r1d}/{r1n} | {r1rate} | {r2d}/{r2n} | {r2rate} |".format(
                bucket=row["margin_bucket"],
                r1d=row["run1_disagreements"],
                r1n=row["run1_steps"],
                r1rate=format_optional(row["run1_disagreement_rate"], 6),
                r2d=row["run2_disagreements"],
                r2n=row["run2_steps"],
                r2rate=format_optional(row["run2_disagreement_rate"], 6),
            )
        )
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    lines.extend(
        [
            "",
            "`reference_mismatch_steps.csv` contains every step where at least one "
            "aligned engine would not naturally choose the forced source token.",
            "`cross_disagreement_steps.csv` contains every FP16/KVarN raw-top1 flip.",
            "",
        ]
    )
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    create_plots(
        rows,
        analysis_dir,
        args.step_bin_size,
        source_run,
        margin_table,
    )
    print(f"Supplemental aligned analysis written to {analysis_dir}")


if __name__ == "__main__":
    main()
