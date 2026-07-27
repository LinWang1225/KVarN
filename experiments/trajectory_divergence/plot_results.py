#!/usr/bin/env python3
"""Generate publication-ready plots from per_sample_comparison.csv."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGGER = logging.getLogger("plot_trajectory_results")

BUCKET_ORDER = (
    "[0,512)",
    "[512,1024)",
    "[1024,2048)",
    "[2048,4096)",
    "[4096,+inf)",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot FP16/KVarN trajectory divergence results."
    )
    parser.add_argument("--comparison-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--boundary-step",
        type=int,
        default=512,
        help="Vertical boundary spacing in the divergence histogram.",
    )
    parser.add_argument(
        "--annotate-top-k",
        type=int,
        default=8,
        help="Number of strongest length-inflation samples to label.",
    )
    return parser.parse_args()


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_int(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(float(value))


def parse_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    **row,
                    "diverged": parse_bool(row.get("diverged")),
                    "first_divergence_step": parse_int(row.get("first_divergence_step")),
                    "fp16_output_tokens": parse_int(row.get("fp16_output_tokens")) or 0,
                    "kvarn_output_tokens": parse_int(row.get("kvarn_output_tokens")) or 0,
                    "length_ratio": parse_float(row.get("length_ratio")) or 0.0,
                }
            )
    return rows


def save_figure(fig: Any, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_empty_message(ax: Any, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_length_scatter(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: Path,
    annotate_top_k: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    if not rows:
        add_empty_message(ax, "No valid paired samples")
        save_figure(fig, output_dir, "figure_length_scatter")
        return

    identical = [row for row in rows if not row["diverged"]]
    diverged = [row for row in rows if row["diverged"]]
    if identical:
        ax.scatter(
            [row["fp16_output_tokens"] for row in identical],
            [row["kvarn_output_tokens"] for row in identical],
            alpha=0.7,
            label="Identical token trajectory",
        )
    if diverged:
        ax.scatter(
            [row["fp16_output_tokens"] for row in diverged],
            [row["kvarn_output_tokens"] for row in diverged],
            alpha=0.8,
            label="Diverged",
        )

    maximum = max(
        max(row["fp16_output_tokens"], row["kvarn_output_tokens"]) for row in rows
    )
    ax.plot([0, maximum], [0, maximum], linestyle="--", linewidth=1, label="y = x")
    ax.set_xlabel("FP16 output tokens")
    ax.set_ylabel("KVarN output tokens")
    divergence_rate = summary.get("divergence_rate")
    rate_text = "N/A" if divergence_rate is None else f"{100 * divergence_rate:.1f}%"
    ax.set_title(f"Output length comparison (n={len(rows)}, divergence={rate_text})")
    ax.legend()
    ax.grid(alpha=0.25)

    inflation = sorted(
        diverged,
        key=lambda row: row["length_ratio"],
        reverse=True,
    )[: max(0, annotate_top_k)]
    for row in inflation:
        if row["length_ratio"] <= 1.1:
            continue
        ax.annotate(
            str(row["sample_id"])[:8],
            (row["fp16_output_tokens"], row["kvarn_output_tokens"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    save_figure(fig, output_dir, "figure_length_scatter")


def plot_divergence_hist(
    rows: list[dict[str, Any]], output_dir: Path, boundary_step: int
) -> None:
    steps = [
        row["first_divergence_step"]
        for row in rows
        if row["diverged"] and row["first_divergence_step"] is not None
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if not steps:
        add_empty_message(ax, "No diverged samples")
    else:
        bins = min(40, max(8, int(math.sqrt(len(steps)) * 3)))
        ax.hist(steps, bins=bins, alpha=0.85)
        if boundary_step > 0:
            maximum = max(steps)
            for boundary in range(boundary_step, maximum + 1, boundary_step):
                ax.axvline(boundary, linestyle="--", linewidth=0.6, alpha=0.35)
        ax.set_xlabel("First divergence step (0-based output token)")
        ax.set_ylabel("Samples")
        ax.set_title("Distribution of first trajectory divergence")
        ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_dir, "figure_first_divergence_hist")


def plot_divergence_cdf(rows: list[dict[str, Any]], output_dir: Path) -> None:
    steps = sorted(
        row["first_divergence_step"]
        for row in rows
        if row["diverged"] and row["first_divergence_step"] is not None
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if not steps:
        add_empty_message(ax, "No diverged samples")
    else:
        cdf = [(index + 1) / len(steps) for index in range(len(steps))]
        ax.step(steps, cdf, where="post")
        median = steps[(len(steps) - 1) // 2]
        p90 = steps[min(len(steps) - 1, math.ceil(0.9 * len(steps)) - 1)]
        ax.axvline(median, linestyle="--", linewidth=1, label=f"Median={median}")
        ax.axvline(p90, linestyle=":", linewidth=1, label=f"P90={p90}")
        ax.set_xlabel("First divergence step")
        ax.set_ylabel("Cumulative fraction of diverged samples")
        ax.set_ylim(0, 1.02)
        ax.set_title("CDF of first trajectory divergence")
        ax.legend()
        ax.grid(alpha=0.25)
    save_figure(fig, output_dir, "figure_first_divergence_cdf")


def plot_divergence_by_length(summary: dict[str, Any], output_dir: Path) -> None:
    bucket_data = summary.get("length_buckets", {})
    labels: list[str] = []
    rates: list[float] = []
    annotations: list[str] = []
    for label in BUCKET_ORDER:
        values = bucket_data.get(label, {})
        count = int(values.get("sample_count", 0) or 0)
        diverged = int(values.get("divergence_count", 0) or 0)
        rate = values.get("divergence_rate")
        if count == 0 or rate is None:
            continue
        labels.append(label)
        rates.append(float(rate))
        annotations.append(f"{diverged}/{count}")

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    if not labels:
        add_empty_message(ax, "No populated output-length buckets")
    else:
        bars = ax.bar(labels, rates)
        for bar, annotation in zip(bars, annotations):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                annotation,
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_ylim(0, min(1.05, max(rates) * 1.2 + 0.05))
        ax.set_xlabel("FP16 output-token bucket")
        ax.set_ylabel("Divergence rate")
        ax.set_title("Trajectory divergence versus FP16 output length")
        ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_dir, "figure_divergence_by_length")


def plot_length_ratio_hist(rows: list[dict[str, Any]], output_dir: Path) -> None:
    mismatch = [
        row["length_ratio"]
        for row in rows
        if row.get("divergence_type") == "token_mismatch"
    ]
    length_only = [
        row["length_ratio"]
        for row in rows
        if row.get("divergence_type") == "length_only"
    ]
    identical = [
        row["length_ratio"]
        for row in rows
        if row.get("divergence_type") == "identical"
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    values = mismatch + length_only + identical
    if not values:
        add_empty_message(ax, "No valid paired samples")
    else:
        maximum = max(values)
        bins = min(45, max(12, int(math.sqrt(len(values)) * 4)))
        if identical:
            ax.hist(identical, bins=bins, alpha=0.45, label="Identical")
        if mismatch:
            ax.hist(mismatch, bins=bins, alpha=0.6, label="Token mismatch")
        if length_only:
            ax.hist(length_only, bins=bins, alpha=0.6, label="Length only")
        for threshold, label in ((1.0, "1.0"), (1.5, "1.5"), (2.0, "2.0")):
            if threshold <= maximum * 1.05:
                ax.axvline(threshold, linestyle="--", linewidth=0.8, label=label)
        ax.set_xlabel("KVarN output tokens / FP16 output tokens")
        ax.set_ylabel("Samples")
        ax.set_title("Output-length ratio distribution")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_dir, "figure_length_ratio_hist")


def plot_divergence_vs_ratio(rows: list[dict[str, Any]], output_dir: Path) -> None:
    diverged = [
        row
        for row in rows
        if row["diverged"] and row["first_divergence_step"] is not None
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if not diverged:
        add_empty_message(ax, "No diverged samples")
    else:
        ax.scatter(
            [row["first_divergence_step"] for row in diverged],
            [row["length_ratio"] for row in diverged],
            alpha=0.8,
        )
        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.set_xlabel("First divergence step")
        ax.set_ylabel("KVarN / FP16 output-length ratio")
        ax.set_title("First divergence location versus output-length change")
        ax.grid(alpha=0.25)
    save_figure(fig, output_dir, "figure_divergence_vs_length_ratio")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    rows = load_rows(args.comparison_csv)
    with args.summary_json.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_length_scatter(rows, summary, args.output_dir, args.annotate_top_k)
    plot_divergence_hist(rows, args.output_dir, args.boundary_step)
    plot_divergence_cdf(rows, args.output_dir)
    plot_divergence_by_length(summary, args.output_dir)
    plot_length_ratio_hist(rows, args.output_dir)
    plot_divergence_vs_ratio(rows, args.output_dir)
    LOGGER.info("Saved plots to %s", args.output_dir)


if __name__ == "__main__":
    main()
