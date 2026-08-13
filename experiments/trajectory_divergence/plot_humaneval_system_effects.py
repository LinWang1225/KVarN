#!/usr/bin/env python3
"""Plot HumanEval trajectory/system consequence diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot HumanEval system-effect analysis")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    **row,
                    "diverged": parse_bool(row.get("diverged")),
                    "either_censored": parse_bool(row.get("either_censored")),
                    "normalized_first_divergence": parse_float(row.get("normalized_first_divergence")),
                    "token_inflation_ratio": parse_float(row.get("token_inflation_ratio")),
                    "latency_inflation_ratio": parse_float(row.get("latency_inflation_ratio")),
                    "decode_latency_inflation_ratio": parse_float(row.get("decode_latency_inflation_ratio")),
                    "delta_thinking_tokens": parse_float(row.get("delta_thinking_tokens")),
                    "delta_final_tokens": parse_float(row.get("delta_final_tokens")),
                }
            )
    return rows


def save(fig: Any, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def empty(ax: Any, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_tir_cdf(rows: list[dict[str, Any]], output_dir: Path) -> None:
    values = sorted(
        row["token_inflation_ratio"]
        for row in rows
        if not row["either_censored"] and row["token_inflation_ratio"] is not None
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if not values:
        empty(ax, "No uncensored token-inflation values")
    else:
        cdf = [(index + 1) / len(values) for index in range(len(values))]
        ax.step(values, cdf, where="post")
        for threshold in (0.0, 0.10, 0.25, 0.50, 1.00):
            ax.axvline(threshold, linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Token inflation ratio: (KVarN - FP16) / FP16")
        ax.set_ylabel("Cumulative fraction")
        ax.set_title("HumanEval token-inflation CDF")
        ax.grid(alpha=0.25)
    save(fig, output_dir, "figure_humaneval_tir_cdf")


def plot_tir_vs_lir(rows: list[dict[str, Any]], output_dir: Path) -> None:
    usable = [
        row
        for row in rows
        if not row["either_censored"]
        and row["token_inflation_ratio"] is not None
        and row["decode_latency_inflation_ratio"] is not None
    ]
    use_decode = bool(usable)
    if not usable:
        usable = [
            row
            for row in rows
            if not row["either_censored"]
            and row["token_inflation_ratio"] is not None
            and row["latency_inflation_ratio"] is not None
        ]
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    if not usable:
        empty(ax, "No paired token/latency inflation values")
    else:
        y_key = "decode_latency_inflation_ratio" if use_decode else "latency_inflation_ratio"
        ax.scatter(
            [row["token_inflation_ratio"] for row in usable],
            [row[y_key] for row in usable],
            alpha=0.75,
        )
        ax.axhline(0.0, linestyle="--", linewidth=0.8)
        ax.axvline(0.0, linestyle="--", linewidth=0.8)
        ax.set_xlabel("Token inflation ratio")
        ax.set_ylabel("Decode latency inflation ratio" if use_decode else "E2E latency inflation ratio")
        ax.set_title("Generation inflation versus latency inflation")
        ax.grid(alpha=0.25)
    save(fig, output_dir, "figure_humaneval_tir_vs_lir")


def plot_divergence_vs_tir(rows: list[dict[str, Any]], output_dir: Path) -> None:
    usable = [
        row
        for row in rows
        if row["diverged"]
        and not row["either_censored"]
        and row["normalized_first_divergence"] is not None
        and row["token_inflation_ratio"] is not None
    ]
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    if not usable:
        empty(ax, "No uncensored diverged samples")
    else:
        ax.scatter(
            [row["normalized_first_divergence"] for row in usable],
            [row["token_inflation_ratio"] for row in usable],
            alpha=0.75,
        )
        ax.axhline(0.0, linestyle="--", linewidth=0.8)
        ax.set_xlabel("Normalized first divergence: step / FP16 output tokens")
        ax.set_ylabel("Token inflation ratio")
        ax.set_title("First divergence location versus generation inflation")
        ax.grid(alpha=0.25)
    save(fig, output_dir, "figure_humaneval_divergence_vs_tir")


def plot_group_box(rows: list[dict[str, Any]], output_dir: Path) -> None:
    order = ["no_divergence", "early_[0,0.25)", "middle_[0.25,0.5)", "late_[0.5,1+]", "diverged_unclassified"]
    grouped: list[list[float]] = []
    labels: list[str] = []
    for group in order:
        values = [
            row["token_inflation_ratio"]
            for row in rows
            if row.get("divergence_group") == group
            and not row["either_censored"]
            and row["token_inflation_ratio"] is not None
        ]
        if values:
            grouped.append(values)
            labels.append(group.replace("_", "\n"))
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    if not grouped:
        empty(ax, "No populated divergence groups")
    else:
        #ax.boxplot(grouped, labels=labels, showfliers=True)
        ax.boxplot(grouped, tick_labels=labels, showfliers=True)
        ax.axhline(0.0, linestyle="--", linewidth=0.8)
        ax.set_ylabel("Token inflation ratio")
        ax.set_title("Token inflation by trajectory-divergence group")
        ax.grid(axis="y", alpha=0.25)
    save(fig, output_dir, "figure_humaneval_tir_by_divergence_group")


def plot_thinking_vs_final(rows: list[dict[str, Any]], output_dir: Path) -> None:
    usable = [
        row
        for row in rows
        if not row["either_censored"]
        and row["delta_thinking_tokens"] is not None
        and row["delta_final_tokens"] is not None
    ]
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    if not usable:
        empty(ax, "No thinking/final token split available")
    else:
        ax.scatter(
            [row["delta_thinking_tokens"] for row in usable],
            [row["delta_final_tokens"] for row in usable],
            alpha=0.75,
        )
        ax.axhline(0.0, linestyle="--", linewidth=0.8)
        ax.axvline(0.0, linestyle="--", linewidth=0.8)
        ax.set_xlabel("Δ thinking tokens (KVarN - FP16)")
        ax.set_ylabel("Δ final/code-region tokens (KVarN - FP16)")
        ax.set_title("Where does HumanEval output inflation come from?")
        ax.grid(alpha=0.25)
    save(fig, output_dir, "figure_humaneval_thinking_vs_final_delta")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv)
    plot_tir_cdf(rows, args.output_dir)
    plot_tir_vs_lir(rows, args.output_dir)
    plot_divergence_vs_tir(rows, args.output_dir)
    plot_group_box(rows, args.output_dir)
    plot_thinking_vs_final(rows, args.output_dir)


if __name__ == "__main__":
    main()
