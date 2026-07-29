# SPDX-License-Identifier: Apache-2.0
"""Render the three panels used by the Figure 5 protocol."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

METHOD_LABELS = {
    "kvarn": "KVarN K4/V2 G128",
    "turboquant": "TurboQuant 3-bit NC",
}
REGIME_LABELS = {"accumulated": "Accumulated", "static": "Static"}


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _curve(rows: list[dict[str, Any]], predicate) -> list[dict[str, Any]]:
    selected = [row for row in rows if predicate(row)]
    return sorted(selected, key=lambda row: int(row["context_tokens"]))


def _plot_with_ci(ax, rows, *, label: str, linestyle: str, marker: str) -> None:
    x = [int(row["context_tokens"]) / 1000.0 for row in rows]
    y = [float(row.get("mean_attention_output_mae") or row["mean_difference"]) for row in rows]
    low = [float(row["ci95_low"]) for row in rows]
    high = [float(row["ci95_high"]) for row in rows]
    line = ax.plot(x, y, label=label, linestyle=linestyle, marker=marker)[0]
    ax.fill_between(x, low, high, alpha=0.12, color=line.get_color())


def plot_figure5(summary_csv: Path, output_prefix: Path, *, title: str) -> None:
    rows = _read(summary_csv)
    figure, axes = plt.subplots(1, 3, figsize=(17.2, 4.8))

    ax = axes[0]
    for method, marker in (("kvarn", "o"), ("turboquant", "s")):
        for regime, linestyle in (("accumulated", "-"), ("static", "--")):
            curve = _curve(
                rows,
                lambda row, m=method, r=regime: row.get("method") == m
                and row.get("regime") == r
                and row.get("mean_attention_output_mae", "") != "",
            )
            _plot_with_ci(
                ax,
                curve,
                label=f"{METHOD_LABELS[method]} — {REGIME_LABELS[regime]}",
                linestyle=linestyle,
                marker=marker,
            )
    ax.set_title("(a) Average attention-output reconstruction error")
    ax.set_xlabel("Context length (kTokens)")
    ax.set_ylabel("MAE(attention-output)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    for regime, linestyle, marker in (
        ("accumulated", "-", "o"),
        ("static", "--", "s"),
    ):
        curve = _curve(
            rows,
            lambda row, r=regime: row.get("panel") == "method_gap"
            and row.get("regime") == r,
        )
        _plot_with_ci(
            ax,
            curve,
            label=REGIME_LABELS[regime],
            linestyle=linestyle,
            marker=marker,
        )
    ax.axhline(0.0, linewidth=1.0, linestyle=":")
    ax.set_title("(b) Difference of KVarN to TurboQuant")
    ax.set_xlabel("Context length (kTokens)")
    ax.set_ylabel(r"MAE$_{KVarN}$ − MAE$_{TurboQuant}$")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[2]
    curve = _curve(
        rows,
        lambda row: row.get("panel") == "regime_gap"
        and row.get("regime") == "static_minus_accumulated",
    )
    _plot_with_ci(
        ax,
        curve,
        label="Static − accumulated",
        linestyle="-",
        marker="o",
    )
    ax.axhline(0.0, linewidth=1.0, linestyle=":")
    ax.set_title("(c) Difference between the two curves in (b)")
    ax.set_xlabel("Context length (kTokens)")
    ax.set_ylabel("Static − accumulated")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    figure.suptitle(title)
    figure.tight_layout()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output_prefix.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    plt.close(figure)
