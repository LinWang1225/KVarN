# SPDX-License-Identifier: Apache-2.0
"""Plot the backend-native KVarN/TurboQuant comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_figure(input_csv: Path, output_prefix: Path, title: str | None = None) -> None:
    frame = pd.read_csv(input_csv).sort_values(["method", "context_tokens"])
    required = {
        "label",
        "context_tokens",
        "mean_abs_logprob_error",
        "top1_agreement",
        "prompt_tokens_per_second",
        "fp16_prompt_tokens_per_second",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.3), constrained_layout=True)

    for label, group in frame.groupby("label", sort=False):
        x = group["context_tokens"] / 1000.0
        axes[0].plot(x, group["mean_abs_logprob_error"], marker="o", label=label)
        axes[1].plot(x, 100.0 * group["top1_agreement"], marker="o", label=label)
        axes[2].plot(x, group["prompt_tokens_per_second"], marker="o", label=label)

    fp16 = frame[["context_tokens", "fp16_prompt_tokens_per_second"]].drop_duplicates()
    fp16 = fp16.sort_values("context_tokens")
    axes[2].plot(
        fp16["context_tokens"] / 1000.0,
        fp16["fp16_prompt_tokens_per_second"],
        marker="o",
        linestyle="--",
        label="FP16",
    )

    axes[0].set_title("(a) Fixed-trajectory log-probability drift")
    axes[0].set_ylabel("Mean |Δ log p(target)|")
    axes[1].set_title("(b) Top-1 agreement with FP16")
    axes[1].set_ylabel("Agreement (%)")
    axes[1].set_ylim(0, 101)
    axes[2].set_title("(c) End-to-end prompt scoring throughput")
    axes[2].set_ylabel("Prompt tokens / second")

    for axis in axes:
        axis.set_xlabel("Context length (kTokens)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    if title:
        fig.suptitle(title)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_prefix.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--title", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_figure(args.input, args.output_prefix, args.title)
