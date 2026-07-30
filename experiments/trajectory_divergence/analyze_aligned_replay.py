#!/usr/bin/env python3
"""Compare four teacher-forced aligned replays and create statistics/plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

PAIR_SPECS = {
    "fp16_self": ("fp16_run1", "fp16_run2"),
    "kvarn_self": ("kvarn_run1", "kvarn_run2"),
    "cross_run1": ("fp16_run1", "kvarn_run1"),
    "cross_run2": ("fp16_run2", "kvarn_run2"),
}
RUN_NAMES = ("fp16_run1", "fp16_run2", "kvarn_run1", "kvarn_run2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze four aligned replay runs.")
    parser.add_argument("--aligned-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--step-bin-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed line {line_number} in {path}") from exc
    return records


def successful_replay_map(run_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(run_dir / "replay_records.jsonl"):
        if record.get("error") is None and record.get("replay_verified") is True:
            result[str(record["sample_id"])] = record
    return result


def metric_map(run_dir: Path, sample_id: str) -> dict[int, dict[str, Any]]:
    records = read_jsonl(run_dir / "metrics" / f"{sample_id}.jsonl")
    return {int(record["step"]): record for record in records}


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
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def safe_mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else None


def topk_jaccard(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_ids = set(int(value) for value in left["raw_topk_token_ids"])
    right_ids = set(int(value) for value in right["raw_topk_token_ids"])
    union = left_ids | right_ids
    return len(left_ids & right_ids) / len(union) if union else 1.0


def first_disagreement(
    metrics_left: dict[int, dict[str, Any]],
    metrics_right: dict[int, dict[str, Any]],
) -> int | None:
    for step in sorted(set(metrics_left) & set(metrics_right)):
        if int(metrics_left[step]["raw_top1_token_id"]) != int(
            metrics_right[step]["raw_top1_token_id"]
        ):
            return step
    return None


def load_tokenizer(name_or_path: str | None) -> Any | None:
    if not name_or_path:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        name_or_path,
        trust_remote_code=True,
        local_files_only=True,
    )


def token_text(tokenizer: Any | None, token_id: int) -> str:
    if tokenizer is None:
        return ""
    return tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for record in records:
        for key in record:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def pair_summary(
    name: str,
    first_steps: dict[str, int | None],
    pair_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    valid_first = [float(value) for value in first_steps.values() if value is not None]
    disagreement_rows = [row for row in pair_rows if row[f"{name}_top1_disagree"]]
    return {
        "valid_samples": len(first_steps),
        "samples_with_raw_top1_disagreement": len(valid_first),
        "sample_disagreement_rate": len(valid_first) / len(first_steps) if first_steps else None,
        "first_disagreement_step_median": percentile(valid_first, 0.5),
        "first_disagreement_step_p90": percentile(valid_first, 0.9),
        "observed_steps": len(pair_rows),
        "top1_disagreement_steps": len(disagreement_rows),
        "step_disagreement_rate": len(disagreement_rows) / len(pair_rows) if pair_rows else None,
        "mean_abs_forced_logprob_difference": safe_mean(
            abs(float(row[f"{name}_forced_logprob_delta"])) for row in pair_rows
        ),
        "mean_topk_jaccard": safe_mean(float(row[f"{name}_topk_jaccard"]) for row in pair_rows),
        "first_disagreement_region_counts": dict(
            Counter(
                row["region"]
                for row in pair_rows
                if row["sample_id"] in first_steps
                and first_steps[row["sample_id"]] == row["step"]
            )
        ),
    }


def create_plots(
    rows: list[dict[str, Any]],
    first_steps_by_pair: dict[str, dict[str, int | None]],
    output_dir: Path,
    step_bin_size: int,
    block_size: int,
) -> None:
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for pair_name in ("kvarn_self", "cross_run1", "cross_run2"):
        values = sorted(
            value
            for value in first_steps_by_pair[pair_name].values()
            if value is not None
        )
        if values:
            y = [(index + 1) / len(values) for index in range(len(values))]
            ax.step(values, y, where="post", label=pair_name)
    ax.set_xlabel("First raw-top1 disagreement step")
    ax.set_ylabel("Cumulative sample fraction")
    ax.set_title("Aligned replay: first raw-top1 disagreement")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(plot_dir / f"figure_aligned_first_disagreement_cdf.{suffix}", dpi=180)
    plt.close(fig)

    bins: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bins[(int(row["step"]) // step_bin_size) * step_bin_size].append(row)
    fig, ax = plt.subplots(figsize=(9, 5))
    for pair_name in ("kvarn_self", "cross_run1", "cross_run2"):
        xs: list[int] = []
        ys: list[float] = []
        for start in sorted(bins):
            bucket = bins[start]
            xs.append(start)
            ys.append(
                sum(bool(row[f"{pair_name}_top1_disagree"]) for row in bucket)
                / len(bucket)
            )
        ax.plot(xs, ys, label=pair_name)
    ax.set_xlabel(f"Replay step (binned every {step_bin_size})")
    ax.set_ylabel("Raw-top1 disagreement rate")
    ax.set_title("Aligned replay disagreement rate over generation")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(plot_dir / f"figure_aligned_disagreement_by_step.{suffix}", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for run_index in (1, 2):
        key = f"cross_run{run_index}_reference_logprob_drop"
        xs: list[int] = []
        medians: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for start in sorted(bins):
            values = [float(row[key]) for row in bins[start]]
            xs.append(start)
            medians.append(float(percentile(values, 0.5) or 0.0))
            lows.append(float(percentile(values, 0.25) or 0.0))
            highs.append(float(percentile(values, 0.75) or 0.0))
        line = ax.plot(xs, medians, label=f"cross_run{run_index}")[0]
        ax.fill_between(xs, lows, highs, alpha=0.18, color=line.get_color())
    ax.axhline(0.0, linewidth=1, linestyle="--")
    ax.set_xlabel(f"Replay step (binned every {step_bin_size})")
    ax.set_ylabel("FP16 ref logprob - KVarN ref logprob")
    ax.set_title("Reference-token log-probability drop under KVarN")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(plot_dir / f"figure_aligned_reference_logprob_drop.{suffix}", dpi=180)
    plt.close(fig)

    offsets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        offsets[int(row["absolute_offset_in_block"])].append(row)
    fig, ax = plt.subplots(figsize=(10, 5))
    for pair_name in ("kvarn_self", "cross_run1", "cross_run2"):
        xs = sorted(offsets)
        ys = [
            sum(bool(row[f"{pair_name}_top1_disagree"]) for row in offsets[offset])
            / len(offsets[offset])
            for offset in xs
        ]
        ax.plot(xs, ys, label=pair_name)
    ax.set_xlim(0, block_size - 1)
    ax.set_xlabel("Absolute token offset within 128-token KV block")
    ax.set_ylabel("Raw-top1 disagreement rate")
    ax.set_title("Aligned disagreement versus KV-block offset")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(plot_dir / f"figure_aligned_block_offset_disagreement.{suffix}", dpi=180)
    plt.close(fig)

    flip_margins = [
        float(row["fp16_run1_raw_top1_margin"])
        for row in rows
        if row["cross_run1_top1_disagree"]
    ]
    stable_margins = [
        float(row["fp16_run1_raw_top1_margin"])
        for row in rows
        if not row["cross_run1_top1_disagree"]
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    if stable_margins:
        ax.hist(stable_margins, bins=50, alpha=0.55, density=True, label="cross top1 agrees")
    if flip_margins:
        ax.hist(flip_margins, bins=50, alpha=0.55, density=True, label="cross top1 differs")
    ax.set_xlabel("FP16 raw top1-top2 margin")
    ax.set_ylabel("Density")
    ax.set_title("Decision margin at aligned cross-mode disagreement steps")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(plot_dir / f"figure_aligned_margin_at_cross_disagreement.{suffix}", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} is non-empty; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.tokenizer)

    run_dirs = {name: args.aligned_root / name for name in RUN_NAMES}
    replay_maps = {name: successful_replay_map(path) for name, path in run_dirs.items()}
    common_samples = sorted(set.intersection(*(set(records) for records in replay_maps.values())))
    if not common_samples:
        raise ValueError("No samples are complete in all four aligned replay runs")

    all_rows: list[dict[str, Any]] = []
    first_steps_by_pair: dict[str, dict[str, int | None]] = {
        name: {} for name in PAIR_SPECS
    }
    verification = {
        run_name: {
            "successful_samples": len(replay_maps[run_name]),
            "all_prompt_token_ids_match": all(
                record.get("prompt_token_ids_match_reference") is True
                for record in replay_maps[run_name].values()
            ),
            "all_metrics_complete": all(
                record.get("metrics_complete") is True
                for record in replay_maps[run_name].values()
            ),
        }
        for run_name in RUN_NAMES
    }

    for sample_id in common_samples:
        metrics = {
            run_name: metric_map(run_dirs[run_name], sample_id)
            for run_name in RUN_NAMES
        }
        common_steps = sorted(set.intersection(*(set(value) for value in metrics.values())))
        for pair_name, (left_name, right_name) in PAIR_SPECS.items():
            first_steps_by_pair[pair_name][sample_id] = first_disagreement(
                metrics[left_name], metrics[right_name]
            )
        for step in common_steps:
            base = metrics["fp16_run1"][step]
            forced_token_id = int(base["forced_token_id"])
            row: dict[str, Any] = {
                "sample_id": sample_id,
                "step": step,
                "absolute_position": int(base["absolute_position"]),
                "absolute_block": int(base["absolute_block"]),
                "absolute_offset_in_block": int(base["absolute_offset_in_block"]),
                "region": base.get("region", "unknown"),
                "forced_token_id": forced_token_id,
                "forced_token_text": token_text(tokenizer, forced_token_id),
            }
            for run_name in RUN_NAMES:
                metric = metrics[run_name][step]
                row[f"{run_name}_raw_top1_token_id"] = int(metric["raw_top1_token_id"])
                row[f"{run_name}_raw_top1_token_text"] = token_text(
                    tokenizer, int(metric["raw_top1_token_id"])
                )
                for key in (
                    "raw_top1_is_forced",
                    "raw_top1_logprob",
                    "raw_top2_logprob",
                    "raw_top1_margin",
                    "forced_token_logprob",
                    "forced_token_rank",
                ):
                    row[f"{run_name}_{key}"] = metric[key]
            for pair_name, (left_name, right_name) in PAIR_SPECS.items():
                left = metrics[left_name][step]
                right = metrics[right_name][step]
                row[f"{pair_name}_top1_disagree"] = int(left["raw_top1_token_id"]) != int(
                    right["raw_top1_token_id"]
                )
                row[f"{pair_name}_forced_logprob_delta"] = float(
                    right["forced_token_logprob"]
                ) - float(left["forced_token_logprob"])
                row[f"{pair_name}_topk_jaccard"] = topk_jaccard(left, right)
            row["cross_run1_reference_logprob_drop"] = -float(
                row["cross_run1_forced_logprob_delta"]
            )
            row["cross_run2_reference_logprob_drop"] = -float(
                row["cross_run2_forced_logprob_delta"]
            )
            all_rows.append(row)

    pair_summaries = {
        name: pair_summary(name, first_steps_by_pair[name], all_rows)
        for name in PAIR_SPECS
    }
    region_cross = defaultdict(lambda: {"steps": 0, "run1_disagree": 0, "run2_disagree": 0})
    for row in all_rows:
        region = str(row["region"])
        region_cross[region]["steps"] += 1
        region_cross[region]["run1_disagree"] += int(row["cross_run1_top1_disagree"])
        region_cross[region]["run2_disagree"] += int(row["cross_run2_top1_disagree"])
    summary = {
        "schema_version": 1,
        "common_samples": len(common_samples),
        "common_sample_ids": common_samples,
        "observed_aligned_steps": len(all_rows),
        "verification": verification,
        "pairs": pair_summaries,
        "cross_disagreement_by_region": dict(region_cross),
        "interpretation": {
            "positive_reference_logprob_drop": (
                "The FP16 reference token is less probable under KVarN at the same aligned history."
            ),
            "raw_top1_disagreement": (
                "The unmodified next-token argmax differs before teacher forcing."
            ),
        },
    }

    write_jsonl(args.output_dir / "per_step_aligned.jsonl", all_rows)
    write_csv(args.output_dir / "per_step_aligned.csv", all_rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    lines = [
        "# Teacher-forced aligned replay summary",
        "",
        f"- Samples complete in all four runs: **{len(common_samples)}**",
        f"- Aligned step records: **{len(all_rows)}**",
        "",
        "## Pairwise raw-logit decisions",
        "",
        "| Pair | Samples with top1 disagreement | Rate | Median first step | P90 | Step disagreement rate | Mean abs ref-logprob difference |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, data in pair_summaries.items():
        lines.append(
            "| {name} | {count}/{valid} | {rate} | {median} | {p90} | {step_rate} | {logprob} |".format(
                name=name,
                count=data["samples_with_raw_top1_disagreement"],
                valid=data["valid_samples"],
                rate="N/A" if data["sample_disagreement_rate"] is None else f"{data['sample_disagreement_rate']:.4f}",
                median="N/A" if data["first_disagreement_step_median"] is None else f"{data['first_disagreement_step_median']:.1f}",
                p90="N/A" if data["first_disagreement_step_p90"] is None else f"{data['first_disagreement_step_p90']:.1f}",
                step_rate="N/A" if data["step_disagreement_rate"] is None else f"{data['step_disagreement_rate']:.6f}",
                logprob="N/A" if data["mean_abs_forced_logprob_difference"] is None else f"{data['mean_abs_forced_logprob_difference']:.6f}",
            )
        )
    lines.extend(
        [
            "",
            "A raw-top1 disagreement is measured before the logits processor masks the vocabulary to the FP16 reference token.",
            "All four runs therefore consume the same reference history even after their unmodified argmax decisions differ.",
            "",
        ]
    )
    (args.output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    create_plots(
        all_rows,
        first_steps_by_pair,
        args.output_dir,
        args.step_bin_size,
        args.block_size,
    )


if __name__ == "__main__":
    main()
