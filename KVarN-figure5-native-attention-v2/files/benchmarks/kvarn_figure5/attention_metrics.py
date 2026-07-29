# SPDX-License-Identifier: Apache-2.0
"""Aggregation for the native-kernel Figure 5 attention probe."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

METHOD_ORDER = ("kvarn", "turboquant")
REGIME_ORDER = ("accumulated", "static")


def read_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
    if not rows:
        raise ValueError("No attention probe rows were produced")
    return rows


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot summarize an empty sample")
    average = mean(values)
    sample_std = stdev(values) if len(values) > 1 else 0.0
    sem = sample_std / math.sqrt(len(values)) if values else 0.0
    half = 1.96 * sem
    return {
        "num_samples": len(values),
        "mean": average,
        "sample_std": sample_std,
        "sem": sem,
        "ci95_low": average - half,
        "ci95_high": average + half,
    }


def _complete_fp16_layers(
    raw_rows: list[dict[str, Any]],
    *,
    num_hidden_layers: int,
) -> list[dict[str, Any]]:
    """Add zero-error rows for preset layers intentionally kept in FP16.

    TurboQuant presets may route first/last layers to the normal FP16 backend,
    so the native TurboQuant hook is not called for them.  Figure 5 averages
    over all attention layers; missing layers are therefore included with zero
    reconstruction error.  A missing layer is filled only when the observed
    layer set is stable for every request of that method/regime.
    """
    groups: dict[tuple[str, str, int, int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        groups[
            (
                str(row["method"]),
                str(row["regime"]),
                int(row["request_index"]),
                int(row["sample_id"]),
                int(row["context_tokens"]),
                int(row["chunk_start"]),
                int(row["chunk_end"]),
            )
        ].append(row)

    completed = list(raw_rows)
    observed_sets: dict[tuple[str, str], set[int]] = {}
    for key, rows in groups.items():
        (
            method,
            regime,
            request_index,
            sample_id,
            context_tokens,
            chunk_start,
            chunk_end,
        ) = key
        observed = {int(row["layer_index"]) for row in rows if int(row["layer_index"]) >= 0}
        signature = (method, regime)
        previous = observed_sets.setdefault(signature, observed)
        if observed != previous:
            raise ValueError(
                f"Observed layer set changes for {signature}: {sorted(previous)} vs "
                f"{sorted(observed)}. Refusing to interpret missing layers as FP16 skips."
            )
        if any(int(row["layer_index"]) < 0 for row in rows):
            raise ValueError(f"Could not parse one or more layer indices for {key}")
        missing = sorted(set(range(num_hidden_layers)) - observed)
        if not missing:
            continue
        template_numel = int(rows[0]["num_elements"])
        query_tokens = int(rows[0]["query_tokens"])
        for layer_index in missing:
            completed.append(
                {
                    "method": method,
                    "regime": regime,
                    "request_index": request_index,
                    "sample_id": sample_id,
                    "context_tokens": context_tokens,
                    "chunk_start": chunk_start,
                    "chunk_end": chunk_end,
                    "layer_name": f"synthetic_fp16_layer.{layer_index}",
                    "layer_index": layer_index,
                    "query_tokens": query_tokens,
                    "sum_abs_error": 0.0,
                    "num_elements": template_numel,
                    "attention_output_mae": 0.0,
                    "max_abs_error": 0.0,
                    "synthetic_fp16_layer": True,
                }
            )
    return completed


def aggregate_figure5(
    raw_rows: list[dict[str, Any]],
    *,
    num_hidden_layers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    completed = _complete_fp16_layers(
        raw_rows, num_hidden_layers=num_hidden_layers
    )

    layer_keys: set[tuple[str, str, int, int, int, int, int]] = set()
    for row in completed:
        key = (
            str(row["method"]),
            str(row["regime"]),
            int(row["request_index"]),
            int(row["sample_id"]),
            int(row["context_tokens"]),
            int(row["chunk_end"]),
            int(row["layer_index"]),
        )
        if key in layer_keys:
            raise ValueError(f"Duplicate layer observation: {key}")
        layer_keys.add(key)

    grouped: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in completed:
        grouped[
            (
                str(row["method"]),
                str(row["regime"]),
                int(row["sample_id"]),
                int(row["context_tokens"]),
            )
        ].append(row)

    sample_rows: list[dict[str, Any]] = []
    expected_layers = set(range(num_hidden_layers))
    for (method, regime, sample_id, context_tokens), rows in sorted(grouped.items()):
        by_chunk: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_chunk[(int(row["chunk_start"]), int(row["chunk_end"]))].append(row)

        chunk_ranges = sorted(by_chunk)
        if not chunk_ranges or chunk_ranges[0][0] != 0:
            raise ValueError(
                f"Chunk coverage does not start at zero for "
                f"{(method, regime, sample_id, context_tokens)}: {chunk_ranges[:3]}"
            )
        for previous, current in zip(chunk_ranges, chunk_ranges[1:]):
            if previous[1] != current[0]:
                raise ValueError(
                    f"Non-contiguous chunk coverage for "
                    f"{(method, regime, sample_id, context_tokens)}: "
                    f"{previous} then {current}"
                )
        if chunk_ranges[-1][1] != context_tokens:
            raise ValueError(
                f"Chunk coverage ends at {chunk_ranges[-1][1]}, expected "
                f"{context_tokens} for {(method, regime, sample_id)}"
            )

        for chunk_range, chunk_rows in by_chunk.items():
            indices = {int(row["layer_index"]) for row in chunk_rows}
            if indices != expected_layers:
                raise ValueError(
                    f"Expected {num_hidden_layers} layers for "
                    f"{(method, regime, sample_id, context_tokens, chunk_range)}, "
                    f"got {sorted(indices)}"
                )

        sum_abs = sum(float(row["sum_abs_error"]) for row in rows)
        numel = sum(int(row["num_elements"]) for row in rows)
        sample_rows.append(
            {
                "method": method,
                "regime": regime,
                "sample_id": sample_id,
                "context_tokens": context_tokens,
                "num_layers": num_hidden_layers,
                "num_chunks": len(chunk_ranges),
                "sum_abs_error": sum_abs,
                "num_elements": numel,
                "attention_output_mae": sum_abs / numel,
            }
        )

    curve_values: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    sample_lookup: dict[tuple[str, str, int, int], float] = {}
    for row in sample_rows:
        method = str(row["method"])
        regime = str(row["regime"])
        context = int(row["context_tokens"])
        sample_id = int(row["sample_id"])
        value = float(row["attention_output_mae"])
        curve_values[(method, regime, context)].append(value)
        sample_lookup[(method, regime, sample_id, context)] = value

    summary_rows: list[dict[str, Any]] = []
    for (method, regime, context), values in sorted(curve_values.items()):
        stats = _summary(values)
        summary_rows.append(
            {
                "method": method,
                "regime": regime,
                "context_tokens": context,
                "num_samples": stats["num_samples"],
                "mean_attention_output_mae": stats["mean"],
                "sample_std": stats["sample_std"],
                "sem": stats["sem"],
                "ci95_low": stats["ci95_low"],
                "ci95_high": stats["ci95_high"],
            }
        )

    contexts = sorted({int(row["context_tokens"]) for row in sample_rows})
    sample_ids = sorted({int(row["sample_id"]) for row in sample_rows})
    difference_rows: list[dict[str, Any]] = []
    for context in contexts:
        regime_gaps: dict[str, list[float]] = {regime: [] for regime in REGIME_ORDER}
        paired_by_sample: dict[int, dict[str, float]] = defaultdict(dict)
        for regime in REGIME_ORDER:
            for sample_id in sample_ids:
                kvarn_key = ("kvarn", regime, sample_id, context)
                turbo_key = ("turboquant", regime, sample_id, context)
                if kvarn_key not in sample_lookup or turbo_key not in sample_lookup:
                    raise ValueError(
                        f"Missing paired observation at context={context}, sample={sample_id}, regime={regime}"
                    )
                gap = sample_lookup[kvarn_key] - sample_lookup[turbo_key]
                regime_gaps[regime].append(gap)
                paired_by_sample[sample_id][regime] = gap
            stats = _summary(regime_gaps[regime])
            difference_rows.append(
                {
                    "panel": "method_gap",
                    "regime": regime,
                    "context_tokens": context,
                    "num_samples": stats["num_samples"],
                    "mean_difference": stats["mean"],
                    "sample_std": stats["sample_std"],
                    "sem": stats["sem"],
                    "ci95_low": stats["ci95_low"],
                    "ci95_high": stats["ci95_high"],
                    "definition": "MAE_KVarN - MAE_TurboQuant",
                }
            )

        static_minus_accumulated = [
            paired_by_sample[sample_id]["static"]
            - paired_by_sample[sample_id]["accumulated"]
            for sample_id in sample_ids
        ]
        stats = _summary(static_minus_accumulated)
        difference_rows.append(
            {
                "panel": "regime_gap",
                "regime": "static_minus_accumulated",
                "context_tokens": context,
                "num_samples": stats["num_samples"],
                "mean_difference": stats["mean"],
                "sample_std": stats["sample_std"],
                "sem": stats["sem"],
                "ci95_low": stats["ci95_low"],
                "ci95_high": stats["ci95_high"],
                "definition": "(KVarN-TurboQuant)_static - (KVarN-TurboQuant)_accumulated",
            }
        )

    completed.sort(
        key=lambda row: (
            str(row["method"]),
            str(row["regime"]),
            int(row["sample_id"]),
            int(row["context_tokens"]),
            int(row["chunk_end"]),
            int(row["layer_index"]),
        )
    )
    return completed, sample_rows, summary_rows + difference_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
