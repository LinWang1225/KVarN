#!/usr/bin/env python3
"""Prepare a fixed HumanEval manifest shared by FP16 and KVarN runs."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("prepare_humaneval_samples")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare OpenAI HumanEval tasks for trajectory/system experiments."
    )
    parser.add_argument("--dataset-name", default="openai/openai_humaneval")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=164,
        help="Use 164 for the full HumanEval set. Samples are kept in dataset order.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_string(row: dict[str, Any], key: str, index: int) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"HumanEval row {index} is missing non-empty field {key!r}")
    return value


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if args.output.exists() and not args.overwrite:
        LOGGER.info("Manifest already exists; reusing %s", args.output)
        return
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency 'datasets'. Install trajectory_divergence requirements.") from exc

    LOGGER.info(
        "Loading dataset=%s config=%s split=%s",
        args.dataset_name,
        args.dataset_config,
        args.dataset_split,
    )
    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    total = len(dataset)
    if args.num_samples > total:
        raise ValueError(f"Requested {args.num_samples} samples, dataset contains {total}")

    samples: list[dict[str, Any]] = []
    for index in range(args.num_samples):
        row = dict(dataset[index])
        task_id = require_string(row, "task_id", index)
        prompt = require_string(row, "prompt", index)
        canonical_solution = require_string(row, "canonical_solution", index)
        test = require_string(row, "test", index)
        entry_point = require_string(row, "entry_point", index)
        samples.append(
            {
                "selection_order": index,
                "dataset_index": index,
                "sample_id": task_id,
                "source_id": task_id,
                # Generic fields retained for compatibility with the existing
                # trajectory comparison/audit scripts.
                "problem": prompt,
                "reference_answer": None,
                "reference_solution": canonical_solution,
                "subject": "code",
                "level": None,
                # HumanEval-native fields.
                "task_id": task_id,
                "prompt": prompt,
                "canonical_solution": canonical_solution,
                "test": test,
                "entry_point": entry_point,
            }
        )

    manifest = {
        "schema_version": 1,
        "task_type": "humaneval",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "dataset_size": total,
        "num_samples": len(samples),
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(args.output)
    LOGGER.info("Saved %d HumanEval tasks to %s", len(samples), args.output)


if __name__ == "__main__":
    main()
