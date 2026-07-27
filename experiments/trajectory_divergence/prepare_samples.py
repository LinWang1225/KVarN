#!/usr/bin/env python3
"""Prepare a fixed MATH-500 sample manifest shared by all experiment modes."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("prepare_samples")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a deterministic subset from a Hugging Face dataset and save "
            "it as a manifest shared by FP16 and KVarN runs."
        )
    )
    parser.add_argument(
        "--dataset-name",
        default="HuggingFaceH4/MATH-500",
        help="Hugging Face dataset name.",
    )
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing manifest instead of reusing it.",
    )
    return parser.parse_args()


def first_present(row: dict[str, Any], names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None:
            return value
    return default


def stable_sample_id(dataset_name: str, split: str, index: int, problem: str) -> str:
    payload = f"{dataset_name}\n{split}\n{index}\n{problem}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if args.output.exists() and not args.overwrite:
        LOGGER.info("Manifest already exists; reusing %s", args.output)
        return

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Missing dependency 'datasets'. Install requirements.txt first."
        ) from exc

    LOGGER.info(
        "Loading dataset=%s config=%s split=%s",
        args.dataset_name,
        args.dataset_config,
        args.dataset_split,
    )
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.dataset_split,
    )

    total = len(dataset)
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.num_samples > total:
        raise ValueError(
            f"Requested {args.num_samples} samples, but dataset has only {total}."
        )

    rng = random.Random(args.seed)
    indices = rng.sample(range(total), args.num_samples)

    samples: list[dict[str, Any]] = []
    for order, index in enumerate(indices):
        row = dict(dataset[index])
        problem = first_present(row, ("problem", "question", "prompt", "input"))
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(
                f"Could not find a non-empty problem/question field at dataset index {index}."
            )

        reference_answer = first_present(
            row,
            ("answer", "reference_answer", "target", "final_answer"),
        )
        solution = first_present(row, ("solution", "reference_solution", "rationale"))
        source_id = first_present(row, ("unique_id", "id", "sample_id"), default=index)

        samples.append(
            {
                "selection_order": order,
                "dataset_index": index,
                "sample_id": stable_sample_id(
                    args.dataset_name,
                    args.dataset_split,
                    index,
                    problem,
                ),
                "source_id": source_id,
                "problem": problem,
                "reference_answer": reference_answer,
                "reference_solution": solution,
                "subject": row.get("subject"),
                "level": row.get("level"),
            }
        )

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "dataset_size": total,
        "num_samples": len(samples),
        "seed": args.seed,
        "samples": samples,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temp_path.replace(args.output)
    LOGGER.info("Saved %d selected samples to %s", len(samples), args.output)


if __name__ == "__main__":
    main()
