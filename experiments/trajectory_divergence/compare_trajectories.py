#!/usr/bin/env python3
"""Compare FP16 and KVarN output token trajectories sample by sample."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

LOGGER = logging.getLogger("compare_trajectories")

LENGTH_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("[0,512)", 0, 512),
    ("[512,1024)", 512, 1024),
    ("[1024,2048)", 1024, 2048),
    ("[2048,4096)", 2048, 4096),
    ("[4096,+inf)", 4096, None),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare token trajectories from two generation JSONL files."
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-repeat", type=Path, default=None)
    parser.add_argument("--candidate-repeat", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "Tokenizer path/ID for decoding divergence windows. If omitted, the "
            "model field in the reference JSONL is used."
        ),
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--window-radius", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument(
        "--skip-window-decoding",
        action="store_true",
        help="Do not initialize a Transformers tokenizer for local window text.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
            sample_id = record.get("sample_id")
            if sample_id is None:
                LOGGER.warning("Ignoring record without sample_id at %s:%d", path, line_number)
                continue
            # Resume runs may contain a failed record followed by a successful retry.
            previous = records.get(str(sample_id))
            if previous is None or previous.get("error") is not None:
                records[str(sample_id)] = record
            elif record.get("error") is None:
                records[str(sample_id)] = record
    return records


def longest_common_prefix(left: list[int], right: list[int]) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


def trajectory_relation(left: list[int], right: list[int]) -> dict[str, Any]:
    lcp = longest_common_prefix(left, right)
    if lcp < min(len(left), len(right)):
        divergence_type = "token_mismatch"
        first_divergence_step: int | None = lcp
    elif len(left) != len(right):
        divergence_type = "length_only"
        first_divergence_step = lcp
    else:
        divergence_type = "identical"
        first_divergence_step = None
    return {
        "common_prefix_length": lcp,
        "divergence_type": divergence_type,
        "first_divergence_step": first_divergence_step,
        "diverged": divergence_type != "identical",
    }


def safe_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def correctness_transition(reference: Any, candidate: Any) -> str | None:
    ref = safe_bool(reference)
    cand = safe_bool(candidate)
    if ref is None or cand is None:
        return None
    return f"{'correct' if ref else 'wrong'}_to_{'correct' if cand else 'wrong'}"


def make_decoder(
    tokenizer_name: str | None,
    trust_remote_code: bool,
) -> Callable[[list[int]], str | None]:
    if not tokenizer_name:
        return lambda _: None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        LOGGER.warning("Could not load tokenizer %s: %s", tokenizer_name, exc)
        return lambda _: None

    def decode(token_ids: list[int]) -> str | None:
        try:
            return tokenizer.decode(token_ids, skip_special_tokens=False)
        except Exception:
            return None

    return decode


def compare_pair(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    block_size: int,
    window_radius: int,
    decode: Callable[[list[int]], str | None],
) -> dict[str, Any]:
    reference_ids = [int(value) for value in reference.get("output_token_ids", [])]
    candidate_ids = [int(value) for value in candidate.get("output_token_ids", [])]
    relation = trajectory_relation(reference_ids, candidate_ids)
    step = relation["first_divergence_step"]

    if step is None:
        window_start = None
        window_end = None
        reference_window: list[int] = []
        candidate_window: list[int] = []
    else:
        window_start = max(0, step - window_radius)
        window_end = min(max(len(reference_ids), len(candidate_ids)), step + window_radius + 1)
        reference_window = reference_ids[window_start:window_end]
        candidate_window = candidate_ids[window_start:window_end]

    reference_length = len(reference_ids)
    candidate_length = len(candidate_ids)
    reference_correct = safe_bool(reference.get("approx_correct"))
    candidate_correct = safe_bool(candidate.get("approx_correct"))
    min_length = min(reference_length, candidate_length)

    normalized_ref = reference.get("normalized_extracted_answer")
    normalized_cand = candidate.get("normalized_extracted_answer")
    same_extracted_answer = (
        normalized_ref is not None
        and normalized_cand is not None
        and normalized_ref == normalized_cand
    )

    return {
        "sample_id": reference.get("sample_id"),
        "source_id": reference.get("source_id"),
        "dataset_index": reference.get("dataset_index"),
        "selection_order": reference.get("selection_order"),
        "subject": reference.get("subject"),
        "level": reference.get("level"),
        "prompt_sha256_reference": reference.get("prompt_sha256"),
        "prompt_sha256_candidate": candidate.get("prompt_sha256"),
        "same_prompt_sha256": reference.get("prompt_sha256") == candidate.get("prompt_sha256"),
        "reference_mode": reference.get("mode"),
        "candidate_mode": candidate.get("mode"),
        "reference_run_name": reference.get("run_name"),
        "candidate_run_name": candidate.get("run_name"),
        **relation,
        "first_divergence_block": step // block_size if step is not None else None,
        "offset_in_block": step % block_size if step is not None else None,
        "lcp_ratio": relation["common_prefix_length"] / max(1, min_length),
        "reference_output_tokens": reference_length,
        "candidate_output_tokens": candidate_length,
        "fp16_output_tokens": reference_length,
        "kvarn_output_tokens": candidate_length,
        "length_difference": candidate_length - reference_length,
        "length_ratio": candidate_length / max(1, reference_length),
        "reference_finish_reason": reference.get("finish_reason"),
        "candidate_finish_reason": candidate.get("finish_reason"),
        "fp16_finish_reason": reference.get("finish_reason"),
        "kvarn_finish_reason": candidate.get("finish_reason"),
        "reference_stop_reason": reference.get("stop_reason"),
        "candidate_stop_reason": candidate.get("stop_reason"),
        "reference_approx_correct": reference_correct,
        "candidate_approx_correct": candidate_correct,
        "fp16_approx_correct": reference_correct,
        "kvarn_approx_correct": candidate_correct,
        "correctness_transition": correctness_transition(reference_correct, candidate_correct),
        "reference_extracted_answer": reference.get("extracted_answer"),
        "candidate_extracted_answer": candidate.get("extracted_answer"),
        "same_extracted_answer": same_extracted_answer,
        "same_output_text": reference.get("output_text") == candidate.get("output_text"),
        "same_token_ids": reference_ids == candidate_ids,
        "divergence_window_start": window_start,
        "divergence_window_end": window_end,
        "reference_token_ids_window": reference_window,
        "candidate_token_ids_window": candidate_window,
        "fp16_token_ids_window": reference_window,
        "kvarn_token_ids_window": candidate_window,
        "reference_decoded_window": decode(reference_window) if reference_window else None,
        "candidate_decoded_window": decode(candidate_window) if candidate_window else None,
        "fp16_decoded_window": decode(reference_window) if reference_window else None,
        "kvarn_decoded_window": decode(candidate_window) if candidate_window else None,
    }


def mean_or_none(values: Iterable[float | int]) -> float | None:
    values_list = list(values)
    return statistics.fmean(values_list) if values_list else None


def median_or_none(values: Iterable[float | int]) -> float | None:
    values_list = list(values)
    return statistics.median(values_list) if values_list else None


def percentile(values: Iterable[float | int], probability: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def accuracy(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [record[field] for record in records if isinstance(record.get(field), bool)]
    correct = sum(bool(value) for value in values)
    return {
        "correct": correct,
        "evaluated": len(values),
        "accuracy": correct / len(values) if values else None,
    }


def self_divergence_summary(
    first: dict[str, dict[str, Any]],
    second: dict[str, dict[str, Any]] | None,
) -> dict[str, Any] | None:
    if second is None:
        return None
    common_ids = sorted(set(first).intersection(second))
    valid = 0
    diverged = 0
    divergent_ids: list[str] = []
    for sample_id in common_ids:
        left = first[sample_id]
        right = second[sample_id]
        if left.get("error") is not None or right.get("error") is not None:
            continue
        valid += 1
        relation = trajectory_relation(
            [int(value) for value in left.get("output_token_ids", [])],
            [int(value) for value in right.get("output_token_ids", [])],
        )
        if relation["diverged"]:
            diverged += 1
            divergent_ids.append(sample_id)
    return {
        "valid_pairs": valid,
        "diverged_count": diverged,
        "self_divergence_rate": diverged / valid if valid else None,
        "divergent_sample_ids": divergent_ids,
    }


def bucket_for_length(length: int) -> str:
    for label, lower, upper in LENGTH_BUCKETS:
        if length >= lower and (upper is None or length < upper):
            return label
    raise AssertionError(f"Unhandled length: {length}")


def summarize(
    comparisons: list[dict[str, Any]],
    total_sample_ids: set[str],
    missing_reference: list[str],
    missing_candidate: list[str],
    failed_reference: list[str],
    failed_candidate: list[str],
    reference_self: dict[str, Any] | None,
    candidate_self: dict[str, Any] | None,
) -> dict[str, Any]:
    diverged = [record for record in comparisons if record["diverged"]]
    divergence_steps = [
        int(record["first_divergence_step"])
        for record in diverged
        if record["first_divergence_step"] is not None
    ]
    length_ratios = [float(record["length_ratio"]) for record in comparisons]
    length_differences = [int(record["length_difference"]) for record in comparisons]

    buckets: dict[str, dict[str, Any]] = {}
    for label, _, _ in LENGTH_BUCKETS:
        subset = [
            record
            for record in comparisons
            if bucket_for_length(int(record["fp16_output_tokens"])) == label
        ]
        subset_diverged = [record for record in subset if record["diverged"]]
        subset_steps = [
            int(record["first_divergence_step"])
            for record in subset_diverged
            if record["first_divergence_step"] is not None
        ]
        buckets[label] = {
            "sample_count": len(subset),
            "divergence_count": len(subset_diverged),
            "divergence_rate": len(subset_diverged) / len(subset) if subset else None,
            "median_first_divergence_step": median_or_none(subset_steps),
            "mean_length_ratio": mean_or_none(
                float(record["length_ratio"]) for record in subset
            ),
            "correct_to_wrong_count": sum(
                record.get("correctness_transition") == "correct_to_wrong"
                for record in subset
            ),
        }

    transition_counts = Counter(
        record.get("correctness_transition") or "unavailable" for record in comparisons
    )
    reference_finish = Counter(
        str(record.get("reference_finish_reason")) for record in comparisons
    )
    candidate_finish = Counter(
        str(record.get("candidate_finish_reason")) for record in comparisons
    )

    return {
        "schema_version": 1,
        "total_samples": len(total_sample_ids),
        "valid_pairs": len(comparisons),
        "missing_reference_count": len(missing_reference),
        "missing_reference_sample_ids": missing_reference,
        "missing_candidate_count": len(missing_candidate),
        "missing_candidate_sample_ids": missing_candidate,
        "failed_reference_count": len(failed_reference),
        "failed_reference_sample_ids": failed_reference,
        "failed_candidate_count": len(failed_candidate),
        "failed_candidate_sample_ids": failed_candidate,
        "prompt_hash_mismatch_count": sum(
            not record["same_prompt_sha256"] for record in comparisons
        ),
        "identical_count": sum(record["divergence_type"] == "identical" for record in comparisons),
        "diverged_count": len(diverged),
        "divergence_rate": len(diverged) / len(comparisons) if comparisons else None,
        "token_mismatch_count": sum(
            record["divergence_type"] == "token_mismatch" for record in comparisons
        ),
        "length_only_count": sum(
            record["divergence_type"] == "length_only" for record in comparisons
        ),
        "mean_first_divergence_step": mean_or_none(divergence_steps),
        "median_first_divergence_step": median_or_none(divergence_steps),
        "first_divergence_step_percentiles": {
            "p25": percentile(divergence_steps, 0.25),
            "p50": percentile(divergence_steps, 0.50),
            "p75": percentile(divergence_steps, 0.75),
            "p90": percentile(divergence_steps, 0.90),
        },
        "mean_lcp_ratio": mean_or_none(
            float(record["lcp_ratio"]) for record in comparisons
        ),
        "median_lcp_ratio": median_or_none(
            float(record["lcp_ratio"]) for record in comparisons
        ),
        "mean_fp16_output_tokens": mean_or_none(
            int(record["fp16_output_tokens"]) for record in comparisons
        ),
        "mean_kvarn_output_tokens": mean_or_none(
            int(record["kvarn_output_tokens"]) for record in comparisons
        ),
        "mean_length_difference": mean_or_none(length_differences),
        "median_length_difference": median_or_none(length_differences),
        "mean_length_ratio": mean_or_none(length_ratios),
        "fraction_length_ratio_gt_1_1": (
            sum(value > 1.1 for value in length_ratios) / len(length_ratios)
            if length_ratios
            else None
        ),
        "fraction_length_ratio_gt_1_5": (
            sum(value > 1.5 for value in length_ratios) / len(length_ratios)
            if length_ratios
            else None
        ),
        "fraction_length_ratio_gt_2_0": (
            sum(value > 2.0 for value in length_ratios) / len(length_ratios)
            if length_ratios
            else None
        ),
        "fp16_approx_accuracy": accuracy(comparisons, "fp16_approx_correct"),
        "kvarn_approx_accuracy": accuracy(comparisons, "kvarn_approx_correct"),
        "correctness_transition_counts": dict(transition_counts),
        "correct_to_wrong_count": transition_counts.get("correct_to_wrong", 0),
        "wrong_to_correct_count": transition_counts.get("wrong_to_correct", 0),
        "reference_finish_reason_counts": dict(reference_finish),
        "candidate_finish_reason_counts": dict(candidate_finish),
        "length_buckets": buckets,
        "fp16_self_divergence": reference_self,
        "kvarn_self_divergence": candidate_self,
    }


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_comparisons(output_dir: Path, comparisons: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "per_sample_comparison.jsonl"
    csv_path = output_dir / "per_sample_comparison.csv"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in comparisons:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    fieldnames = list(comparisons[0].keys()) if comparisons else ["sample_id"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in comparisons:
            writer.writerow({key: csv_value(value) for key, value in record.items()})


def format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Trajectory divergence summary",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total sample IDs | {summary['total_samples']} |",
        f"| Valid FP16/KVarN pairs | {summary['valid_pairs']} |",
        f"| Diverged | {summary['diverged_count']} |",
        f"| Divergence rate | {format_number(summary['divergence_rate'])} |",
        f"| Identical | {summary['identical_count']} |",
        f"| Token mismatch | {summary['token_mismatch_count']} |",
        f"| Length-only mismatch | {summary['length_only_count']} |",
        f"| Median first divergence step | {format_number(summary['median_first_divergence_step'], 1)} |",
        f"| P90 first divergence step | {format_number(summary['first_divergence_step_percentiles']['p90'], 1)} |",
        f"| Mean FP16 output tokens | {format_number(summary['mean_fp16_output_tokens'], 1)} |",
        f"| Mean KVarN output tokens | {format_number(summary['mean_kvarn_output_tokens'], 1)} |",
        f"| Mean length ratio | {format_number(summary['mean_length_ratio'])} |",
        f"| Correct→wrong | {summary['correct_to_wrong_count']} |",
        f"| Prompt hash mismatches | {summary['prompt_hash_mismatch_count']} |",
        "",
        "## Divergence by FP16 output length",
        "",
        "| FP16 output tokens | Samples | Diverged | Rate | Median first step | Mean length ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, values in summary["length_buckets"].items():
        lines.append(
            "| {label} | {sample_count} | {divergence_count} | {rate} | {median} | {ratio} |".format(
                label=label,
                sample_count=values["sample_count"],
                divergence_count=values["divergence_count"],
                rate=format_number(values["divergence_rate"]),
                median=format_number(values["median_first_divergence_step"], 1),
                ratio=format_number(values["mean_length_ratio"]),
            )
        )

    lines.extend(["", "## Determinism controls", ""])
    for label, key in (("FP16", "fp16_self_divergence"), ("KVarN", "kvarn_self_divergence")):
        values = summary.get(key)
        if values is None:
            lines.append(f"- {label}: repeat run not supplied.")
        else:
            lines.append(
                f"- {label}: {values['diverged_count']}/{values['valid_pairs']} self-diverged "
                f"(rate={format_number(values['self_divergence_rate'])})."
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")
    if args.window_radius < 0:
        raise ValueError("--window-radius cannot be negative")

    reference = load_jsonl(args.reference)
    candidate = load_jsonl(args.candidate)
    reference_repeat = load_jsonl(args.reference_repeat) if args.reference_repeat else None
    candidate_repeat = load_jsonl(args.candidate_repeat) if args.candidate_repeat else None

    all_ids = set(reference).union(candidate)
    missing_reference = sorted(set(candidate) - set(reference))
    missing_candidate = sorted(set(reference) - set(candidate))
    failed_reference = sorted(
        sample_id for sample_id, record in reference.items() if record.get("error") is not None
    )
    failed_candidate = sorted(
        sample_id for sample_id, record in candidate.items() if record.get("error") is not None
    )

    common_valid_ids = sorted(
        sample_id
        for sample_id in set(reference).intersection(candidate)
        if reference[sample_id].get("error") is None
        and candidate[sample_id].get("error") is None
    )

    tokenizer_name = args.tokenizer
    if not tokenizer_name and common_valid_ids:
        tokenizer_name = reference[common_valid_ids[0]].get("model")
    decode = (
        (lambda _: None)
        if args.skip_window_decoding
        else make_decoder(tokenizer_name, args.trust_remote_code)
    )

    comparisons = [
        compare_pair(
            reference=reference[sample_id],
            candidate=candidate[sample_id],
            block_size=args.block_size,
            window_radius=args.window_radius,
            decode=decode,
        )
        for sample_id in common_valid_ids
    ]
    comparisons.sort(key=lambda record: (record.get("selection_order") is None, record.get("selection_order", 0)))

    reference_self = self_divergence_summary(reference, reference_repeat)
    candidate_self = self_divergence_summary(candidate, candidate_repeat)
    summary = summarize(
        comparisons=comparisons,
        total_sample_ids=all_ids,
        missing_reference=missing_reference,
        missing_candidate=missing_candidate,
        failed_reference=failed_reference,
        failed_candidate=failed_candidate,
        reference_self=reference_self,
        candidate_self=candidate_self,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_comparisons(args.output_dir, comparisons)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    write_summary_markdown(args.output_dir / "summary.md", summary)

    LOGGER.info(
        "Compared %d valid pairs; %d diverged (rate=%s)",
        summary["valid_pairs"],
        summary["diverged_count"],
        format_number(summary["divergence_rate"]),
    )
    if summary["prompt_hash_mismatch_count"]:
        LOGGER.warning(
            "%d valid pairs used different prompts. Do not attribute those differences to KV quantization.",
            summary["prompt_hash_mismatch_count"],
        )
    for label, values in (("FP16", reference_self), ("KVarN", candidate_self)):
        if values and values["diverged_count"]:
            LOGGER.warning(
                "%s repeat runs self-diverged on %d/%d samples. Cross-mode results include runtime nondeterminism.",
                label,
                values["diverged_count"],
                values["valid_pairs"],
            )


if __name__ == "__main__":
    main()
