#!/usr/bin/env python3
"""Audit exact outputs and repeat determinism for FP16 and KVarN runs.

This stage-1 analysis consumes the four generation files produced by
``run_full_experiment.sh`` with ``NUM_REPEATS=2``. It does not rerun the model.
It preserves each run's exact vLLM output text and token IDs, detects whether
Qwen's visible thinking trace is present, and diagnoses FP16/KVarN self-
divergence and cross-mode divergence sample by sample.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import logging
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

LOGGER = logging.getLogger("audit_repeat_outputs")
DEFAULT_RUNS = ("fp16_run1", "fp16_run2", "kvarn_run1", "kvarn_run2")
PREFIX_BUDGETS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preserve and inspect exact outputs/tokens from FP16×2 and KVarN×2 "
            "trajectory runs."
        )
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fp16-run1", default="fp16_run1")
    parser.add_argument("--fp16-run2", default="fp16_run2")
    parser.add_argument("--kvarn-run1", default="kvarn_run1")
    parser.add_argument("--kvarn-run2", default="kvarn_run2")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "Tokenizer path/ID. If omitted, infer it from experiment_config.json "
            "or the generation records."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--window-radius", type=int, default=24)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument(
        "--skip-tokenizer-load",
        action="store_true",
        help="Write IDs/text without decoding token windows or marker token positions.",
    )
    parser.add_argument(
        "--include-token-strings",
        action="store_true",
        help=(
            "Also store tokenizer.convert_ids_to_tokens output for every token. "
            "This can make raw token JSON files much larger."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing audit directory.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
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
            sample_key = str(sample_id)
            previous = records.get(sample_key)
            # Resume files can contain a failed attempt followed by a successful one.
            if previous is None or previous.get("error") is not None or record.get("error") is None:
                records[sample_key] = record
    return records


def safe_int_list(value: Iterable[Any] | None) -> list[int]:
    if value is None:
        return []
    result: list[int] = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_ids(ids: Sequence[int]) -> str:
    payload = ",".join(str(value) for value in ids)
    return sha256_text(payload)


def longest_common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return index
    return min(len(left), len(right))


def trajectory_relation(left: Sequence[int], right: Sequence[int]) -> dict[str, Any]:
    lcp = longest_common_prefix(left, right)
    minimum = min(len(left), len(right))
    if lcp < minimum:
        relation = "token_mismatch"
        first_step: int | None = lcp
    elif len(left) != len(right):
        relation = "length_only"
        first_step = minimum
    else:
        relation = "identical"
        first_step = None
    return {
        "relation": relation,
        "diverged": relation != "identical",
        "first_divergence_step": first_step,
        "longest_common_prefix": lcp,
        "left_length": len(left),
        "right_length": len(right),
        "length_difference": len(right) - len(left),
        "length_ratio": len(right) / max(1, len(left)),
    }


def find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None
    limit = len(haystack) - len(needle) + 1
    first = needle[0]
    for start in range(limit):
        if haystack[start] == first and list(haystack[start : start + len(needle)]) == list(needle):
            return start
    return None


def decode_ids(tokenizer: Any | None, ids: Sequence[int]) -> str | None:
    if tokenizer is None:
        return None
    try:
        return str(
            tokenizer.decode(
                list(ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
    except TypeError:
        return str(tokenizer.decode(list(ids), skip_special_tokens=False))


def encode_marker(tokenizer: Any | None, marker: str) -> list[int]:
    if tokenizer is None:
        return []
    try:
        return safe_int_list(tokenizer.encode(marker, add_special_tokens=False))
    except Exception:
        LOGGER.exception("Could not encode thinking marker %r", marker)
        return []


def infer_tokenizer_name(output_root: Path, run_names: Sequence[str], records: dict[str, dict[str, dict[str, Any]]]) -> str | None:
    for run_name in run_names:
        config_path = output_root / run_name / "experiment_config.json"
        if config_path.exists():
            config = read_json(config_path)
            tokenizer_name = config.get("tokenizer") or config.get("model")
            if tokenizer_name:
                return str(tokenizer_name)
    for run_name in run_names:
        for record in records[run_name].values():
            tokenizer_name = record.get("tokenizer") or record.get("model")
            if tokenizer_name:
                return str(tokenizer_name)
    return None


def load_tokenizer(name: str | None, trust_remote_code: bool) -> Any | None:
    if not name:
        LOGGER.warning("No tokenizer path/ID could be inferred; token-window decoding is disabled.")
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        LOGGER.warning("transformers is unavailable; token-window decoding is disabled.")
        return None
    LOGGER.info("Loading tokenizer for output audit: %s", name)
    return AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)


def split_visible_output(prompt_text: str, output_text: str, decoded_text: str | None) -> dict[str, Any]:
    """Split Qwen's visible reasoning trace from its final answer conservatively.

    Qwen chat templates may place ``<think>`` in the prompt, while ``</think>``
    is generated. The exact vLLM ``output_text`` is always preserved separately;
    this function only provides an audit-oriented split and never mutates it.
    """

    prompt_has_open = "<think>" in prompt_text
    prompt_has_close = "</think>" in prompt_text
    visible = output_text
    source = "output_text"
    # Some decoding paths can hide special markers from candidate.text. Prefer the
    # decoded token sequence only when it reveals a marker absent from output_text.
    if decoded_text and (
        ("<think>" in decoded_text and "<think>" not in visible)
        or ("</think>" in decoded_text and "</think>" not in visible)
    ):
        visible = decoded_text
        source = "decoded_output_token_ids"

    open_index = visible.find("<think>")
    close_index = visible.find("</think>")
    reasoning_text: str | None
    final_text: str | None
    status: str

    if open_index >= 0:
        reasoning_start = open_index + len("<think>")
        if close_index >= reasoning_start:
            reasoning_text = visible[reasoning_start:close_index]
            final_text = visible[close_index + len("</think>") :]
            status = "complete_open_and_close_in_output"
        else:
            reasoning_text = visible[reasoning_start:]
            final_text = None
            status = "open_in_output_without_close"
    elif prompt_has_open:
        if close_index >= 0:
            reasoning_text = visible[:close_index]
            final_text = visible[close_index + len("</think>") :]
            status = "open_in_prompt_close_in_output"
        else:
            reasoning_text = visible
            final_text = None
            status = "open_in_prompt_without_close"
    elif close_index >= 0:
        reasoning_text = visible[:close_index]
        final_text = visible[close_index + len("</think>") :]
        status = "close_in_output_without_detected_open"
    else:
        reasoning_text = None
        final_text = visible
        status = "no_thinking_markers_detected"

    return {
        "split_source": source,
        "status": status,
        "prompt_contains_think_open": prompt_has_open,
        "prompt_contains_think_close": prompt_has_close,
        "output_contains_think_open": "<think>" in output_text,
        "output_contains_think_close": "</think>" in output_text,
        "decoded_contains_think_open": bool(decoded_text and "<think>" in decoded_text),
        "decoded_contains_think_close": bool(decoded_text and "</think>" in decoded_text),
        "reasoning_text": reasoning_text,
        "final_answer_text": final_text,
        "reasoning_characters": len(reasoning_text) if reasoning_text is not None else None,
        "final_answer_characters": len(final_text) if final_text is not None else None,
    }


def token_thinking_boundaries(
    tokenizer: Any | None,
    prompt_ids: Sequence[int],
    output_ids: Sequence[int],
) -> dict[str, Any]:
    open_ids = encode_marker(tokenizer, "<think>")
    close_ids = encode_marker(tokenizer, "</think>")
    prompt_open = find_subsequence(prompt_ids, open_ids)
    output_open = find_subsequence(output_ids, open_ids)
    output_close = find_subsequence(output_ids, close_ids)

    reasoning_start: int | None = None
    reasoning_end: int | None = None
    final_start: int | None = None
    status: str

    if output_open is not None:
        reasoning_start = output_open + len(open_ids)
        if output_close is not None and output_close >= reasoning_start:
            reasoning_end = output_close
            final_start = output_close + len(close_ids)
            status = "complete_open_and_close_in_output"
        else:
            reasoning_end = len(output_ids)
            status = "open_in_output_without_close"
    elif prompt_open is not None:
        reasoning_start = 0
        if output_close is not None:
            reasoning_end = output_close
            final_start = output_close + len(close_ids)
            status = "open_in_prompt_close_in_output"
        else:
            reasoning_end = len(output_ids)
            status = "open_in_prompt_without_close"
    elif output_close is not None:
        reasoning_start = 0
        reasoning_end = output_close
        final_start = output_close + len(close_ids)
        status = "close_in_output_without_detected_open"
    else:
        status = "no_thinking_marker_tokens_detected"
        final_start = 0

    return {
        "status": status,
        "think_open_token_ids": open_ids,
        "think_close_token_ids": close_ids,
        "prompt_think_open_start": prompt_open,
        "output_think_open_start": output_open,
        "output_think_close_start": output_close,
        "reasoning_token_start": reasoning_start,
        "reasoning_token_end": reasoning_end,
        "reasoning_tokens": (
            reasoning_end - reasoning_start
            if reasoning_start is not None and reasoning_end is not None
            else None
        ),
        "final_answer_token_start": final_start,
        "final_answer_tokens": (
            len(output_ids) - final_start if final_start is not None else None
        ),
    }


def classify_step_region(step: int | None, boundary: dict[str, Any]) -> str | None:
    if step is None:
        return None
    reasoning_start = boundary.get("reasoning_token_start")
    reasoning_end = boundary.get("reasoning_token_end")
    final_start = boundary.get("final_answer_token_start")
    close_start = boundary.get("output_think_close_start")
    close_ids = boundary.get("think_close_token_ids") or []
    if close_start is not None and close_start <= step < close_start + len(close_ids):
        return "thinking_boundary"
    if reasoning_start is not None and reasoning_end is not None:
        if reasoning_start <= step < reasoning_end:
            return "thinking"
    if final_start is not None and step >= final_start:
        return "final_answer"
    return "unknown"


def percentile(values: Sequence[int | float], q: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def safe_filename(sample_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._")
    return safe or sha256_text(sample_id)[:16]


def write_text(path: Path, value: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(value or "")
        if value and not value.endswith("\n"):
            handle.write("\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")


def relation_with_window(
    left_record: dict[str, Any],
    right_record: dict[str, Any],
    left_boundary: dict[str, Any],
    right_boundary: dict[str, Any],
    tokenizer: Any | None,
    radius: int,
    block_size: int,
) -> dict[str, Any]:
    left_ids = safe_int_list(left_record.get("output_token_ids"))
    right_ids = safe_int_list(right_record.get("output_token_ids"))
    relation = trajectory_relation(left_ids, right_ids)
    step = relation["first_divergence_step"]
    if step is None:
        window_start = None
        window_end = None
        left_window: list[int] = []
        right_window: list[int] = []
    else:
        window_start = max(0, step - radius)
        window_end = min(max(len(left_ids), len(right_ids)), step + radius + 1)
        left_window = left_ids[window_start:window_end]
        right_window = right_ids[window_start:window_end]
    input_tokens = left_record.get("input_tokens")
    absolute_position = (
        int(input_tokens) + step
        if step is not None and input_tokens is not None
        else None
    )
    return {
        **relation,
        "first_divergence_block": step // block_size if step is not None else None,
        "offset_in_output_block": step % block_size if step is not None else None,
        "absolute_divergence_position": absolute_position,
        "absolute_divergence_block": (
            absolute_position // block_size if absolute_position is not None else None
        ),
        "absolute_offset_in_block": (
            absolute_position % block_size if absolute_position is not None else None
        ),
        "left_divergence_region": classify_step_region(step, left_boundary),
        "right_divergence_region": classify_step_region(step, right_boundary),
        "window_start": window_start,
        "window_end": window_end,
        "left_token_ids_window": left_window,
        "right_token_ids_window": right_window,
        "left_decoded_window": decode_ids(tokenizer, left_window),
        "right_decoded_window": decode_ids(tokenizer, right_window),
    }


def run_audit(
    run_name: str,
    record: dict[str, Any],
    tokenizer: Any | None,
    include_token_strings: bool,
) -> dict[str, Any]:
    prompt_text = str(record.get("prompt_text") or "")
    output_text = str(record.get("output_text") or "")
    input_ids = safe_int_list(record.get("input_token_ids"))
    output_ids = safe_int_list(record.get("output_token_ids"))
    decoded_text = decode_ids(tokenizer, output_ids)
    visible_split = split_visible_output(prompt_text, output_text, decoded_text)
    token_boundary = token_thinking_boundaries(tokenizer, input_ids, output_ids)
    token_strings: list[str] | None = None
    if tokenizer is not None and include_token_strings:
        try:
            token_strings = [str(value) for value in tokenizer.convert_ids_to_tokens(output_ids)]
        except Exception:
            LOGGER.exception("Could not convert token IDs to token strings for %s", run_name)
    return {
        "run_name": run_name,
        "mode": record.get("mode"),
        "sample_id": str(record.get("sample_id")),
        "error": record.get("error"),
        "prompt_sha256": record.get("prompt_sha256") or sha256_text(prompt_text),
        "input_tokens": record.get("input_tokens", len(input_ids)),
        "input_token_ids_sha256": sha256_ids(input_ids),
        "output_tokens": record.get("output_tokens", len(output_ids)),
        "output_token_ids_sha256": sha256_ids(output_ids),
        "output_text_sha256": sha256_text(output_text),
        "finish_reason": record.get("finish_reason"),
        "stop_reason": record.get("stop_reason"),
        "extracted_answer": record.get("extracted_answer"),
        "normalized_extracted_answer": record.get("normalized_extracted_answer"),
        "approx_correct": record.get("approx_correct"),
        "enable_thinking_requested": record.get("enable_thinking_requested"),
        "enable_thinking_argument_applied": record.get(
            "enable_thinking_argument_applied"
        ),
        "prompt_text": prompt_text,
        "input_token_ids": input_ids,
        "output_token_ids": output_ids,
        "output_text": output_text,
        "decoded_from_output_token_ids": decoded_text,
        "token_strings": token_strings,
        "visible_thinking_split": visible_split,
        "token_thinking_boundaries": token_boundary,
    }


def markdown_escape(value: Any) -> str:
    text = str(value) if value is not None else "N/A"
    return text.replace("|", "\\|").replace("\n", " ")


def html_pre(value: str | None, limit: int | None = None) -> str:
    text = value or ""
    if limit is not None and len(text) > limit:
        text = text[:limit] + "\n...[truncated in report; see raw .txt file]..."
    return f"<pre>{html.escape(text)}</pre>"


def write_sample_report(
    path: Path,
    sample_id: str,
    records: dict[str, dict[str, Any]],
    audits: dict[str, dict[str, Any]],
    relations: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    representative = records[next(iter(records))]
    lines = [
        f"# Output audit: `{sample_id}`",
        "",
        f"- Dataset index: `{representative.get('dataset_index')}`",
        f"- Reference answer: `{markdown_escape(representative.get('reference_answer'))}`",
        f"- Problem: {markdown_escape(representative.get('problem'))}",
        "",
        "## Pairwise trajectory relations",
        "",
        "| Pair | Relation | First step | Region (left/right) | LCP | Lengths |",
        "|---|---|---:|---|---:|---:|",
    ]
    for pair_name, relation in relations.items():
        lines.append(
            "| {pair} | {relation} | {step} | {left_region}/{right_region} | "
            "{lcp} | {left_len}/{right_len} |".format(
                pair=pair_name,
                relation=relation["relation"],
                step=markdown_escape(relation.get("first_divergence_step")),
                left_region=markdown_escape(relation.get("left_divergence_region")),
                right_region=markdown_escape(relation.get("right_divergence_region")),
                lcp=relation["longest_common_prefix"],
                left_len=relation["left_length"],
                right_len=relation["right_length"],
            )
        )
    lines.extend(
        [
            "",
            "## Per-run output audit",
            "",
            "| Run | Output tokens | Finish | Think split status | Reasoning tokens | Final tokens | Answer | Correct |",
            "|---|---:|---|---|---:|---:|---|---|",
        ]
    )
    for run_name, audit in audits.items():
        boundary = audit["token_thinking_boundaries"]
        lines.append(
            "| {run} | {tokens} | {finish} | {status} | {reasoning} | {final} | "
            "{answer} | {correct} |".format(
                run=run_name,
                tokens=audit.get("output_tokens"),
                finish=markdown_escape(audit.get("finish_reason")),
                status=markdown_escape(boundary.get("status")),
                reasoning=markdown_escape(boundary.get("reasoning_tokens")),
                final=markdown_escape(boundary.get("final_answer_tokens")),
                answer=markdown_escape(audit.get("extracted_answer")),
                correct=markdown_escape(audit.get("approx_correct")),
            )
        )

    lines.extend(["", "## Exact files", ""])
    safe_id = safe_filename(sample_id)
    for run_name in records:
        rel_output = Path("raw_outputs") / run_name / f"{safe_id}.output.txt"
        rel_reasoning = Path("raw_outputs") / run_name / f"{safe_id}.reasoning.txt"
        rel_final = Path("raw_outputs") / run_name / f"{safe_id}.final_answer.txt"
        rel_tokens = Path("raw_tokens") / run_name / f"{safe_id}.tokens.json"
        lines.append(
            f"- `{run_name}`: `{rel_output}`; `{rel_reasoning}`; `{rel_final}`; `{rel_tokens}`"
        )

    lines.extend(["", "## Divergence windows", ""])
    for pair_name, relation in relations.items():
        lines.append(f"### {pair_name}")
        lines.append("")
        if relation["relation"] == "identical":
            lines.append("The token sequences are identical.")
        else:
            lines.append(
                f"First divergence step: `{relation.get('first_divergence_step')}`; "
                f"absolute position: `{relation.get('absolute_divergence_position')}`."
            )
            lines.append("")
            lines.append("Left window:")
            lines.append(html_pre(relation.get("left_decoded_window") or str(relation.get("left_token_ids_window"))))
            lines.append("")
            lines.append("Right window:")
            lines.append(html_pre(relation.get("right_decoded_window") or str(relation.get("right_token_ids_window"))))
        lines.append("")

    lines.extend(["## Visible outputs (preview)", ""])
    for run_name, audit in audits.items():
        lines.append(f"### {run_name}")
        lines.append("")
        lines.append(html_pre(audit.get("output_text"), limit=4000))
        lines.append("")
    write_text(path, "\n".join(lines))


def relation_summary(relations: list[dict[str, Any]]) -> dict[str, Any]:
    first_steps = [
        int(relation["first_divergence_step"])
        for relation in relations
        if relation.get("first_divergence_step") is not None
    ]
    region_counts = Counter(
        relation.get("left_divergence_region") or "none" for relation in relations
    )
    prefix_rates: dict[str, dict[str, Any]] = {}
    for budget in PREFIX_BUDGETS:
        count = sum(
            1
            for relation in relations
            if relation.get("first_divergence_step") is not None
            and int(relation["first_divergence_step"]) < budget
        )
        prefix_rates[str(budget)] = {
            "diverged_count": count,
            "total": len(relations),
            "rate": count / len(relations) if relations else None,
        }
    return {
        "valid_pairs": len(relations),
        "diverged_count": sum(bool(value.get("diverged")) for value in relations),
        "divergence_rate": (
            sum(bool(value.get("diverged")) for value in relations) / len(relations)
            if relations
            else None
        ),
        "relation_counts": dict(Counter(value["relation"] for value in relations)),
        "mean_first_divergence_step": statistics.fmean(first_steps) if first_steps else None,
        "median_first_divergence_step": statistics.median(first_steps) if first_steps else None,
        "first_divergence_step_percentiles": {
            "p25": percentile(first_steps, 0.25),
            "p50": percentile(first_steps, 0.50),
            "p75": percentile(first_steps, 0.75),
            "p90": percentile(first_steps, 0.90),
        },
        "divergence_region_counts": dict(region_counts),
        "prefix_divergence": prefix_rates,
    }


def write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Repeat/output audit summary",
        "",
        f"- Samples present in all four runs: **{summary['complete_sample_count']}**",
        f"- Prompt hash mismatches across four runs: **{summary['prompt_hash_mismatch_count']}**",
        f"- Input token mismatches across four runs: **{summary['input_token_mismatch_count']}**",
        f"- Runs with detected complete thinking boundary: **{summary['complete_thinking_run_count']}/{summary['total_run_records']}**",
        "",
        "## Pairwise divergence",
        "",
        "| Pair | Diverged | Rate | Median first step | P90 | Thinking | Boundary | Final answer | Unknown |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pair_name, values in summary["pairs"].items():
        regions = values["divergence_region_counts"]
        lines.append(
            "| {pair} | {count}/{total} | {rate} | {median} | {p90} | {thinking} | "
            "{boundary} | {final} | {unknown} |".format(
                pair=pair_name,
                count=values["diverged_count"],
                total=values["valid_pairs"],
                rate=(
                    f"{values['divergence_rate']:.4f}"
                    if values["divergence_rate"] is not None
                    else "N/A"
                ),
                median=markdown_escape(values.get("median_first_divergence_step")),
                p90=markdown_escape(values["first_divergence_step_percentiles"].get("p90")),
                thinking=regions.get("thinking", 0),
                boundary=regions.get("thinking_boundary", 0),
                final=regions.get("final_answer", 0),
                unknown=regions.get("unknown", 0) + regions.get("none", 0),
            )
        )
    lines.extend(
        [
            "",
            "## Prefix divergence rates",
            "",
            "This table explains why a short 3-sample/512-token smoke test can be stable "
            "while longer generation later self-diverges.",
            "",
            "| Pair | 128 | 256 | 512 | 1K | 2K | 4K | 8K | 16K |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for pair_name, values in summary["pairs"].items():
        prefix = values["prefix_divergence"]
        cells = []
        for budget in PREFIX_BUDGETS:
            item = prefix[str(budget)]
            cells.append(f"{item['diverged_count']}/{item['total']}")
        lines.append(f"| {pair_name} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Manual inspection",
            "",
            "- Exact vLLM output text: `raw_outputs/<run>/<sample>.output.txt`",
            "- Extracted visible reasoning: `raw_outputs/<run>/<sample>.reasoning.txt`",
            "- Extracted final-answer section: `raw_outputs/<run>/<sample>.final_answer.txt`",
            "- Exact input/output token IDs: `raw_tokens/<run>/<sample>.tokens.json`",
            "- Four-run comparison report: `samples/<sample>.md`",
            "",
            "The reasoning/final split is an audit aid based on `<think>`/`</think>` "
            "markers in the prompt, output text, and token sequence. The untouched raw "
            "output remains the source of truth.",
        ]
    )
    write_text(path, "\n".join(lines))


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if args.window_radius < 0:
        raise ValueError("--window-radius must be non-negative")
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")

    output_root = args.output_root.resolve()
    output_dir = (args.output_dir or (output_root / "output_audit")).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} already contains files. Use --overwrite or choose a new directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_names = (
        args.fp16_run1,
        args.fp16_run2,
        args.kvarn_run1,
        args.kvarn_run2,
    )
    run_records: dict[str, dict[str, dict[str, Any]]] = {}
    for run_name in run_names:
        path = output_root / run_name / "generations.jsonl"
        run_records[run_name] = load_jsonl(path)
        LOGGER.info("Loaded %d records from %s", len(run_records[run_name]), path)

    tokenizer_name = args.tokenizer or infer_tokenizer_name(output_root, run_names, run_records)
    tokenizer = None if args.skip_tokenizer_load else load_tokenizer(
        tokenizer_name, args.trust_remote_code
    )

    all_sample_ids = set().union(*(set(records) for records in run_records.values()))
    complete_sample_ids = sorted(
        sample_id
        for sample_id in all_sample_ids
        if all(sample_id in run_records[run_name] for run_name in run_names)
        and all(
            run_records[run_name][sample_id].get("error") is None
            for run_name in run_names
        )
    )
    missing_by_run = {
        run_name: sorted(all_sample_ids - set(records))
        for run_name, records in run_records.items()
    }
    failed_by_run = {
        run_name: sorted(
            sample_id for sample_id, record in records.items() if record.get("error") is not None
        )
        for run_name, records in run_records.items()
    }
    if not complete_sample_ids:
        raise RuntimeError("No sample is successful and present in all four runs")

    pair_definitions = {
        "fp16_self_run1_vs_run2": (args.fp16_run1, args.fp16_run2),
        "kvarn_self_run1_vs_run2": (args.kvarn_run1, args.kvarn_run2),
        "cross_fp16_run1_vs_kvarn_run1": (args.fp16_run1, args.kvarn_run1),
        "cross_fp16_run2_vs_kvarn_run2": (args.fp16_run2, args.kvarn_run2),
    }
    pair_relations: dict[str, list[dict[str, Any]]] = {
        pair_name: [] for pair_name in pair_definitions
    }
    audit_rows: list[dict[str, Any]] = []
    prompt_hash_mismatch_count = 0
    input_token_mismatch_count = 0
    complete_thinking_run_count = 0
    total_run_records = 0

    per_sample_jsonl_path = output_dir / "per_sample_audit.jsonl"
    with per_sample_jsonl_path.open("w", encoding="utf-8") as jsonl_handle:
        for sample_id in complete_sample_ids:
            records = {
                run_name: run_records[run_name][sample_id] for run_name in run_names
            }
            audits = {
                run_name: run_audit(
                    run_name,
                    records[run_name],
                    tokenizer,
                    args.include_token_strings,
                )
                for run_name in run_names
            }
            total_run_records += len(audits)
            complete_thinking_statuses = {
                "complete_open_and_close_in_output",
                "open_in_prompt_close_in_output",
                "close_in_output_without_detected_open",
            }
            complete_thinking_run_count += sum(
                audit["token_thinking_boundaries"]["status"]
                in complete_thinking_statuses
                or audit["visible_thinking_split"]["status"]
                in complete_thinking_statuses
                for audit in audits.values()
            )

            prompt_hashes = {str(audit.get("prompt_sha256")) for audit in audits.values()}
            input_hashes = {str(audit.get("input_token_ids_sha256")) for audit in audits.values()}
            prompt_match = len(prompt_hashes) == 1
            input_match = len(input_hashes) == 1
            prompt_hash_mismatch_count += int(not prompt_match)
            input_token_mismatch_count += int(not input_match)

            relations: dict[str, dict[str, Any]] = {}
            for pair_name, (left_run, right_run) in pair_definitions.items():
                relation = relation_with_window(
                    records[left_run],
                    records[right_run],
                    audits[left_run]["token_thinking_boundaries"],
                    audits[right_run]["token_thinking_boundaries"],
                    tokenizer,
                    args.window_radius,
                    args.block_size,
                )
                relation["left_run"] = left_run
                relation["right_run"] = right_run
                relations[pair_name] = relation
                pair_relations[pair_name].append(relation)

            safe_id = safe_filename(sample_id)
            prompt_text = str(records[args.fp16_run1].get("prompt_text") or "")
            write_text(output_dir / "prompts" / f"{safe_id}.prompt.txt", prompt_text)
            for run_name, audit in audits.items():
                raw_base = output_dir / "raw_outputs" / run_name
                token_base = output_dir / "raw_tokens" / run_name
                split = audit["visible_thinking_split"]
                write_text(raw_base / f"{safe_id}.output.txt", audit.get("output_text"))
                write_text(
                    raw_base / f"{safe_id}.decoded_with_special_tokens.txt",
                    audit.get("decoded_from_output_token_ids"),
                )
                write_text(
                    raw_base / f"{safe_id}.reasoning.txt",
                    split.get("reasoning_text"),
                )
                write_text(
                    raw_base / f"{safe_id}.final_answer.txt",
                    split.get("final_answer_text"),
                )
                write_json(token_base / f"{safe_id}.tokens.json", audit)

            sample_payload = {
                "schema_version": 1,
                "sample_id": sample_id,
                "dataset_index": records[args.fp16_run1].get("dataset_index"),
                "problem": records[args.fp16_run1].get("problem"),
                "reference_answer": records[args.fp16_run1].get("reference_answer"),
                "all_prompt_hashes_match": prompt_match,
                "all_input_token_ids_match": input_match,
                "runs": {
                    run_name: {
                        key: audits[run_name].get(key)
                        for key in (
                            "mode",
                            "run_name",
                            "output_tokens",
                            "output_token_ids_sha256",
                            "output_text_sha256",
                            "finish_reason",
                            "stop_reason",
                            "extracted_answer",
                            "normalized_extracted_answer",
                            "approx_correct",
                            "enable_thinking_requested",
                            "enable_thinking_argument_applied",
                            "visible_thinking_split",
                            "token_thinking_boundaries",
                        )
                    }
                    for run_name in run_names
                },
                "relations": relations,
            }
            jsonl_handle.write(json.dumps(sample_payload, ensure_ascii=False, default=str) + "\n")
            write_sample_report(
                output_dir / "samples" / f"{safe_id}.md",
                sample_id,
                records,
                audits,
                relations,
                output_dir,
            )

            row = {
                "sample_id": sample_id,
                "dataset_index": records[args.fp16_run1].get("dataset_index"),
                "all_prompt_hashes_match": prompt_match,
                "all_input_token_ids_match": input_match,
            }
            for run_name, audit in audits.items():
                boundary = audit["token_thinking_boundaries"]
                row.update(
                    {
                        f"{run_name}_output_tokens": audit.get("output_tokens"),
                        f"{run_name}_finish_reason": audit.get("finish_reason"),
                        f"{run_name}_think_status": boundary.get("status"),
                        f"{run_name}_reasoning_tokens": boundary.get("reasoning_tokens"),
                        f"{run_name}_final_answer_tokens": boundary.get("final_answer_tokens"),
                        f"{run_name}_extracted_answer": audit.get("extracted_answer"),
                        f"{run_name}_approx_correct": audit.get("approx_correct"),
                    }
                )
            for pair_name, relation in relations.items():
                row.update(
                    {
                        f"{pair_name}_relation": relation["relation"],
                        f"{pair_name}_first_step": relation.get("first_divergence_step"),
                        f"{pair_name}_left_region": relation.get("left_divergence_region"),
                        f"{pair_name}_right_region": relation.get("right_divergence_region"),
                        f"{pair_name}_lcp": relation.get("longest_common_prefix"),
                    }
                )
            audit_rows.append(row)

    csv_path = output_dir / "per_sample_audit.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(audit_rows)

    summary = {
        "schema_version": 1,
        "output_root": str(output_root),
        "output_dir": str(output_dir),
        "tokenizer": tokenizer_name,
        "run_names": list(run_names),
        "all_sample_count": len(all_sample_ids),
        "complete_sample_count": len(complete_sample_ids),
        "complete_sample_ids": complete_sample_ids,
        "missing_by_run": missing_by_run,
        "failed_by_run": failed_by_run,
        "prompt_hash_mismatch_count": prompt_hash_mismatch_count,
        "input_token_mismatch_count": input_token_mismatch_count,
        "complete_thinking_run_count": complete_thinking_run_count,
        "total_run_records": total_run_records,
        "pairs": {
            pair_name: relation_summary(relations)
            for pair_name, relations in pair_relations.items()
        },
    }
    write_json(output_dir / "summary.json", summary)
    write_summary_markdown(output_dir / "summary.md", summary)
    LOGGER.info("Output audit complete: %s", output_dir)


if __name__ == "__main__":
    main()
