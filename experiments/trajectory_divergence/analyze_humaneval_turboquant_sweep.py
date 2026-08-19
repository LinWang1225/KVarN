#!/usr/bin/env python3
"""Aggregate HumanEval TurboQuant precision-sweep results.

The analysis intentionally emphasizes both signed inflation and absolute
trajectory perturbation. Lower precision can make some requests longer and
others shorter, so mean signed TIR alone can hide a stronger quantization
effect through cancellation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

METHOD_META: dict[str, dict[str, Any]] = {
    "turboquant_k8v4": {
        "severity_rank": 1,
        "key_bits": 8,
        "value_bits": 4,
        "nominal_kv_bits": 12,
        "norm_correction": False,
        "compression_note": "2.6x",
        "repo_ppl_note": "+1.17%",
    },
    "turboquant_4bit_nc": {
        "severity_rank": 2,
        "key_bits": 4,
        "value_bits": 4,
        "nominal_kv_bits": 8,
        "norm_correction": True,
        "compression_note": "3.8x",
        "repo_ppl_note": "+2.71%",
    },
    "turboquant_k3v4_nc": {
        "severity_rank": 3,
        "key_bits": 3,
        "value_bits": 4,
        "nominal_kv_bits": 7,
        "norm_correction": True,
        "compression_note": "~3.5x",
        "repo_ppl_note": "+10.63%",
    },
    "turboquant_3bit_nc": {
        "severity_rank": 4,
        "key_bits": 3,
        "value_bits": 3,
        "nominal_kv_bits": 6,
        "norm_correction": True,
        "compression_note": "4.9x",
        "repo_ppl_note": "+20.59%",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-root", type=Path, required=True)
    p.add_argument("--seeds", required=True, help="Comma-separated seeds")
    p.add_argument("--methods", required=True, help="Comma-separated TurboQuant presets")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--fp16-source-pattern",
        default="",
        help="Optional existing FP16 generations path containing literal {seed}.",
    )
    return p.parse_args()


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row.get("sample_id")
            if sid is None:
                continue
            sid = str(sid)
            prev = out.get(sid)
            # Prefer a successful later record when --resume appended a retry.
            if prev is None or prev.get("error") is not None or row.get("error") is None:
                out[sid] = row
    return out


def number(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def integer(v: Any) -> int | None:
    x = number(v)
    return None if x is None else int(x)


def boolean(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    t = str(v).strip().lower()
    if t in {"1", "true", "yes"}:
        return True
    if t in {"0", "false", "no"}:
        return False
    return None


def ratio(candidate: float | int | None, reference: float | int | None) -> float | None:
    if candidate is None or reference is None or float(reference) == 0:
        return None
    return (float(candidate) - float(reference)) / float(reference)


def percentile(values: Iterable[float], q: float) -> float | None:
    xs = sorted(float(x) for x in values)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def summarize(values: Iterable[float | int | None]) -> dict[str, Any]:
    xs = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not xs:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "p90": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(xs),
        "mean": statistics.fmean(xs),
        "std": statistics.stdev(xs) if len(xs) > 1 else 0.0,
        "median": statistics.median(xs),
        "p90": percentile(xs, 0.90),
        "p95": percentile(xs, 0.95),
        "min": min(xs),
        "max": max(xs),
    }


def is_valid(row: dict[str, Any] | None) -> bool:
    return bool(row) and row.get("error") is None and integer(row.get("output_tokens")) is not None


def is_censored(row: dict[str, Any]) -> bool:
    if str(row.get("finish_reason") or "").lower() == "length":
        return True
    n = integer(row.get("output_tokens"))
    cap = integer(row.get("max_tokens"))
    return n is not None and cap is not None and n >= cap


def transition(a: bool | None, b: bool | None) -> str:
    if a is None or b is None:
        return "unavailable"
    return f"{'pass' if a else 'fail'}_to_{'pass' if b else 'fail'}"


def fp16_path(args: argparse.Namespace, seed: int) -> Path:
    if args.fp16_source_pattern:
        if "{seed}" not in args.fp16_source_pattern:
            raise ValueError("--fp16-source-pattern must contain literal {seed}")
        return Path(args.fp16_source_pattern.replace("{seed}", str(seed)))
    return args.experiment_root / f"seed_{seed}" / "fp16" / "generations.jsonl"


def candidate_path(root: Path, seed: int, method: str) -> Path:
    return root / f"seed_{seed}" / method / "generations.jsonl"


def build_pair(seed: int, method: str, fp: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
    fp_tokens = integer(fp.get("output_tokens"))
    q_tokens = integer(cand.get("output_tokens"))
    assert fp_tokens is not None and q_tokens is not None
    fp_censored = is_censored(fp)
    q_censored = is_censored(cand)
    fp_pass = boolean(fp.get("execution_passed"))
    q_pass = boolean(cand.get("execution_passed"))
    tir = ratio(q_tokens, fp_tokens)

    fp_think = integer(fp.get("thinking_tokens"))
    q_think = integer(cand.get("thinking_tokens"))
    fp_final = integer(fp.get("final_tokens"))
    q_final = integer(cand.get("final_tokens"))
    delta = q_tokens - fp_tokens
    delta_think = None if fp_think is None or q_think is None else q_think - fp_think
    delta_final = None if fp_final is None or q_final is None else q_final - fp_final
    think_share = None
    if delta > 0 and delta_think is not None:
        think_share = delta_think / delta

    fp_latency = number(fp.get("latency_seconds"))
    q_latency = number(cand.get("latency_seconds"))
    fp_decode = number(fp.get("decode_time_seconds"))
    q_decode = number(cand.get("decode_time_seconds"))
    fp_tpot = number(fp.get("tpot_seconds"))
    q_tpot = number(cand.get("tpot_seconds"))
    fp_ttft = number(fp.get("ttft_seconds"))
    q_ttft = number(cand.get("ttft_seconds"))

    return {
        "seed": seed,
        "method": method,
        "severity_rank": METHOD_META[method]["severity_rank"],
        "sample_id": str(fp.get("sample_id")),
        "task_id": str(fp.get("task_id") or fp.get("sample_id")),
        "prompt_hash_match": fp.get("prompt_sha256") == cand.get("prompt_sha256"),
        "fp16_output_tokens": fp_tokens,
        "candidate_output_tokens": q_tokens,
        "delta_tokens": delta,
        "tir": tir,
        "abs_tir": abs(tir) if tir is not None else None,
        "fp16_censored": fp_censored,
        "candidate_censored": q_censored,
        "either_censored": fp_censored or q_censored,
        "fp16_pass": fp_pass,
        "candidate_pass": q_pass,
        "correctness_transition": transition(fp_pass, q_pass),
        "fp16_thinking_tokens": fp_think,
        "candidate_thinking_tokens": q_think,
        "delta_thinking_tokens": delta_think,
        "fp16_final_tokens": fp_final,
        "candidate_final_tokens": q_final,
        "delta_final_tokens": delta_final,
        "thinking_contribution_to_extra_tokens": think_share,
        "fp16_latency_seconds": fp_latency,
        "candidate_latency_seconds": q_latency,
        "e2e_lir": ratio(q_latency, fp_latency),
        "fp16_decode_seconds": fp_decode,
        "candidate_decode_seconds": q_decode,
        "decode_lir": ratio(q_decode, fp_decode),
        "fp16_tpot_seconds": fp_tpot,
        "candidate_tpot_seconds": q_tpot,
        "tpot_lir": ratio(q_tpot, fp_tpot),
        "fp16_ttft_seconds": fp_ttft,
        "candidate_ttft_seconds": q_ttft,
        "ttft_lir": ratio(q_ttft, fp_ttft),
        "requested_kv_cache_dtype": cand.get("requested_kv_cache_dtype"),
        "resolved_kv_cache_dtype": cand.get("resolved_kv_cache_dtype"),
        "backend_verified": boolean(cand.get("backend_verified")),
    }


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and order[j][1] == order[i][1]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k][0]] = rank
        i = j
    return ranks


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(x) != len(y):
        return None
    mx, my = statistics.fmean(x), statistics.fmean(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    denom = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    if denom == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denom


def spearman(x: list[float], y: list[float]) -> float | None:
    return pearson(average_ranks(x), average_ranks(y))


def monotone_non_decreasing(xs: list[float], tol: float = 0.0) -> bool:
    return all(xs[i + 1] + tol >= xs[i] for i in range(len(xs) - 1))


def fmt_pct(v: float | None, digits: int = 2) -> str:
    return "NA" if v is None else f"{100*v:.{digits}f}%"


def fmt_num(v: float | None, digits: int = 2) -> str:
    return "NA" if v is None else f"{v:.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    unknown = [m for m in methods if m not in METHOD_META]
    if unknown:
        raise ValueError(f"Unknown TurboQuant methods: {unknown}")
    methods.sort(key=lambda m: METHOD_META[m]["severity_rank"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fp_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in seeds:
        path = fp16_path(args, seed)
        if not path.exists():
            raise FileNotFoundError(f"Missing FP16 generations: {path}")
        fp_by_seed[seed] = load_jsonl(path)

    pair_rows: list[dict[str, Any]] = []
    method_seed_stats: dict[str, list[dict[str, Any]]] = defaultdict(list)
    method_task_pairs: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    method_summaries: dict[str, dict[str, Any]] = {}

    for method in methods:
        all_rows: list[dict[str, Any]] = []
        for seed in seeds:
            cpath = candidate_path(args.experiment_root, seed, method)
            if not cpath.exists():
                raise FileNotFoundError(f"Missing candidate generations: {cpath}")
            cand = load_jsonl(cpath)
            fp = fp_by_seed[seed]
            common = sorted(set(fp).intersection(cand))
            rows: list[dict[str, Any]] = []
            for sid in common:
                if not is_valid(fp[sid]) or not is_valid(cand[sid]):
                    continue
                row = build_pair(seed, method, fp[sid], cand[sid])
                rows.append(row)
                pair_rows.append(row)
                method_task_pairs[method][row["task_id"]].append(row)
            all_rows.extend(rows)

            unc = [r for r in rows if not r["either_censored"]]
            fp_mean_all = statistics.fmean(r["fp16_output_tokens"] for r in rows) if rows else None
            q_mean_all = statistics.fmean(r["candidate_output_tokens"] for r in rows) if rows else None
            fp_acc_values = [r["fp16_pass"] for r in rows if isinstance(r["fp16_pass"], bool)]
            q_acc_values = [r["candidate_pass"] for r in rows if isinstance(r["candidate_pass"], bool)]
            method_seed_stats[method].append({
                "seed": seed,
                "valid_pairs": len(rows),
                "uncensored_pairs": len(unc),
                "fp16_mean_output_tokens": fp_mean_all,
                "candidate_mean_output_tokens": q_mean_all,
                "output_change": ratio(q_mean_all, fp_mean_all),
                "fp16_pass_at_1": (sum(fp_acc_values) / len(fp_acc_values)) if fp_acc_values else None,
                "candidate_pass_at_1": (sum(q_acc_values) / len(q_acc_values)) if q_acc_values else None,
                "tir_mean": summarize(r["tir"] for r in unc)["mean"],
                "tir_p95": summarize(r["tir"] for r in unc)["p95"],
                "abs_tir_mean": summarize(r["abs_tir"] for r in unc)["mean"],
                "abs_tir_p95": summarize(r["abs_tir"] for r in unc)["p95"],
                "decode_lir_p95": summarize(r["decode_lir"] for r in unc)["p95"],
                "tpot_lir_mean": summarize(r["tpot_lir"] for r in unc)["mean"],
            })

        unc = [r for r in all_rows if not r["either_censored"]]
        tir = [r["tir"] for r in unc if r["tir"] is not None]
        abs_tir = [r["abs_tir"] for r in unc if r["abs_tir"] is not None]
        seed_stats = method_seed_stats[method]
        fp_mode_means = [r["fp16_mean_output_tokens"] for r in seed_stats if r["fp16_mean_output_tokens"] is not None]
        q_mode_means = [r["candidate_mean_output_tokens"] for r in seed_stats if r["candidate_mean_output_tokens"] is not None]
        fp_acc = [r["fp16_pass_at_1"] for r in seed_stats if r["fp16_pass_at_1"] is not None]
        q_acc = [r["candidate_pass_at_1"] for r in seed_stats if r["candidate_pass_at_1"] is not None]
        fp_mode_mean = statistics.fmean(fp_mode_means) if fp_mode_means else None
        q_mode_mean = statistics.fmean(q_mode_means) if q_mode_means else None
        fp_acc_mean = statistics.fmean(fp_acc) if fp_acc else None
        q_acc_mean = statistics.fmean(q_acc) if q_acc else None

        severe = [r for r in unc if r["tir"] is not None and r["tir"] > 0.50]
        severe_split = [r for r in severe if r["delta_thinking_tokens"] is not None and r["delta_tokens"] > 0]
        severe_extra = sum(r["delta_tokens"] for r in severe_split)
        severe_extra_think = sum(r["delta_thinking_tokens"] for r in severe_split)
        severe_passpass = [r for r in severe_split if r["correctness_transition"] == "pass_to_pass"]
        pp_extra = sum(r["delta_tokens"] for r in severe_passpass)
        pp_think = sum(r["delta_thinking_tokens"] for r in severe_passpass)

        robust_rows: list[dict[str, Any]] = []
        majority = len(seeds) // 2 + 1
        for task_id, rows in method_task_pairs[method].items():
            rows = sorted(rows, key=lambda r: r["seed"])
            unc_rows = [r for r in rows if not r["either_censored"]]
            all_unc = len(unc_rows) == len(seeds)
            task = {
                "method": method,
                "task_id": task_id,
                "all_seeds_uncensored": all_unc,
                "uncensored_seed_pairs": len(unc_rows),
                "median_tir": statistics.median([r["tir"] for r in unc_rows]) if unc_rows else None,
                "mean_abs_tir": statistics.fmean([r["abs_tir"] for r in unc_rows]) if unc_rows else None,
                "tir_gt_10_seed_count": sum(r["tir"] > 0.10 for r in unc_rows),
                "tir_gt_25_seed_count": sum(r["tir"] > 0.25 for r in unc_rows),
                "tir_gt_50_seed_count": sum(r["tir"] > 0.50 for r in unc_rows),
                "abs_tir_gt_25_seed_count": sum(r["abs_tir"] > 0.25 for r in unc_rows),
                "abs_tir_gt_50_seed_count": sum(r["abs_tir"] > 0.50 for r in unc_rows),
                "tir_lt_minus_10_seed_count": sum(r["tir"] < -0.10 for r in unc_rows),
            }
            robust_rows.append(task)

        all_unc_tasks = [r for r in robust_rows if r["all_seeds_uncensored"]]
        robust = {
            "all_seeds_uncensored_task_count": len(all_unc_tasks),
            "majority_seed_threshold": majority,
            "longer_gt_10_majority": sum(r["tir_gt_10_seed_count"] >= majority for r in all_unc_tasks),
            "longer_gt_25_majority": sum(r["tir_gt_25_seed_count"] >= majority for r in all_unc_tasks),
            "longer_gt_50_majority": sum(r["tir_gt_50_seed_count"] >= majority for r in all_unc_tasks),
            "abs_change_gt_25_majority": sum(r["abs_tir_gt_25_seed_count"] >= majority for r in all_unc_tasks),
            "abs_change_gt_50_majority": sum(r["abs_tir_gt_50_seed_count"] >= majority for r in all_unc_tasks),
            "longer_gt_50_all": sum(r["tir_gt_50_seed_count"] == len(seeds) for r in all_unc_tasks),
        }

        transitions: dict[str, int] = defaultdict(int)
        for r in all_rows:
            transitions[r["correctness_transition"]] += 1

        method_summaries[method] = {
            "method": method,
            **METHOD_META[method],
            "valid_pairs": len(all_rows),
            "uncensored_pairs": len(unc),
            "paper_style": {
                "fp16_output_tokens": summarize(fp_mode_means),
                "candidate_output_tokens": summarize(q_mode_means),
                "output_change": ratio(q_mode_mean, fp_mode_mean),
                "fp16_pass_at_1": summarize(fp_acc),
                "candidate_pass_at_1": summarize(q_acc),
                "accuracy_drop_points": ((fp_acc_mean - q_acc_mean) * 100) if fp_acc_mean is not None and q_acc_mean is not None else None,
            },
            "pooled_uncensored": {
                "tir": summarize(tir),
                "abs_tir": summarize(abs_tir),
                "e2e_lir": summarize(r["e2e_lir"] for r in unc),
                "decode_lir": summarize(r["decode_lir"] for r in unc),
                "tpot_lir": summarize(r["tpot_lir"] for r in unc),
                "tail": {
                    "tir_gt_10": sum(x > 0.10 for x in tir) / len(tir) if tir else None,
                    "tir_gt_25": sum(x > 0.25 for x in tir) / len(tir) if tir else None,
                    "tir_gt_50": sum(x > 0.50 for x in tir) / len(tir) if tir else None,
                    "tir_gt_100": sum(x > 1.00 for x in tir) / len(tir) if tir else None,
                    "tir_lt_minus_10": sum(x < -0.10 for x in tir) / len(tir) if tir else None,
                    "abs_tir_gt_25": sum(x > 0.25 for x in abs_tir) / len(abs_tir) if abs_tir else None,
                    "abs_tir_gt_50": sum(x > 0.50 for x in abs_tir) / len(abs_tir) if abs_tir else None,
                },
            },
            "severe_inflation": {
                "pair_count": len(severe),
                "thinking_split_count": len(severe_split),
                "extra_tokens": severe_extra,
                "extra_thinking_tokens": severe_extra_think,
                "thinking_share": severe_extra_think / severe_extra if severe_extra else None,
                "pass_to_pass_pair_count": len(severe_passpass),
                "pass_to_pass_thinking_share": pp_think / pp_extra if pp_extra else None,
            },
            "correctness_transitions": dict(transitions),
            "robust_across_seeds": robust,
            "per_seed": seed_stats,
            "per_task": robust_rows,
        }

    # Flat comparison table, ordered from highest to lowest precision.
    sweep_rows: list[dict[str, Any]] = []
    for method in methods:
        s = method_summaries[method]
        pooled = s["pooled_uncensored"]
        paper = s["paper_style"]
        robust = s["robust_across_seeds"]
        severe = s["severe_inflation"]
        sweep_rows.append({
            "severity_rank": s["severity_rank"],
            "method": method,
            "K_bits": s["key_bits"],
            "V_bits": s["value_bits"],
            "nominal_K_plus_V_bits": s["nominal_kv_bits"],
            "NC": s["norm_correction"],
            "compression_note": s["compression_note"],
            "repo_PPL_note": s["repo_ppl_note"],
            "mean_output_change": paper["output_change"],
            "accuracy_drop_points": paper["accuracy_drop_points"],
            "median_TIR": pooled["tir"]["median"],
            "mean_TIR": pooled["tir"]["mean"],
            "P95_TIR": pooled["tir"]["p95"],
            "mean_abs_TIR": pooled["abs_tir"]["mean"],
            "P95_abs_TIR": pooled["abs_tir"]["p95"],
            "TIR_gt_50_rate": pooled["tail"]["tir_gt_50"],
            "abs_TIR_gt_50_rate": pooled["tail"]["abs_tir_gt_50"],
            "P95_decode_LIR": pooled["decode_lir"]["p95"],
            "mean_TPOT_overhead": pooled["tpot_lir"]["mean"],
            "severe_thinking_share": severe["thinking_share"],
            "stable_gt50_majority_tasks": robust["longer_gt_50_majority"],
            "stable_abs_gt50_majority_tasks": robust["abs_change_gt_50_majority"],
            "all_seed_uncensored_tasks": robust["all_seeds_uncensored_task_count"],
        })

    # Aggregate monotonicity diagnostics: severity rank should correlate positively
    # with disturbance metrics if lower precision truly has a stronger effect.
    ranks = [float(r["severity_rank"]) for r in sweep_rows]
    monotonic_metrics = [
        "mean_abs_TIR",
        "P95_abs_TIR",
        "TIR_gt_50_rate",
        "abs_TIR_gt_50_rate",
        "accuracy_drop_points",
        "mean_TPOT_overhead",
    ]
    monotonicity: dict[str, Any] = {}
    for metric in monotonic_metrics:
        vals = [number(r[metric]) for r in sweep_rows]
        if all(v is not None for v in vals):
            clean = [float(v) for v in vals if v is not None]
            monotonicity[metric] = {
                "spearman_vs_severity_rank": spearman(ranks, clean),
                "non_decreasing": monotone_non_decreasing(clean),
                "values_high_to_low_precision": clean,
            }
        else:
            monotonicity[metric] = {
                "spearman_vs_severity_rank": None,
                "non_decreasing": None,
                "values_high_to_low_precision": vals,
            }

    # Task-level monotonicity of median |TIR| across precision, restricted to
    # tasks with all seeds uncensored for every method.
    task_maps = {
        method: {r["task_id"]: r for r in method_summaries[method]["per_task"]}
        for method in methods
    }
    common_tasks = set.intersection(*(set(m) for m in task_maps.values())) if task_maps else set()
    eligible_tasks: list[str] = []
    monotone_tasks = 0
    monotone_tasks_tol5 = 0
    task_spearmans: list[float] = []
    for task_id in sorted(common_tasks):
        rows = [task_maps[m][task_id] for m in methods]
        if not all(r["all_seeds_uncensored"] for r in rows):
            continue
        vals = [number(r["mean_abs_tir"]) for r in rows]
        if not all(v is not None for v in vals):
            continue
        xs = [float(v) for v in vals if v is not None]
        eligible_tasks.append(task_id)
        monotone_tasks += int(monotone_non_decreasing(xs))
        monotone_tasks_tol5 += int(monotone_non_decreasing(xs, tol=0.05))
        rho = spearman(ranks, xs)
        if rho is not None:
            task_spearmans.append(rho)

    task_level_monotonicity = {
        "eligible_task_count": len(eligible_tasks),
        "strict_non_decreasing_abs_tir_tasks": monotone_tasks,
        "non_decreasing_abs_tir_tasks_with_5pp_tolerance": monotone_tasks_tol5,
        "mean_task_spearman_severity_vs_abs_tir": statistics.fmean(task_spearmans) if task_spearmans else None,
    }

    # Flatten task summaries for inspection.
    task_rows = [
        row
        for method in methods
        for row in method_summaries[method]["per_task"]
    ]

    write_csv(args.output_dir / "precision_sweep.csv", sweep_rows)
    write_csv(args.output_dir / "per_pair_metrics.csv", pair_rows)
    write_csv(args.output_dir / "per_task_method.csv", task_rows)

    summary = {
        "experiment": "humaneval_turboquant_precision_sweep",
        "seeds": seeds,
        "methods_high_to_low_precision": methods,
        "method_summaries": method_summaries,
        "aggregate_monotonicity": monotonicity,
        "task_level_monotonicity": task_level_monotonicity,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# HumanEval TurboQuant precision sweep",
        "",
        "Methods are ordered from higher to lower nominal KV precision. Signed TIR and absolute TIR are both shown because longer/shorter trajectories can cancel in the mean.",
        "",
        "| Method | K/V | Mean output Δ | Median TIR | P95 TIR | Mean |TIR| | P95 |TIR| | TIR>50% | |TIR|>50% | Acc drop | Mean TPOT Δ | Stable >50% (majority seeds) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sweep_rows:
        lines.append(
            f"| {r['method']} | {r['K_bits']}/{r['V_bits']} | {fmt_pct(r['mean_output_change'])} | "
            f"{fmt_pct(r['median_TIR'])} | {fmt_pct(r['P95_TIR'])} | {fmt_pct(r['mean_abs_TIR'])} | "
            f"{fmt_pct(r['P95_abs_TIR'])} | {fmt_pct(r['TIR_gt_50_rate'])} | {fmt_pct(r['abs_TIR_gt_50_rate'])} | "
            f"{fmt_num(r['accuracy_drop_points'])} pp | {fmt_pct(r['mean_TPOT_overhead'])} | "
            f"{r['stable_gt50_majority_tasks']}/{r['all_seed_uncensored_tasks']} |"
        )

    lines += [
        "",
        "## Severe-inflation source (TIR > 50%)",
        "",
        "| Method | Severe pairs | Pass→pass severe | Thinking share of extra tokens |",
        "|---|---:|---:|---:|",
    ]
    for method in methods:
        x = method_summaries[method]["severe_inflation"]
        lines.append(
            f"| {method} | {x['pair_count']} | {x['pass_to_pass_pair_count']} | {fmt_pct(x['thinking_share'])} |"
        )

    lines += [
        "",
        "## Does lower precision produce a stronger effect?",
        "",
        "Positive Spearman rho means the metric tends to increase as precision becomes more aggressive (rank 1 → 4). With only four presets, treat this as a descriptive trend, not a significance test.",
        "",
        "| Metric | Spearman ρ vs severity | Strictly non-decreasing? |",
        "|---|---:|---:|",
    ]
    for metric in monotonic_metrics:
        x = monotonicity[metric]
        rho = x["spearman_vs_severity_rank"]
        lines.append(
            f"| {metric} | {fmt_num(rho, 3)} | {x['non_decreasing']} |"
        )

    tl = task_level_monotonicity
    lines += [
        "",
        "## Task-level monotonicity",
        "",
        f"- Tasks uncensored for all seeds and all methods: **{tl['eligible_task_count']}**",
        f"- Strictly non-decreasing mean |TIR| from high→low precision: **{tl['strict_non_decreasing_abs_tir_tasks']}**",
        f"- Non-decreasing within 5 percentage-point tolerance: **{tl['non_decreasing_abs_tir_tasks_with_5pp_tolerance']}**",
        f"- Mean per-task Spearman(severity, mean |TIR|): **{fmt_num(tl['mean_task_spearman_severity_vs_abs_tir'], 3)}**",
        "",
        "Interpretation priority: first inspect mean/P95 |TIR| and |TIR|>50%; then positive-only inflation, correctness, and TPOT. If absolute disturbance grows monotonically while signed mean TIR stays near zero, lower precision is increasing trajectory instability rather than only making outputs longer.",
    ]
    (args.output_dir / "precision_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
