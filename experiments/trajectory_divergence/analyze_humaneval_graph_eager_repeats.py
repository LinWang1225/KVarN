#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CONDITIONS = ("fp16_graph", "fp16_eager", "kvarn_graph", "kvarn_eager")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--num-repeats", type=int, default=5)
    return p.parse_args()


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[str(row["sample_id"])] = row
    return out


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def h(ids: list[int]) -> str:
    raw = json.dumps(ids, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def first_diff(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def env_fp(config: dict[str, Any]) -> dict[str, Any]:
    env = config.get("environment") or {}
    pkgs = env.get("packages") or {}
    cmd = env.get("command") or []
    return {
        "git_commit": env.get("git_commit"),
        "vllm": pkgs.get("vllm"),
        "torch": pkgs.get("torch"),
        "cuda_version": env.get("cuda_version"),
        "gpu_names": env.get("gpu_names"),
        "model": config.get("model"),
        "dtype": config.get("dtype"),
        "kv_cache_dtype": config.get("resolved_kv_cache_dtype"),
        "max_tokens": config.get("max_tokens"),
        "max_model_len": config.get("max_model_len"),
        "seed": config.get("seed"),
        "temperature": config.get("temperature"),
        "top_p": config.get("top_p"),
        "top_k": config.get("top_k"),
        "min_p": config.get("min_p"),
        "block_size": config.get("block_size"),
        "tp": config.get("tensor_parallel_size"),
        "gpu_memory_utilization": config.get("gpu_memory_utilization"),
        "prefix_caching": config.get("prefix_caching"),
        "flashinfer_sampler": env.get("vllm_use_flashinfer_sampler"),
        "enforce_eager": "--enforce-eager" in cmd,
    }


def fp_key(fp: dict[str, Any], drop_condition: bool = False) -> str:
    fp = dict(fp)
    if drop_condition:
        fp.pop("kv_cache_dtype", None)
        fp.pop("enforce_eager", None)
    return json.dumps(fp, sort_keys=True, ensure_ascii=False, default=str)


def analyze_sample(sample_id: str, records: list[tuple[int, dict[str, Any]]], expected: int) -> dict[str, Any]:
    valid = [(i, r) for i, r in records if r.get("error") is None]
    clusters: dict[str, list[int]] = defaultdict(list)
    run_rows = []
    for i, r in valid:
        ids = [int(x) for x in (r.get("output_token_ids") or [])]
        th = h(ids)
        clusters[th].append(i)
        run_rows.append({
            "run": i,
            "trajectory_hash": th,
            "output_tokens": len(ids),
            "finish_reason": r.get("finish_reason"),
            "hit_length_cap": str(r.get("finish_reason") or "").lower() == "length",
            "thinking_boundary": bool(r.get("thinking_boundary_detected")),
            "thinking_tokens": r.get("thinking_tokens"),
            "execution_passed": r.get("execution_passed"),
            "candidate_code_sha256": r.get("candidate_code_sha256"),
        })

    pair_steps = []
    pair_total = 0
    pair_div = 0
    for (_, left), (_, right) in itertools.combinations(valid, 2):
        a = [int(x) for x in (left.get("output_token_ids") or [])]
        b = [int(x) for x in (right.get("output_token_ids") or [])]
        step = first_diff(a, b)
        pair_total += 1
        if step is not None:
            pair_div += 1
            pair_steps.append(step)

    lengths = [r["output_tokens"] for r in run_rows]
    finish = Counter(str(r["finish_reason"]) for r in run_rows)
    passed = Counter(str(r["execution_passed"]) for r in run_rows)
    thinking = Counter(str(r["thinking_boundary"]) for r in run_rows)
    candidate_hashes = {r["candidate_code_sha256"] for r in run_rows if r["candidate_code_sha256"]}
    cluster_sizes = sorted((len(v) for v in clusters.values()), reverse=True)

    return {
        "sample_id": sample_id,
        "expected_repeats": expected,
        "valid_repeats": len(valid),
        "distinct_trajectory_count": len(clusters),
        "exactly_deterministic": len(valid) == expected and len(clusters) == 1,
        "dominant_trajectory_count": cluster_sizes[0] if cluster_sizes else 0,
        "trajectory_clusters": dict(clusters),
        "pairwise_comparison_count": pair_total,
        "pairwise_diverged_count": pair_div,
        "pairwise_divergence_rate": pair_div / pair_total if pair_total else None,
        "first_divergence_steps": pair_steps,
        "min_first_divergence_step": min(pair_steps) if pair_steps else None,
        "median_first_divergence_step": statistics.median(pair_steps) if pair_steps else None,
        "max_first_divergence_step": max(pair_steps) if pair_steps else None,
        "finish_reason_counts": dict(finish),
        "finish_reason_stable": len(finish) <= 1,
        "execution_pass_counts": dict(passed),
        "execution_pass_stable": len(passed) <= 1,
        "thinking_boundary_counts": dict(thinking),
        "thinking_boundary_stable": len(thinking) <= 1,
        "hit_length_cap_runs": sum(r["hit_length_cap"] for r in run_rows),
        "min_output_tokens": min(lengths) if lengths else None,
        "mean_output_tokens": statistics.fmean(lengths) if lengths else None,
        "max_output_tokens": max(lengths) if lengths else None,
        "distinct_candidate_code_count": len(candidate_hashes),
        "run_details": run_rows,
    }


def analyze_condition(root: Path, condition: str, n: int, analysis_root: Path):
    runs = []
    fps = []
    for i in range(1, n + 1):
        d = root / condition / f"run{i}"
        runs.append(load_jsonl(d / "generations.jsonl"))
        fps.append(env_fp(load_json(d / "experiment_config.json")))

    sample_ids = sorted(set().union(*(set(r) for r in runs)))
    rows = []
    for sid in sample_ids:
        records = [(i, runs[i-1][sid]) for i in range(1, n+1) if sid in runs[i-1]]
        rows.append(analyze_sample(sid, records, n))

    pair_total = sum(r["pairwise_comparison_count"] for r in rows)
    pair_div = sum(r["pairwise_diverged_count"] for r in rows)
    steps = [s for r in rows for s in r["first_divergence_steps"]]
    stable = sum(r["exactly_deterministic"] for r in rows)
    summary = {
        "condition": condition,
        "mode": condition.split("_", 1)[0],
        "execution_mode": condition.split("_", 1)[1],
        "num_repeats": n,
        "sample_count": len(rows),
        "exactly_deterministic_samples": stable,
        "unstable_samples": len(rows) - stable,
        "sample_instability_rate": (len(rows)-stable)/len(rows) if rows else None,
        "pairwise_comparison_count": pair_total,
        "pairwise_diverged_count": pair_div,
        "pairwise_divergence_rate": pair_div/pair_total if pair_total else None,
        "min_first_divergence_step": min(steps) if steps else None,
        "median_first_divergence_step": statistics.median(steps) if steps else None,
        "max_first_divergence_step": max(steps) if steps else None,
        "samples_with_finish_reason_instability": sum(not r["finish_reason_stable"] for r in rows),
        "samples_with_execution_pass_instability": sum(not r["execution_pass_stable"] for r in rows),
        "samples_with_thinking_boundary_instability": sum(not r["thinking_boundary_stable"] for r in rows),
        "total_runs_hitting_length_cap": sum(r["hit_length_cap_runs"] for r in rows),
        "environment_identical_within_condition": len({fp_key(x) for x in fps}) == 1,
        "environment_fingerprints": fps,
    }

    out = analysis_root / condition
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_sample.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    lines = [
        f"# {condition}",
        "",
        f"- stable samples: {stable}/{len(rows)}",
        f"- pairwise divergence: {pair_div}/{pair_total}",
        f"- first divergence min/median/max: {summary['min_first_divergence_step']}/"
        f"{summary['median_first_divergence_step']}/{summary['max_first_divergence_step']}",
        f"- finish instability samples: {summary['samples_with_finish_reason_instability']}",
        f"- pass instability samples: {summary['samples_with_execution_pass_instability']}",
        f"- thinking-boundary instability samples: {summary['samples_with_thinking_boundary_instability']}",
        f"- length-cap hits: {summary['total_runs_hitting_length_cap']}",
        f"- identical environment within condition: {summary['environment_identical_within_condition']}",
        "",
        "| sample | distinct traj | dominant | pair div | first step min/med/max | token range |",
        "|---|---:|---:|---:|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['sample_id']} | {r['distinct_trajectory_count']} | "
            f"{r['dominant_trajectory_count']}/{r['valid_repeats']} | "
            f"{r['pairwise_diverged_count']}/{r['pairwise_comparison_count']} | "
            f"{r['min_first_divergence_step']}/{r['median_first_divergence_step']}/"
            f"{r['max_first_divergence_step']} | {r['min_output_tokens']}-{r['max_output_tokens']} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary, fps


def interpretation(summaries: dict[str, dict[str, Any]]) -> list[str]:
    fg, fe = summaries["fp16_graph"], summaries["fp16_eager"]
    kg, ke = summaries["kvarn_graph"], summaries["kvarn_eager"]
    notes = []
    if fg["unstable_samples"] == 0 and fe["unstable_samples"] == 0 and kg["unstable_samples"] > 0 and ke["unstable_samples"] == 0:
        notes.append("Pattern supports a KVarN × graph-execution interaction.")
    elif kg["unstable_samples"] > ke["unstable_samples"]:
        notes.append("KVarN is less stable in graph mode than eager mode, but the strongest interaction claim is not fully isolated.")
    else:
        notes.append("The matrix does not isolate graph execution as the dominant source of KVarN repeat instability.")
    if fg["unstable_samples"] > 0:
        notes.append("FP16 graph is also unstable, so graph-mode nondeterminism is not KVarN-specific.")
    return notes


def main():
    args = parse_args()
    if args.num_repeats < 2:
        raise ValueError("--num-repeats must be >= 2")
    analysis_root = args.root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)

    summaries = {}
    all_fps = []
    for condition in CONDITIONS:
        summary, fps = analyze_condition(args.root, condition, args.num_repeats, analysis_root)
        summaries[condition] = summary
        all_fps.extend(fps)

    cross_keys = {fp_key(x, drop_condition=True) for x in all_fps}
    env_clean = len(cross_keys) == 1
    payload = {
        "environment_control_clean": env_clean,
        "non_condition_environment_fingerprint_count": len(cross_keys),
        "conditions": [summaries[x] for x in CONDITIONS],
        "interpretation": interpretation(summaries),
    }
    (analysis_root / "matrix_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    fields = [
        "condition","mode","execution_mode","num_repeats","sample_count",
        "exactly_deterministic_samples","unstable_samples","sample_instability_rate",
        "pairwise_comparison_count","pairwise_diverged_count","pairwise_divergence_rate",
        "min_first_divergence_step","median_first_divergence_step","max_first_divergence_step",
        "samples_with_finish_reason_instability","samples_with_execution_pass_instability",
        "samples_with_thinking_boundary_instability","total_runs_hitting_length_cap",
        "environment_identical_within_condition",
    ]
    with (analysis_root / "matrix_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(summaries[x] for x in CONDITIONS)

    lines = [
        "# HumanEval graph/eager 5-repeat determinism matrix",
        "",
        f"- environment control clean: {env_clean}",
        f"- non-condition environment fingerprints: {len(cross_keys)}",
        "",
        "| condition | stable samples | pairwise divergence | first step min/med/max | finish unstable | pass unstable | cap hits |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for c in CONDITIONS:
        s = summaries[c]
        lines.append(
            f"| {c} | {s['exactly_deterministic_samples']}/{s['sample_count']} | "
            f"{s['pairwise_diverged_count']}/{s['pairwise_comparison_count']} | "
            f"{s['min_first_divergence_step']}/{s['median_first_divergence_step']}/{s['max_first_divergence_step']} | "
            f"{s['samples_with_finish_reason_instability']} | "
            f"{s['samples_with_execution_pass_instability']} | "
            f"{s['total_runs_hitting_length_cap']} |"
        )
    lines += ["", "## Interpretation", ""] + [f"- {x}" for x in interpretation(summaries)]
    lines += [
        "",
        "## Decision rule",
        "",
        "- If FP16 graph/eager are stable, KVarN graph is unstable, KVarN eager is stable, and environment control is clean: use eager for the causal trajectory-mechanism experiment.",
        "- Keep normal graph-mode vLLM for production/system-effect validation, but aggregate across seeds/repeats.",
        "- If KVarN eager is also unstable, stop before the 164-task causal run and isolate KVarN kernel/store/decode determinism.",
    ]
    (analysis_root / "matrix_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
