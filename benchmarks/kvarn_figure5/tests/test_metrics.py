from dataclasses import dataclass

from metrics import (
    compare_worker_results,
    extract_prompt_window,
)


@dataclass
class LP:
    logprob: float
    rank: int


def test_extract_prompt_window():
    tokens = [10, 11, 12]
    logprobs = [None, {11: LP(-1.0, 2), 99: LP(-0.2, 1)}, {12: LP(-0.5, 1)}]
    rows = extract_prompt_window(tokens, logprobs, start=1, end=3)
    assert rows[0]["top1_token_id"] == 99
    assert rows[0]["target_rank"] == 2
    assert rows[1]["top1_token_id"] == 12


def test_compare_worker_results():
    baseline = {
        "points": [
            {
                "context_tokens": 4,
                "elapsed_seconds": 2.0,
                "prompt_tokens_per_second": 2.0,
                "window": [
                    {"position": 2, "token_id": 5, "target_logprob": -1.0, "target_rank": 1, "top1_token_id": 5},
                    {"position": 3, "token_id": 6, "target_logprob": -2.0, "target_rank": 2, "top1_token_id": 7},
                ],
            }
        ]
    }
    candidate = {
        "method": "kvarn",
        "label": "KVarN K4/V2 G128",
        "kv_cache_dtype": "kvarn_k4v2_g128",
        "points": [
            {
                "context_tokens": 4,
                "elapsed_seconds": 1.0,
                "prompt_tokens_per_second": 4.0,
                "window": [
                    {"position": 2, "token_id": 5, "target_logprob": -1.2, "target_rank": 2, "top1_token_id": 5},
                    {"position": 3, "token_id": 6, "target_logprob": -1.6, "target_rank": 1, "top1_token_id": 6},
                ],
            }
        ],
    }
    raw, summary = compare_worker_results(baseline, candidate)
    assert len(raw) == 2
    assert abs(summary[0]["mean_abs_logprob_error"] - 0.3) < 1e-9
    assert summary[0]["top1_agreement"] == 0.5
    assert summary[0]["throughput_ratio_vs_fp16"] == 2.0
