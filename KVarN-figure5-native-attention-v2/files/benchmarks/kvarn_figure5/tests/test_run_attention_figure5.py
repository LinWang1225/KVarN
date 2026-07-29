# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from benchmarks.kvarn_figure5.run_attention_figure5 import (
    estimate_shadow_cache_bytes,
    reserve_probe_memory,
)


def test_shadow_cache_estimate():
    config = SimpleNamespace(
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_size=4096,
        head_dim=128,
    )
    expected = 2 * 36 * 32768 * 8 * 128 * 2
    assert estimate_shadow_cache_bytes(config, 32768) == expected


def test_vllm_memory_reservation_subtracts_shadow_and_safety():
    gib = 1024**3
    effective = reserve_probe_memory(
        requested_utilization=0.9,
        shadow_bytes=4 * gib,
        total_gpu_bytes=24 * gib,
        safety_bytes=1 * gib,
    )
    assert abs(effective - (0.9 - 5 / 24)) < 1e-12
