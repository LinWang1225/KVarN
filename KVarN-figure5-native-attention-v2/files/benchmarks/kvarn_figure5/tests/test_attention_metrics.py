# SPDX-License-Identifier: Apache-2.0

from benchmarks.kvarn_figure5.attention_metrics import aggregate_figure5


def _row(method, regime, sample, context, layer, error, *, chunk_start=0, chunk_end=None):
    if chunk_end is None:
        chunk_end = context
    return {
        "method": method,
        "regime": regime,
        "request_index": sample,
        "sample_id": sample,
        "context_tokens": context,
        "chunk_start": chunk_start,
        "chunk_end": chunk_end,
        "layer_name": f"model.layers.{layer}.self_attn",
        "layer_index": layer,
        "query_tokens": 2,
        "sum_abs_error": error * 4,
        "num_elements": 4,
        "attention_output_mae": error,
        "max_abs_error": error,
    }


def test_figure5_differences_are_paired():
    rows = []
    for sample in (0, 1):
        for method, base in (("kvarn", 1.0), ("turboquant", 3.0)):
            for regime, offset in (("static", 0.0), ("accumulated", 1.0)):
                rows.append(_row(method, regime, sample, 4096, 0, base + offset))
                rows.append(_row(method, regime, sample, 4096, 1, base + offset))
    _, sample_rows, summary = aggregate_figure5(rows, num_hidden_layers=2)
    assert len(sample_rows) == 8
    gaps = [row for row in summary if row.get("panel") == "method_gap"]
    assert {row["mean_difference"] for row in gaps} == {-2.0}
    panel_c = [row for row in summary if row.get("panel") == "regime_gap"]
    assert len(panel_c) == 1
    assert panel_c[0]["mean_difference"] == 0.0


def test_missing_preset_layer_is_counted_as_zero_error():
    rows = []
    for method in ("kvarn", "turboquant"):
        for regime in ("static", "accumulated"):
            # Layer 1 is intentionally absent, as for an FP16 skip layer.
            rows.append(_row(method, regime, 0, 4096, 0, 2.0))
    completed, sample_rows, _ = aggregate_figure5(rows, num_hidden_layers=2)
    assert len(completed) == 8
    assert all(row["attention_output_mae"] == 1.0 for row in sample_rows)


def test_all_chunks_are_element_weighted():
    rows = []
    for method in ("kvarn", "turboquant"):
        for regime in ("static", "accumulated"):
            # Two contiguous chunks and two layers. The first chunk has MAE 1,
            # the second MAE 3, so the full-sequence mean is 2.
            for layer in (0, 1):
                rows.append(
                    _row(
                        method,
                        regime,
                        0,
                        256,
                        layer,
                        1.0,
                        chunk_start=0,
                        chunk_end=128,
                    )
                )
                rows.append(
                    _row(
                        method,
                        regime,
                        0,
                        256,
                        layer,
                        3.0,
                        chunk_start=128,
                        chunk_end=256,
                    )
                )
    _, sample_rows, _ = aggregate_figure5(rows, num_hidden_layers=2)
    assert all(row["num_chunks"] == 2 for row in sample_rows)
    assert all(row["attention_output_mae"] == 2.0 for row in sample_rows)
