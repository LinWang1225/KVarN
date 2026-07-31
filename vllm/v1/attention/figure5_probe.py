# SPDX-License-Identifier: Apache-2.0
"""Benchmark-only probe for the paper's Figure 5 semantics.

The probe is dormant unless ``VLLM_FIGURE5_PROBE_OUTPUT`` is set. It runs
inside the native KVarN/TurboQuant attention backends *after* they have already
produced the quantized-cache attention output.

Two regimes are supported and mirror the paper's protocol:

``static``
    Measure the local reconstruction error of the current chunk against a
    shadow FP16 reference, but feed the FP16 reference forward so later chunks
    do not inherit the quantization error from earlier chunks.

``accumulated``
    Measure the same local reconstruction error, but feed the native quantized
    output forward so later chunks observe the full downstream error
    propagation.

This benchmark pins one request per batch, disables prefix caching, and uses
128-token chunked prefill. Supporting general mixed batches here would add
substantial state-management complexity and is intentionally rejected.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class _LayerState:
    key: torch.Tensor
    value: torch.Tensor
    request_index: int = -1
    sample_id: int = -1
    target_context: int = -1


_LAYER_STATES: dict[str, _LayerState] = {}
_RECORDED: set[tuple[str, str, int, int, int, str]] = set()
_CU_CACHE: dict[tuple[str, int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _single_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RuntimeError(f"Expected one sequence length, got shape {tuple(value.shape)}")
        return int(value.reshape(-1)[0].item())
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            if len(converted) != 1:
                raise RuntimeError(f"Expected one sequence length, got {converted!r}")
            return int(converted[0])
        return int(converted)
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise RuntimeError(f"Expected one sequence length, got {value!r}")
        return int(value[0])
    return int(value)


def _sequence_length(attn_metadata: Any) -> int:
    cpu_value = getattr(attn_metadata, "seq_lens_cpu", None)
    if cpu_value is not None:
        return _single_int(cpu_value)
    return _single_int(attn_metadata.seq_lens)


def _query_length(attn_metadata: Any, actual_tokens: int) -> int:
    cpu_value = getattr(attn_metadata, "query_start_loc_cpu", None)
    if cpu_value is None:
        cpu_value = getattr(attn_metadata, "query_start_loc", None)
    if cpu_value is None:
        return actual_tokens
    if isinstance(cpu_value, torch.Tensor):
        values = cpu_value.tolist()
    elif hasattr(cpu_value, "tolist"):
        values = cpu_value.tolist()
    else:
        values = list(cpu_value)
    if len(values) != 2:
        raise RuntimeError(
            "Figure 5 probe requires max_num_seqs=1; "
            f"query_start_loc has {len(values) - 1} requests"
        )
    q_len = int(values[1]) - int(values[0])
    if q_len != actual_tokens:
        raise RuntimeError(
            "Figure 5 probe requires a single unmixed request; "
            f"q_len={q_len}, num_actual_tokens={actual_tokens}"
        )
    return q_len


def _read_control() -> dict[str, int]:
    path = os.environ.get("VLLM_FIGURE5_PROBE_CONTROL")
    if not path:
        raise RuntimeError("VLLM_FIGURE5_PROBE_CONTROL is not set")

    control_path = Path(path)
    last_error: Exception | None = None
    # The controller writes this file immediately before each generate() call,
    # but EngineCore may enter the first attention hook slightly earlier on some
    # launches. A short bounded retry avoids a spurious startup race without
    # masking genuine missing-file bugs.
    for _ in range(50):
        try:
            with control_path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            required = ("request_index", "sample_id", "target_context")
            missing = [name for name in required if name not in payload]
            if missing:
                raise RuntimeError(
                    f"Probe control file is missing {missing}: {control_path}"
                )
            return {name: int(payload[name]) for name in required}
        except FileNotFoundError as exc:
            last_error = exc
            import time

            time.sleep(0.02)
        except json.JSONDecodeError as exc:
            last_error = exc
            import time

            time.sleep(0.02)

    raise RuntimeError(
        f"Unable to read Figure 5 probe control file after retries: {control_path}"
    ) from last_error


def _layer_name(method: str, impl: Any, layer: Any) -> str:
    for owner in (impl, layer):
        name = getattr(owner, "layer_name", None)
        if name:
            return str(name)
    return f"{method}.impl_{id(impl)}"


def _layer_index(name: str) -> int:
    matches = re.findall(r"(?:layers|layer)\.(\d+)", name)
    return int(matches[-1]) if matches else -1


def _cu_pair(device: torch.device, q_len: int, kv_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    key = (str(device), q_len, kv_len)
    pair = _CU_CACHE.get(key)
    if pair is None:
        pair = (
            torch.tensor([0, q_len], device=device, dtype=torch.int32),
            torch.tensor([0, kv_len], device=device, dtype=torch.int32),
        )
        _CU_CACHE[key] = pair
    return pair


def _sdpa_reference(
    impl: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Bottom-right causal SDPA fallback for q_len <= kv_len."""
    q_len = query.shape[0]
    kv_len = key.shape[0]
    q = query.transpose(0, 1).unsqueeze(0)
    k = key.transpose(0, 1).unsqueeze(0)
    v = value.transpose(0, 1).unsqueeze(0)
    # Query row i represents absolute position kv_len - q_len + i.
    q_positions = torch.arange(
        kv_len - q_len, kv_len, device=query.device, dtype=torch.int64
    ).unsqueeze(1)
    k_positions = torch.arange(kv_len, device=query.device, dtype=torch.int64).unsqueeze(0)
    allowed = k_positions <= q_positions
    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=allowed,
        is_causal=False,
        scale=float(getattr(impl, "scale", query.shape[-1] ** -0.5)),
        enable_gqa=key.shape[1] < query.shape[1],
    )
    return out.squeeze(0).transpose(0, 1)


def _fp16_reference(
    impl: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Compute the FP16 attention output on the shadow unquantized history."""
    q_len = int(query.shape[0])
    kv_len = int(key.shape[0])
    cu_q, cu_k = _cu_pair(query.device, q_len, kv_len)

    # Reuse the backend's own FlashAttention wrapper where possible.  This keeps
    # mask orientation, GQA handling and softmax scaling aligned with the native
    # first-prefill path while leaving KV quantization entirely to the backend.
    if hasattr(impl, "_flash_varlen"):
        return impl._flash_varlen(
            query,
            key,
            value,
            cu_q,
            cu_k,
            q_len,
            kv_len,
            causal=True,
        )
    if hasattr(impl, "_flash_attn_varlen"):
        return impl._flash_attn_varlen(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=q_len,
            max_seqlen_k=kv_len,
        )
    return _sdpa_reference(impl, query, key, value)


def _append_record(record: dict[str, Any]) -> None:
    path = Path(os.environ["VLLM_FIGURE5_PROBE_OUTPUT"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def probe_attention_output(
    *,
    method: str,
    impl: Any,
    layer: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: Any,
    native_output: torch.Tensor,
) -> torch.Tensor:
    """Record Figure 5 attention-output MAE and return the selected regime.

    The probe measures the same local reconstruction error in both regimes.
    The only semantic difference is whether the native quantized output is fed
    forward (``accumulated``) or replaced by the local FP16 reference
    (``static``) so later chunks either do or do not inherit the earlier error
    propagation, matching the paper's protocol.
    """
    regime = os.environ.get("VLLM_FIGURE5_PROBE_REGIME", "accumulated")
    if regime not in {"static", "accumulated"}:
        raise RuntimeError(f"Unsupported Figure 5 regime: {regime!r}")

    actual_tokens = int(query.shape[0])
    q_len = _query_length(attn_metadata, actual_tokens)
    seq_len = _sequence_length(attn_metadata)
    start = seq_len - q_len
    if start < 0:
        raise RuntimeError(f"Invalid seq_len={seq_len}, q_len={q_len}")

    max_context = int(os.environ.get("VLLM_FIGURE5_PROBE_MAX_CONTEXT", seq_len))
    if seq_len > max_context + 1:
        raise RuntimeError(
            f"Probe sequence {seq_len} exceeds configured maximum {max_context}"
        )

    # Ignore tiny startup/warmup attention calls.
    if seq_len < 128 or q_len < 128:
        return native_output

    name = _layer_name(method, impl, layer)
    state = _LAYER_STATES.get(name)
    shape = (max_context, key.shape[1], key.shape[2])
    if (
        state is None
        or tuple(state.key.shape) != shape
        or state.key.device != key.device
        or state.key.dtype != key.dtype
    ):
        state = _LayerState(
            key=torch.empty(shape, device=key.device, dtype=key.dtype),
            value=torch.empty(shape, device=value.device, dtype=value.dtype),
        )
        _LAYER_STATES[name] = state

    if start == 0:
        control = _read_control()
        state.request_index = control["request_index"]
        state.sample_id = control["sample_id"]
        state.target_context = control["target_context"]

    if state.target_context <= 0:
        raise RuntimeError(f"Probe state for {name} was not initialized by a first chunk")
    if seq_len > state.target_context:
        return native_output

    state.key[start:seq_len].copy_(key[:q_len])
    state.value[start:seq_len].copy_(value[:q_len])

    reference = _fp16_reference(
        impl,
        query[:q_len],
        state.key[:seq_len],
        state.value[:seq_len],
    )

    record_key = (
        method,
        regime,
        state.request_index,
        state.target_context,
        seq_len,
        name,
    )
    if record_key not in _RECORDED:
        difference = native_output[:q_len].float() - reference.float()
        absolute = difference.abs()
        sum_abs = float(absolute.sum().item())
        numel = int(absolute.numel())
        _append_record(
            {
                "method": method,
                "regime": regime,
                "request_index": state.request_index,
                "sample_id": state.sample_id,
                "context_tokens": state.target_context,
                "chunk_start": start,
                "chunk_end": seq_len,
                "layer_name": name,
                "layer_index": _layer_index(name),
                "query_tokens": q_len,
                "sum_abs_error": sum_abs,
                "num_elements": numel,
                "attention_output_mae": sum_abs / numel,
                "max_abs_error": float(absolute.max().item()),
            }
        )
        _RECORDED.add(record_key)

    if regime == "static":
        return reference.to(native_output.dtype)
    return native_output
