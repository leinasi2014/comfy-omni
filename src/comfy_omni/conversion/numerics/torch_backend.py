"""Lazy Torch backend for bounded inverse ConvRot to dense BF16.

Derived from Apache-2.0 ``h3_forge.convrot`` at commit
e9cb011d00b028c149db3978de246c54f6e34acc (blob
8b4b9eebacd8bdaf64b251d5635b0147e7d790db). No vLLM import is required.
"""

from __future__ import annotations

import importlib
from typing import Any

from comfy_omni.conversion.numerics.errors import ConvRotNumericsError
from comfy_omni.conversion.numerics.reference import row_blocks, validate_group_size, validate_max_rows


def _torch() -> Any:
    try:
        return importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        raise ConvRotNumericsError(
            "the inverse-ConvRot backend requires Torch supplied by the conversion container"
        ) from exc


def _regular_hadamard(torch: Any, size: int, *, device: Any) -> Any:
    validate_group_size(size)
    h4 = torch.tensor(
        ((1, 1, 1, -1), (1, 1, -1, 1), (1, -1, 1, 1), (-1, 1, 1, 1)),
        device=device,
        dtype=torch.float32,
    )
    matrix = h4
    current = 4
    while current < size:
        matrix = torch.kron(matrix, h4)
        current *= 4
    return matrix / size**0.5


def regular_hadamard(size: int, *, device: Any = None) -> Any:
    """Build the normalized regular Hadamard matrix in float32."""

    torch = _torch()
    with torch.inference_mode():
        return _regular_hadamard(torch, size, device=device)


def _validate_inputs(torch: Any, qweight: Any, rowwise_scale: Any, group_size: int) -> tuple[int, int]:
    validate_group_size(group_size)
    if not isinstance(qweight, torch.Tensor) or qweight.dtype != torch.int8 or qweight.ndim != 2:
        raise ConvRotNumericsError("qweight must be a rank-2 torch.int8 tensor")
    rows, columns = qweight.shape
    if rows <= 0 or columns <= 0 or columns % group_size:
        raise ConvRotNumericsError("qweight dimensions must be nonzero and width divisible by group_size")
    if (
        not isinstance(rowwise_scale, torch.Tensor)
        or rowwise_scale.dtype != torch.float32
        or tuple(rowwise_scale.shape) != (rows, 1)
    ):
        raise ConvRotNumericsError("source scale must be torch.float32 shaped [rows, 1]")
    if qweight.device != rowwise_scale.device:
        raise ConvRotNumericsError("qweight and source scale must be on the same device")
    if not bool(torch.isfinite(rowwise_scale).all()) or not bool((rowwise_scale > 0).all()):
        raise ConvRotNumericsError("source scales must be finite and positive")
    return rows, columns


def _inverse_rows(torch: Any, qweight: Any, rowwise_scale: Any, matrix: Any, group_size: int) -> Any:
    dequantized = qweight.to(dtype=torch.float32).mul_(rowwise_scale)
    grouped = dequantized.reshape(qweight.shape[0], -1, group_size)
    return torch.matmul(grouped, matrix.T).reshape_as(dequantized)


def _fast_regular_hadamard(torch: Any, values: Any, group_size: int) -> Any:
    """Apply the regular Hadamard in base-four stages without materializing a dense matmul."""

    output = values.reshape(values.shape[0], -1, group_size)
    stride = 1
    while stride < group_size:
        original_shape = output.shape
        staged = output.reshape(*original_shape[:-1], group_size // (4 * stride), 4, stride)
        a, b, c, d = staged.unbind(dim=-2)
        ab = a + b
        cd = c - d
        ac = a - b
        bd = c + d
        output = torch.stack((ab + cd, ab - cd, ac + bd, bd - ac), dim=-2).mul_(0.5).reshape(original_shape)
        stride *= 4
    return output.reshape_as(values)


def inverse_convrot_rows(qweight: Any, rowwise_scale: Any, *, group_size: int = 256) -> Any:
    """Inverse one already-bounded row block and return float32."""

    torch = _torch()
    _validate_inputs(torch, qweight, rowwise_scale, group_size)
    with torch.inference_mode():
        matrix = _regular_hadamard(torch, group_size, device=qweight.device)
        return _inverse_rows(torch, qweight, rowwise_scale, matrix, group_size)


def fast_inverse_convrot_rows(qweight: Any, rowwise_scale: Any, *, group_size: int = 256) -> Any:
    """Inverse one bounded row block in O(n log n) base-four Hadamard stages."""

    torch = _torch()
    _validate_inputs(torch, qweight, rowwise_scale, group_size)
    with torch.inference_mode():
        dequantized = qweight.to(dtype=torch.float32).mul_(rowwise_scale)
        return _fast_regular_hadamard(torch, dequantized, group_size)


def inverse_convrot_to_bf16(
    qweight: Any,
    rowwise_scale: Any,
    *,
    max_rows: int = 128,
    group_size: int = 256,
) -> Any:
    """Inverse one validated ConvRot weight to dense BF16 in bounded row blocks."""

    validate_max_rows(max_rows)
    torch = _torch()
    rows, _ = _validate_inputs(torch, qweight, rowwise_scale, group_size)
    with torch.inference_mode():
        matrix = _regular_hadamard(torch, group_size, device=qweight.device)
        dense = torch.empty(qweight.shape, dtype=torch.bfloat16, device=qweight.device)
        for start, end in row_blocks(rows, max_rows):
            decoded = _inverse_rows(
                torch,
                qweight[start:end],
                rowwise_scale[start:end],
                matrix,
                group_size,
            )
            dense[start:end].copy_(decoded)
    return dense


__all__ = [
    "fast_inverse_convrot_rows",
    "inverse_convrot_rows",
    "inverse_convrot_to_bf16",
    "regular_hadamard",
]
