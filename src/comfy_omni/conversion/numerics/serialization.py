"""Lazy Torch serialization adapter for one bounded inverse-ConvRot block.

The adapter is intentionally separate from artifact I/O and producer scheduling. It is derived
from Apache-2.0 ``h3_forge.convrot`` at commit
e9cb011d00b028c149db3978de246c54f6e34acc (blob
8b4b9eebacd8bdaf64b251d5635b0147e7d790db).
"""

from __future__ import annotations

import sys

from comfy_omni.conversion.numerics.errors import ConvRotNumericsError
from comfy_omni.conversion.numerics.torch_backend import _torch, fast_inverse_convrot_rows


def torch_convrot_bf16_block(
    qweight: bytes,
    rowwise_scale: bytes,
    *,
    rows: int,
    columns: int,
    group_size: int,
) -> bytes:
    """Decode one bounded I8/F32 block and return exact little-endian BF16 bytes."""

    if sys.byteorder != "little":
        raise ConvRotNumericsError("safetensors BF16 serialization requires a little-endian conversion container")
    if type(rows) is not int or type(columns) is not int or rows <= 0 or columns <= 0:
        raise ConvRotNumericsError("serialized ConvRot block dimensions must be positive integers")
    if not isinstance(qweight, bytes) or len(qweight) != rows * columns:
        raise ConvRotNumericsError("serialized ConvRot I8 block has the wrong byte length")
    if not isinstance(rowwise_scale, bytes) or len(rowwise_scale) != rows * 4:
        raise ConvRotNumericsError("serialized ConvRot F32 scale block has the wrong byte length")
    torch = _torch()
    weight = torch.frombuffer(bytearray(qweight), dtype=torch.int8).reshape(rows, columns)
    scale = torch.frombuffer(bytearray(rowwise_scale), dtype=torch.float32).reshape(rows, 1)
    decoded = (
        fast_inverse_convrot_rows(weight, scale, group_size=group_size).to(dtype=torch.bfloat16).contiguous().cpu()
    )
    payload = bytes(decoded.view(torch.uint8).untyped_storage())
    expected = rows * columns * 2
    if len(payload) != expected:
        raise ConvRotNumericsError(f"serialized ConvRot BF16 block produced {len(payload)} bytes; expected {expected}")
    return payload


__all__ = ["torch_convrot_bf16_block"]
