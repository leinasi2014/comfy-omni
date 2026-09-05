"""Bounded CPU TE decoding, lazily requiring the conversion container's Torch.

Adapted from comfy-kitchen b678fdf63378409676aa5596721445d33794d0ea:
eager/quantization.py blob1df1514e38216b0deeb1977075a187fdda5886ad,
float_utils.py blob29077a7b5375a596eab64bab449bfc2e842beb9d.
Apache-2.0 and torchao BSD-3-Clause attribution: THIRD_PARTY_NOTICES.md.
The independent acceptance oracle does not import this implementation.
"""

from __future__ import annotations

import importlib

from comfy_omni.contracts.models import ContractError


def _torch():
    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        raise ContractError("TE decoding requires Torch in the offline conversion container") from exc
    torch.set_flush_denormal(False)
    return torch


def _bytes(torch, output) -> bytes:
    if not torch.isfinite(output).all().item():
        raise ContractError("TE payload contains nonfinite values")
    return output.contiguous().view(torch.uint8).numpy().tobytes()


def nvfp4_bf16_stripe(packed: bytes, block_scales: bytes, global_scale: bytes, *, rows: int, columns: int) -> bytes:
    if (
        type(rows) is not int
        or type(columns) is not int
        or not 1 <= rows <= 128
        or not 1 <= columns <= 25600
        or columns % 64
    ):
        raise ContractError("invalid bounded NVFP4 stripe dimensions")
    if len(packed) != rows * columns // 2 or len(block_scales) != 128 * (columns // 16) or len(global_scale) != 4:
        raise ContractError("NVFP4 stripe byte lengths disagree with dimensions")
    torch = _torch()
    with torch.inference_mode():
        q = torch.frombuffer(bytearray(packed), dtype=torch.uint8).reshape(rows, columns // 2)
        scales = torch.frombuffer(bytearray(block_scales), dtype=torch.float8_e4m3fn)
        # Exact inverse128x4 tile order; this stripe contains one128-row tile band.
        scales = scales.reshape(-1, 32, 4, 4).transpose(1, 2)
        scales = scales.reshape(1, columns // 64, 128, 4).permute(0, 2, 1, 3)
        scales = scales.reshape(128, columns // 16)[:rows].to(torch.bfloat16)
        global_value = torch.frombuffer(bytearray(global_scale), dtype=torch.float32).to(torch.bfloat16)
        total_scale = global_value * scales
        if not torch.isfinite(total_scale).all().item():
            raise ContractError("NVFP4 scales contain nonfinite values")
        codes = torch.stack((q >> 4, q & 15), dim=-1).reshape(rows, columns)
        lut = torch.tensor(
            (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0),
            dtype=torch.bfloat16,
        )
        values = lut[codes.to(torch.int64)].reshape(rows, columns // 16, 16)
        output = (values * total_scale.unsqueeze(-1)).reshape(rows, columns)
        return _bytes(torch, output)


def int8_bf16_chunk(packed: bytes, global_scale: bytes) -> bytes:
    if not packed or len(packed) > 4 * 1024**2 or len(global_scale) != 4:
        raise ContractError("invalid bounded INT8 embedding chunk")
    torch = _torch()
    with torch.inference_mode():
        values = torch.frombuffer(bytearray(packed), dtype=torch.int8).to(torch.float32)
        scale = torch.frombuffer(bytearray(global_scale), dtype=torch.float32)
        return _bytes(torch, (values * scale).to(torch.bfloat16))


def validate_bf16_chunk(raw: bytes) -> None:
    if not raw or len(raw) % 2 or len(raw) > 8 * 1024**2:
        raise ContractError("invalid bounded BF16 passthrough chunk")
    torch = _torch()
    with torch.inference_mode():
        values = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16)
        if not torch.isfinite(values).all().item():
            raise ContractError("BF16 passthrough contains nonfinite values")
