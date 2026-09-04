"""Pure comfy_quant (ComfyUI) quantized-weight decoding for the H3 text encoder.

Provenance: the decode math mirrors the ComfyUI reference implementations:

- ``comfy/float.py`` of Comfy-Org/ComfyUI (Apache-2.0) --
  ``stochastic_round_quantize_nvfp4_by_block`` / ``to_blocked`` semantics;
- ``comfy_kitchen/float_utils.py`` (Comfy Org, Apache-2.0; portions derived
  from PyTorch AO, BSD-3-Clause, see NOTICE) -- ``_floatx_unpacked_to_f32``,
  ``unpack_uint4``, ``F4_E2M1_*`` constants;
- ``comfy_kitchen/backends/eager/quantization.py`` -- ``E2M1_LUT`` and the
  eager ``dequantize_nvfp4`` reference.

The on-disk ``strict.safetensors`` layout (Abiray
``Qwen3-VL-32B-Heretic-MiniMax-H3-nvfp4-ComfyUI``) uses the **natural**
block-scale grid ``[rows, cols/16]`` (not the cuBLAS swizzle): weight
``U8 [rows, cols/2]`` (two 4-bit values per byte), ``weight_scale``
``F8_E4M3 [rows, cols/16]``, ``weight_scale_2`` ``F32 [1]``, and a
``{layer}.comfy_quant`` JSON marker (``{"format": "nvfp4"}`` or
``{"format": "int8_tensorwise"}``).

Torch is imported lazily so this module stays importable in lightweight
environments (wire/CLI/unit gates).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

#: NVFP4 E2M1 decode constants (bias=1 -> eps 0.5, max 6.0).
F4_E2M1_EPS = 0.5
F4_E2M1_MAX = 6.0
F8_E4M3_MAX = 448.0

#: Supported comfy_quant marker formats.
SUPPORTED_FORMATS = frozenset({"nvfp4", "int8_tensorwise"})

#: E2M1 lookup: nibble -> value (sign bit = bit 3).
E2M1_LUT = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def parse_comfy_marker(payload: bytes | bytearray | Sequence[int]) -> dict[str, Any]:
    """Parse the ``{layer}.comfy_quant`` JSON marker bytes.

    Raises ``ValueError`` on malformed payloads; the caller fails closed.
    """
    try:
        raw = bytes(payload).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("comfy_quant marker is not utf-8 JSON") from exc
    try:
        conf = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("comfy_quant marker is not valid JSON") from exc
    if not isinstance(conf, dict):
        raise ValueError("comfy_quant marker must be a JSON object")
    fmt = conf.get("format")
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"unsupported comfy_quant format {fmt!r}")
    return conf


def _torch():
    import torch

    return torch


def decode_int8_tensorwise(
    qdata: Any,
    scale: Any,
    *,
    output_dtype: Any = None,
) -> Any:
    """Dequantize ``int8_tensorwise`` weight: ``qdata * scale`` (scalar)."""
    torch = _torch()
    w = qdata.to(torch.float32)
    s = scale.to(torch.float32)
    if s.numel() != 1:
        raise ValueError(f"int8_tensorwise scale must be a scalar, got {tuple(s.shape)}")
    out = w * s.reshape(())
    if output_dtype is not None:
        out = out.to(output_dtype)
    return out


def decode_nvfp4(
    qdata: Any,
    per_tensor_scale: Any,
    block_scale: Any,
    *,
    hi_first: bool = True,
    output_dtype: Any = None,
) -> Any:
    """Dequantize NVFP4 weights in natural (unswizzled) block-scale layout.

    ``qdata`` is ``U8 [R, C/2]``, ``block_scale`` is ``F8_E4M3 [R, C/16]``
    (natural grid), ``per_tensor_scale`` is a scalar ``F32``. The even-indexed
    logical element is packed in the high nibble when ``hi_first`` (the
    reference default).
    """
    torch = _torch()
    if qdata.dim() != 2:
        raise ValueError(f"nvfp4 qdata must be 2-D, got {tuple(qdata.shape)}")
    rows, stored_cols = qdata.shape
    logical_cols = stored_cols * 2
    if logical_cols % 16:
        raise ValueError(f"nvfp4 logical columns {logical_cols} are not a multiple of 16")
    blocks_per_row = logical_cols // 16
    if tuple(block_scale.shape) != (rows, blocks_per_row):
        raise ValueError(
            f"nvfp4 block_scale shape {tuple(block_scale.shape)} does not match "
            f"natural grid ({rows}, {blocks_per_row})"
        )
    if per_tensor_scale.numel() != 1:
        raise ValueError(f"nvfp4 per-tensor scale must be a scalar, got {tuple(per_tensor_scale.shape)}")

    # On-disk qdata may be stored signed (I8) with two-complement nibbles;
    # force the unsigned bit pattern before splitting nibbles.
    import os as _os
    if _os.environ.get("COMFY_OMNI_TE_DEBUG") == "1":
        print(f"NVFP4-DECODE qdata dtype={qdata.dtype} shape={tuple(qdata.shape)} ", flush=True)
    qdata = qdata.to(torch.uint8)
    hi = qdata >> 4
    lo = qdata & 0x0F
    if hi_first:
        unpacked = torch.stack([hi, lo], dim=-1).reshape(rows, logical_cols)
    else:
        unpacked = torch.stack([lo, hi], dim=-1).reshape(rows, logical_cols)
    lut = torch.tensor(E2M1_LUT, dtype=torch.float32).reshape(-1, 1)
    if qdata.device.type != "cpu":
        lut = lut.to(qdata.device)
    if bool(qdata.numel()) and (int(lo.min()) < 0 or int(hi.max()) > 15):
        raise ValueError(
            f"nvfp4 qdata nibble range [{int(lo.min())}, {int(hi.max())}] is invalid for "
            f"dtype={qdata.dtype} shape={tuple(qdata.shape)}"
        )
    values = torch.nn.functional.embedding(unpacked.to(torch.long), lut).squeeze(-1)
    bs = block_scale.to(torch.float32)
    total = per_tensor_scale.to(torch.float32).reshape(()) * bs.unsqueeze(-1)
    out = (values.reshape(rows, blocks_per_row, 16) * total.unsqueeze(-1)).reshape(rows, logical_cols)
    if output_dtype is not None:
        out = out.to(output_dtype)
    return out


def decode_weight(
    qdata: Any,
    marker: dict[str, Any],
    *,
    weight_scale: Any | None = None,
    weight_scale_2: Any | None = None,
    output_dtype: Any = None,
) -> Any:
    """Decode one quantized weight by its comfy_quant marker format."""
    fmt = marker["format"]
    if fmt == "int8_tensorwise":
        if weight_scale is None:
            raise ValueError("int8_tensorwise weight requires weight_scale")
        return decode_int8_tensorwise(qdata, weight_scale, output_dtype=output_dtype)
    if fmt == "nvfp4":
        if weight_scale is None or weight_scale_2 is None:
            raise ValueError("nvfp4 weight requires weight_scale and weight_scale_2")
        return decode_nvfp4(qdata, weight_scale_2, weight_scale, output_dtype=output_dtype)
    raise ValueError(f"unsupported comfy_quant format {fmt!r}")


__all__ = [
    "F4_E2M1_EPS",
    "F4_E2M1_MAX",
    "F8_E4M3_MAX",
    "SUPPORTED_FORMATS",
    "E2M1_LUT",
    "parse_comfy_marker",
    "decode_int8_tensorwise",
    "decode_nvfp4",
    "decode_weight",
]
