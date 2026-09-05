"""Independent stdlib TE scalar oracle; never imports the producer or Torch.

Arithmetic is independently expressed from the characterized E2M1/E4M3FN and
BF16 contract. Reference/version/attribution: docs/migration/source-attribution.md.
"""

from __future__ import annotations

import math
import struct


def fp32(value: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError as exc:
        raise ValueError("nonfinite FP32 result") from exc


def bf16_bits(value: float) -> int:
    value = fp32(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite BF16 operand")
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    result = ((bits + 32767 + ((bits >> 16) & 1)) >> 16) & 65535
    if result & 0x7F80 == 0x7F80:
        raise ValueError("nonfinite BF16 result")
    return result


def bf16_value(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def round_bf16(value: float) -> float:
    return bf16_value(bf16_bits(value))


def e2m1(code: int) -> float:
    exponent, mantissa = (code >> 1) & 3, code & 1
    value = mantissa / 2 if exponent == 0 else math.ldexp(1 + mantissa / 2, exponent - 1)
    return math.copysign(value, -1 if code & 8 else 1)


def e4m3(code: int) -> float:
    exponent, mantissa = (code >> 3) & 15, code & 7
    if exponent == 15 and mantissa == 7:
        raise ValueError("nonfinite E4M3FN scale")
    value = math.ldexp(mantissa / 8, -6) if exponent == 0 else math.ldexp(1 + mantissa / 8, exponent - 7)
    return math.copysign(value, -1 if code & 128 else 1)


def nvfp4_row(packed: bytes, blocked_band: bytes, global_raw: bytes, *, row_in_band: int) -> bytes:
    """Decode one full row using its entire128-row scale band, in exact BF16 bits."""
    columns = len(packed) * 2
    if columns % 64 or not 0 <= row_in_band < 128 or len(blocked_band) != 128 * columns // 16:
        raise ValueError("invalid NVFP4 scalar row geometry")
    global_value = round_bf16(struct.unpack("<f", global_raw)[0])
    output = bytearray(columns * 2)
    for block in range(columns // 16):
        tile_address = (block // 4) * 512
        within_tile = (row_in_band % 32) * 16 + (row_in_band // 32) * 4 + block % 4
        scale = e4m3(blocked_band[tile_address + within_tile])
        total = round_bf16(global_value * round_bf16(scale))
        for pair in range(8):
            byte = packed[block * 8 + pair]
            struct.pack_into(
                "<HH",
                output,
                (block * 16 + pair * 2) * 2,
                bf16_bits(e2m1(byte >> 4) * total),
                bf16_bits(e2m1(byte & 15) * total),
            )
    return bytes(output)


def int8_values(packed: bytes, global_raw: bytes) -> bytes:
    scale = struct.unpack("<f", global_raw)[0]
    if not math.isfinite(scale):
        raise ValueError("nonfinite INT8 scale")
    output = bytearray(len(packed) * 2)
    for index, byte in enumerate(packed):
        value = byte if byte < 128 else byte - 256
        struct.pack_into("<H", output, index * 2, bf16_bits(fp32(value * scale)))
    return bytes(output)
