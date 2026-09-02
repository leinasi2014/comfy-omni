"""Strict, metadata-only safetensors descriptor I/O.

This module owns bounded filesystem/header access and structural validation. It never imports Torch,
reads tensor payload bytes, classifies model families, or performs conversion/publication.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any, BinaryIO

from comfy_omni.domain.checkpoints import TensorDescriptor

MAX_HEADER_BYTES = 64 * 1024 * 1024
MAX_TENSOR_COUNT = 100_000
MAX_TENSOR_RANK = 64
MAX_DIMENSION = (1 << 63) - 1
MAX_JSON_INTEGER_DIGITS = 20

DTYPE_BITS = {
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "U16": 16,
    "I16": 16,
    "F16": 16,
    "BF16": 16,
    "U32": 32,
    "I32": 32,
    "F32": 32,
    "U64": 64,
    "I64": 64,
    "F64": 64,
    "C64": 64,
    "F4": 4,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
    "F8_E4M3": 8,
    "F8_E5M2": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E5M2FNUZ": 8,
    "F8_E8M0": 8,
}


class _DuplicateKeyError(ValueError):
    pass


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _bounded_json_int(value: str) -> int:
    digits = value.removeprefix("-")
    if len(digits) > MAX_JSON_INTEGER_DIGITS:
        raise ValueError(f"JSON integer exceeds {MAX_JSON_INTEGER_DIGITS} digits")
    result = int(value)
    if result < -(1 << 63) or result > (1 << 64) - 1:
        raise ValueError("JSON integer is outside the supported 64-bit range")
    return result


def _finite_json_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("JSON floating-point value is outside the finite range")
    return result


def _validate_unicode_strings(value: Any) -> None:
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            if any(0xD800 <= ord(character) <= 0xDFFF for character in item):
                raise ValueError("JSON strings must not contain unpaired Unicode surrogates")
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)


def _decode_json_header(path: Path, raw_header: bytes) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _DuplicateKeyError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        header = json.loads(
            raw_header.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
            parse_int=_bounded_json_int,
        )
        _validate_unicode_strings(header)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, _DuplicateKeyError, ValueError) as exc:
        raise ValueError(f"{path}: invalid safetensors JSON header: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"{path}: safetensors header must be an object")
    return header


def _metadata_from_header(path: Path, header: dict[str, Any]) -> dict[str, str]:
    raw_metadata = header.pop("__metadata__", {})
    if not isinstance(raw_metadata, dict):
        raise ValueError(f"{path}: __metadata__ must be an object")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in raw_metadata.items()):
        raise ValueError(f"{path}: __metadata__ keys and values must be strings")
    return dict(raw_metadata)


def _descriptor_from_record(path: Path, name: str, record: Any, payload_bytes: int) -> TensorDescriptor:
    if not isinstance(record, dict):
        raise ValueError(f"{path}: tensor {name!r} has an invalid descriptor")
    dtype = record.get("dtype")
    shape = record.get("shape")
    offsets = record.get("data_offsets")
    if (
        not isinstance(dtype, str)
        or dtype not in DTYPE_BITS
        or not isinstance(shape, list)
        or not isinstance(offsets, list)
        or len(offsets) != 2
    ):
        raise ValueError(f"{path}: tensor {name!r} has an incomplete descriptor")
    if not all(type(value) is int and value >= 0 for value in [*shape, *offsets]):
        raise ValueError(f"{path}: tensor {name!r} has invalid shape or offsets")
    if len(shape) > MAX_TENSOR_RANK:
        raise ValueError(f"{path}: tensor {name!r} exceeds rank limit {MAX_TENSOR_RANK}")
    if any(dimension > MAX_DIMENSION for dimension in shape):
        raise ValueError(f"{path}: tensor {name!r} exceeds dimension limit {MAX_DIMENSION}")
    if offsets[1] < offsets[0]:
        raise ValueError(f"{path}: tensor {name!r} has reversed offsets")
    if offsets[1] > payload_bytes:
        raise ValueError(f"{path}: tensor {name!r} extends beyond end of file")
    _validate_tensor_span(path, name, dtype, shape, offsets)
    return TensorDescriptor(name, dtype, tuple(shape), (offsets[0], offsets[1]))


def _validate_tensor_span(path: Path, name: str, dtype: str, shape: list[int], offsets: list[int]) -> None:
    span = offsets[1] - offsets[0]
    dtype_bits = DTYPE_BITS[dtype]
    payload_bits = span * 8
    if payload_bits % dtype_bits:
        raise ValueError(f"{path}: tensor {name!r} byte range does not match dtype {dtype} and shape {shape}")
    expected_elements = payload_bits // dtype_bits
    actual_elements = 0 if 0 in shape else 1
    if actual_elements:
        for dimension in shape:
            if actual_elements > expected_elements // dimension:
                raise ValueError(f"{path}: tensor {name!r} byte range does not match dtype {dtype} and shape {shape}")
            actual_elements *= dimension
    if actual_elements != expected_elements:
        raise ValueError(f"{path}: tensor {name!r} byte range does not match dtype {dtype} and shape {shape}")


def _validate_contiguous_payload(path: Path, tensors: list[TensorDescriptor], payload_bytes: int) -> None:
    cursor = 0
    for tensor in sorted(tensors, key=lambda item: item.data_offsets):
        start, end = tensor.data_offsets
        if start != cursor:
            relation = "overlap" if start < cursor else "gap"
            raise ValueError(f"{path}: tensor data contains an offset {relation} before {tensor.name!r}")
        if end > payload_bytes:
            raise ValueError(f"{path}: tensor {tensor.name!r} extends beyond end of file")
        cursor = end
    if cursor != payload_bytes:
        raise ValueError(f"{path}: safetensors payload contains unindexed bytes")


def _decode_safetensors_header(
    path: Path, size: int, header_length: int, raw_header: bytes
) -> tuple[dict[str, str], tuple[TensorDescriptor, ...]]:
    header = _decode_json_header(path, raw_header)
    metadata = _metadata_from_header(path, header)
    if len(header) > MAX_TENSOR_COUNT:
        raise ValueError(f"{path}: safetensors header exceeds tensor-count limit {MAX_TENSOR_COUNT}")

    payload_bytes = size - 8 - header_length
    tensors = [_descriptor_from_record(path, str(name), record, payload_bytes) for name, record in header.items()]
    _validate_contiguous_payload(path, tensors, payload_bytes)
    return metadata, tuple(tensors)


def read_safetensors_header_stream(
    stream: BinaryIO, path: Path, size: int
) -> tuple[dict[str, str], tuple[TensorDescriptor, ...], int]:
    """Read a validated header, leaving the open stream at the payload boundary."""

    if size < 8:
        raise ValueError(f"{path}: file is too small to be safetensors")
    raw_length = stream.read(8)
    if len(raw_length) != 8:
        raise ValueError(f"{path}: truncated safetensors header length")
    (header_length,) = struct.unpack("<Q", raw_length)
    if header_length == 0 or header_length > MAX_HEADER_BYTES:
        raise ValueError(f"{path}: unsafe safetensors header length {header_length}")
    if 8 + header_length > size:
        raise ValueError(f"{path}: truncated safetensors header")
    raw_header = stream.read(header_length)
    if len(raw_header) != header_length:
        raise ValueError(f"{path}: truncated safetensors header")
    metadata, tensors = _decode_safetensors_header(path, size, header_length, raw_header)
    return metadata, tensors, header_length


def read_safetensors_header(path: Path) -> tuple[dict[str, str], tuple[TensorDescriptor, ...]]:
    """Read and validate one safetensors header without reading tensor payload bytes."""

    size = path.stat().st_size
    with path.open("rb") as stream:
        metadata, tensors, _ = read_safetensors_header_stream(stream, path, size)
    return metadata, tensors


__all__ = [
    "DTYPE_BITS",
    "MAX_DIMENSION",
    "MAX_HEADER_BYTES",
    "MAX_JSON_INTEGER_DIGITS",
    "MAX_TENSOR_COUNT",
    "MAX_TENSOR_RANK",
    "read_safetensors_header",
    "read_safetensors_header_stream",
]
