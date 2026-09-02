"""Deterministic, exclusive safetensors writing and independent verification.

The behavior is characterized from Apache-2.0 ``h3-forge`` source
``native_export.py@475cee5523be64e5b24a95e16c5de3f371cbdf67``. The legacy monolith is not
copied: this module owns only generic artifact serialization and strict rereads.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import BinaryIO

from comfy_omni.artifacts.safetensors import DTYPE_BITS, MAX_HEADER_BYTES
from comfy_omni.artifacts.sources import SafeTensorSources
from comfy_omni.contracts.models import ContractError
from comfy_omni.domain.checkpoints import TensorDescriptor


@dataclass(frozen=True)
class TensorPayload:
    """One target tensor and a fresh, repeatable byte-chunk producer."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    byte_length: int
    chunks: Callable[[], Iterable[bytes]]


@dataclass(frozen=True)
class SafetensorsFileRecord:
    """Identity and strict descriptor inventory for one written shard."""

    name: str
    size: int
    sha256: str
    descriptors: tuple[TensorDescriptor, ...]

    @property
    def tensor_count(self) -> int:
        return len(self.descriptors)


def _fail(detail: str) -> None:
    raise ContractError(detail, evidence={"stage": "safetensors-write"})


def _validate_payloads(tensors: Sequence[TensorPayload]) -> tuple[TensorPayload, ...]:
    values = tuple(tensors)
    if not values:
        _fail("a safetensors shard must contain at least one tensor")
    names: set[str] = set()
    for tensor in values:
        if not tensor.name or tensor.name == "__metadata__" or tensor.name in names:
            _fail(f"invalid or duplicate target tensor name: {tensor.name!r}")
        names.add(tensor.name)
        bits = DTYPE_BITS.get(tensor.dtype)
        if bits is None or any(type(dimension) is not int or dimension < 0 for dimension in tensor.shape):
            _fail(f"invalid descriptor for target tensor {tensor.name!r}")
        expected_bits = (0 if 0 in tensor.shape else prod(tensor.shape)) * bits
        if expected_bits % 8 or tensor.byte_length != expected_bits // 8:
            _fail(f"target tensor {tensor.name!r} byte length disagrees with dtype and shape")
        if not callable(tensor.chunks):
            _fail(f"target tensor {tensor.name!r} has no chunk producer")
    return values


def _header(tensors: tuple[TensorPayload, ...]) -> tuple[bytes, tuple[TensorDescriptor, ...]]:
    document: dict[str, object] = {}
    descriptors: list[TensorDescriptor] = []
    cursor = 0
    for tensor in tensors:
        end = cursor + tensor.byte_length
        document[tensor.name] = {
            "data_offsets": [cursor, end],
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
        }
        descriptors.append(TensorDescriptor(tensor.name, tensor.dtype, tensor.shape, (cursor, end)))
        cursor = end
    encoded = json.dumps(document, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    if not encoded or len(encoded) > MAX_HEADER_BYTES:
        _fail("generated safetensors header exceeds the supported size")
    return struct.pack("<Q", len(encoded)) + encoded, tuple(descriptors)


def _write_all(stream: BinaryIO, value: bytes) -> None:
    view = memoryview(value)
    while view:
        written = stream.write(view)
        if written is None or written <= 0:
            raise OSError("safetensors staging write made no progress")
        view = view[written:]


def write_safetensors_file(path: Path, tensors: Sequence[TensorPayload]) -> SafetensorsFileRecord:
    """Write one fresh shard durably; refuse overwrite and producer length drift."""

    values = _validate_payloads(tensors)
    prefix, descriptors = _header(values)
    digest = hashlib.sha256()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise ContractError(f"refusing to overwrite safetensors shard: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb", buffering=0) as stream:
            _write_all(stream, prefix)
            digest.update(prefix)
            for tensor in values:
                written = 0
                for raw_chunk in tensor.chunks():
                    if not isinstance(raw_chunk, bytes):
                        _fail(f"producer for {tensor.name!r} yielded a non-bytes chunk")
                    if written + len(raw_chunk) > tensor.byte_length:
                        _fail(f"producer for {tensor.name!r} exceeded {tensor.byte_length} bytes")
                    _write_all(stream, raw_chunk)
                    digest.update(raw_chunk)
                    written += len(raw_chunk)
                if written != tensor.byte_length:
                    _fail(f"producer for {tensor.name!r} wrote {written} bytes; expected {tensor.byte_length}")
            stream.flush()
            os.fsync(stream.fileno())
            size = os.fstat(stream.fileno()).st_size
    except OSError as exc:
        raise ContractError(f"safetensors staging write failed: {path}: {exc}") from exc
    if os.name != "nt":
        path.chmod(0o444)
    return SafetensorsFileRecord(path.name, size, digest.hexdigest(), descriptors)


def verify_safetensors_file(
    path: Path,
    expected_descriptors: Sequence[TensorDescriptor],
    expected_sha256: str,
) -> SafetensorsFileRecord:
    """Independently reopen, strictly parse, rehash, and compare one staged shard."""

    expected = tuple(expected_descriptors)
    with SafeTensorSources([path]) as sources:
        observed = tuple(item.descriptor for item in sources.tensors.values())
        if observed != expected:
            raise ContractError(
                f"independent safetensors descriptor verification failed: {path.name}",
                evidence={"stage": "safetensors-verify"},
            )
        if sources.hashes != [expected_sha256]:
            raise ContractError(
                f"independent safetensors SHA256 verification failed: {path.name}",
                evidence={"stage": "safetensors-verify"},
            )
        sources.verify_unchanged()
        return SafetensorsFileRecord(path.name, sources.sizes[0], sources.hashes[0], observed)


__all__ = [
    "SafetensorsFileRecord",
    "TensorPayload",
    "verify_safetensors_file",
    "write_safetensors_file",
]
