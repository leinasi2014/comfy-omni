"""Pinned, read-only safetensors source-set access for contract workflows."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from comfy_omni.artifacts.fileops import HASH_CHUNK_BYTES, fd_identity, reject_linked_ancestors
from comfy_omni.artifacts.safetensors import read_safetensors_header_stream
from comfy_omni.contracts.models import ContractError
from comfy_omni.domain.checkpoints import TensorDescriptor

INDEX_NAME = "model.safetensors.index.json"
MAX_SOURCE_FILES = 128
MAX_TOTAL_SOURCE_BYTES = 64 * 1024**4


@dataclass(frozen=True)
class LocatedTensor:
    descriptor: TensorDescriptor
    source_index: int
    payload_offset: int


@dataclass
class _OpenSource:
    path: Path
    stream: BinaryIO
    identity: tuple[int, int, int, int, int]
    size: int
    sha256: str


def _hash_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    stream.seek(0)
    while chunk := stream.read(HASH_CHUNK_BYTES):
        digest.update(chunk)
    return digest.hexdigest()


def _path_identity(path: Path) -> tuple[int, int, int]:
    try:
        status = path.lstat()
        return status.st_dev, status.st_ino, status.st_size
    except OSError as exc:
        raise ContractError(f"source path could not be inspected: {path}", evidence={"stage": "source-open"}) from exc


class SafeTensorSources:
    """One logical source held through stable descriptors for the entire scan."""

    def __init__(self, paths: Sequence[Path]) -> None:
        if not paths or len(paths) > MAX_SOURCE_FILES:
            raise ContractError(f"source file count must be in 1..{MAX_SOURCE_FILES}")
        self.paths = tuple(reject_linked_ancestors(Path(path)) for path in paths)
        if len(set(self.paths)) != len(self.paths):
            raise ContractError("duplicate source safetensors path")
        self.tensors: dict[str, LocatedTensor] = {}
        self.metadata: list[dict[str, str]] = []
        self.hashes: list[str] = []
        self.sizes: list[int] = []
        self._sources: list[_OpenSource] = []
        try:
            self._open_all()
        except Exception:
            self.close()
            raise

    def _open_all(self) -> None:
        total_size = 0
        for source_index, path in enumerate(self.paths):
            before = path.lstat()
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
                raise ContractError(f"source must be a non-linked regular file: {path}")
            stream = path.open("rb", buffering=0)
            opened = os.fstat(stream.fileno())
            identity = fd_identity(opened)
            if (before.st_dev, before.st_ino, before.st_size) != identity[:3]:
                stream.close()
                raise ContractError(f"source changed while opening: {path}")
            total_size += opened.st_size
            if total_size > MAX_TOTAL_SOURCE_BYTES:
                stream.close()
                raise ContractError("aggregate source size exceeds safety limit")
            self._ingest(source_index, path, stream, identity, opened.st_size)

    def _ingest(
        self, source_index: int, path: Path, stream: BinaryIO, identity: tuple[int, int, int, int, int], size: int
    ) -> None:
        metadata, descriptors, header_length = read_safetensors_header_stream(stream, path, size)
        payload_offset = 8 + header_length
        for descriptor in descriptors:
            if descriptor.name in self.tensors:
                stream.close()
                raise ContractError(f"duplicate tensor across source shards: {descriptor.name!r}")
            self.tensors[descriptor.name] = LocatedTensor(descriptor, source_index, payload_offset)
        digest = _hash_stream(stream)
        if fd_identity(os.fstat(stream.fileno())) != identity or _path_identity(path) != identity[:3]:
            stream.close()
            raise ContractError(f"source changed during hashing: {path}")
        self._sources.append(_OpenSource(path, stream, identity, size, digest))
        self.metadata.append(metadata)
        self.hashes.append(digest)
        self.sizes.append(size)

    def read_raw(self, tensor: LocatedTensor) -> bytes:
        start, end = tensor.descriptor.data_offsets
        source = self._sources[tensor.source_index]
        source.stream.seek(tensor.payload_offset + start)
        raw = source.stream.read(end - start)
        if len(raw) != end - start:
            raise ContractError(f"truncated source tensor {tensor.descriptor.name!r}")
        return raw

    def verify_unchanged(self) -> None:
        for source in self._sources:
            if fd_identity(os.fstat(source.stream.fileno())) != source.identity:
                raise ContractError(f"source descriptor changed during scan: {source.path}")
            if _path_identity(source.path) != source.identity[:3]:
                raise ContractError(f"source path changed during scan: {source.path}")

    def close(self) -> None:
        for source in self._sources:
            source.stream.close()
        self._sources.clear()

    def __enter__(self) -> SafeTensorSources:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["INDEX_NAME", "LocatedTensor", "SafeTensorSources"]
