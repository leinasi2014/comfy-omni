"""Read-only fixed TE planning and held-source reauthorization."""
from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from math import prod
from pathlib import Path

from comfy_omni.artifacts import fileops
from comfy_omni.artifacts.sources import SafeTensorSources
from comfy_omni.contracts import te_nvfp4 as contract
from comfy_omni.contracts.models import ContractError


@dataclass(frozen=True)
class TETensorPlan:
    source_name: str
    target_name: str
    operation: str
    shape: tuple[int, ...]
    byte_length: int


@dataclass(frozen=True)
class TEExportPlan:
    source_path: str
    config_path: str
    source_sha256: str
    source_bytes: int
    config_sha256: str
    config_bytes: int
    source_schema_sha256: str
    target_schema_sha256: str
    target_payload_bytes: int
    tensors: tuple[TETensorPlan, ...]
    profile: str = contract.PROFILE
    consumer: str = contract.CONSUMER
    max_rows: int = contract.MAX_ROWS
    schema: str = "comfy_omni.te_dense.plan/v1"
    content_sha256: str = ""

    def to_dict(self, *, include_content_sha256: bool = True) -> dict:
        document = asdict(self)
        if not include_content_sha256:
            document.pop("content_sha256")
        return document


@contextmanager
def held_config(path: Path) -> Iterator[tuple[bytes, Callable[[], None]]]:
    """Retain the tiny exact config FD throughout hashing, conversion and final checks."""
    path = fileops.reject_linked_ancestors(path)
    descriptor, before = fileops._open_pinned(path)
    identity = fileops.fd_identity(before)

    def verify() -> None:
        os.lseek(descriptor, 0, os.SEEK_SET)
        final = os.read(descriptor, contract.CONFIG_BYTES + 1)
        if (
            fileops.fd_identity(os.fstat(descriptor)) != identity
            or fileops.fd_identity(path.lstat()) != identity
            or hashlib.sha256(final).hexdigest() != contract.CONFIG_SHA256
        ):
            raise ContractError("TE config changed while held")

    try:
        if before.st_size != contract.CONFIG_BYTES:
            raise ContractError("TE config size is not the fixed asset")
        raw = os.read(descriptor, contract.CONFIG_BYTES + 1)
        if len(raw) != contract.CONFIG_BYTES or hashlib.sha256(raw).hexdigest() != contract.CONFIG_SHA256:
            raise ContractError("TE config SHA256 is not the fixed asset")
        if not isinstance(fileops.parse_json_strict(raw), dict):
            raise ContractError("TE config must be an object")
        yield raw, verify
    finally:
        try:
            verify()
        finally:
            os.close(descriptor)


def plan_from_held(source: SafeTensorSources, config_path: Path) -> TEExportPlan:
    if len(source.paths) != 1 or source.sizes != [contract.SOURCE_BYTES] or source.hashes != [contract.SOURCE_SHA256]:
        raise ContractError("TE source size or SHA256 is not the fixed strict asset")
    observed = {name: (located.descriptor.dtype, located.descriptor.shape) for name, located in source.tensors.items()}
    if observed != contract.SOURCE_INVENTORY or contract.schema_sha256(observed) != contract.SOURCE_SCHEMA_SHA256:
        raise ContractError("TE source descriptor inventory is not the complete fixed 1954 schema")
    plans = []
    consumed = set()
    for name, (dtype, shape) in sorted(observed.items()):
        if name in consumed:
            continue
        if name.endswith((".comfy_quant", ".weight_scale", ".weight_scale_2")):
            continue
        if dtype == "BF16":
            operation = "copy-bf16"
            consumed.add(name)
        else:
            module = name.removesuffix(".weight")
            marker_name = module + ".comfy_quant"
            marker = source.tensors[marker_name]
            if marker.descriptor.data_offsets[1] - marker.descriptor.data_offsets[0] > 4096:
                raise ContractError("TE quantization declaration exceeds its bound")
            declaration = fileops.parse_json_strict(source.read_raw(marker))
            if name == "model.embed_tokens.weight":
                operation = "int8-f32-to-bf16"
                expected = {"format": "int8_tensorwise"}
                members = {name, marker_name, module + ".weight_scale"}
            else:
                operation = "nvfp4-blocked-to-bf16"
                expected = {"format": "nvfp4"}
                members = {name, marker_name, module + ".weight_scale", module + ".weight_scale_2"}
                shape = (shape[0], shape[1] * 2)
                if shape[0] % 128 or shape[1] % 64:
                    raise ContractError("fixed TE stripe geometry is invalid")
            if declaration != expected or not members <= observed.keys() or members & consumed:
                raise ContractError("TE quantization group is incomplete, duplicated or unsupported")
            consumed.update(members)
        plans.append(TETensorPlan(name, contract.native_name(name), operation, shape, prod(shape) * 2))
    targets = {item.target_name: ("BF16", item.shape) for item in plans}
    if (
        consumed != observed.keys() or len(targets) != len(plans)
        or targets != contract.TARGET_INVENTORY
        or contract.schema_sha256(targets) != contract.TARGET_SCHEMA_SHA256
        or sum(item.byte_length for item in plans) != contract.TARGET_PAYLOAD_BYTES
    ):
        raise ContractError("TE source accounting or complete native target schema disagrees")
    draft = TEExportPlan(
        str(source.paths[0]), str(fileops.reject_linked_ancestors(config_path)),
        source.hashes[0], source.sizes[0], contract.CONFIG_SHA256, contract.CONFIG_BYTES,
        contract.SOURCE_SCHEMA_SHA256, contract.TARGET_SCHEMA_SHA256, contract.TARGET_PAYLOAD_BYTES,
        tuple(sorted(plans, key=lambda item: item.target_name)),
    )
    return replace(draft, content_sha256=hashlib.sha256(fileops.canonical_json(draft.to_dict(include_content_sha256=False))).hexdigest())


def plan_te_dense_export(source_path: Path, config_path: Path) -> TEExportPlan:
    """Authorize only the exact fixed TE, retaining all source descriptors through validation."""
    with held_config(config_path), SafeTensorSources([source_path]) as source:
        try:
            return plan_from_held(source, config_path)
        finally:
            source.verify_unchanged()
