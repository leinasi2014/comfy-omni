"""Direct, in-memory loading of authenticated H3 ConvRot transformer sources.

The primary beta4 contract remains exactly BF16. The shared reader also
preserves native F16/F32 passthrough weights for the standard H3 binding.
Establishment hashes the read-only source once; later loads only recheck its
file-descriptor/path identity. No safetensors export is created.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Any

from comfy_omni.artifacts import fileops
from comfy_omni.artifacts.safetensors import read_safetensors_header_stream
from comfy_omni.artifacts.sources import SafeTensorSources
from comfy_omni.contracts.beta4 import (
    BETA4_SOURCE_BYTES,
    BETA4_SOURCE_NAME,
    BETA4_SOURCE_SCHEMA_SHA256,
    BETA4_SOURCE_SHA256,
    BETA4_TARGET_INVENTORY,
    BETA4_TARGET_SCHEMA_SHA256,
)
from comfy_omni.contracts.models import ContractError
from comfy_omni.conversion.contract_workflows.census import FileRecord, census_tensors, schema_sha256
from comfy_omni.conversion.exporters.beta4 import build_beta4_dense_plan
from comfy_omni.conversion.exporters.models import NativeExportPlan, TensorAction
from comfy_omni.conversion.exporters.payloads import validate_qkv_layout
from comfy_omni.conversion.exporters.planning import (
    OP_COPY_QKV_TO_GROUPED,
    OP_COPY_RAW,
    OP_INVERSE_CONVROT_BF16,
    OP_INVERSE_CONVROT_BF16_QKV_TO_GROUPED,
    OP_OMIT_SCALE,
)
from comfy_omni.conversion.numerics.serialization import torch_convrot_bf16_block
from comfy_omni.domain.checkpoints import TensorDescriptor

_FLOAT_BYTES = {"BF16": 2, "F16": 2, "F32": 4}


def _fail(detail: str, *, stage: str) -> None:
    raise ContractError(detail, evidence={"stage": stage})


@dataclass(frozen=True)
class RawBeta4Identity:
    """Trusted identity captured by an earlier full-source verification."""

    name: str
    size: int
    sha256: str
    source_schema_sha256: str
    tensor_count: int | None = None
    target_schema_sha256: str | None = None
    target_tensor_count: int | None = None


PRIMARY_RAW_BETA4_IDENTITY = RawBeta4Identity(
    BETA4_SOURCE_NAME,
    BETA4_SOURCE_BYTES,
    BETA4_SOURCE_SHA256,
    BETA4_SOURCE_SCHEMA_SHA256,
    934,
    BETA4_TARGET_SCHEMA_SHA256,
    534,
)


def primary_target_descriptors() -> tuple[TensorDescriptor, ...]:
    """Return the immutable 534 logical host input descriptors."""

    return tuple(
        TensorDescriptor(name, dtype, shape, (0, prod(shape) * 2))
        for name, (dtype, shape) in sorted(BETA4_TARGET_INVENTORY.items())
    )


def _identity(path: Path) -> tuple[int, int, int, int, int]:
    try:
        return fileops.fd_identity(path.lstat())
    except OSError as exc:
        raise ContractError(
            f"raw beta4 source could not be inspected: {path}", evidence={"stage": "raw-identity"}
        ) from exc


def _assert_same_identity(path: Path, stream: Any, expected: tuple[int, int, int, int, int]) -> None:
    try:
        opened = fileops.fd_identity(os.fstat(stream.fileno()))
        named = _identity(path)
    except OSError as exc:
        raise ContractError(
            "raw beta4 source identity could not be rechecked", evidence={"stage": "raw-identity"}
        ) from exc
    if opened != expected or named != expected:
        _fail("raw beta4 source identity changed since authentication", stage="raw-identity")


def _action_bytes(action: TensorAction) -> int:
    return prod(action.shape) * _FLOAT_BYTES[action.target_dtype]


def _actions_by_target(plan: NativeExportPlan) -> tuple[TensorAction, ...]:
    actions = [action for action in plan.actions if action.target_name is not None]
    if not actions or any(action.target_dtype not in _FLOAT_BYTES for action in actions):
        _fail("raw H3 plan must expose non-empty BF16/F16/F32 targets", stage="raw-plan")
    names = [action.target_name for action in actions]
    if len(names) != len(set(names)):
        _fail("raw beta4 plan has duplicate target names", stage="raw-plan")
    if any(action.target_bytes != _action_bytes(action) for action in actions):
        _fail("raw beta4 plan has an invalid target byte geometry", stage="raw-plan")
    for action in actions:
        if action.operation in {OP_COPY_RAW, OP_COPY_QKV_TO_GROUPED} and (
            action.target_dtype != action.source_dtype or action.target_bytes != action.source_bytes
        ):
            _fail("raw H3 copy must preserve its source dtype and byte size", stage="raw-plan")
        if action.operation in {OP_INVERSE_CONVROT_BF16, OP_INVERSE_CONVROT_BF16_QKV_TO_GROUPED} and (
            action.target_dtype != "BF16"
        ):
            _fail("raw H3 inverse ConvRot targets must be BF16", stage="raw-plan")
    return tuple(sorted(actions, key=lambda action: action.target_name or ""))


def _validate_plan(
    plan: NativeExportPlan,
    identity: RawBeta4Identity,
    path: Path,
    descriptors: tuple[TensorDescriptor, ...],
) -> None:
    if len(plan.source_files) != 1:
        _fail("raw beta4 plan must bind exactly one source", stage="raw-plan")
    source_file = plan.source_files[0]
    if (source_file.path, source_file.size, source_file.sha256) != (
        str(path),
        identity.size,
        identity.sha256,
    ):
        _fail("raw beta4 plan source differs from trusted identity", stage="raw-plan")
    targets = _actions_by_target(plan)
    if identity.target_tensor_count is not None and len(targets) != identity.target_tensor_count:
        _fail("raw beta4 target tensor count differs from trusted identity", stage="raw-plan")
    if identity.target_schema_sha256 is not None and plan.target_schema_sha256 != identity.target_schema_sha256:
        _fail("raw beta4 target schema differs from trusted identity", stage="raw-plan")
    if identity.name == BETA4_SOURCE_NAME:
        observed = {action.target_name: (action.target_dtype, action.shape) for action in targets}
        if observed != dict(BETA4_TARGET_INVENTORY):
            _fail("raw beta4 target geometry differs from the fixed 534-slot contract", stage="raw-plan")
    by_source = {item.name: item for item in descriptors}
    action_names = {action.source_name for action in plan.actions}
    if set(by_source) != action_names or len(plan.actions) != len(action_names):
        _fail("raw beta4 actions do not exactly cover source descriptors", stage="raw-plan")
    for action in plan.actions:
        source = by_source[action.source_name]
        source_bytes = source.data_offsets[1] - source.data_offsets[0]
        if (source.dtype, source.shape, source_bytes) != (
            action.source_dtype,
            action.shape,
            action.source_bytes,
        ):
            _fail("raw beta4 action differs from source descriptor geometry", stage="raw-plan")


def _read_into(stream: Any, offset: int, target: memoryview) -> None:
    stream.seek(offset)
    written = 0
    while written < len(target):
        count = stream.readinto(target[written:])
        if not count:
            _fail("raw beta4 source payload is truncated", stage="raw-payload")
        written += count


def _read_exact(stream: Any, offset: int, size: int) -> bytes:
    stream.seek(offset)
    result = stream.read(size)
    if len(result) != size:
        _fail("raw beta4 source payload is truncated", stage="raw-payload")
    return result


def _runs(order: tuple[int, ...], max_rows: int) -> Iterator[tuple[int, int]]:
    if not order or type(max_rows) is not int or max_rows <= 0:
        _fail("raw beta4 has an invalid bounded row schedule", stage="raw-plan")
    start = previous = order[0]
    count = 1
    for current in order[1:]:
        if current == previous + 1 and count < max_rows:
            count += 1
        else:
            yield start, count
            start, count = current, 1
        previous = current
    yield start, count


@dataclass(frozen=True)
class RawBeta4Binding:
    """A fully authenticated source plus immutable output geometry.

    ``establish`` is the only operation that computes a whole-file digest.
    ``open_weights`` yields complete tensors and releases its previous target
    before allocating the next. A streaming host loop can still retain the
    previous target while the current target is built: budget both targets,
    bounded decoder scratch, and any explicit consumer cache or dtype cast.
    The provider never materializes a checkpoint-sized object or writes an
    intermediate artifact.
    """

    source_path: Path
    trusted_identity: RawBeta4Identity
    source_file_identity: tuple[int, int, int, int, int]
    source_descriptors: tuple[TensorDescriptor, ...]
    target_descriptors: tuple[TensorDescriptor, ...]
    plan: NativeExportPlan

    @classmethod
    def establish(
        cls,
        source_path: Path | str,
        *,
        identity: RawBeta4Identity = PRIMARY_RAW_BETA4_IDENTITY,
    ) -> RawBeta4Binding:
        path = fileops.reject_linked_ancestors(Path(source_path)).resolve(strict=True)
        with SafeTensorSources((path,)) as sources:
            if sources.sizes != [identity.size] or sources.hashes != [identity.sha256]:
                _fail("raw beta4 source size or SHA256 differs from trusted identity", stage="raw-authentication")
            descriptors = tuple(
                sorted((item.descriptor for item in sources.tensors.values()), key=lambda item: item.name)
            )
            if schema_sha256(descriptors) != identity.source_schema_sha256:
                _fail("raw beta4 source descriptor schema differs from trusted identity", stage="raw-authentication")
            if identity.tensor_count is not None and len(descriptors) != identity.tensor_count:
                _fail("raw beta4 source tensor count differs from trusted identity", stage="raw-authentication")
            report = census_tensors(
                descriptors,
                {
                    name: sources.read_raw(item)
                    for name, item in sources.tensors.items()
                    if name.endswith(".comfy_quant")
                },
                files=(FileRecord(str(path), sources.sizes[0], sources.hashes[0]),),
            )
            plan = build_beta4_dense_plan(report)
            _validate_plan(plan, identity, path, descriptors)
            # SafeTensorSources has already guarded this open descriptor while
            # hashing. Preserve that identity and rebind its path after close;
            # do not call verify_unchanged(), which intentionally rehashes.
            source_file_identity = sources._sources[0].identity
        if _identity(path) != source_file_identity:
            _fail("raw beta4 source changed after authentication", stage="raw-authentication")
        targets = _actions_by_target(plan)
        target_descriptors = tuple(
            TensorDescriptor(
                action.target_name or "", action.target_dtype or "", action.shape, (0, action.target_bytes)
            )
            for action in targets
        )
        return cls(path, identity, source_file_identity, descriptors, target_descriptors, plan)

    def open_weights(self) -> Iterator[tuple[str, Any]]:
        """Yield sorted CPU tensors in each authenticated action's target dtype."""

        return self._iter_weights()

    def verify_unchanged(self) -> None:
        """Perform the repeatable identity/header check without rehashing payloads."""

        path = fileops.reject_linked_ancestors(self.source_path)
        if _identity(path) != self.source_file_identity:
            _fail("raw beta4 source identity changed since authentication", stage="raw-identity")
        with path.open("rb", buffering=0) as stream:
            _assert_same_identity(path, stream, self.source_file_identity)
            _, descriptors, _ = read_safetensors_header_stream(stream, path, self.trusted_identity.size)
            descriptors = tuple(sorted(descriptors, key=lambda item: item.name))
            if (
                descriptors != self.source_descriptors
                or schema_sha256(descriptors) != self.trusted_identity.source_schema_sha256
            ):
                _fail("raw beta4 source header geometry changed since authentication", stage="raw-header")
            _assert_same_identity(path, stream, self.source_file_identity)

    def _iter_weights(self) -> Iterator[tuple[str, Any]]:
        import torch

        torch_dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
        path = fileops.reject_linked_ancestors(self.source_path)
        self.verify_unchanged()
        with path.open("rb", buffering=0) as stream:
            _assert_same_identity(path, stream, self.source_file_identity)
            _, descriptors, header_size = read_safetensors_header_stream(stream, path, self.trusted_identity.size)
            descriptors = tuple(sorted(descriptors, key=lambda item: item.name))
            if (
                descriptors != self.source_descriptors
                or schema_sha256(descriptors) != self.trusted_identity.source_schema_sha256
            ):
                _fail("raw beta4 source header geometry changed since authentication", stage="raw-header")
            located = {item.name: item for item in descriptors}
            payload_offset = 8 + header_size
            actions = _actions_by_target(self.plan)
            scales = {action.group_prefix: action for action in self.plan.actions if action.operation == OP_OMIT_SCALE}
            for action in actions:
                output = torch.empty(action.shape, dtype=torch_dtypes[action.target_dtype], device="cpu")
                destination = memoryview(output.view(torch.uint8).numpy()).cast("B")
                source = located[action.source_name]
                source_offset = payload_offset + source.data_offsets[0]
                if action.operation == OP_COPY_RAW:
                    _read_into(stream, source_offset, destination)
                elif action.operation == OP_COPY_QKV_TO_GROUPED:
                    rows, columns = action.shape
                    order = validate_qkv_layout(self.plan.qkv_layout)
                    if len(order) != rows:
                        _fail("raw beta4 QKV rows disagree with the bound layout", stage="raw-plan")
                    position = 0
                    element_bytes = output.element_size()
                    for start, count in _runs(order, self.plan.resource_envelope.max_rows):
                        size = count * columns * element_bytes
                        _read_into(
                            stream,
                            source_offset + start * columns * element_bytes,
                            destination[position : position + size],
                        )
                        position += size
                elif action.operation in {OP_INVERSE_CONVROT_BF16, OP_INVERSE_CONVROT_BF16_QKV_TO_GROUPED}:
                    if action.group_prefix is None or action.group_size is None:
                        _fail("raw beta4 ConvRot action lacks a group binding", stage="raw-plan")
                    scale_action = scales.get(action.group_prefix)
                    if scale_action is None:
                        _fail("raw beta4 ConvRot action lacks its scale", stage="raw-plan")
                    scale = located[scale_action.source_name]
                    scale_offset = payload_offset + scale.data_offsets[0]
                    rows, columns = action.shape
                    order = (
                        validate_qkv_layout(self.plan.qkv_layout)
                        if action.operation == OP_INVERSE_CONVROT_BF16_QKV_TO_GROUPED
                        else tuple(range(rows))
                    )
                    position = 0
                    for start, count in _runs(order, self.plan.resource_envelope.max_rows):
                        block = torch_convrot_bf16_block(
                            _read_exact(stream, source_offset + start * columns, count * columns),
                            _read_exact(stream, scale_offset + start * 4, count * 4),
                            rows=count,
                            columns=columns,
                            group_size=action.group_size,
                        )
                        destination[position : position + len(block)] = block
                        position += len(block)
                else:
                    _fail("raw beta4 plan has an unsupported target action", stage="raw-plan")
                _assert_same_identity(path, stream, self.source_file_identity)
                try:
                    yield action.target_name or "", output
                finally:
                    destination.release()
                    del destination, output
            _assert_same_identity(path, stream, self.source_file_identity)


__all__ = [
    "PRIMARY_RAW_BETA4_IDENTITY",
    "RawBeta4Binding",
    "RawBeta4Identity",
    "primary_target_descriptors",
]
