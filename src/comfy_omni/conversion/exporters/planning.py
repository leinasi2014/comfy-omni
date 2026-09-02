"""Fail-closed, read-only planning for the first H3 native export route.

No payload is decoded or written here. The planner turns an exact census and
an explicit source authority into a deterministic list of future operations.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from math import prod

from comfy_omni.artifacts import fileops
from comfy_omni.contracts.conversion import (
    EXPORT_SCHEMA,
    NATIVE_EXPORT_PROFILES,
    PROFILE_DENSE_BF16_ONLINE_INT8,
    NativeExportProfile,
)
from comfy_omni.contracts.models import ArchitectureTemplate, ContractError, ContractRecord
from comfy_omni.contracts.templates import template_digest
from comfy_omni.conversion.contract_workflows.census import CensusReport, schema_sha256
from comfy_omni.conversion.contract_workflows.matching import validate_level3
from comfy_omni.conversion.exporters.models import (
    NativeExportPlan,
    QkvLayoutPlan,
    ResourceEnvelope,
    ShardPlan,
    SourceBinding,
    TensorAction,
)
from comfy_omni.domain.checkpoints import TensorDescriptor
from comfy_omni.domain.qkv import qkv_to_grouped_row_indices

PLAN_SCHEMA = "comfy_omni.native_export.plan/v2"
DEFAULT_MAX_ROWS = 128
DEFAULT_MAX_SHARD_BYTES = 4 * 1024**3
MAX_ROWS_LIMIT = 4096
MAX_SHARD_BYTES_LIMIT = 16 * 1024**3

OP_COPY_RAW = "copy-raw"
OP_COPY_QKV_TO_GROUPED = "copy-runtime-qkv-to-grouped"
OP_INVERSE_CONVROT_BF16 = "inverse-convrot-to-bf16"
OP_INVERSE_CONVROT_BF16_QKV_TO_GROUPED = "inverse-convrot-to-bf16-runtime-qkv-to-grouped"
OP_OMIT_MARKER = "omit-comfy-quant-marker"
OP_OMIT_SCALE = "omit-source-rowwise-scale"


class ConversionPlanError(ContractError):
    """An export plan could not be authorized without assumptions."""


def _fail(message: str, stage: str, **evidence: object) -> None:
    raise ConversionPlanError(message, evidence={"stage": stage, **evidence})


def _is_sha256(value: str | None) -> bool:
    return value is not None and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_limits(max_rows: int, max_shard_bytes: int) -> None:
    if type(max_rows) is not int or not 1 <= max_rows <= MAX_ROWS_LIMIT:
        _fail(f"max_rows must be in 1..{MAX_ROWS_LIMIT}", "resource-envelope", max_rows=max_rows)
    if type(max_shard_bytes) is not int or not 1 <= max_shard_bytes <= MAX_SHARD_BYTES_LIMIT:
        _fail(
            f"max_shard_bytes must be in 1..{MAX_SHARD_BYTES_LIMIT}",
            "resource-envelope",
            max_shard_bytes=max_shard_bytes,
        )


def _validate_files(report: CensusReport) -> tuple[SourceBinding, ...]:
    if not report.files:
        _fail("an executable plan requires digest-bound source files", "source-binding")
    bindings = tuple(
        sorted(
            (SourceBinding(item.path, item.size, item.sha256) for item in report.files),
            key=lambda item: item.path,
        )
    )
    if len({item.path for item in bindings}) != len(bindings):
        _fail("source file paths are not unique", "source-binding")
    for item in bindings:
        if item.size <= 0 or not _is_sha256(item.sha256):
            _fail("source file binding is invalid", "source-binding", path=item.path)
    return bindings


def _validate_authority(
    report: CensusReport,
    record: ContractRecord,
    template: ArchitectureTemplate,
    profile: NativeExportProfile,
) -> tuple[str, str | None]:
    contract = record.contract
    if contract.schema_sha256 is None:
        _fail("source contract has no exact schema authorization", "contract-authorization", contract=contract.name)
    if record.storage_kind != profile.source_storage_kind or report.storage_kind != profile.source_storage_kind:
        _fail("source storage is not authorized by the output profile", "contract-authorization")
    if record.template_name != template.template_name or contract.component != template.component:
        _fail("source contract and architecture template disagree", "contract-authorization")
    if profile.component != contract.component:
        _fail("output profile is for a different component", "contract-authorization")
    observed_schema = schema_sha256(report.descriptors)
    expected = contract.schema_sha256
    checks = {
        "tensor_count": (contract.tensor_count, report.tensor_count),
        "convrot_group_count": (contract.convrot_group_count, report.convrot_group_count),
        "schema_sha256": (expected, report.observed_schema_sha256),
        "recomputed_schema_sha256": (expected, observed_schema),
    }
    mismatches = {
        name: {"expected": pair[0], "observed": pair[1]} for name, pair in checks.items() if pair[0] != pair[1]
    }
    if mismatches:
        _fail("source census does not match the exact contract", "contract-authorization", mismatches=mismatches)
    level3 = validate_level3(report, template)
    if not level3.passed:
        _fail("source fails its exact architecture template", "architecture-template", diff=level3.census_diff)
    snapshot = record.snapshot_manifest_sha256
    if snapshot is None:
        if record.snapshot_payload is not None:
            _fail("compile-time contract unexpectedly carries snapshot bytes", "contract-authorization")
        return "compile-time", None
    if not _is_sha256(snapshot) or not record.snapshot_payload:
        _fail("external contract snapshot is not digest-bound", "contract-authorization")
    return "external-snapshot", hashlib.sha256(record.snapshot_payload).hexdigest()


def _qkv_plan(profile: NativeExportProfile) -> QkvLayoutPlan:
    layout = profile.qkv
    indices = qkv_to_grouped_row_indices(
        num_query_groups=layout.num_query_groups,
        heads_per_group=layout.heads_per_group,
        head_dim=layout.head_dim,
    )
    digest = hashlib.sha256(fileops.canonical_json(list(indices))).hexdigest()
    return QkvLayoutPlan(
        layout.source_layout,
        layout.target_layout,
        layout.num_query_groups,
        layout.heads_per_group,
        layout.head_dim,
        layout.row_count,
        digest,
    )


def _raw_bytes(descriptor: TensorDescriptor) -> int:
    start, end = descriptor.data_offsets
    if start < 0 or end <= start:
        _fail("tensor has an invalid source byte range", "tensor-action", tensor=descriptor.name)
    return end - start


def _converted_bytes(descriptor: TensorDescriptor) -> int:
    if len(descriptor.shape) != 2 or any(dimension <= 0 for dimension in descriptor.shape):
        _fail("converted ConvRot weight must be a non-empty matrix", "tensor-action", tensor=descriptor.name)
    return prod(descriptor.shape) * 2


def _qkv_check(descriptor: TensorDescriptor, qkv: QkvLayoutPlan) -> None:
    if len(descriptor.shape) != 2 or descriptor.shape[0] != qkv.row_count:
        _fail(
            "QKV tensor rows disagree with the pinned loader layout",
            "qkv-layout",
            tensor=descriptor.name,
            expected_rows=qkv.row_count,
            observed_shape=list(descriptor.shape),
        )


def _action_for(
    descriptor: TensorDescriptor,
    group_role: tuple[str, str, int] | None,
    qkv: QkvLayoutPlan,
) -> TensorAction:
    source_bytes = _raw_bytes(descriptor)
    prefix, role, group_size = group_role or (None, "copy", None)
    is_qkv = descriptor.name.endswith(".attn.qkv_proj.weight")
    if role == "marker":
        return TensorAction(
            descriptor.name,
            None,
            descriptor.dtype,
            None,
            descriptor.shape,
            source_bytes,
            0,
            OP_OMIT_MARKER,
            prefix,
            group_size,
        )
    if role == "scale":
        return TensorAction(
            descriptor.name,
            None,
            descriptor.dtype,
            None,
            descriptor.shape,
            source_bytes,
            0,
            OP_OMIT_SCALE,
            prefix,
            group_size,
        )
    if role == "weight":
        if is_qkv:
            _qkv_check(descriptor, qkv)
        operation = OP_INVERSE_CONVROT_BF16_QKV_TO_GROUPED if is_qkv else OP_INVERSE_CONVROT_BF16
        return TensorAction(
            descriptor.name,
            descriptor.name,
            descriptor.dtype,
            "BF16",
            descriptor.shape,
            source_bytes,
            _converted_bytes(descriptor),
            operation,
            prefix,
            group_size,
        )
    if is_qkv:
        _qkv_check(descriptor, qkv)
        return TensorAction(
            descriptor.name,
            descriptor.name,
            descriptor.dtype,
            descriptor.dtype,
            descriptor.shape,
            source_bytes,
            source_bytes,
            OP_COPY_QKV_TO_GROUPED,
            descriptor.name.removesuffix(".weight"),
        )
    return TensorAction(
        descriptor.name,
        descriptor.name,
        descriptor.dtype,
        descriptor.dtype,
        descriptor.shape,
        source_bytes,
        source_bytes,
        OP_COPY_RAW,
    )


def _build_actions(report: CensusReport, qkv: QkvLayoutPlan) -> tuple[TensorAction, ...]:
    roles: dict[str, tuple[str, str, int]] = {}
    for group in report.groups:
        for name, role in (
            (group.weight.name, "weight"),
            (group.scale.name, "scale"),
            (group.marker.name, "marker"),
        ):
            if name in roles:
                _fail("a source tensor belongs to multiple ConvRot groups", "tensor-action", tensor=name)
            roles[name] = (group.prefix, role, group.group_size)
    actions = tuple(
        _action_for(item, roles.get(item.name), qkv)
        for item in sorted(report.descriptors, key=lambda descriptor: descriptor.name)
    )
    targets = [item.target_name for item in actions if item.target_name is not None]
    if len(targets) != len(set(targets)):
        _fail("planned target tensor names are not unique", "tensor-action")
    expected = report.tensor_count - 2 * report.convrot_group_count
    if len(targets) != expected:
        _fail("planned output census is inconsistent", "tensor-action", expected=expected, observed=len(targets))
    return actions


def _assign_shards(actions: tuple[TensorAction, ...], limit: int) -> tuple[ShardPlan, ...]:
    outputs = sorted(
        (item for item in actions if item.target_name is not None),
        key=lambda item: item.target_name or "",
    )
    too_large = [item.source_name for item in outputs if item.target_bytes > limit]
    if too_large:
        _fail("a target tensor exceeds the shard resource limit", "shard-plan", tensors=too_large[:8])
    groups: list[tuple[list[str], int]] = []
    names: list[str] = []
    payload_bytes = 0
    for item in outputs:
        assert item.target_name is not None
        if names and payload_bytes + item.target_bytes > limit:
            groups.append((names, payload_bytes))
            names, payload_bytes = [], 0
        names.append(item.target_name)
        payload_bytes += item.target_bytes
    if names:
        groups.append((names, payload_bytes))
    total = len(groups)
    return tuple(
        ShardPlan(f"model-{index:05d}-of-{total:05d}.safetensors", tuple(tensor_names), size)
        for index, (tensor_names, size) in enumerate(groups, start=1)
    )


def build_native_export_plan(
    report: CensusReport,
    record: ContractRecord,
    template: ArchitectureTemplate,
    *,
    profile_name: str = PROFILE_DENSE_BF16_ONLINE_INT8,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
) -> NativeExportPlan:
    """Authorize and describe an export without reading or writing payload tensors."""

    _validate_limits(max_rows, max_shard_bytes)
    profile = NATIVE_EXPORT_PROFILES.get(profile_name)
    if profile is None:
        _fail("unsupported native export profile", "profile", profile=profile_name)
    bindings = _validate_files(report)
    origin, snapshot_file_sha256 = _validate_authority(report, record, template, profile)
    qkv = _qkv_plan(profile)
    actions = _build_actions(report, qkv)
    shards = _assign_shards(actions, max_shard_bytes)
    target_actions = tuple(item for item in actions if item.target_name is not None)
    largest = max(item.target_bytes for item in target_actions)
    draft = NativeExportPlan(
        schema=PLAN_SCHEMA,
        output_schema=EXPORT_SCHEMA,
        component=record.contract.component,
        profile=profile.name,
        source_contract=record.name,
        source_contract_origin=origin,
        source_contract_schema_sha256=record.contract.schema_sha256 or "",
        source_snapshot_manifest_sha256=record.snapshot_manifest_sha256,
        source_snapshot_file_sha256=snapshot_file_sha256,
        template_name=template.template_name,
        template_version=template.template_version,
        template_sha256=template_digest(template),
        source_files=bindings,
        qkv_layout=qkv,
        resource_envelope=ResourceEnvelope(max_rows, max_shard_bytes, largest),
        actions=actions,
        shards=shards,
        target_tensor_count=len(target_actions),
        target_payload_bytes=sum(item.target_bytes for item in target_actions),
        runtime_quant_method=profile.runtime_quant_method,
        runtime_ignored_layers=profile.runtime_ignored_layers,
        payload_semantics=profile.payload_semantics,
        content_sha256="",
    )
    content_sha256 = hashlib.sha256(fileops.canonical_json(draft.to_dict(include_content_sha256=False))).hexdigest()
    return replace(draft, content_sha256=content_sha256)


__all__ = [
    "DEFAULT_MAX_ROWS",
    "DEFAULT_MAX_SHARD_BYTES",
    "PLAN_SCHEMA",
    "ConversionPlanError",
    "build_native_export_plan",
]
