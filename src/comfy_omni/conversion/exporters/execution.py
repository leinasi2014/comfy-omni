"""Execute every payload operation in a verified native-export plan."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable
from math import prod
from pathlib import Path
from typing import Any

from comfy_omni.artifacts import fileops
from comfy_omni.artifacts.safetensors_writer import (
    TensorPayload,
    verify_safetensors_file,
    write_safetensors_file,
)
from comfy_omni.artifacts.snapshot_schema import CONTRACT_SNAPSHOT_COPY_NAME, load_snapshot
from comfy_omni.artifacts.sources import SafeTensorSources
from comfy_omni.contracts.beta4 import BETA4_SOURCE_NAME, BETA4_TARGET_NAME
from comfy_omni.contracts.conversion import (
    EXPORT_SCHEMA,
    PROFILE_BETA4_DENSE_BF16,
    PROFILE_DENSE_BF16_ONLINE_INT8,
)
from comfy_omni.contracts.models import ContractError
from comfy_omni.conversion.contract_workflows.census import FileRecord, census_tensors
from comfy_omni.conversion.contract_workflows.convrot import discover_convrot_groups
from comfy_omni.conversion.exporters.beta4 import build_beta4_dense_plan
from comfy_omni.conversion.exporters.models import NativeExportPlan, TensorAction
from comfy_omni.conversion.exporters.payloads import (
    ConvRotBlockBackend,
    convrot_bf16_chunks,
    qkv_raw_chunks,
    validate_qkv_layout,
)
from comfy_omni.conversion.exporters.planning import (
    OP_COPY_QKV_TO_GROUPED,
    OP_COPY_RAW,
    OP_INVERSE_CONVROT_BF16,
    OP_INVERSE_CONVROT_BF16_QKV_TO_GROUPED,
    OP_OMIT_MARKER,
    OP_OMIT_SCALE,
    PLAN_SCHEMA,
)
from comfy_omni.conversion.numerics.serialization import torch_convrot_bf16_block
from comfy_omni.conversion.packaging.native_export import (
    NativeExportPublication,
    StagedArtifact,
    prepare_native_export,
    publish_native_export,
    stage_document,
)
from comfy_omni.domain.normalization import ToolIdentity

INDEX_NAME = "model.safetensors.index.json"
PLAN_NAME = "export.plan.json"
CONFIG_NAME = "config.patch.json"
_SHARD_PATTERN = re.compile(r"model-(\d{5})-of-(\d{5})\.safetensors")


def _fail(detail: str, stage: str) -> None:
    raise ContractError(detail, evidence={"stage": stage})


def _verify_plan_digest(plan: NativeExportPlan) -> None:
    if (
        plan.source_contract == BETA4_SOURCE_NAME or plan.target_contract == BETA4_TARGET_NAME
    ) and plan.profile != PROFILE_BETA4_DENSE_BF16:
        _fail("beta4 authority cannot be relabeled with another execution profile", "plan-binding")
    observed = hashlib.sha256(fileops.canonical_json(plan.to_dict(include_content_sha256=False))).hexdigest()
    if observed != plan.content_sha256:
        _fail("native export plan content SHA256 mismatch", "plan-binding")
    if (
        plan.schema != PLAN_SCHEMA
        or plan.output_schema != EXPORT_SCHEMA
        or plan.profile not in {PROFILE_DENSE_BF16_ONLINE_INT8, PROFILE_BETA4_DENSE_BF16}
    ):
        _fail("native export plan schema or profile is not executable by this slice", "plan-binding")
    snapshot_digests = (plan.source_snapshot_manifest_sha256, plan.source_snapshot_file_sha256)
    if plan.source_contract_origin == "compile-time":
        if any(value is not None for value in snapshot_digests):
            _fail("compile-time plan carries external snapshot digests", "plan-binding")
    elif plan.source_contract_origin == "external-snapshot":
        if any(value is None or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in snapshot_digests):
            _fail("external plan has incomplete snapshot digest bindings", "plan-binding")
    else:
        _fail("native export plan has an unknown contract authority origin", "plan-binding")
    if not plan.source_files or len({item.path for item in plan.source_files}) != len(plan.source_files):
        _fail("native export source file bindings are empty or duplicated", "plan-binding")
    for binding in plan.source_files:
        if (
            not Path(binding.path).is_absolute()
            or binding.size <= 0
            or re.fullmatch(r"[0-9a-f]{64}", binding.sha256) is None
        ):
            _fail(f"invalid native export source binding: {binding.path!r}", "plan-binding")


def _bind_snapshot(
    plan: NativeExportPlan,
    source_contract_snapshot: Path | None,
) -> tuple[bytes | None, dict[str, str] | None]:
    if plan.source_contract_origin == "compile-time":
        if source_contract_snapshot is not None:
            _fail("compile-time plan rejects a supplied contract snapshot", "snapshot-binding")
        return None, None
    if source_contract_snapshot is None:
        _fail("external plan requires its exact contract snapshot", "snapshot-binding")
    snapshot = load_snapshot(source_contract_snapshot)
    block = snapshot.contract_block
    pin_template = snapshot.document["pin"]["template"]
    file_digest = hashlib.sha256(snapshot.payload).hexdigest()
    expected = (
        plan.source_contract,
        plan.component,
        plan.source_contract_schema_sha256,
        plan.template_name,
        plan.template_version,
        plan.template_sha256,
        plan.source_snapshot_manifest_sha256,
        plan.source_snapshot_file_sha256,
    )
    observed = (
        block["name"],
        block["component"],
        block["schema_sha256"],
        block["template_name"],
        block["template_version"],
        pin_template["digest"],
        snapshot.manifest_sha256,
        file_digest,
    )
    if observed != expected:
        _fail("external contract snapshot disagrees with the immutable plan", "snapshot-binding")
    return snapshot.payload, {
        "manifest_sha256": snapshot.manifest_sha256,
        "name": plan.source_contract,
        "origin": "external-snapshot",
        "snapshot_file": CONTRACT_SNAPSHOT_COPY_NAME,
        "snapshot_file_sha256": file_digest,
    }


def _bind_sources(plan: NativeExportPlan, sources: SafeTensorSources) -> None:
    expected_paths = tuple(str(path) for path in sources.paths)
    planned_paths = tuple(item.path for item in plan.source_files)
    if planned_paths != expected_paths:
        _fail("native export source paths disagree with the immutable plan", "source-binding")
    observed = tuple(zip(sources.sizes, sources.hashes, strict=True))
    expected = tuple((item.size, item.sha256) for item in plan.source_files)
    if observed != expected:
        _fail("native export source size or SHA256 disagrees with the immutable plan", "source-binding")


def _validate_action(action: TensorAction, sources: SafeTensorSources) -> None:
    located = sources.tensors.get(action.source_name)
    if located is None:
        _fail(f"planned source tensor is missing: {action.source_name!r}", "action-binding")
    descriptor = located.descriptor
    source_bytes = descriptor.data_offsets[1] - descriptor.data_offsets[0]
    if (descriptor.dtype, descriptor.shape, source_bytes) != (
        action.source_dtype,
        action.shape,
        action.source_bytes,
    ):
        _fail(f"planned descriptor disagrees with source tensor: {action.source_name!r}", "action-binding")
    if action.operation == OP_COPY_RAW:
        if action.group_prefix is not None or action.group_size is not None:
            _fail(f"raw copy unexpectedly belongs to a ConvRot group: {action.source_name!r}", "action-binding")
        if (
            action.target_name is None
            or action.target_dtype != action.source_dtype
            or action.target_bytes != action.source_bytes
        ):
            _fail(f"copy action changes tensor semantics: {action.source_name!r}", "action-binding")
        return
    if action.operation == OP_COPY_QKV_TO_GROUPED:
        expected_prefix = action.source_name.removesuffix(".weight")
        if (
            not action.source_name.endswith(".attn.qkv_proj.weight")
            or action.group_prefix != expected_prefix
            or action.group_size is not None
            or action.target_name != action.source_name
            or action.target_dtype != action.source_dtype
            or action.target_bytes != action.source_bytes
            or len(action.shape) != 2
        ):
            _fail(f"QKV copy action has inconsistent semantics: {action.source_name!r}", "action-binding")
        return
    group_operations = {
        OP_INVERSE_CONVROT_BF16,
        OP_INVERSE_CONVROT_BF16_QKV_TO_GROUPED,
        OP_OMIT_MARKER,
        OP_OMIT_SCALE,
    }
    if action.operation not in group_operations:
        _fail(f"native export operation is unsupported: {action.operation!r}", "action-operation")
    if not action.group_prefix or type(action.group_size) is not int or action.group_size <= 0:
        _fail(f"ConvRot action lacks an exact group binding: {action.source_name!r}", "action-binding")
    if action.operation in {OP_OMIT_MARKER, OP_OMIT_SCALE}:
        if action.target_name is not None or action.target_dtype is not None or action.target_bytes != 0:
            _fail(f"omitted ConvRot metadata became an output: {action.source_name!r}", "action-binding")
        return
    expected_operation = (
        OP_INVERSE_CONVROT_BF16_QKV_TO_GROUPED
        if action.source_name.endswith(".attn.qkv_proj.weight")
        else OP_INVERSE_CONVROT_BF16
    )
    if (
        action.operation != expected_operation
        or action.source_name != f"{action.group_prefix}.weight"
        or action.source_dtype != "I8"
        or action.target_name != action.source_name
        or action.target_dtype != "BF16"
        or len(action.shape) != 2
        or action.target_bytes != prod(action.shape) * 2
    ):
        _fail(f"ConvRot weight action has inconsistent semantics: {action.source_name!r}", "action-binding")


def _validate_convrot_groups(
    actions: tuple[TensorAction, ...],
    sources: SafeTensorSources,
) -> dict[str, Any]:
    operations = {
        OP_INVERSE_CONVROT_BF16: "weight",
        OP_INVERSE_CONVROT_BF16_QKV_TO_GROUPED: "weight",
        OP_OMIT_SCALE: "scale",
        OP_OMIT_MARKER: "marker",
    }
    grouped: dict[str, dict[str, TensorAction]] = {}
    for action in actions:
        role = operations.get(action.operation)
        if role is None:
            continue
        assert action.group_prefix is not None
        roles = grouped.setdefault(action.group_prefix, {})
        if role in roles:
            _fail(f"duplicate ConvRot {role} action for {action.group_prefix!r}", "action-binding")
        roles[role] = action
    scales: dict[str, Any] = {}
    for prefix, roles in grouped.items():
        if set(roles) != {"weight", "scale", "marker"}:
            _fail(f"incomplete ConvRot action triplet for {prefix!r}", "action-binding")
        weight, scale, marker = (roles[role] for role in ("weight", "scale", "marker"))
        if len({weight.group_size, scale.group_size, marker.group_size}) != 1:
            _fail(f"ConvRot action group sizes disagree for {prefix!r}", "action-binding")
        if (
            weight.source_name != f"{prefix}.weight"
            or scale.source_name != f"{prefix}.weight_scale"
            or marker.source_name != f"{prefix}.comfy_quant"
            or scale.source_dtype != "F32"
            or marker.source_dtype != "U8"
        ):
            _fail(f"ConvRot action names or dtypes disagree for {prefix!r}", "action-binding")
        marker_tensor = sources.tensors[marker.source_name]
        group_size = weight.group_size
        assert group_size is not None
        discover_convrot_groups(
            tuple(sources.tensors[item.source_name].descriptor for item in (weight, scale, marker)),
            {marker.source_name: sources.read_raw(marker_tensor)},
            expected_groups=1,
            expected_group_sizes={prefix: group_size},
        )
        scales[prefix] = sources.tensors[scale.source_name]
    return scales


def _validate_plan(
    plan: NativeExportPlan,
    sources: SafeTensorSources,
) -> tuple[dict[str, TensorAction], dict[str, Any]]:
    _bind_sources(plan, sources)
    if plan.profile == PROFILE_BETA4_DENSE_BF16:
        report = census_tensors(
            tuple(item.descriptor for item in sources.tensors.values()),
            {name: sources.read_raw(item) for name, item in sources.tensors.items() if name.endswith(".comfy_quant")},
            files=tuple(
                FileRecord(str(path), size, digest)
                for path, size, digest in zip(
                    sources.paths,
                    sources.sizes,
                    sources.hashes,
                    strict=True,
                )
            ),
        )
        expected = build_beta4_dense_plan(
            report,
            max_rows=plan.resource_envelope.max_rows,
            max_shard_bytes=plan.resource_envelope.max_shard_bytes,
        )
        if expected != plan:
            _fail("beta4 plan disagrees with its reconstructed compiled authority", "plan-binding")
    qkv_operations = {OP_COPY_QKV_TO_GROUPED, OP_INVERSE_CONVROT_BF16_QKV_TO_GROUPED}
    if any(action.operation in qkv_operations for action in plan.actions):
        validate_qkv_layout(plan.qkv_layout)
    if len(plan.actions) != len(sources.tensors):
        _fail("native export action coverage is incomplete", "action-binding")
    actions: dict[str, TensorAction] = {}
    by_target: dict[str, TensorAction] = {}
    for action in plan.actions:
        if action.source_name in actions:
            _fail(f"duplicate source action: {action.source_name!r}", "action-binding")
        actions[action.source_name] = action
        _validate_action(action, sources)
        if action.target_name is None:
            continue
        if action.target_name in by_target:
            _fail(f"duplicate target action: {action.target_name!r}", "action-binding")
        by_target[action.target_name] = action
    if set(actions) != set(sources.tensors):
        _fail("native export action names do not exactly cover source tensors", "action-binding")
    if not by_target:
        _fail("native export plan has no target tensors", "action-binding")
    shard_targets: list[str] = []
    shard_count = len(plan.shards)
    for index, shard in enumerate(plan.shards, start=1):
        match = _SHARD_PATTERN.fullmatch(shard.name)
        if match is None or (int(match.group(1)), int(match.group(2))) != (index, shard_count):
            _fail(f"invalid shard name or sequence: {shard.name!r}", "shard-binding")
        if len(shard.tensor_names) != len(set(shard.tensor_names)):
            _fail(f"duplicate tensor in shard {shard.name!r}", "shard-binding")
        try:
            size = sum(by_target[name].target_bytes for name in shard.tensor_names)
        except KeyError as exc:
            _fail(f"shard references an unknown target tensor: {exc.args[0]!r}", "shard-binding")
        if size != shard.payload_bytes or size > plan.resource_envelope.max_shard_bytes:
            _fail(f"shard payload budget mismatch: {shard.name!r}", "shard-binding")
        shard_targets.extend(shard.tensor_names)
    if shard_targets != sorted(by_target):
        _fail("shards do not exactly cover targets in deterministic order", "shard-binding")
    if plan.target_tensor_count != len(by_target) or plan.target_payload_bytes != sum(
        item.target_bytes for item in by_target.values()
    ):
        _fail("native export target census disagrees with actions", "action-binding")
    largest = max(item.target_bytes for item in by_target.values())
    if largest != plan.resource_envelope.largest_target_tensor_bytes:
        _fail("native export largest tensor envelope is inconsistent", "action-binding")
    return by_target, _validate_convrot_groups(tuple(actions.values()), sources)


def _chunks_for_action(
    action: TensorAction,
    sources: SafeTensorSources,
    plan: NativeExportPlan,
    scales: dict[str, Any],
    convrot_backend: ConvRotBlockBackend,
) -> Iterable[bytes]:
    located = sources.tensors[action.source_name]
    if action.operation == OP_COPY_RAW:
        return sources.iter_raw(located)
    if action.operation == OP_COPY_QKV_TO_GROUPED:
        return qkv_raw_chunks(
            sources,
            located,
            action,
            plan.qkv_layout,
            max_rows=plan.resource_envelope.max_rows,
        )
    if action.operation in {OP_INVERSE_CONVROT_BF16, OP_INVERSE_CONVROT_BF16_QKV_TO_GROUPED}:
        assert action.group_prefix is not None
        return convrot_bf16_chunks(
            sources,
            located,
            scales[action.group_prefix],
            action,
            plan.qkv_layout,
            convrot_backend,
            max_rows=plan.resource_envelope.max_rows,
            reorder_qkv=action.operation == OP_INVERSE_CONVROT_BF16_QKV_TO_GROUPED,
        )
    _fail(f"target action has no payload producer: {action.operation!r}", "action-operation")


def _config_patch(plan: NativeExportPlan) -> dict[str, Any]:
    config = {
        "_comfy_omni": {
            "output_schema": plan.output_schema,
            "plan_content_sha256": plan.content_sha256,
            "profile": plan.profile,
        },
    }
    if plan.profile == PROFILE_BETA4_DENSE_BF16:
        config["quantization_config"] = None
    if plan.runtime_quant_method is not None:
        config["quantization_config"] = {
            "ignored_layers": list(plan.runtime_ignored_layers),
            "quant_method": plan.runtime_quant_method,
        }
    return config


def execute_native_export(
    plan: NativeExportPlan,
    output_dir: Path,
    *,
    tool: ToolIdentity,
    convrot_backend: ConvRotBlockBackend = torch_convrot_bf16_block,
    source_contract_snapshot: Path | None = None,
    before_publication: Callable[[Path], None] | None = None,
) -> NativeExportPublication:
    """Execute one exact plan through private staging and manifest-last publication."""

    _verify_plan_digest(plan)
    snapshot_payload, snapshot_record = _bind_snapshot(plan, source_contract_snapshot)
    stage = prepare_native_export(Path(output_dir))
    source_paths = tuple(Path(item.path) for item in plan.source_files)
    with SafeTensorSources(source_paths) as sources:
        by_target, scales = _validate_plan(plan, sources)
        staged: list[StagedArtifact] = []
        weight_map: dict[str, str] = {}
        for shard in plan.shards:
            payloads: list[TensorPayload] = []
            for target_name in shard.tensor_names:
                action = by_target[target_name]
                payloads.append(
                    TensorPayload(
                        target_name,
                        action.target_dtype or "",
                        action.shape,
                        action.target_bytes,
                        lambda action=action: _chunks_for_action(
                            action,
                            sources,
                            plan,
                            scales,
                            convrot_backend,
                        ),
                    )
                )
                weight_map[target_name] = shard.name
            written = write_safetensors_file(stage.path / shard.name, tuple(payloads))
            verified = verify_safetensors_file(
                stage.path / shard.name,
                written.descriptors,
                written.sha256,
            )
            if written != verified:
                _fail(f"staged shard changed after writing: {shard.name}", "staging")
            staged.append(
                StagedArtifact(shard.name, verified.size, verified.sha256, "safetensors", verified.tensor_count)
            )
        index = {"metadata": {"total_size": plan.target_payload_bytes}, "weight_map": weight_map}
        staged.append(stage_document(stage, INDEX_NAME, fileops.canonical_json(index), kind="safetensors-index"))
        staged.append(
            stage_document(stage, PLAN_NAME, fileops.canonical_json(plan.to_dict()), kind="native-export-plan")
        )
        staged.append(
            stage_document(
                stage,
                CONFIG_NAME,
                fileops.canonical_json(_config_patch(plan)),
                kind="runtime-config-patch",
            )
        )
        if snapshot_payload is not None:
            staged.append(
                stage_document(
                    stage,
                    CONTRACT_SNAPSHOT_COPY_NAME,
                    snapshot_payload,
                    kind="contract-snapshot",
                )
            )
        manifest = {
            "component": plan.component,
            "files": [item.to_dict() for item in sorted(staged, key=lambda value: value.name)],
            "output_schema": plan.output_schema,
            "plan_content_sha256": plan.content_sha256,
            "profile": plan.profile,
            "schema": "comfy_omni.native_export.receipt/v1",
            "source_files": [item.to_dict() for item in plan.source_files],
            "status": "COMMITTED",
            "target": {
                "payload_bytes": plan.target_payload_bytes,
                "tensor_count": plan.target_tensor_count,
            },
            "tool": tool.to_dict(),
        }
        if plan.profile == PROFILE_BETA4_DENSE_BF16:
            manifest["target"].update(contract=plan.target_contract, schema_sha256=plan.target_schema_sha256)
            manifest["runtime_quantization"] = plan.to_dict()["runtime_quantization"]
            manifest["qkv_layout"] = plan.qkv_layout.to_dict()
        if snapshot_record is not None:
            manifest["source_contract"] = snapshot_record
        if before_publication is not None:
            before_publication(stage.path)
        return publish_native_export(stage, tuple(staged), manifest, before_manifest=sources.verify_unchanged)


__all__ = [
    "CONFIG_NAME",
    "INDEX_NAME",
    "PLAN_NAME",
    "execute_native_export",
]
