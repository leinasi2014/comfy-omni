"""Execute the verified copy-only portion of an immutable native export plan.

This module deliberately rejects numerical and QKV operations. Later Issue #8 slices can add
producers without weakening the source, shard, verification, or publication transaction.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from comfy_omni.artifacts import fileops
from comfy_omni.artifacts.safetensors_writer import (
    TensorPayload,
    verify_safetensors_file,
    write_safetensors_file,
)
from comfy_omni.artifacts.sources import SafeTensorSources
from comfy_omni.contracts.conversion import (
    EXPORT_SCHEMA,
    PROFILE_DENSE_BF16_ONLINE_INT8,
)
from comfy_omni.contracts.models import ContractError
from comfy_omni.conversion.exporters.models import NativeExportPlan, TensorAction
from comfy_omni.conversion.exporters.planning import OP_COPY_RAW, PLAN_SCHEMA
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
    observed = hashlib.sha256(fileops.canonical_json(plan.to_dict(include_content_sha256=False))).hexdigest()
    if observed != plan.content_sha256:
        _fail("native export plan content SHA256 mismatch", "plan-binding")
    if (
        plan.schema != PLAN_SCHEMA
        or plan.output_schema != EXPORT_SCHEMA
        or plan.profile != PROFILE_DENSE_BF16_ONLINE_INT8
    ):
        _fail("native export plan schema or profile is not executable by this slice", "plan-binding")
    if plan.source_contract_origin != "compile-time" or any(
        value is not None
        for value in (plan.source_snapshot_manifest_sha256, plan.source_snapshot_file_sha256)
    ):
        _fail("external contract snapshot publication is not implemented by this slice", "plan-binding")
    if not plan.source_files or len({item.path for item in plan.source_files}) != len(plan.source_files):
        _fail("native export source file bindings are empty or duplicated", "plan-binding")
    for binding in plan.source_files:
        if (
            not Path(binding.path).is_absolute()
            or binding.size <= 0
            or re.fullmatch(r"[0-9a-f]{64}", binding.sha256) is None
        ):
            _fail(f"invalid native export source binding: {binding.path!r}", "plan-binding")


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
    if action.operation != OP_COPY_RAW:
        _fail(f"transaction slice does not implement operation {action.operation!r}", "action-operation")
    if (
        action.target_name is None
        or action.target_dtype != action.source_dtype
        or action.target_bytes != action.source_bytes
    ):
        _fail(f"copy action changes tensor semantics: {action.source_name!r}", "action-binding")


def _validate_plan(plan: NativeExportPlan, sources: SafeTensorSources) -> dict[str, TensorAction]:
    _bind_sources(plan, sources)
    if len(plan.actions) != len(sources.tensors):
        _fail("native export action coverage is incomplete", "action-binding")
    actions: dict[str, TensorAction] = {}
    by_target: dict[str, TensorAction] = {}
    for action in plan.actions:
        if action.source_name in actions:
            _fail(f"duplicate source action: {action.source_name!r}", "action-binding")
        actions[action.source_name] = action
        _validate_action(action, sources)
        assert action.target_name is not None
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
    return by_target


def _config_patch(plan: NativeExportPlan) -> dict[str, Any]:
    return {
        "_comfy_omni": {
            "output_schema": plan.output_schema,
            "plan_content_sha256": plan.content_sha256,
            "profile": plan.profile,
        },
        "quantization_config": {
            "ignored_layers": list(plan.runtime_ignored_layers),
            "quant_method": plan.runtime_quant_method,
        },
    }


def execute_native_export(
    plan: NativeExportPlan,
    output_dir: Path,
    *,
    tool: ToolIdentity,
) -> NativeExportPublication:
    """Execute a copy-only plan through private staging and manifest-last publication."""

    _verify_plan_digest(plan)
    stage = prepare_native_export(Path(output_dir))
    source_paths = tuple(Path(item.path) for item in plan.source_files)
    with SafeTensorSources(source_paths) as sources:
        by_target = _validate_plan(plan, sources)
        staged: list[StagedArtifact] = []
        weight_map: dict[str, str] = {}
        for shard in plan.shards:
            payloads: list[TensorPayload] = []
            for target_name in shard.tensor_names:
                action = by_target[target_name]
                located = sources.tensors[action.source_name]
                payloads.append(
                    TensorPayload(
                        target_name,
                        action.target_dtype or "",
                        action.shape,
                        action.target_bytes,
                        lambda located=located: sources.iter_raw(located),
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
        sources.verify_unchanged()
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
        return publish_native_export(stage, tuple(staged), manifest)


__all__ = [
    "CONFIG_NAME",
    "INDEX_NAME",
    "PLAN_NAME",
    "execute_native_export",
]
