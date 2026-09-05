"""One fixed offline TE producer: bounded payloads, strict reread, receipt last."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from comfy_omni.artifacts import fileops
from comfy_omni.artifacts.safetensors_writer import TensorPayload, verify_safetensors_file, write_safetensors_file
from comfy_omni.artifacts.sources import SafeTensorSources
from comfy_omni.contracts import te_nvfp4 as contract
from comfy_omni.contracts.models import ContractError
from comfy_omni.conversion.exporters.te_nvfp4_plan import TEExportPlan, TETensorPlan, held_config, plan_from_held
from comfy_omni.conversion.numerics.te_nvfp4 import int8_bf16_chunk, nvfp4_bf16_stripe, validate_bf16_chunk
from comfy_omni.conversion.packaging.native_export import (
    NativeExportPublication,
    StagedArtifact,
    prepare_native_export,
    publish_native_export,
    stage_document,
)
from comfy_omni.domain.normalization import ToolIdentity


def _space(path: Path, *, starting: bool = False) -> None:
    required = contract.MIN_FREE_BYTES if starting else contract.RESERVE_BYTES
    if shutil.disk_usage(path).free < required:
        raise ContractError("TE output filesystem lacks its required free-space reserve")


def _raw(source: SafeTensorSources, name: str, offset: int, length: int) -> bytes:
    if not 0 < length <= contract.MAX_CHUNK_BYTES:
        raise ContractError("TE raw read exceeds its bounded chunk size")
    return b"".join(source.iter_raw_range(source.tensors[name], offset, length, chunk_bytes=contract.MAX_CHUNK_BYTES))


def _chunks(source: SafeTensorSources, action: TETensorPlan):
    name = action.source_name
    module = name.removesuffix(".weight")
    if action.operation == "copy-bf16":
        for chunk in source.iter_raw(source.tensors[name], chunk_bytes=contract.MAX_CHUNK_BYTES):
            validate_bf16_chunk(chunk)
            yield chunk
    elif action.operation == "int8-f32-to-bf16":
        scale = _raw(source, module + ".weight_scale", 0, 4)
        for chunk in source.iter_raw(source.tensors[name], chunk_bytes=contract.MAX_CHUNK_BYTES // 2):
            yield int8_bf16_chunk(chunk, scale)
    elif action.operation == "nvfp4-blocked-to-bf16":
        rows, columns = action.shape
        scale = _raw(source, module + ".weight_scale_2", 0, 4)
        for start in range(0, rows, contract.MAX_ROWS):
            height = min(contract.MAX_ROWS, rows - start)
            packed = _raw(source, name, start * columns // 2, height * columns // 2)
            scales = _raw(source, module + ".weight_scale", start * columns // 16, 128 * columns // 16)
            yield nvfp4_bf16_stripe(packed, scales, scale, rows=height, columns=columns)
    else:
        raise ContractError("unknown TE plan operation")


def execute_te_dense_export(plan: TEExportPlan, output_dir: Path, *, tool: ToolIdentity) -> NativeExportPublication:
    """Rebuild all authority from held sources; never accept caller-edited plan semantics."""
    output_dir = fileops.reject_linked_ancestors(output_dir, allow_missing_final=True)
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite TE output")
    if not isinstance(plan, TEExportPlan) or not isinstance(tool, ToolIdentity) or tool.distribution != "comfy-omni":
        raise ContractError("TE execution requires a fixed plan and ComfyOmni tool identity")
    source_path, config_path = Path(plan.source_path), Path(plan.config_path)
    if any(output_dir == p or output_dir in p.parents for p in (source_path, config_path)):
        raise ContractError("TE output cannot overlap either source")
    with held_config(config_path) as (config, verify_config), SafeTensorSources([source_path]) as source:
        try:
            authoritative = plan_from_held(source, config_path)
            if plan != authoritative:
                raise ContractError("TE plan disagrees with authority reconstructed from held sources")
            if plan.target_payload_bytes + 4 * 1024**2 > contract.MAX_OUTPUT_BYTES:
                raise ContractError("TE output allocation exceeds its fixed bound")
            _space(output_dir.parent, starting=True)
            stage = prepare_native_export(output_dir)
            hashes = {}
            max_chunk = 0

            def tracked(action):
                def chunks():
                    nonlocal max_chunk
                    digest = hashlib.sha256()
                    for chunk in _chunks(source, action):
                        if len(chunk) > contract.MAX_CHUNK_BYTES:
                            raise ContractError("TE decoder exceeded bounded output chunk size")
                        _space(stage.path)
                        digest.update(chunk)
                        max_chunk = max(max_chunk, len(chunk))
                        yield chunk
                    hashes[action.target_name] = digest.hexdigest()

                return chunks

            payloads = [TensorPayload(a.target_name, "BF16", a.shape, a.byte_length, tracked(a)) for a in plan.tensors]
            written = write_safetensors_file(stage.path / "model.safetensors", payloads)
            verified = verify_safetensors_file(stage.path / written.name, written.descriptors, written.sha256)
            if set(hashes) != contract.TARGET_INVENTORY.keys():
                raise ContractError("TE output tensor digest coverage is incomplete")
            artifacts = (
                StagedArtifact(verified.name, verified.size, verified.sha256, "safetensors", verified.tensor_count),
                stage_document(stage, "config.json", config, kind="config"),
                stage_document(stage, "export.plan.json", fileops.canonical_json(plan.to_dict()), kind="plan"),
            )

            def final_check():
                _space(stage.path)
                source.verify_unchanged()
                verify_config()

            manifest = {
                "schema": "comfy_omni.te_dense.export/v1",
                "component": "text_encoder",
                "profile": contract.PROFILE,
                "consumer": contract.CONSUMER,
                "historical_writer_identity_proven": False,
                "plan_content_sha256": plan.content_sha256,
                "source": {
                    "size": plan.source_bytes,
                    "sha256": plan.source_sha256,
                    "schema_sha256": plan.source_schema_sha256,
                },
                "config": {"size": plan.config_bytes, "sha256": plan.config_sha256},
                "target": {
                    "schema_sha256": plan.target_schema_sha256,
                    "tensor_count": len(plan.tensors),
                    "payload_bytes": plan.target_payload_bytes,
                },
                "source_tensor_count": len(source.tensors),
                "consumed_auxiliary_count": len(source.tensors) - len(plan.tensors),
                "tool": tool.to_dict(),
                "files": [a.to_dict() for a in artifacts],
                "tensor_sha256": hashes,
                "max_emitted_chunk_bytes": max_chunk,
                "numerical_semantics": "NVFP4 BF16 materialized steps; INT8 FP32 multiply then BF16",
            }
            return publish_native_export(stage, artifacts, manifest, before_manifest=final_check)
        except BaseException:
            source.verify_unchanged()
            raise
