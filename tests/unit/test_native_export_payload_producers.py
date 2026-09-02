from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from comfy_omni.artifacts import fileops
from comfy_omni.artifacts.snapshot_schema import (
    DECISION_INHERITED_ENFORCED_PIN,
    PIN_BLOCK_SCHEMA,
    SNAPSHOT_SCHEMA,
    contract_block,
)
from comfy_omni.artifacts.snapshot_store import write_snapshot
from comfy_omni.artifacts.sources import SafeTensorSources
from comfy_omni.contracts import ARCHITECTURE_TEMPLATES
from comfy_omni.contracts.models import STORAGE_INT8_CONVROT, ContractError, NativeSourceContract
from comfy_omni.contracts.templates import template_digest
from comfy_omni.conversion.exporters.execution import execute_native_export
from comfy_omni.conversion.exporters.models import (
    NativeExportPlan,
    QkvLayoutPlan,
    ResourceEnvelope,
    ShardPlan,
    SourceBinding,
    TensorAction,
)
from comfy_omni.conversion.exporters.planning import (
    OP_COPY_QKV_TO_GROUPED,
    OP_INVERSE_CONVROT_BF16,
    OP_INVERSE_CONVROT_BF16_QKV_TO_GROUPED,
    OP_OMIT_MARKER,
    OP_OMIT_SCALE,
)
from comfy_omni.domain.normalization import ToolIdentity
from comfy_omni.domain.qkv import qkv_to_grouped_row_indices

SNAPSHOT_COPY_NAME = "source-contract.snapshot.json"
QKV_NAME = "token_refiner.blocks.0.attn.qkv_proj.weight"
CONVROT_PREFIX = "blocks.0.mlp"
CONVROT_QKV_PREFIX = "blocks.0.attn.qkv_proj"


def _write_safetensors(path: Path, tensors: tuple[tuple[str, str, tuple[int, ...], bytes], ...]) -> None:
    header: dict[str, object] = {}
    payload = bytearray()
    for name, dtype, shape, raw in sorted(tensors):
        start = len(payload)
        payload.extend(raw)
        header[name] = {"data_offsets": [start, len(payload)], "dtype": dtype, "shape": list(shape)}
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _tool() -> ToolIdentity:
    return ToolIdentity("comfy-omni", "0.2.0a1", "1" * 40, "2" * 64)


def _digest_plan(plan: NativeExportPlan) -> NativeExportPlan:
    digest = hashlib.sha256(fileops.canonical_json(plan.to_dict(include_content_sha256=False))).hexdigest()
    return replace(plan, content_sha256=digest)


def _layout() -> QkvLayoutPlan:
    indices = qkv_to_grouped_row_indices(num_query_groups=2, heads_per_group=1, head_dim=1)
    digest = hashlib.sha256(fileops.canonical_json(list(indices))).hexdigest()
    return QkvLayoutPlan("runtime-qkv", "grouped-for-official-loader", 2, 1, 1, len(indices), digest)


def _base_plan(
    source: Path,
    actions: tuple[TensorAction, ...],
    *,
    largest: int,
    payload_bytes: int,
    contract_name: str = "tiny-producer-v1",
    contract_schema: str = "a" * 64,
    snapshot_manifest_sha256: str | None = None,
    snapshot_file_sha256: str | None = None,
) -> NativeExportPlan:
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    targets = tuple(sorted(item.target_name for item in actions if item.target_name is not None))
    draft = NativeExportPlan(
        schema="comfy_omni.native_export.plan/v2",
        output_schema="h3-comfy-int8-export/v2",
        component="transformer",
        profile="dense-bf16-online-int8",
        source_contract=contract_name,
        source_contract_origin="external-snapshot" if snapshot_manifest_sha256 else "compile-time",
        source_contract_schema_sha256=contract_schema,
        source_snapshot_manifest_sha256=snapshot_manifest_sha256,
        source_snapshot_file_sha256=snapshot_file_sha256,
        template_name="tiny-producer",
        template_version=1,
        template_sha256="b" * 64,
        source_files=(SourceBinding(str(source), source.stat().st_size, source_digest),),
        qkv_layout=_layout(),
        resource_envelope=ResourceEnvelope(max_rows=2, max_shard_bytes=4096, largest_target_tensor_bytes=largest),
        actions=actions,
        shards=(ShardPlan("model-00001-of-00001.safetensors", targets, payload_bytes),),
        target_tensor_count=len(targets),
        target_payload_bytes=payload_bytes,
        runtime_quant_method="compressed-tensors",
        runtime_ignored_layers=(),
        payload_semantics="test-bounded-producers",
        content_sha256="",
    )
    return _digest_plan(draft)


def _qkv_fixture(tmp_path: Path) -> tuple[Path, NativeExportPlan, tuple[bytes, ...]]:
    rows = tuple(bytes((row, row, row, row)) for row in range(6))
    source = tmp_path / "qkv.safetensors"
    _write_safetensors(source, ((QKV_NAME, "BF16", (6, 2), b"".join(rows)),))
    action = TensorAction(
        QKV_NAME,
        QKV_NAME,
        "BF16",
        "BF16",
        (6, 2),
        24,
        24,
        OP_COPY_QKV_TO_GROUPED,
        QKV_NAME.removesuffix(".weight"),
    )
    return source, _base_plan(source, (action,), largest=24, payload_bytes=24), rows


def _convrot_fixture(tmp_path: Path, *, qkv: bool) -> tuple[Path, NativeExportPlan, bytes, str]:
    prefix = CONVROT_QKV_PREFIX if qkv else CONVROT_PREFIX
    rows = 6 if qkv else 3
    width = 256
    weight_name = f"{prefix}.weight"
    scale_name = f"{prefix}.weight_scale"
    marker_name = f"{prefix}.comfy_quant"
    weight = b"".join(bytes((row,)) * width for row in range(rows))
    scale = struct.pack(f"<{rows}f", *(float(row + 1) for row in range(rows)))
    marker = b'{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}'
    source = tmp_path / ("convrot-qkv.safetensors" if qkv else "convrot.safetensors")
    _write_safetensors(
        source,
        (
            (marker_name, "U8", (len(marker),), marker),
            (weight_name, "I8", (rows, width), weight),
            (scale_name, "F32", (rows, 1), scale),
        ),
    )
    operation = OP_INVERSE_CONVROT_BF16_QKV_TO_GROUPED if qkv else OP_INVERSE_CONVROT_BF16
    actions = (
        TensorAction(marker_name, None, "U8", None, (len(marker),), len(marker), 0, OP_OMIT_MARKER, prefix, 256),
        TensorAction(
            weight_name,
            weight_name,
            "I8",
            "BF16",
            (rows, width),
            len(weight),
            rows * width * 2,
            operation,
            prefix,
            256,
        ),
        TensorAction(scale_name, None, "F32", None, (rows, 1), len(scale), 0, OP_OMIT_SCALE, prefix, 256),
    )
    plan = _base_plan(source, actions, largest=rows * width * 2, payload_bytes=rows * width * 2)
    return source, plan, weight, weight_name


def _read_only_payload(output: Path, name: str) -> bytes:
    with SafeTensorSources([output / "model-00001-of-00001.safetensors"]) as sources:
        assert set(sources.tensors) == {name}
        return sources.read_raw(sources.tensors[name])


def _fake_convrot(calls: list[tuple[int, bytes]]) -> Any:
    def convert(qweight: bytes, rowwise_scale: bytes, *, rows: int, columns: int, group_size: int) -> bytes:
        assert len(qweight) == rows * columns
        assert len(rowwise_scale) == rows * 4
        assert group_size == 256
        calls.append((rows, qweight))
        return b"".join(bytes((value, 0)) for value in qweight)

    return convert


def _snapshot_for(source: Path, directory: Path) -> Any:
    template = ARCHITECTURE_TEMPLATES["h3-transformer-50l-convrot"]
    contract = NativeSourceContract(
        "external-transformer-v1",
        "transformer",
        1,
        len(template.convrot_table()),
        "c" * 64,
    )
    source_bytes = source.read_bytes()
    document = {
        "schema": SNAPSHOT_SCHEMA,
        "status": "PINNED",
        "pending_review": False,
        "contract": contract_block(
            contract,
            template_name=template.template_name,
            template_version=template.template_version,
            storage_kind=STORAGE_INT8_CONVROT,
        ),
        "pin": {
            "schema": PIN_BLOCK_SCHEMA,
            "reviewed_by": "reviewer-bob",
            "evidence_sha256": "d" * 64,
            "generated_by": {"operator": "generator-alice"},
            "draft_sha256": "e" * 64,
            "source_files": [
                {
                    "path": str(source),
                    "size": len(source_bytes),
                    "sha256": hashlib.sha256(source_bytes).hexdigest(),
                }
            ],
            "census_sha256": "f" * 64,
            "template": {
                "name": template.template_name,
                "version": template.template_version,
                "digest": template_digest(template),
            },
            "enforced_schema_decision": DECISION_INHERITED_ENFORCED_PIN,
        },
    }
    return write_snapshot(directory, document)


def test_qkv_copy_reorders_complete_rows_using_plan_bound_permutation(tmp_path: Path) -> None:
    _, plan, rows = _qkv_fixture(tmp_path)

    publication = execute_native_export(plan, tmp_path / "out", tool=_tool())

    expected = b"".join(rows[index] for index in (0, 2, 4, 1, 3, 5))
    assert _read_only_payload(publication.output_dir, QKV_NAME) == expected


@pytest.mark.parametrize("qkv", [False, True])
def test_convrot_producer_is_bounded_and_combined_qkv_uses_grouped_order(tmp_path: Path, qkv: bool) -> None:
    _, plan, source_weight, weight_name = _convrot_fixture(tmp_path, qkv=qkv)
    calls: list[tuple[int, bytes]] = []

    publication = execute_native_export(
        plan,
        tmp_path / "out",
        tool=_tool(),
        convrot_backend=_fake_convrot(calls),
    )

    source_rows = tuple(source_weight[index : index + 256] for index in range(0, len(source_weight), 256))
    row_order = (0, 2, 4, 1, 3, 5) if qkv else tuple(range(3))
    expected_i8 = b"".join(source_rows[index] for index in row_order)
    expected_bf16 = b"".join(bytes((value, 0)) for value in expected_i8)
    assert _read_only_payload(publication.output_dir, weight_name) == expected_bf16
    assert all(1 <= rows <= plan.resource_envelope.max_rows for rows, _ in calls)
    assert b"".join(raw for _, raw in calls) == expected_i8


def test_execution_rejects_qkv_permutation_digest_drift_before_publication(tmp_path: Path) -> None:
    _, plan, _ = _qkv_fixture(tmp_path)
    plan = replace(plan, qkv_layout=replace(plan.qkv_layout, permutation_sha256="0" * 64))
    plan = _digest_plan(plan)

    with pytest.raises(ContractError, match="permutation"):
        execute_native_export(plan, tmp_path / "out", tool=_tool())

    assert not (tmp_path / "out" / "manifest.json").exists()


def test_external_contract_snapshot_is_revalidated_carried_and_receipted(tmp_path: Path) -> None:
    source, plan, _ = _qkv_fixture(tmp_path)
    snapshot = _snapshot_for(source, tmp_path / "contracts")
    template = ARCHITECTURE_TEMPLATES["h3-transformer-50l-convrot"]
    plan = replace(
        plan,
        source_contract="external-transformer-v1",
        source_contract_origin="external-snapshot",
        source_contract_schema_sha256="c" * 64,
        source_snapshot_manifest_sha256=snapshot.manifest_sha256,
        source_snapshot_file_sha256=hashlib.sha256(snapshot.payload).hexdigest(),
        template_name=template.template_name,
        template_version=template.template_version,
        template_sha256=template_digest(template),
    )
    plan = _digest_plan(plan)

    publication = execute_native_export(
        plan,
        tmp_path / "out",
        tool=_tool(),
        source_contract_snapshot=snapshot.path,
    )

    assert (publication.output_dir / SNAPSHOT_COPY_NAME).read_bytes() == snapshot.payload
    manifest = json.loads(publication.manifest_path.read_bytes())
    assert manifest["source_contract"] == {
        "manifest_sha256": snapshot.manifest_sha256,
        "name": "external-transformer-v1",
        "origin": "external-snapshot",
        "snapshot_file": SNAPSHOT_COPY_NAME,
        "snapshot_file_sha256": hashlib.sha256(snapshot.payload).hexdigest(),
    }


def test_snapshot_authority_mismatch_and_compile_time_snapshot_fail_closed(tmp_path: Path) -> None:
    source, compile_time, _ = _qkv_fixture(tmp_path)
    snapshot = _snapshot_for(source, tmp_path / "contracts")

    with pytest.raises(ContractError, match="compile-time"):
        execute_native_export(
            compile_time,
            tmp_path / "compile-time-out",
            tool=_tool(),
            source_contract_snapshot=snapshot.path,
        )

    external = replace(
        compile_time,
        source_contract="external-transformer-v1",
        source_contract_origin="external-snapshot",
        source_contract_schema_sha256="c" * 64,
        source_snapshot_manifest_sha256=snapshot.manifest_sha256,
        source_snapshot_file_sha256="0" * 64,
        template_name="h3-transformer-50l-convrot",
        template_sha256=template_digest(ARCHITECTURE_TEMPLATES["h3-transformer-50l-convrot"]),
    )
    external = _digest_plan(external)
    with pytest.raises(ContractError, match="snapshot"):
        execute_native_export(
            external,
            tmp_path / "external-out",
            tool=_tool(),
            source_contract_snapshot=snapshot.path,
        )
    assert not (tmp_path / "external-out" / "manifest.json").exists()
