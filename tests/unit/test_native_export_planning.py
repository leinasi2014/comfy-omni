from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from comfy_omni.artifacts import fileops
from comfy_omni.contracts.models import (
    STORAGE_INT8_CONVROT,
    ArchitectureTemplate,
    ContractRecord,
    NativeSourceContract,
)
from comfy_omni.conversion.contract_workflows.census import FileRecord, census_tensors, schema_sha256
from comfy_omni.conversion.exporters.planning import (
    OP_COPY_QKV_TO_GROUPED,
    OP_COPY_RAW,
    OP_INVERSE_CONVROT_BF16_QKV_TO_GROUPED,
    OP_OMIT_MARKER,
    OP_OMIT_SCALE,
    PLAN_SCHEMA,
    ConversionPlanError,
    build_native_export_plan,
)
from comfy_omni.domain.checkpoints import TensorDescriptor

MARKER = b'{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}'
QKV_SHAPE = (21_504, 256)


def _descriptor(name: str, dtype: str, shape: tuple[int, ...], byte_length: int) -> TensorDescriptor:
    return TensorDescriptor(name, dtype, shape, (0, byte_length))


def _fixture() -> tuple[object, ContractRecord, ArchitectureTemplate]:
    qkv_elements = QKV_SHAPE[0] * QKV_SHAPE[1]
    descriptors = (
        _descriptor("blocks.0.attn.qkv_proj.weight", "I8", QKV_SHAPE, qkv_elements),
        _descriptor("blocks.0.attn.qkv_proj.weight_scale", "F32", (QKV_SHAPE[0], 1), QKV_SHAPE[0] * 4),
        _descriptor("blocks.0.attn.qkv_proj.comfy_quant", "U8", (len(MARKER),), len(MARKER)),
        _descriptor("final_layer.bias", "BF16", (2,), 4),
        _descriptor("token_refiner.blocks.0.attn.qkv_proj.weight", "BF16", QKV_SHAPE, qkv_elements * 2),
    )
    report = census_tensors(
        descriptors,
        {"blocks.0.attn.qkv_proj.comfy_quant": MARKER},
        files=(FileRecord("/models/source.safetensors", 123_456_789, "a" * 64),),
    )
    template = ArchitectureTemplate(
        template_name="test-h3-transformer",
        template_version=1,
        component="transformer",
        layer_topology=(0,),
        layer_prefix_template="blocks.{layer}.{suffix}",
        convrot_suffixes={"attn.qkv_proj": (QKV_SHAPE, 256)},
        scale_shape_census={(QKV_SHAPE[0], 1): 1},
    )
    contract = NativeSourceContract(
        name="test-exact-transformer",
        component="transformer",
        tensor_count=len(descriptors),
        convrot_group_count=1,
        schema_sha256=schema_sha256(descriptors),
    )
    return report, ContractRecord(contract, template.template_name, STORAGE_INT8_CONVROT), template


def test_plan_is_deterministic_complete_and_explicit_about_semantics() -> None:
    report, record, template = _fixture()

    first = build_native_export_plan(report, record, template, max_shard_bytes=12 * 1024**2)
    second = build_native_export_plan(report, record, template, max_shard_bytes=12 * 1024**2)
    operations = {item.source_name: item.operation for item in first.actions}

    assert first == second
    assert first.schema == PLAN_SCHEMA
    assert first.target_tensor_count == 3
    assert operations == {
        "blocks.0.attn.qkv_proj.comfy_quant": OP_OMIT_MARKER,
        "blocks.0.attn.qkv_proj.weight": OP_INVERSE_CONVROT_BF16_QKV_TO_GROUPED,
        "blocks.0.attn.qkv_proj.weight_scale": OP_OMIT_SCALE,
        "final_layer.bias": OP_COPY_RAW,
        "token_refiner.blocks.0.attn.qkv_proj.weight": OP_COPY_QKV_TO_GROUPED,
    }
    grouped = {item.source_name: item.group_size for item in first.actions}
    assert grouped == {
        "blocks.0.attn.qkv_proj.comfy_quant": 256,
        "blocks.0.attn.qkv_proj.weight": 256,
        "blocks.0.attn.qkv_proj.weight_scale": 256,
        "final_layer.bias": None,
        "token_refiner.blocks.0.attn.qkv_proj.weight": None,
    }
    assert len(first.shards) == 2
    assert first.to_dict()["semantics"] == {
        "description": "inverse-convrot-to-dense-bf16; runtime-int8-required; not-payload-preserving",
        "payload_preserving": False,
        "lossless_claim": False,
        "direct_convrot_loading": False,
    }
    expected_digest = hashlib.sha256(fileops.canonical_json(first.to_dict(include_content_sha256=False))).hexdigest()
    assert first.content_sha256 == expected_digest


def test_plan_rejects_any_schema_drift_before_actions_are_authorized() -> None:
    report, record, template = _fixture()
    changed_contract = replace(
        record,
        contract=replace(record.contract, schema_sha256="b" * 64),
    )

    with pytest.raises(ConversionPlanError, match="exact contract") as failure:
        build_native_export_plan(report, changed_contract, template)

    assert failure.value.evidence["stage"] == "contract-authorization"
    assert "schema_sha256" in failure.value.evidence["mismatches"]


def test_plan_rejects_contract_without_exact_schema_authority() -> None:
    report, record, template = _fixture()
    unpinned = replace(record, contract=replace(record.contract, schema_sha256=None))

    with pytest.raises(ConversionPlanError, match="no exact schema") as failure:
        build_native_export_plan(report, unpinned, template)

    assert failure.value.evidence["stage"] == "contract-authorization"


def test_plan_rejects_a_tensor_larger_than_the_shard_envelope() -> None:
    report, record, template = _fixture()

    with pytest.raises(ConversionPlanError, match="exceeds the shard") as failure:
        build_native_export_plan(report, record, template, max_shard_bytes=1024)

    assert failure.value.evidence["stage"] == "shard-plan"
