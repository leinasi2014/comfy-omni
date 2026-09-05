"""Fixed beta4 observations; no checkpoint payload is embedded in this fixture."""

from __future__ import annotations

import hashlib
import struct
from collections import Counter
from dataclasses import replace
from math import prod
from pathlib import Path
from types import SimpleNamespace

import pytest

from comfy_omni.artifacts import fileops
from comfy_omni.contracts.beta4 import (
    BETA4_SOURCE_INVENTORY,
    BETA4_SOURCE_RECORD,
    BETA4_SOURCE_TEMPLATE,
    BETA4_TARGET_INVENTORY,
    BETA4_TARGET_SCHEMA_SHA256,
)
from comfy_omni.contracts.conversion import PROFILE_BETA4_DENSE_BF16
from comfy_omni.contracts.models import ContractError
from comfy_omni.contracts.registry import COMPILE_TIME_CATALOG
from comfy_omni.contracts.templates import ARCHITECTURE_TEMPLATES
from comfy_omni.conversion.contract_workflows.census import FileRecord, census_tensors
from comfy_omni.conversion.exporters.beta4 import build_beta4_dense_plan
from comfy_omni.conversion.exporters.execution import (
    _config_patch,
    _validate_plan,
    _verify_plan_digest,
    execute_native_export,
)
from comfy_omni.conversion.exporters.planning import build_native_export_plan
from comfy_omni.domain.checkpoints import TensorDescriptor
from comfy_omni.domain.normalization import ToolIdentity

SOURCE_SHA = "54d56b15c65923b54c9ca16b494dae641bfe9455cfcb1c19c49b1008e270bbc1"
SOURCE_BYTES = 20_967_637_320
SOURCE_SCHEMA = "ae2456bc6ac904929a4b773f703f8a1baa99b6356b5a389994faf64a1a2d80f2"
MARKER = b'{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}'


def beta4_report():
    # The existing Apache-2.0 beta3 template is the independent comparison
    # inventory. The observed beta4 difference is precisely one missing grid
    # and the 200 declared INT8 ConvRot triplets.
    inventory = dict(ARCHITECTURE_TEMPLATES["h3-transformer-50l-hybrid8-bf16-plain"].non_quantized_inventory)
    inventory.pop("silu_t_emb_grid")
    quantized = ARCHITECTURE_TEMPLATES["h3-transformer-50l-convrot"].convrot_table()
    markers = {}
    for prefix, (shape, _) in quantized.items():
        inventory[f"{prefix}.weight"] = ("I8", shape)
        inventory[f"{prefix}.weight_scale"] = ("F32", (shape[0], 1))
        inventory[f"{prefix}.comfy_quant"] = ("U8", (72,))
        markers[f"{prefix}.comfy_quant"] = MARKER.ljust(72)
    widths = {"BF16": 2, "I8": 1, "U8": 1, "F32": 4}
    offset = 0
    descriptors = []
    for name, (dtype, shape) in sorted(inventory.items()):
        length = prod(shape) * widths[dtype]
        descriptors.append(TensorDescriptor(name, dtype, shape, (offset, offset + length)))
        offset += length
    report = census_tensors(
        tuple(descriptors),
        markers,
        files=(FileRecord("/models/primary.safetensors", SOURCE_BYTES, SOURCE_SHA),),
    )
    assert report.observed_schema_sha256 == SOURCE_SCHEMA
    assert report.tensor_count == 934
    return report


def test_fixed_beta4_can_plan_exact_534_dense_tensors_without_inventing_grid():
    report = beta4_report()
    plan = build_beta4_dense_plan(report)
    assert plan.target_tensor_count == 534
    assert all(action.target_name != "silu_t_emb_grid" for action in plan.actions)
    assert plan.runtime_quant_method is None
    assert plan.target_schema_sha256 == BETA4_TARGET_SCHEMA_SHA256
    assert plan.target_payload_bytes == 40_222_925_872
    assert Counter(action.operation for action in plan.actions) == {
        "copy-raw": 332,
        "copy-runtime-qkv-to-grouped": 2,
        "inverse-convrot-to-bf16": 150,
        "inverse-convrot-to-bf16-runtime-qkv-to-grouped": 50,
        "omit-comfy-quant-marker": 200,
        "omit-source-rowwise-scale": 200,
    }
    assert plan.to_dict()["runtime_quantization"] == {
        "required": False,
        "method": None,
        "ignored_layers": [],
        "checkpoint_int8_serialized": False,
    }
    assert _config_patch(plan)["quantization_config"] is None
    assert {a.target_name: (a.target_dtype, a.shape) for a in plan.actions if a.target_name} == BETA4_TARGET_INVENTORY


def test_old_contracts_still_reject_beta4_and_retain_their_identity():
    report = beta4_report()
    record = COMPILE_TIME_CATALOG.resolve("transformer")
    with pytest.raises(ContractError, match="exact contract"):
        build_native_export_plan(report, record, ARCHITECTURE_TEMPLATES[record.template_name])
    assert record.contract.tensor_count == 932
    assert len(ARCHITECTURE_TEMPLATES["h3-transformer-50l-hybrid8-bf16-plain"].non_quantized_inventory) == 535
    assert {d.name: (d.dtype, d.shape) for d in report.descriptors} == BETA4_SOURCE_INVENTORY


def test_beta4_cannot_be_relabeled_with_online_int8_policy():
    report = beta4_report()
    with pytest.raises(ContractError, match="explicit dense BF16"):
        build_native_export_plan(report, BETA4_SOURCE_RECORD, BETA4_SOURCE_TEMPLATE)
    plan = replace(build_beta4_dense_plan(report), profile="dense-bf16-online-int8")
    plan = replace(
        plan,
        content_sha256=hashlib.sha256(fileops.canonical_json(plan.to_dict(include_content_sha256=False))).hexdigest(),
    )
    with pytest.raises(ContractError, match="relabeled"):
        _verify_plan_digest(plan)


@pytest.mark.parametrize("field,value", [("sha256", "f" * 64), ("size", SOURCE_BYTES + 1)])
def test_beta4_rejects_wrong_file_identity_even_with_matching_descriptors(field, value):
    report = beta4_report()
    report = replace(report, files=(replace(report.files[0], **{field: value}),))
    with pytest.raises(ContractError, match="fixed asset"):
        build_beta4_dense_plan(report)


def test_beta4_rejects_wrong_descriptor_even_with_claimed_source_schema():
    report = beta4_report()
    first = report.descriptors[0]
    report = replace(report, descriptors=(replace(first, shape=(8, 2689)), *report.descriptors[1:]))
    with pytest.raises(ContractError, match="exact contract"):
        build_beta4_dense_plan(report)


@pytest.mark.parametrize("change", ["record", "template", "snapshot", "groups"])
def test_new_profile_cannot_borrow_other_authority(change):
    report = beta4_report()
    record, template = BETA4_SOURCE_RECORD, BETA4_SOURCE_TEMPLATE
    if change == "record":
        record = replace(record, contract=replace(record.contract, name="pretend"))
    elif change == "template":
        template = replace(template, non_quantized_inventory={})
    elif change == "snapshot":
        record = replace(record, snapshot_payload=b"{}", snapshot_manifest_sha256="a" * 64)
    else:
        report = replace(report, groups=tuple(replace(group, group_size=64) for group in report.groups))
    with pytest.raises(ContractError):
        build_native_export_plan(report, record, template, profile_name=PROFILE_BETA4_DENSE_BF16)


def _held_report(report):
    return SimpleNamespace(
        paths=(Path(report.files[0].path),),
        sizes=[report.files[0].size],
        hashes=[report.files[0].sha256],
        tensors={d.name: SimpleNamespace(descriptor=d) for d in report.descriptors},
        read_raw=lambda tensor: MARKER.ljust(72),
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_contract", "invented"),
        ("template_sha256", "0" * 64),
        ("target_schema_sha256", "0" * 64),
        ("target_contract", "invented"),
        ("runtime_quant_method", "int8"),
        ("runtime_ignored_layers", ("anything",)),
        ("payload_semantics", "lossless"),
    ],
)
def test_executor_reauthorizes_rehashed_plan_from_held_sources(field, value):
    report = beta4_report()
    plan = replace(build_beta4_dense_plan(report), **{field: value})
    digest = hashlib.sha256(fileops.canonical_json(plan.to_dict(include_content_sha256=False))).hexdigest()
    plan = replace(plan, content_sha256=digest)
    with pytest.raises(ContractError, match="reconstructed"):
        _validate_plan(plan, _held_report(report))


def test_executor_checks_real_source_identity_and_complete_targets():
    report = beta4_report()
    plan = build_beta4_dense_plan(report)
    targets, scales = _validate_plan(plan, _held_report(report))
    assert set(targets) == set(BETA4_TARGET_INVENTORY)
    assert len(scales) == 200
    held = _held_report(report)
    held.hashes = ["0" * 64]
    with pytest.raises(ContractError, match="SHA256"):
        _validate_plan(plan, held)


def test_beta4_publication_never_overwrites_existing_output(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    (output / "keep").write_bytes(b"unchanged")
    with pytest.raises(FileExistsError):
        execute_native_export(
            build_beta4_dense_plan(beta4_report()),
            output,
            tool=ToolIdentity("comfy-omni", "0.2.0a1", "1" * 40, "2" * 64),
        )
    assert (output / "keep").read_bytes() == b"unchanged"


def test_real_serialized_backend_matches_independent_small_matrix_oracle():
    pytest.importorskip("torch")
    from comfy_omni.conversion.numerics.reference import inverse_convrot_reference
    from comfy_omni.conversion.numerics.serialization import torch_convrot_bf16_block

    # Two complete 256-column groups exercise the actual beta4 group size.
    rows = tuple(tuple((column * 7 + row * 3) % 29 - 14 for column in range(512)) for row in range(3))
    scales = (0.125, 0.5, 2.0)
    expected = inverse_convrot_reference(rows, scales, group_size=256)
    raw = bytes(value & 255 for row in rows for value in row)
    actual = torch_convrot_bf16_block(raw, struct.pack("<3f", *scales), rows=3, columns=512, group_size=256)
    golden = bytearray()
    for row in expected:
        for value in row:
            bits = struct.unpack("<I", struct.pack("<f", value))[0]
            golden.extend(struct.pack("<H", ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF))
    assert actual == bytes(golden)
