from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from comfy_omni.conversion.contract_workflows.census import CensusEngine, ContractScanError, census_tensors
from comfy_omni.domain.checkpoints import TensorDescriptor


def _marker(group_size: int = 256) -> bytes:
    return json.dumps(
        {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": group_size},
        separators=(",", ":"),
    ).encode()


def _triplet() -> tuple[list[TensorDescriptor], dict[str, bytes]]:
    marker = _marker()
    descriptors = [
        TensorDescriptor("blocks.0.linear.weight", "I8", (2, 256), (0, 512)),
        TensorDescriptor("blocks.0.linear.weight_scale", "F32", (2, 1), (512, 520)),
        TensorDescriptor("blocks.0.linear.comfy_quant", "U8", (len(marker),), (520, 520 + len(marker))),
    ]
    return descriptors, {"blocks.0.linear.comfy_quant": marker}


def _write_safetensors(path: Path, tensors: list[tuple[str, str, list[int], bytes]]) -> None:
    offset = 0
    header: dict[str, object] = {}
    payload = bytearray()
    for name, dtype, shape, raw in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        payload.extend(raw)
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def test_descriptor_census_discovers_one_strict_convrot_group() -> None:
    descriptors, payloads = _triplet()
    report = census_tensors(descriptors, payloads)
    assert report.storage_kind == "int8-convrot"
    assert report.convrot_group_count == 1
    assert report.convrot_group_size_census == {"256": 1}
    assert report.scale_shape_census == {"2x1": 1}


def test_descriptor_census_rejects_missing_marker_payload_and_marker_free_int8() -> None:
    descriptors, _ = _triplet()
    with pytest.raises(ContractScanError, match="coverage mismatch"):
        census_tensors(descriptors, {})
    with pytest.raises(ContractScanError, match="INT8 tensors"):
        census_tensors((TensorDescriptor("weight", "I8", (1,), (0, 1)),), {})


def test_single_file_scan_binds_real_file_sha256(tmp_path: Path) -> None:
    marker = _marker()
    source = tmp_path / "tiny.safetensors"
    _write_safetensors(
        source,
        [
            ("blocks.0.linear.weight", "I8", [2, 256], bytes(512)),
            ("blocks.0.linear.weight_scale", "F32", [2, 1], bytes(8)),
            ("blocks.0.linear.comfy_quant", "U8", [len(marker)], marker),
        ],
    )
    report = CensusEngine().scan(source)
    assert report.input_mode == "single-file"
    assert report.files[0].sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report.files[0].size == source.stat().st_size


def test_index_mode_rejects_extra_safetensors(tmp_path: Path) -> None:
    root = tmp_path / "shards"
    root.mkdir()
    _write_safetensors(root / "model-1.safetensors", [("weight", "BF16", [1], b"\0\0")])
    _write_safetensors(root / "extra.safetensors", [("extra", "BF16", [1], b"\0\0")])
    index = {"weight_map": {"weight": "model-1.safetensors"}}
    (root / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ContractScanError, match="outside the index-declared"):
        CensusEngine().scan(root)
