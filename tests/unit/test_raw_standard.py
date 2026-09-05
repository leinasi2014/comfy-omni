"""Small-source tests for the direct standard H3 ConvRot provider."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

import pytest

from comfy_omni.artifacts.fileops import canonical_json
from comfy_omni.artifacts.sources import SafeTensorSources
from comfy_omni.contracts.models import ContractError
from comfy_omni.conversion.contract_workflows.census import schema_sha256
from comfy_omni.conversion.exporters.models import (
    NativeExportPlan,
    QkvLayoutPlan,
    ResourceEnvelope,
    SourceBinding,
    TensorAction,
)
from comfy_omni.conversion.exporters.planning import (
    OP_COPY_QKV_TO_GROUPED,
    OP_COPY_RAW,
    OP_INVERSE_CONVROT_BF16,
    OP_OMIT_MARKER,
    OP_OMIT_SCALE,
)
from comfy_omni.domain.qkv import qkv_to_grouped_row_indices

torch = pytest.importorskip("torch")


def _write_safetensors(path: Path, tensors: tuple[tuple[str, str, tuple[int, ...], bytes], ...]) -> None:
    header: dict[str, object] = {}
    payload = bytearray()
    for name, dtype, shape, raw in tensors:
        start = len(payload)
        payload.extend(raw)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [start, len(payload)]}
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _bytes(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def _fixture(tmp_path: Path):
    source = tmp_path / "tiny-standard.safetensors"
    plain = torch.tensor([[2.0, 3.0]], dtype=torch.bfloat16)
    qkv = torch.arange(12, dtype=torch.float32).reshape(6, 2).to(torch.bfloat16)
    marker = b'{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}'
    _write_safetensors(
        source,
        (
            ("plain", "BF16", (1, 2), _bytes(plain)),
            ("qkv", "BF16", (6, 2), _bytes(qkv)),
            ("conv.weight", "I8", (1, 256), bytes([1]) * 256),
            ("conv.weight_scale", "F32", (1, 1), struct.pack("<f", 0.5)),
            ("conv.comfy_quant", "U8", (len(marker),), marker),
        ),
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with SafeTensorSources((source,)) as sources:
        source_schema = schema_sha256(tuple(item.descriptor for item in sources.tensors.values()))
    indices = qkv_to_grouped_row_indices(num_query_groups=2, heads_per_group=1, head_dim=1)
    layout = QkvLayoutPlan(
        "runtime-qkv",
        "grouped-for-official-loader",
        2,
        1,
        1,
        len(indices),
        hashlib.sha256(canonical_json(list(indices))).hexdigest(),
    )
    actions = (
        TensorAction("plain", "plain", "BF16", "BF16", (1, 2), 4, 4, OP_COPY_RAW),
        TensorAction("qkv", "qkv", "BF16", "BF16", (6, 2), 24, 24, OP_COPY_QKV_TO_GROUPED, "qkv"),
        TensorAction(
            "conv.weight", "conv.weight", "I8", "BF16", (1, 256), 256, 512, OP_INVERSE_CONVROT_BF16, "conv", 256
        ),
        TensorAction("conv.weight_scale", None, "F32", None, (1, 1), 4, 0, OP_OMIT_SCALE, "conv", 256),
        TensorAction("conv.comfy_quant", None, "U8", None, (len(marker),), len(marker), 0, OP_OMIT_MARKER, "conv", 256),
    )
    plan = NativeExportPlan(
        "test",
        "test",
        "transformer",
        "standard",
        "test",
        "compile-time",
        "schema",
        None,
        None,
        "test",
        1,
        "template",
        (SourceBinding(str(source), source.stat().st_size, digest),),
        layout,
        ResourceEnvelope(2, 4096, 512),
        actions,
        (),
        3,
        540,
        None,
        (),
        "test",
        "digest",
        None,
        None,
    )
    return source, digest, source_schema, plan, plain, qkv, indices


def test_standard_provider_yields_only_its_logical_532_style_targets(tmp_path, monkeypatch):
    from comfy_omni.runtime.h3 import raw_standard

    source, digest, source_schema, plan, plain, qkv, indices = _fixture(tmp_path)
    monkeypatch.setattr(raw_standard, "build_native_export_plan", lambda *args: plan)
    identity = raw_standard.RawStandardIdentity("tiny", source.stat().st_size, digest, source_schema, 5, 3)
    binding = raw_standard.RawStandardBinding.establish(source, identity=identity)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    monkeypatch.setattr(raw_standard, "SafeTensorSources", lambda *_: pytest.fail("runtime rehashed source"))

    weights = dict(binding.open_weights())

    assert tuple(weights) == ("conv.weight", "plain", "qkv")
    assert tuple(item.name for item in binding.target_descriptors) == tuple(weights)
    assert all(value.device.type == "cpu" and value.dtype == torch.bfloat16 for value in weights.values())
    assert torch.equal(weights["plain"], plain)
    assert indices == (0, 2, 4, 1, 3, 5)
    assert torch.equal(weights["qkv"], qkv.index_select(0, torch.tensor(indices)))
    assert torch.equal(weights["conv.weight"], torch.tensor([[0.5] * 256], dtype=torch.bfloat16))
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_standard_provider_rejects_file_identity_drift(tmp_path, monkeypatch):
    from comfy_omni.runtime.h3 import raw_standard

    source, digest, source_schema, plan, *_ = _fixture(tmp_path)
    monkeypatch.setattr(raw_standard, "build_native_export_plan", lambda *args: plan)
    binding = raw_standard.RawStandardBinding.establish(
        source, identity=raw_standard.RawStandardIdentity("tiny", source.stat().st_size, digest, source_schema, 5, 3)
    )
    current = source.stat()
    os.utime(source, ns=(current.st_atime_ns, current.st_mtime_ns + 1))
    with pytest.raises(ContractError, match="identity"):
        next(binding.open_weights())


def test_primary_standard_binding_is_fixed_to_932_source_and_532_logical_targets():
    from comfy_omni.runtime.h3 import raw_beta4, raw_standard

    assert issubclass(raw_standard.RawStandardBinding, raw_beta4.RawBeta4Binding)
    assert raw_standard.PRIMARY_RAW_STANDARD_IDENTITY.tensor_count == 932
    assert raw_standard.PRIMARY_RAW_STANDARD_IDENTITY.target_tensor_count == 532
