"""Small-source contract tests for the direct beta4 runtime provider."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import weakref
from dataclasses import replace
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


def _bytes(tensor) -> bytes:
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def _base4_digits(value: int) -> str:
    return f"{value // 64}{value // 16 % 4}{value // 4 % 4}{value % 4}"


def _fixture(tmp_path: Path, *, conv_rows: bytes | None = None):
    import torch

    source = tmp_path / "tiny-beta4.safetensors"
    qkv = torch.arange(12, dtype=torch.float32).reshape(6, 2).to(torch.bfloat16)
    raw = torch.tensor([[2.0, 3.0]], dtype=torch.bfloat16)
    i8 = conv_rows if conv_rows is not None else bytes([1] * (2 * 256))
    assert len(i8) == 2 * 256
    scales = struct.pack("<2f", 0.5, 0.25)
    marker = b'{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}'
    _write_safetensors(
        source,
        (
            ("plain", "BF16", (1, 2), _bytes(raw)),
            ("qkv", "BF16", (6, 2), _bytes(qkv)),
            ("conv.weight", "I8", (2, 256), i8),
            ("conv.weight_scale", "F32", (2, 1), scales),
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
            "conv.weight", "conv.weight", "I8", "BF16", (2, 256), 512, 1024, OP_INVERSE_CONVROT_BF16, "conv", 256
        ),
        TensorAction("conv.weight_scale", None, "F32", None, (2, 1), 8, 0, OP_OMIT_SCALE, "conv", 256),
        TensorAction("conv.comfy_quant", None, "U8", None, (len(marker),), len(marker), 0, OP_OMIT_MARKER, "conv", 256),
    )
    plan = NativeExportPlan(
        "test",
        "test",
        "transformer",
        "beta4",
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
        ResourceEnvelope(2, 4096, 1024),
        actions,
        (),
        3,
        1052,
        None,
        (),
        "test",
        "digest",
        "target",
        "target-schema",
    )
    return source, digest, source_schema, plan, raw, qkv, indices


def test_direct_provider_yields_cpu_bf16_targets_without_an_output_file(tmp_path, monkeypatch):
    from comfy_omni.runtime.h3 import raw_beta4

    source, digest, source_schema, plan, plain, qkv, indices = _fixture(tmp_path)
    monkeypatch.setattr(raw_beta4, "build_beta4_dense_plan", lambda report: plan)
    identity = raw_beta4.RawBeta4Identity("tiny", source.stat().st_size, digest, source_schema, 5)
    binding = raw_beta4.RawBeta4Binding.establish(source, identity=identity)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    # Establishment is the only full-hash operation. Iteration must reopen the
    # already authenticated source with its recorded FD/path identity instead.
    monkeypatch.setattr(raw_beta4, "SafeTensorSources", lambda *_: pytest.fail("runtime rehashed source"))
    weights = dict(binding.open_weights())

    assert tuple(weights) == ("conv.weight", "plain", "qkv")
    assert all(value.device.type == "cpu" and value.dtype == torch.bfloat16 for value in weights.values())
    assert torch.equal(weights["plain"], plain)

    # Independent tiny QKV oracle: source rows are Q0, Q1, K0, K1, V0,
    # V1, while the grouped loader consumes Q0, K0, V0, Q1, K1, V1.
    assert indices == (0, 2, 4, 1, 3, 5)
    assert weights["qkv"][:, 0].float().tolist() == [0.0, 4.0, 8.0, 2.0, 6.0, 10.0]
    assert torch.equal(weights["qkv"], qkv.index_select(0, torch.tensor(indices)))

    # The fixed regular-Hadamard base has a negative final column rather
    # than a DC-first Sylvester row. A 256-wide all-one row therefore maps
    # to its scale in every output column.
    conv = weights["conv.weight"]
    assert conv.shape == (2, 256)
    assert torch.equal(conv, torch.tensor([[0.5] * 256, [0.25] * 256], dtype=torch.bfloat16))
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_direct_provider_rejects_drift_before_consuming_payload(tmp_path, monkeypatch):
    from comfy_omni.runtime.h3 import raw_beta4

    source, digest, source_schema, plan, *_ = _fixture(tmp_path)
    monkeypatch.setattr(raw_beta4, "build_beta4_dense_plan", lambda report: plan)
    binding = raw_beta4.RawBeta4Binding.establish(
        source, identity=raw_beta4.RawBeta4Identity("tiny", source.stat().st_size, digest, source_schema, 5)
    )
    current = source.stat()
    os.utime(source, ns=(current.st_atime_ns, current.st_mtime_ns + 1))
    with pytest.raises(ContractError, match="identity"):
        next(binding.open_weights())


def test_primary_contract_remains_exactly_534_target_descriptors():
    from comfy_omni.runtime.h3 import raw_beta4

    assert raw_beta4.PRIMARY_RAW_BETA4_IDENTITY.tensor_count == 934
    assert len(raw_beta4.primary_target_descriptors()) == 534


def test_establish_rejects_action_geometry_drift(tmp_path, monkeypatch):
    from comfy_omni.runtime.h3 import raw_beta4

    source, digest, source_schema, plan, *_ = _fixture(tmp_path)
    drifted_actions = (replace(plan.actions[0], target_bytes=2), *plan.actions[1:])
    monkeypatch.setattr(raw_beta4, "build_beta4_dense_plan", lambda report: replace(plan, actions=drifted_actions))

    with pytest.raises(ContractError, match="target byte geometry"):
        raw_beta4.RawBeta4Binding.establish(
            source, identity=raw_beta4.RawBeta4Identity("tiny", source.stat().st_size, digest, source_schema, 5)
        )


def test_direct_provider_uses_the_fixed_regular_hadamard_sign_basis(tmp_path, monkeypatch):
    from comfy_omni.runtime.h3 import raw_beta4

    # One impulse at input column zero makes the full 256-column sign basis
    # observable. The fixed H4 contributes one negative sign for every base-4
    # digit equal to 3, then normalizes by sqrt(256) == 16.
    source, digest, source_schema, plan, *_ = _fixture(tmp_path, conv_rows=bytes((1,)) + bytes(511))
    monkeypatch.setattr(raw_beta4, "build_beta4_dense_plan", lambda report: plan)
    binding = raw_beta4.RawBeta4Binding.establish(
        source, identity=raw_beta4.RawBeta4Identity("tiny", source.stat().st_size, digest, source_schema, 5)
    )

    conv = dict(binding.open_weights())["conv.weight"]
    expected = torch.tensor(
        [0.5 / 16 * (-1 if sum(digit == "3" for digit in _base4_digits(column)) % 2 else 1) for column in range(256)],
        dtype=torch.bfloat16,
    )
    assert torch.equal(conv[0], expected)
    assert torch.count_nonzero(conv[1]).item() == 0


@pytest.mark.parametrize("dtype_code,torch_dtype", [("F16", torch.float16), ("F32", torch.float32)])
def test_standard_provider_preserves_native_copy_and_qkv_bits(tmp_path, monkeypatch, dtype_code, torch_dtype):
    from comfy_omni.conversion.exporters import planning
    from comfy_omni.runtime.h3 import raw_standard

    source, _, _, skeleton, *_ = _fixture(tmp_path)
    epsilon = 2**-10 if dtype_code == "F16" else 2**-20
    plain = torch.tensor([[1.0 + epsilon, -0.5 - epsilon]], dtype=torch_dtype)
    qkv = (torch.arange(12, dtype=torch.float32).reshape(6, 2) / 8 + epsilon).to(torch_dtype)
    table = torch.tensor([[1.0 + 2**-20, -3.1415925]], dtype=torch.float32)
    names = ("plain.weight", "blocks.0.attn.qkv_proj.weight", "adaln_t_table")
    _write_safetensors(
        source,
        (
            (names[0], dtype_code, tuple(plain.shape), _bytes(plain)),
            (names[1], dtype_code, tuple(qkv.shape), _bytes(qkv)),
            (names[2], "F32", tuple(table.shape), _bytes(table)),
        ),
    )
    original = source.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    with SafeTensorSources((source,)) as sources:
        descriptors = tuple(sorted((item.descriptor for item in sources.tensors.values()), key=lambda item: item.name))
    # Exercise the actual generic planner's dtype and QKV decisions. Only its
    # full-model authority is replaced with a small, explicitly named fixture.
    actions = tuple(planning._action_for(item, None, skeleton.qkv_layout) for item in descriptors)
    plan = replace(
        skeleton,
        source_files=(SourceBinding(str(source), len(original), digest),),
        actions=actions,
        target_payload_bytes=sum(action.target_bytes for action in actions),
        target_contract=None,
        target_schema_sha256=None,
    )
    monkeypatch.setattr(raw_standard, "build_native_export_plan", lambda report, record, template: plan)
    identity = raw_standard.RawStandardIdentity(
        "tiny-standard", len(original), digest, schema_sha256(descriptors), 3, 3
    )
    binding = raw_standard.RawStandardBinding.establish(source, identity=identity)
    observed = dict(binding.open_weights())
    expected = {
        names[0]: plain,
        names[1]: qkv[[0, 2, 4, 1, 3, 5]],
        names[2]: table,
    }
    assert set(observed) == set(expected)
    for name, value in expected.items():
        assert observed[name].dtype == value.dtype
        assert observed[name].device.type == "cpu"
        assert torch.equal(observed[name], value)
        assert _bytes(observed[name]) == _bytes(value)
    for descriptor in binding.target_descriptors:
        value = expected[descriptor.name]
        assert descriptor.data_offsets == (0, value.numel() * value.element_size())
        assert descriptor.dtype == ("F32" if descriptor.name == names[2] else dtype_code)
    assert source.read_bytes() == original
    assert set(tmp_path.iterdir()) == {source}


def test_provider_releases_previous_tensor_and_buffer_before_allocating_next(tmp_path, monkeypatch):
    from comfy_omni.runtime.h3 import raw_beta4

    source, digest, source_schema, plan, *_ = _fixture(tmp_path)
    monkeypatch.setattr(raw_beta4, "build_beta4_dense_plan", lambda report: plan)
    binding = raw_beta4.RawBeta4Binding.establish(
        source, identity=raw_beta4.RawBeta4Identity("tiny", source.stat().st_size, digest, source_schema, 5)
    )
    references = []
    original_empty, original_numpy = torch.empty, torch.Tensor.numpy

    def checked_empty(*args, **kwargs):
        assert not any(reference() is not None for reference in references), "provider retained its previous target"
        value = original_empty(*args, **kwargs)
        references.append(weakref.ref(value))
        return value

    def tracked_numpy(tensor, *args, **kwargs):
        array = original_numpy(tensor, *args, **kwargs)
        references.append(weakref.ref(array))
        return array

    monkeypatch.setattr(torch, "empty", checked_empty)
    monkeypatch.setattr(torch.Tensor, "numpy", tracked_numpy)
    iterator = binding.open_weights()
    try:
        for expected_name in ("conv.weight", "plain", "qkv"):
            item = next(iterator)
            assert item[0] == expected_name
            del item
        with pytest.raises(StopIteration):
            next(iterator)
    finally:
        iterator.close()
    assert not any(reference() is not None for reference in references)
