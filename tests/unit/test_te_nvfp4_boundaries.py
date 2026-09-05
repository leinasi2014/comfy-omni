"""Behavioral failure and streaming checks against real synthetic files."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import replace

import pytest
from test_te_nvfp4_export import bind_fixture, fixture_tensors, oracle, write_fixture

from comfy_omni.artifacts import fileops
from comfy_omni.artifacts.sources import SafeTensorSources
from comfy_omni.contracts.models import ContractError
from comfy_omni.conversion.exporters import te_nvfp4 as execution
from comfy_omni.conversion.exporters.te_nvfp4_plan import plan_te_dense_export
from comfy_omni.conversion.numerics.te_nvfp4 import nvfp4_bf16_stripe, validate_bf16_chunk
from comfy_omni.domain.normalization import ToolIdentity

TOOL = ToolIdentity("comfy-omni", "0.2.0a1", "1" * 40, "2" * 64)


@pytest.mark.parametrize(
    "field,value",
    [("consumer", "another-consumer"), ("profile", "arbitrary"), ("max_rows", 256), ("target_payload_bytes", 1)],
)
def test_rehashed_forged_plan_is_rejected_before_output(tmp_path, monkeypatch, field, value):
    source, config, tensors = write_fixture(tmp_path)
    bind_fixture(monkeypatch, source, config, tensors)
    plan = replace(plan_te_dense_export(source, config), **{field: value})
    plan = replace(
        plan,
        content_sha256=hashlib.sha256(fileops.canonical_json(plan.to_dict(include_content_sha256=False))).hexdigest(),
    )
    with pytest.raises(ContractError, match="authority"):
        execution.execute_te_dense_export(plan, tmp_path / "dense", tool=TOOL)
    assert not (tmp_path / "dense").exists()


@pytest.mark.parametrize("which", ["source", "config"])
def test_late_input_mutation_removes_exclusive_output(tmp_path, monkeypatch, which):
    pytest.importorskip("torch")
    source, config, tensors = write_fixture(tmp_path)
    bind_fixture(monkeypatch, source, config, tensors)
    plan = plan_te_dense_export(source, config)
    output = tmp_path / "dense"
    original = fileops.sha256_file_pinned
    changed = []

    def rehash(path):
        result = original(path)
        if path.parent == output and not changed:
            target = source if which == "source" else config
            raw = bytearray(target.read_bytes())
            raw[-1] ^= 1
            target.chmod(0o600)
            target.write_bytes(raw)
            changed.append(target)
        return result

    monkeypatch.setattr(fileops, "sha256_file_pinned", rehash)
    with pytest.raises(ContractError, match="changed"):
        execution.execute_te_dense_export(plan, output, tool=TOOL)
    assert changed
    assert not output.exists()


def test_second_stripe_is_read_only_after_first_is_written(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    tensors = fixture_tensors()
    name = "model.layers.0.mlp.down_proj.weight"
    dtype, shape, raw = tensors[name]
    tensors[name] = (dtype, (256, shape[1]), raw * 2)
    scales = name.removesuffix(".weight") + ".weight_scale"
    dtype, shape, raw = tensors[scales]
    tensors[scales] = (dtype, (256, shape[1]), raw * 2)
    source, config, tensors = write_fixture(tmp_path, tensors)
    bind_fixture(monkeypatch, source, config, tensors)
    plan = plan_te_dense_export(source, config)
    original = SafeTensorSources.iter_raw_range
    seen = []

    def read(self, located, offset, length, **kwargs):
        if located.descriptor.name == name:
            staged = list(tmp_path.glob(".dense.stage-*/model.safetensors"))
            seen.append((offset, length, staged[0].stat().st_size))
        return original(self, located, offset, length, **kwargs)

    monkeypatch.setattr(SafeTensorSources, "iter_raw_range", read)
    execution.execute_te_dense_export(plan, tmp_path / "dense", tool=TOOL)
    assert len(seen) == 2 and [item[:2] for item in seen] == [(0, 4096), (4096, 4096)]
    assert seen[1][2] - seen[0][2] >= 128 * 64 * 2


@pytest.mark.parametrize("bad", [b"\x80\x7f", b"\xc0\x7f"])
def test_nonfinite_passthrough_is_rejected(bad):
    pytest.importorskip("torch")
    with pytest.raises(ContractError, match="finite"):
        validate_bf16_chunk(bad)


def test_consumer_discriminates_blocked_nibbles_and_bf16_steps():
    pytest.importorskip("torch")
    independent = oracle()
    packed = bytes((i * 17) % 256 for i in range(4096))
    scales = bytes(8 + (i * 7) % 110 for i in range(512))
    global_scale = struct.pack("<f", 1.00390625)
    actual = nvfp4_bf16_stripe(packed, scales, global_scale, rows=128, columns=64)
    expected = b"".join(
        independent.nvfp4_row(packed[r * 32 : (r + 1) * 32], scales, global_scale, row_in_band=r) for r in range(128)
    )
    assert actual == expected
    swapped = bytes((value >> 4) | ((value & 15) << 4) for value in packed)
    assert nvfp4_bf16_stripe(swapped, scales, global_scale, rows=128, columns=64) != expected
    natural = bytes(scales[(row % 32) * 16 + (row // 32) * 4 + block] for row in range(128) for block in range(4))
    assert nvfp4_bf16_stripe(packed, natural, global_scale, rows=128, columns=64) != expected
    rounded = nvfp4_bf16_stripe(bytes([0x33]) * 4096, bytes([0x3A]) * 512, global_scale, rows=128, columns=64)
    # q=1.5, block=1.25: global is rounded to BF16 before either product.
    assert rounded == struct.pack("<H", 0x3FF0) * 8192
    assert independent.bf16_bits(1.5 * 1.25 * 1.00390625) == 0x3FF1


def test_invalid_marker_is_not_accepted_by_a_matching_file_digest(tmp_path, monkeypatch):
    tensors = fixture_tensors()
    name = "model.layers.0.mlp.down_proj.comfy_quant"
    dtype, shape, raw = tensors[name]
    tensors[name] = (dtype, shape, raw.replace(b"nvfp4", b"nvfp5"))
    source, config, tensors = write_fixture(tmp_path, tensors)
    bind_fixture(monkeypatch, source, config, tensors)
    with pytest.raises(ContractError, match="unsupported"):
        plan_te_dense_export(source, config)
