"""Checkpoint-only LoRA evidence, independent of a native runtime package."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest


def _write(path: Path, records, metadata=None):
    header = {"__metadata__": metadata or {"base_model": "MiniMax-H3"}}
    payload = bytearray()
    for name, dtype, shape, raw in records:
        start = len(payload)
        payload.extend(raw)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(payload)]}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size


def _pair(tmp_path):
    base, adapter = tmp_path / "base.safetensors", tmp_path / "adapter.safetensors"
    marker = b'{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}'
    base_pin = _write(
        base,
        [
            ("blocks.0.attn.out_proj.weight", "I8", [4, 256], bytes(4 * 256)),
            ("blocks.0.attn.out_proj.weight_scale", "F32", [4, 1], bytes(16)),
            ("blocks.0.attn.out_proj.comfy_quant", "U8", [len(marker)], marker),
        ],
    )
    adapter_pin = _write(
        adapter,
        [
            ("diffusion_model.blocks.0.attn.out_proj.lora_A.weight", "BF16", [2, 256], bytes(2 * 256 * 2)),
            ("diffusion_model.blocks.0.attn.out_proj.lora_B.weight", "BF16", [4, 2], bytes(4 * 2 * 2)),
        ],
    )
    return base, adapter, base_pin, adapter_pin


def test_checkpoint_pair_retains_both_identities_without_native_package(tmp_path):
    base, adapter, base_pin, adapter_pin = _pair(tmp_path)
    receipt = _invoke(base, adapter).to_dict()
    assert receipt["status"] == "UNSUPPORTED"
    assert receipt["evidence"].get("scope") == "checkpoint-only"
    for role, pin in (("base", base_pin), ("adapter", adapter_pin)):
        assert receipt["evidence"][role]["actual_sha256"] == pin[0]
        assert receipt["evidence"][role]["expected_sha256"] == pin[0]
        assert receipt["evidence"][role]["actual_bytes"] == pin[1]
    assert receipt["evidence"]["promotion_capable"] is False
    assert receipt["evidence"]["offline_fold"] == "NOT_RUN"
    assert receipt["evidence"]["runtime_activation"] == "NOT_RUN"
    assert receipt["reason_code"] == "BASE_REPRESENTATION_UNBINDABLE"
    census = receipt["evidence"]["base"]["census"]
    assert census["convrot_group_count"] == 1
    assert census["convrot_group_size_census"] == {"256": 1}
    assert receipt["evidence"]["mapping"]["modules"][0]["rank"] == 2


def _invoke(base, adapter, **kwargs):
    from comfy_omni.conversion.oracle.checkpoint_preflight import preflight_checkpoint_candidate

    values = {
        "base_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
        "base_bytes": base.stat().st_size,
        "pinned_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
        "pinned_bytes": adapter.stat().st_size,
    }
    values.update(kwargs)
    return preflight_checkpoint_candidate("candidate", base, adapter, **values)


@pytest.mark.parametrize("role", ["base", "adapter"])
def test_pin_mismatch_is_invalid_input_not_unsupported(tmp_path, role):
    from comfy_omni.conversion.oracle.checkpoint_contract import CheckpointInputError

    base, adapter, _, _ = _pair(tmp_path)
    option = "base_sha256" if role == "base" else "pinned_sha256"
    with pytest.raises(CheckpointInputError) as caught:
        _invoke(base, adapter, **{option: "0" * 64})
    failure = caught.value.to_dict()
    assert failure["status"] == "INVALID_INPUT"
    assert failure["reason_code"] == "CHECKPOINT_PIN_MISMATCH"
    assert failure["role"] == role
    assert failure["evidence"]["actual_sha256"] != "0" * 64


@pytest.mark.parametrize(
    "raw",
    [
        b"{}",
        struct.pack("<Q", 100) + b"{}",
        struct.pack("<Q", 10) + b'{"a":0,"a":0}',
    ],
)
def test_malformed_header_is_not_a_compatibility_receipt(tmp_path, raw):
    from comfy_omni.conversion.oracle.checkpoint_contract import CheckpointInputError

    base, adapter, _, _ = _pair(tmp_path)
    adapter.write_bytes(raw)
    with pytest.raises(CheckpointInputError):
        _invoke(base, adapter)


@pytest.mark.parametrize("change", ["tail", "gap", "overlap", "duplicate-key"])
def test_strict_header_rules_still_apply(tmp_path, change):
    from comfy_omni.conversion.oracle.checkpoint_contract import CheckpointInputError

    base, adapter, _, _ = _pair(tmp_path)
    if change == "tail":
        adapter.write_bytes(adapter.read_bytes() + b"x")
    else:
        header = {
            "a": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
            "b": {"dtype": "BF16", "shape": [1], "data_offsets": [2, 4]},
        }
        if change == "gap":
            header["b"]["data_offsets"] = [3, 5]
        if change == "overlap":
            header["b"]["data_offsets"] = [1, 3]
        encoded = json.dumps(header).encode()
        if change == "duplicate-key":
            encoded = encoded.replace(b'"b":', b'"a":')
        adapter.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(5 if change == "gap" else 4))
    with pytest.raises(CheckpointInputError) as caught:
        _invoke(base, adapter)
    assert caught.value.role == "adapter"


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), True, "1", 10**1000])
def test_invalid_request_scale(tmp_path, scale):
    from comfy_omni.conversion.oracle.checkpoint_contract import CheckpointInputError

    base, adapter, _, _ = _pair(tmp_path)
    with pytest.raises(CheckpointInputError, match="REQUEST_SCALE_INVALID"):
        _invoke(base, adapter, scale=scale)


@pytest.mark.parametrize("alpha", [float("nan"), float("inf"), 8.0])
def test_alpha_scalar_is_observed_without_default_multiplier(tmp_path, alpha):
    base, adapter, _, _ = _pair(tmp_path)
    prefix = "diffusion_model.blocks.0.attn.out_proj"
    _write(
        adapter,
        [
            (prefix + ".lora_A.weight", "BF16", [2, 256], bytes(2 * 256 * 2)),
            (prefix + ".lora_B.weight", "BF16", [4, 2], bytes(4 * 2 * 2)),
            (prefix + ".alpha", "F32", [], struct.pack("<f", alpha)),
        ],
    )
    receipt = _invoke(base, adapter, scale=0.3).to_dict()
    if alpha == 8.0:
        record = receipt["evidence"]["mapping"]["modules"][0]
        assert record["alpha"] == {prefix + ".alpha": 8.0}
        assert record["effective_multiplier"] is None
        assert record["requested_scale"] == 0.3
    else:
        assert receipt["reason_code"] == "ADAPTER_SCALE_UNSUPPORTED"


def test_malformed_quantization_refuses_before_receipt(tmp_path):
    from comfy_omni.conversion.oracle.checkpoint_contract import CheckpointInputError

    base, adapter, _, _ = _pair(tmp_path)
    raw = base.read_bytes().replace(b'"convrot_groupsize":256', b'"convrot_groupsize":000')
    base.write_bytes(raw)
    with pytest.raises(CheckpointInputError, match="CHECKPOINT_QUANTIZATION_INVALID"):
        _invoke(base, adapter)


@pytest.mark.parametrize("mutation", ["replace", "rewrite"])
def test_source_changes_during_observation_prevent_receipt(tmp_path, monkeypatch, mutation):
    from comfy_omni.conversion.oracle import checkpoint_preflight as module
    from comfy_omni.conversion.oracle.checkpoint_contract import CheckpointInputError

    base, adapter, _, _ = _pair(tmp_path)
    original = module.observe_mapping

    def changed(*args, **kwargs):
        if mutation == "replace":
            replacement = base.with_suffix(".replacement")
            replacement.write_bytes(base.read_bytes())
            replacement.replace(base)
        else:
            raw = bytearray(adapter.read_bytes())
            raw[-1] ^= 1
            adapter.write_bytes(raw)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "observe_mapping", changed)
    with pytest.raises(CheckpointInputError, match="CHECKPOINT_INPUT_INVALID"):
        _invoke(base, adapter)


def test_same_descriptors_are_hashed_scanned_and_reverified(tmp_path, monkeypatch):
    from comfy_omni.artifacts import sources

    base, adapter, _, _ = _pair(tmp_path)
    seen = {"header": [], "hash": [], "raw": []}
    old_header, old_hash, old_raw = (
        sources.read_safetensors_header_stream,
        sources._hash_stream,
        sources.SafeTensorSources.read_raw,
    )

    def header(stream, *args):
        seen["header"].append(stream.fileno())
        return old_header(stream, *args)

    def hashed(stream):
        seen["hash"].append(stream.fileno())
        return old_hash(stream)

    def raw(self, tensor):
        seen["raw"].append(tensor.descriptor.name)
        return old_raw(self, tensor)

    monkeypatch.setattr(sources, "read_safetensors_header_stream", header)
    monkeypatch.setattr(sources, "_hash_stream", hashed)
    monkeypatch.setattr(sources.SafeTensorSources, "read_raw", raw)
    _invoke(base, adapter)
    assert len(set(seen["header"])) == 2
    assert sorted(seen["hash"]) == sorted(seen["header"] * 2)
    assert seen["raw"] == ["blocks.0.attn.out_proj.comfy_quant"]


def test_receipt_is_deeply_immutable_and_does_not_expose_paths(tmp_path):
    base, adapter, _, _ = _pair(tmp_path)
    before = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in (base, adapter)}
    receipt = _invoke(base, adapter)
    with pytest.raises(TypeError):
        receipt.evidence["base"]["actual_sha256"] = "0" * 64
    document = receipt.to_dict()
    document["evidence"]["mapping"]["modules"].clear()
    assert receipt.evidence["mapping"]["modules"]
    assert str(tmp_path) not in json.dumps(receipt.to_dict())
    assert before == {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in (base, adapter)}


@pytest.mark.parametrize("size", [0, -256, 3])
def test_quant_group_size_is_a_positive_power_of_two(tmp_path, size):
    from comfy_omni.conversion.oracle.checkpoint_contract import CheckpointInputError

    base, adapter, _, _ = _pair(tmp_path)
    marker = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": size}).encode()
    _write(
        base,
        [
            ("blocks.0.attn.out_proj.weight", "I8", [4, 256], bytes(1024)),
            ("blocks.0.attn.out_proj.weight_scale", "F32", [4, 1], bytes(16)),
            ("blocks.0.attn.out_proj.comfy_quant", "U8", [len(marker)], marker),
        ],
    )
    with pytest.raises(CheckpointInputError, match="CHECKPOINT_QUANTIZATION_INVALID"):
        _invoke(base, adapter)


@pytest.mark.parametrize("flag", ["true", 1])
def test_known_quantization_boolean_type_is_strict(tmp_path, flag):
    from comfy_omni.conversion.oracle.checkpoint_contract import CheckpointInputError

    base, adapter, _, _ = _pair(tmp_path)
    marker = json.dumps({"format": "int8_tensorwise", "convrot": flag, "convrot_groupsize": 256}).encode()
    _write(
        base,
        [
            ("blocks.0.attn.out_proj.weight", "I8", [4, 256], bytes(1024)),
            ("blocks.0.attn.out_proj.weight_scale", "F32", [4, 1], bytes(16)),
            ("blocks.0.attn.out_proj.comfy_quant", "U8", [len(marker)], marker),
        ],
    )
    with pytest.raises(CheckpointInputError, match="CHECKPOINT_QUANTIZATION_INVALID"):
        _invoke(base, adapter)


def test_quant_orphan_scale_is_invalid_even_with_a_valid_group(tmp_path):
    from comfy_omni.conversion.oracle.checkpoint_contract import CheckpointInputError

    base, adapter, _, _ = _pair(tmp_path)
    from comfy_omni.artifacts.sources import SafeTensorSources

    with SafeTensorSources([base]) as source:
        records = [
            (name, item.descriptor.dtype, list(item.descriptor.shape), source.read_raw(item))
            for name, item in source.tensors.items()
        ]
    _write(base, records + [("extra.weight_scale", "F32", [1], bytes(4))])
    with pytest.raises(CheckpointInputError, match="CHECKPOINT_QUANTIZATION_INVALID"):
        _invoke(base, adapter)


@pytest.mark.parametrize(
    "metadata,alpha,reason",
    [
        ({"alpha": "8", "ss_network_alpha": "4"}, 8, "ALPHA_DECLARATION_CONFLICT"),
        ({"ss_network_alpha": "4"}, 8, "ALPHA_DECLARATION_CONFLICT"),
        ({"ss_network_dim": "4"}, 8, "RANK_DECLARATION_MISMATCH"),
        ({"rank": "-2"}, 8, "RANK_DECLARATION_INVALID"),
    ],
)
def test_scale_declaration_conflicts_are_explicit(tmp_path, metadata, alpha, reason):
    base, adapter, _, _ = _pair(tmp_path)
    prefix = "diffusion_model.blocks.0.attn.out_proj"
    _write(
        adapter,
        [
            (prefix + ".lora_A.weight", "BF16", [2, 256], bytes(1024)),
            (prefix + ".lora_B.weight", "BF16", [4, 2], bytes(16)),
            (prefix + ".alpha", "F32", [], struct.pack("<f", alpha)),
        ],
        metadata,
    )
    document = _invoke(base, adapter).to_dict()
    assert document["reason_code"] == "ADAPTER_SCALE_UNSUPPORTED"
    assert reason in {item["reason"] for item in document["evidence"]["scale_failures"]}


def test_unproved_quantization_format_is_unsupported_not_corrupt(tmp_path):
    base, adapter, _, _ = _pair(tmp_path)
    marker = b'{"format":"nvfp4"}'
    _write(
        base,
        [
            ("blocks.0.attn.out_proj.weight", "U8", [4, 256], bytes(1024)),
            ("blocks.0.attn.out_proj.comfy_quant", "U8", [len(marker)], marker),
        ],
    )
    document = _invoke(base, adapter).to_dict()
    assert document["status"] == "UNSUPPORTED"
    assert document["reason_code"] == "QUANT_LAYOUT_INCOMPATIBLE"
    assert document["evidence"]["base"]["census"]["refusal"]["marker_declaration_census"] == {"nvfp4": 1}


def _harness():
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "scripts" / "acceptance" / "lora_checkpoint_preflight.py"
    spec = importlib.util.spec_from_file_location("checkpoint_acceptance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_acceptance_harness_binds_identity_and_never_overwrites(tmp_path, monkeypatch):
    from comfy_omni.domain.normalization import ToolIdentity

    base, adapter, base_pin, adapter_pin = _pair(tmp_path)
    harness = _harness()
    tool = ToolIdentity("comfy-omni", "0.2.0a1", "a" * 40, "b" * 64)
    monkeypatch.setattr(harness, "installed_tool_identity", lambda: tool)
    monkeypatch.setattr(harness, "PRIMARY_PIN", (base_pin[1], base_pin[0]))
    monkeypatch.setattr(harness, "CANDIDATE_PINS", {"spatial-physics-lora": (adapter_pin[1], adapter_pin[0])})
    output = tmp_path / "receipt.json"
    argv = [
        "--base",
        str(base),
        "--adapter",
        str(adapter),
        "--candidate-id",
        "spatial-physics-lora",
        "--scale",
        "0.3",
        "--expected-commit",
        "a" * 40,
        "--expected-wheel-sha256",
        "b" * 64,
        "--image-digest",
        "sha256:" + "c" * 64,
        "--result-out",
        str(output),
    ]
    assert harness.main(argv) == 0
    original = output.read_bytes()
    result = json.loads(original)
    digest = result.pop("receipt_sha256")
    from comfy_omni.artifacts.fileops import canonical_json

    assert hashlib.sha256(canonical_json(result)).hexdigest() == digest
    assert result["tool"] == tool.to_dict()
    assert harness.main(argv) == 2
    assert output.read_bytes() == original
    invalid_output = tmp_path / "invalid.json"
    monkeypatch.setattr(harness, "PRIMARY_PIN", (base_pin[1], "0" * 64))
    assert harness.main([*argv[:-1], str(invalid_output)]) == 2
    assert not invalid_output.exists()


def test_harness_pins_match_fixed_model_contract():
    baseline_path = Path(__file__).resolve().parents[2] / "docs" / "testing" / "model-baseline.v1.json"
    assets = {item["id"]: item for item in json.loads(baseline_path.read_text())["assets"]}
    harness = _harness()
    assert harness.PRIMARY_PIN == (assets["primary-dit"]["bytes"], assets["primary-dit"]["sha256"])
    for key, pin in harness.CANDIDATE_PINS.items():
        assert pin == (assets[key]["bytes"], assets[key]["sha256"])


@pytest.mark.parametrize("kind", ["missing", "symlink", "directory"])
def test_source_paths_are_regular_nonlinked_files(tmp_path, kind):
    from comfy_omni.conversion.oracle.checkpoint_contract import CheckpointInputError
    from comfy_omni.conversion.oracle.checkpoint_preflight import preflight_checkpoint_candidate

    base, adapter, base_pin, adapter_pin = _pair(tmp_path)
    target = tmp_path / "other"
    if kind == "symlink":
        target.symlink_to(adapter)
    elif kind == "directory":
        target.mkdir()
    with pytest.raises(CheckpointInputError, match="CHECKPOINT_INPUT_INVALID"):
        preflight_checkpoint_candidate(
            "candidate",
            base,
            target,
            base_sha256=base_pin[0],
            base_bytes=base_pin[1],
            pinned_sha256=adapter_pin[0],
            pinned_bytes=adapter_pin[1],
        )


def test_import_has_no_runtime_or_torch_dependency():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import comfy_omni.conversion.oracle.checkpoint_preflight; "
            "assert not any(name.split('.')[0] in {'torch','fastapi','vllm','vllm_omni'} for name in sys.modules); "
            "assert not any(name.startswith('comfy_omni.integrations') for name in sys.modules)",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
