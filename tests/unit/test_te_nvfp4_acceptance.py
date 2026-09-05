"""Independent acceptance runs against a complete, small on-disk component."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_te_nvfp4_boundaries import TOOL
from test_te_nvfp4_export import bind_fixture, write_fixture

from comfy_omni.artifacts import fileops
from comfy_omni.contracts import te_nvfp4 as contract
from comfy_omni.conversion.exporters import te_nvfp4 as execution
from comfy_omni.conversion.exporters.te_nvfp4_plan import plan_te_dense_export


def acceptance(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts" / "acceptance"))
    return importlib.import_module("verify_te_nvfp4_dense_conversion")


def test_full_real_host_projection_matches_independent_pinned_schema(monkeypatch):
    verifier = acceptance(monkeypatch)
    records = {n: {"dtype": dtype, "shape": list(shape)} for n, (dtype, shape) in contract.TARGET_INVENTORY.items()}
    projection = verifier.host_projection(records)
    assert projection["logical_parameter_count"] == 752
    assert projection["actual_host_loaded"] is False
    wrong = dict(records)
    wrong["model.layers.0.ignored.weight"] = wrong.pop("model.language_model.layers.0.self_attn.q_proj.weight")
    with pytest.raises(ValueError, match="ignore"):
        verifier.host_projection(wrong)


def tiny_host_slots():
    slots = {"text_model.embed_tokens.weight": {"dtype": "BF16", "shape": [4, 64]}}
    for layer in range(50):
        prefix = f"text_model.layers.{layer}."
        for role, shape in {
            "self_attn.qkv_proj": [384, 64],
            "self_attn.o_proj": [128, 64],
            "mlp.gate_up_proj": [256, 64],
            "mlp.down_proj": [128, 64],
            "input_layernorm": [2],
            "post_attention_layernorm": [2],
            "self_attn.q_norm": [2],
            "self_attn.k_norm": [2],
        }.items():
            slots[prefix + role + ".weight"] = {"dtype": "BF16", "shape": shape}
    for name in contract.TARGET_INVENTORY:
        if name.startswith("model.visual."):
            slots["vision." + name.removeprefix("model.visual.")] = {"dtype": "BF16", "shape": [2]}
    return slots


def test_independent_full_component_receipt_and_rehashed_wrong_numeric_rejection(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    source, config, tensors = write_fixture(tmp_path)
    bind_fixture(monkeypatch, source, config, tensors)
    plan = plan_te_dense_export(source, config)
    output = tmp_path / "dense"
    execution.execute_te_dense_export(plan, output, tool=TOOL)
    verifier = acceptance(monkeypatch)
    for key, value in {
        "SOURCE_SIZE": contract.SOURCE_BYTES,
        "SOURCE_SHA": contract.SOURCE_SHA256,
        "CONFIG_SIZE": contract.CONFIG_BYTES,
        "CONFIG_SHA": contract.CONFIG_SHA256,
        "SOURCE_SCHEMA": contract.SOURCE_SCHEMA_SHA256,
        "TARGET_SCHEMA": contract.TARGET_SCHEMA_SHA256,
        "TARGET_BYTES": contract.TARGET_PAYLOAD_BYTES,
        "HOST_SCHEMA": verifier._schema(tiny_host_slots()),
    }.items():
        monkeypatch.setattr(verifier, key, value)
    args = SimpleNamespace(
        source=source,
        config=config,
        output=output,
        expected_commit=TOOL.source_commit,
        expected_wheel_sha256=TOOL.wheel_sha256,
        expected_version=TOOL.version,
    )
    result = verifier.verify(args)
    assert result["status"] == "VERIFIED"
    assert result["all_plain_tensors_byte_equal"] == 551
    assert len(result["sampled_matrices"]) == 351
    nv = [item for item in result["sampled_matrices"] if "embed_tokens" not in item["target_name"]]
    assert len(nv) == 350 and all(item["rows"] == [0, 64, 127] and item["columns_per_row"] == 64 for item in nv)
    assert result["all_numeric_elements_verified"] is False
    # Change a sampled value, then rehash every producer-controlled digest.
    # Only the independent numerical oracle can reject the now self-consistent receipt.
    path = output / "model.safetensors"
    with verifier._held(path, path.stat().st_size) as held:
        descriptors, offset = verifier.header(held)
    name = "model.language_model.layers.49.mlp.down_proj.weight"
    tensor = descriptors[name]
    raw = bytearray(path.read_bytes())
    raw[offset + tensor["start"] + 127 * 64 * 2] ^= 1
    path.chmod(0o600)
    path.write_bytes(raw)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["tensor_sha256"][name] = (
        __import__("hashlib").sha256(raw[offset + tensor["start"] : offset + tensor["end"]]).hexdigest()
    )
    entry = next(item for item in manifest["files"] if item["name"] == "model.safetensors")
    entry["sha256"] = __import__("hashlib").sha256(raw).hexdigest()
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = verifier._digest(manifest)
    manifest_path.chmod(0o600)
    manifest_path.write_bytes(fileops.canonical_json(manifest))
    with pytest.raises(ValueError, match="exact BF16 numerical mismatch"):
        verifier.verify(args)


def test_acceptance_budget_includes_written_header_and_receipts(tmp_path, monkeypatch):
    acceptance(monkeypatch)
    harness = importlib.import_module("te_nvfp4_dense_conversion")
    source, config, tensors = write_fixture(tmp_path)
    bind_fixture(monkeypatch, source, config, tensors)
    budget = harness.estimate(plan_te_dense_export(source, config))
    assert budget["payload_bytes"] < budget["safetensors_file_bytes"] < budget["maximum_output_bytes"]
    monkeypatch.setattr(contract, "MAX_OUTPUT_BYTES", budget["maximum_output_bytes"] - 1)
    with pytest.raises(ValueError, match="allocation"):
        harness.estimate(plan_te_dense_export(source, config))
