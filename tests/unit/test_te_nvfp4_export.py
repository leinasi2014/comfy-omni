"""Complete synthetic TE inventory; values are generated, never model samples."""
from __future__ import annotations

import hashlib
import importlib.util
import struct
from pathlib import Path

import pytest

from comfy_omni.artifacts.safetensors_writer import TensorPayload, write_safetensors_file
from comfy_omni.artifacts.sources import SafeTensorSources
from comfy_omni.contracts import te_nvfp4 as contract
from comfy_omni.contracts.models import ContractError
from comfy_omni.conversion.exporters import te_nvfp4 as execution
from comfy_omni.conversion.exporters.te_nvfp4_plan import plan_te_dense_export
from comfy_omni.domain.normalization import ToolIdentity


def fixture_tensors():
    tensors = {}
    for layer in range(50):
        base = f"model.layers.{layer}"
        for role in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            name = f"{base}.{role}"
            tensors[name + ".weight"] = ("U8", (128, 32), bytes((i * 17 + layer) % 256 for i in range(4096)))
            tensors[name + ".weight_scale"] = ("F8_E4M3", (128, 4), bytes(8 + (i * 7 + layer) % 110 for i in range(512)))
            tensors[name + ".weight_scale_2"] = ("F32", (), struct.pack("<f", 1.00390625))
            marker = b'{"format": "nvfp4"}'
            tensors[name + ".comfy_quant"] = ("U8", (len(marker),), marker)
        for suffix in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"):
            tensors[f"{base}.{suffix}.weight"] = ("BF16", (2,), struct.pack("<2H", 0x3F80, 0xBF00))
    tensors["model.embed_tokens.weight"] = ("I8", (4, 64), bytes(range(256)))
    tensors["model.embed_tokens.weight_scale"] = ("F32", (), struct.pack("<f", 0.010013))
    marker = b'{"format": "int8_tensorwise"}'
    tensors["model.embed_tokens.comfy_quant"] = ("U8", (len(marker),), marker)
    vision = []
    for layer in range(27):
        for suffix in ("attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2", "norm1", "norm2"):
            for member in ("weight", "bias"):
                vision.append(f"visual.blocks.{layer}.{suffix}.{member}")
    for base in ("visual.merger", *(f"visual.deepstack_merger_list.{i}" for i in range(3))):
        for suffix in ("norm", "linear_fc1", "linear_fc2"):
            for member in ("weight", "bias"):
                vision.append(f"{base}.{suffix}.{member}")
    vision += ["visual.patch_embed.proj.weight", "visual.patch_embed.proj.bias", "visual.pos_embed.weight"]
    for index, name in enumerate(vision):
        tensors[name] = ("BF16", (2,), struct.pack("<2H", 0x3F80 + index % 100, 0x8000))
    assert len(vision) == 351 and len(tensors) == 1954
    return tensors


def write_fixture(tmp_path: Path, tensors=None):
    tensors = fixture_tensors() if tensors is None else tensors
    source = tmp_path / "synthetic.safetensors"
    write_safetensors_file(source, [TensorPayload(n, dtype, shape, len(raw), lambda raw=raw: iter((raw,))) for n, (dtype, shape, raw) in sorted(tensors.items())])
    config = tmp_path / "config.json"
    config.write_bytes(b'{"synthetic_complete_te":true}\n')
    return source, config, tensors


def bind_fixture(monkeypatch, source, config, tensors):
    inventory = {name: (dtype, shape) for name, (dtype, shape, _) in tensors.items()}
    target = {}
    for name, (dtype, shape, _) in tensors.items():
        if name.endswith((".comfy_quant", ".weight_scale", ".weight_scale_2")):
            continue
        if dtype == "U8":
            shape = (shape[0], shape[1] * 2)
        target[contract.native_name(name)] = ("BF16", shape)
    for key, value in {
        "SOURCE_INVENTORY": inventory, "SOURCE_SCHEMA_SHA256": contract.schema_sha256(inventory),
        "TARGET_INVENTORY": target, "TARGET_SCHEMA_SHA256": contract.schema_sha256(target),
        "SOURCE_BYTES": source.stat().st_size, "SOURCE_SHA256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "CONFIG_BYTES": config.stat().st_size, "CONFIG_SHA256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "TARGET_PAYLOAD_BYTES": sum(__import__("math").prod(shape) * 2 for _, shape in target.values()),
    }.items():
        monkeypatch.setattr(contract, key, value)
    monkeypatch.setattr(execution, "_space", lambda *_args, **_kwargs: None)


def oracle():
    path = Path(__file__).resolve().parents[2] / "scripts" / "acceptance" / "te_nvfp4_oracle.py"
    spec = importlib.util.spec_from_file_location("independent_te_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_real_descriptor_authority():
    assert len(contract.SOURCE_INVENTORY) == 1954
    assert len(contract.TARGET_INVENTORY) == 902
    assert contract.schema_sha256(contract.SOURCE_INVENTORY) == "807a68e6a06b2bd7f2736aea15b5ef111be8929495d98b9a9b517afd042c3c29"
    assert contract.schema_sha256(contract.TARGET_INVENTORY) == "81262d6f94f41d39c4e1ae0ab0190a8b209f81f62eda3226a89419a11cee8011"


def test_fixed_te_can_plan_the_complete_small_component(tmp_path, monkeypatch):
    source, config, tensors = write_fixture(tmp_path)
    bind_fixture(monkeypatch, source, config, tensors)
    plan = plan_te_dense_export(source, config)
    assert len(plan.tensors) == 902
    assert {x.target_name for x in plan.tensors} == contract.TARGET_INVENTORY.keys()
    assert sum(x.operation == "nvfp4-blocked-to-bf16" for x in plan.tensors) == 350
    assert sum(x.operation == "copy-bf16" for x in plan.tensors) == 551


def test_complete_component_streams_exact_bytes_and_preserves_all_plain_tensors(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    source, config, tensors = write_fixture(tmp_path)
    bind_fixture(monkeypatch, source, config, tensors)
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    plan = plan_te_dense_export(source, config)
    result = execution.execute_te_dense_export(plan, tmp_path / "dense", tool=ToolIdentity("comfy-omni", "0.2.0a1", "1" * 40, "2" * 64))
    independent = oracle()
    with SafeTensorSources([result.output_dir / "model.safetensors"]) as output:
        assert len(output.tensors) == 902
        for action in plan.tensors:
            dtype, shape, raw = tensors[action.source_name]
            module = action.source_name.removesuffix(".weight")
            if dtype == "BF16":
                expected = raw
            elif dtype == "I8":
                expected = independent.int8_values(raw, tensors[module + ".weight_scale"][2])
            else:
                expected = b"".join(independent.nvfp4_row(raw[row * shape[1]:(row + 1) * shape[1]], tensors[module + ".weight_scale"][2], tensors[module + ".weight_scale_2"][2], row_in_band=row) for row in range(shape[0]))
            assert output.read_raw(output.tensors[action.target_name]) == expected
    assert (result.output_dir / "config.json").read_bytes() == config.read_bytes()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_digest
    assert result.manifest_path.is_file()
