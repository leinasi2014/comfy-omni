"""Metadata-only E3 policy and synthetic tiny-file runtime binding coverage.

The full 934-to-534 plan uses the accepted producer and an independent
header fixture, never model payload. File tests monkeypatch only this new
binding module's compiled inventories/pins inside each test. Those tiny
values are not exposed by any production configuration or public API.
"""

from __future__ import annotations

import copy
import hashlib
import json
import struct
import subprocess
import sys
from functools import lru_cache
from math import prod
from types import SimpleNamespace

import pytest
from test_beta4_dense_conversion import beta4_report

from comfy_omni.artifacts import fileops
from comfy_omni.contracts.models import ContractError
from comfy_omni.conversion.exporters.beta4 import build_beta4_dense_plan
from comfy_omni.runtime.h3 import beta4_binding as binding

TOOL = {"distribution": "comfy-omni", "version": "0.2.0a1", "source_commit": "a" * 40, "wheel_sha256": "b" * 64}


def _canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


@lru_cache(maxsize=1)
def _accepted_plan():
    return build_beta4_dense_plan(beta4_report()).to_dict()


def _documents():
    plan = copy.deepcopy(_accepted_plan())
    manifest = {
        "schema": "comfy_omni.native_export.receipt/v1",
        "status": "COMMITTED",
        "component": "transformer",
        "output_schema": plan["output_schema"],
        "profile": plan["profile"],
        "source_files": copy.deepcopy(plan["source_files"]),
        "target": copy.deepcopy(plan["target"]),
        "qkv_layout": copy.deepcopy(plan["qkv_layout"]),
        "runtime_quantization": copy.deepcopy(plan["runtime_quantization"]),
        "tool": dict(TOOL),
        "files": [],
    }
    patch = {
        "_comfy_omni": {"output_schema": plan["output_schema"], "profile": plan["profile"]},
        "quantization_config": None,
    }
    _seal(manifest, plan, patch)
    return manifest, plan, patch


def _seal(manifest, plan, patch):
    plan.pop("content_sha256", None)
    plan["content_sha256"] = _digest(plan)
    manifest["plan_content_sha256"] = plan["content_sha256"]
    patch["_comfy_omni"]["plan_content_sha256"] = plan["content_sha256"]
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = _digest(manifest)


def test_real_e3_metadata_plan_satisfies_runtime_policy_without_model_io():
    manifest, plan, patch = _documents()
    assert len(plan["actions"]) == 934
    assert plan["target"]["tensor_count"] == 534
    assert binding._check_documents(manifest, plan, patch).to_dict() == TOOL


@pytest.mark.parametrize(
    "case",
    [
        "source_schema",
        "source_name",
        "source_size",
        "source_sha",
        "source_path_empty",
        "manifest_source",
        "target_schema",
        "target_name",
        "target_payload",
        "target_count",
        "template",
        "qkv_digest",
        "qkv_layout",
        "action_operation",
        "action_missing",
        "action_extra",
        "quant_required",
        "quant_method",
        "quant_ignored",
        "quant_serialized",
        "config_quant",
        "producer_distribution",
        "producer_commit",
        "producer_wheel",
        "producer_version",
        "producer_missing",
        "producer_extra",
        "uncommitted",
        "unauthorized",
        "wrong_profile",
    ],
)
def test_self_consistent_rehash_does_not_override_fixed_policy(case):
    manifest, plan, patch = _documents()
    if case == "source_schema":
        plan["source_contract"]["schema_sha256"] = "0" * 64
    elif case == "source_name":
        plan["source_contract"]["name"] = "other-source"
    elif case in {"source_size", "source_sha", "source_path_empty"}:
        key, value = {
            "source_size": ("size", 1),
            "source_sha": ("sha256", "0" * 64),
            "source_path_empty": ("path", ""),
        }[case]
        plan["source_files"][0][key] = value
        manifest["source_files"] = copy.deepcopy(plan["source_files"])
    elif case == "manifest_source":
        manifest["source_files"][0]["path"] = "/different/declared-source"
    elif case.startswith("target_"):
        key, value = {
            "target_schema": ("schema_sha256", "0" * 64),
            "target_name": ("contract", "other-target"),
            "target_payload": ("payload_bytes", 1),
            "target_count": ("tensor_count", 1),
        }[case]
        plan["target"][key] = value
        manifest["target"] = copy.deepcopy(plan["target"])
    elif case == "template":
        plan["architecture_template"]["sha256"] = "0" * 64
    elif case.startswith("qkv_"):
        plan["qkv_layout"]["permutation_sha256" if case == "qkv_digest" else "target_layout"] = "0" * 64
        manifest["qkv_layout"] = copy.deepcopy(plan["qkv_layout"])
    elif case == "action_operation":
        next(a for a in plan["actions"] if a["source_dtype"] == "I8")["operation"] = "copy-raw"
    elif case == "action_missing":
        plan["actions"].pop()
    elif case == "action_extra":
        plan["actions"].append(copy.deepcopy(plan["actions"][0]))
    elif case.startswith("quant_"):
        key, value = {
            "quant_required": ("required", True),
            "quant_method": ("method", "int8"),
            "quant_ignored": ("ignored_layers", ["anything"]),
            "quant_serialized": ("checkpoint_int8_serialized", True),
        }[case]
        plan["runtime_quantization"][key] = value
        manifest["runtime_quantization"] = copy.deepcopy(plan["runtime_quantization"])
    elif case == "config_quant":
        patch["quantization_config"] = {"quant_method": "int8"}
    elif case.startswith("producer_"):
        if case == "producer_missing":
            manifest["tool"].pop("wheel_sha256")
        elif case == "producer_extra":
            manifest["tool"]["unbound"] = True
        else:
            key, value = {
                "producer_distribution": ("distribution", "other"),
                "producer_commit": ("source_commit", "abc"),
                "producer_wheel": ("wheel_sha256", "abc"),
                "producer_version": ("version", ""),
            }[case]
            manifest["tool"][key] = value
    elif case == "uncommitted":
        manifest["status"] = "STAGED"
    elif case == "unauthorized":
        plan["status"] = "DRAFT"
    else:
        plan["profile"] = manifest["profile"] = "dense-bf16-online-int8"
    _seal(manifest, plan, patch)
    with pytest.raises(ValueError):
        binding._check_documents(manifest, plan, patch)


@pytest.mark.parametrize("document,key", [("plan", "content_sha256"), ("manifest", "manifest_sha256")])
def test_digest_mismatch_is_rejected(document, key):
    manifest, plan, patch = _documents()
    {"manifest": manifest, "plan": plan}[document][key] = "0" * 64
    with pytest.raises(ContractError, match="digest"):
        binding._check_documents(manifest, plan, patch)


@pytest.mark.parametrize("field", ["required", "checkpoint_int8_serialized"])
def test_integer_zero_cannot_impersonate_boolean_dense_policy(field):
    manifest, plan, patch = _documents()
    plan["runtime_quantization"][field] = 0
    manifest["runtime_quantization"][field] = 0
    _seal(manifest, plan, patch)
    with pytest.raises(ContractError):
        binding._check_documents(manifest, plan, patch)


def _write_shard(path, tensors):
    cursor, header = 0, {}
    for name, (dtype, shape) in sorted(tensors.items()):
        length = prod(shape) * {"BF16": 2, "F32": 4}[dtype]
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [cursor, cursor + length]}
        cursor += length
    encoded = _canonical(header)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(cursor))


def _tiny_export(tmp_path, monkeypatch):
    inventory = {"adaln_basis": ("BF16", (2, 4)), "adaln_mean": ("BF16", (4,)), "adaln_t_table": ("BF16", (3, 2))}
    schema = _digest(
        [{"name": name, "dtype": dtype, "shape": list(shape)} for name, (dtype, shape) in sorted(inventory.items())]
    )
    payload = sum(prod(shape) * 2 for _, shape in inventory.values())
    for name, value in {
        "BETA4_SOURCE_INVENTORY": inventory,
        "BETA4_TARGET_INVENTORY": inventory,
        "BETA4_SOURCE_SCHEMA_SHA256": schema,
        "BETA4_TARGET_SCHEMA_SHA256": schema,
        "BETA4_TARGET_PAYLOAD_BYTES": payload,
    }.items():
        monkeypatch.setattr(binding, name, value)
    manifest, plan, patch = _documents()
    plan["source_contract"]["schema_sha256"] = schema
    plan["target"].update(tensor_count=3, payload_bytes=payload, schema_sha256=schema)
    manifest["target"] = copy.deepcopy(plan["target"])
    plan["actions"] = [
        {
            "source_name": name,
            "target_name": name,
            "source_dtype": "BF16",
            "target_dtype": "BF16",
            "shape": list(shape),
            "source_bytes": prod(shape) * 2,
            "target_bytes": prod(shape) * 2,
            "operation": "copy-raw",
            "group_prefix": None,
            "group_size": None,
        }
        for name, (_, shape) in sorted(inventory.items())
    ]
    root = tmp_path / "component"
    root.mkdir()
    names = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    tensors = [
        {"adaln_basis": inventory["adaln_basis"]},
        {name: inventory[name] for name in ("adaln_mean", "adaln_t_table")},
    ]
    plan["shards"] = []
    weight_map = {}
    for name, items in zip(names, tensors, strict=True):
        _write_shard(root / name, items)
        plan["shards"].append(
            {
                "name": name,
                "tensor_names": sorted(items),
                "payload_bytes": sum(prod(shape) * 2 for _, shape in items.values()),
            }
        )
        weight_map.update(dict.fromkeys(items, name))
    index = {"metadata": {"total_size": payload}, "weight_map": weight_map}
    fixture = SimpleNamespace(
        root=root, manifest=manifest, plan=plan, patch=patch, index=index, names=names, tensors=tensors
    )
    _publish(fixture)
    return fixture


def _publish(fixture, *, alter_manifest=None):
    manifest, plan, patch, root = fixture.manifest, fixture.plan, fixture.patch, fixture.root
    _seal(manifest, plan, patch)
    for name, document in (
        ("export.plan.json", plan),
        ("config.patch.json", patch),
        ("model.safetensors.index.json", fixture.index),
    ):
        (root / name).write_bytes(_canonical(document))
    manifest["files"] = []
    for path in sorted(root.iterdir()):
        if path.name == "manifest.json":
            continue
        raw = path.read_bytes()
        record = {"name": path.name, "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        if path.name in fixture.names:
            record["tensor_count"] = len(fixture.tensors[fixture.names.index(path.name)])
        manifest["files"].append(record)
    if alter_manifest:
        alter_manifest(manifest)
    _seal(manifest, plan, patch)
    (root / "manifest.json").write_bytes(_canonical(manifest))


def test_tiny_files_bind_exact_shards_and_preserve_inputs(tmp_path, monkeypatch):
    fixture = _tiny_export(tmp_path, monkeypatch)
    before = {path.name: path.read_bytes() for path in fixture.root.iterdir()}
    result = binding.verify_beta4_component(fixture.root)
    assert result.producer.to_dict() == TOOL
    assert tuple(path.name for path in result.shard_paths) == fixture.names
    assert {item.name for item in result.files} == set(before)
    assert result.manifest_file_sha256 == hashlib.sha256(before["manifest.json"]).hexdigest()
    binding.verify_beta4_binding_unchanged(result)
    assert {path.name: path.read_bytes() for path in fixture.root.iterdir()} == before


@pytest.mark.parametrize(
    "case",
    [
        "missing_tensor",
        "extra_tensor",
        "duplicate_across_shards",
        "wrong_shape",
        "wrong_dtype",
        "index_target",
        "index_missing",
        "index_extra",
        "index_size",
        "shard_hash",
        "source_identity",
        "duplicate_file",
        "extra_file",
        "shard_escape",
        "shard_order",
        "json_duplicate",
    ],
)
def test_tiny_export_rejects_invalid_closed_file_binding(tmp_path, monkeypatch, case):
    fixture = _tiny_export(tmp_path, monkeypatch)
    if case in {"missing_tensor", "extra_tensor", "duplicate_across_shards", "wrong_shape", "wrong_dtype"}:
        if case == "missing_tensor":
            fixture.tensors[1].pop("adaln_mean")
        elif case == "extra_tensor":
            fixture.tensors[1]["unexpected"] = ("BF16", (1,))
        elif case == "duplicate_across_shards":
            fixture.tensors[1]["adaln_basis"] = ("BF16", (2, 4))
        elif case == "wrong_shape":
            fixture.tensors[1]["adaln_mean"] = ("BF16", (2, 2))
        else:
            fixture.tensors[1]["adaln_mean"] = ("F32", (4,))
        _write_shard(fixture.root / fixture.names[1], fixture.tensors[1])
    elif case.startswith("index_"):
        if case == "index_target":
            fixture.index["weight_map"]["adaln_mean"] = fixture.names[0]
        elif case == "index_missing":
            fixture.index["weight_map"].pop("adaln_mean")
        elif case == "index_extra":
            fixture.index["weight_map"]["invented"] = fixture.names[0]
        else:
            fixture.index["metadata"]["total_size"] += 1
    elif case == "source_identity":
        fixture.plan["source_files"][0]["sha256"] = "0" * 64
        fixture.manifest["source_files"] = copy.deepcopy(fixture.plan["source_files"])
    elif case == "shard_escape":
        fixture.plan["shards"][0]["name"] = "../outside.safetensors"
    elif case == "shard_order":
        fixture.plan["shards"].reverse()
    _publish(
        fixture,
        alter_manifest=(lambda doc: doc["files"].append(copy.deepcopy(doc["files"][0])))
        if case == "duplicate_file"
        else None,
    )
    if case == "shard_hash":
        path = fixture.root / fixture.names[0]
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
    elif case == "extra_file":
        (fixture.root / "undeclared").write_bytes(b"extra")
    elif case == "json_duplicate":
        path = fixture.root / "config.patch.json"
        path.write_bytes(
            path.read_bytes().replace(
                b'"quantization_config":null', b'"quantization_config":null,"quantization_config":null'
            )
        )
    with pytest.raises((ValueError, fileops.FsopsError)):
        binding.verify_beta4_component(fixture.root)


@pytest.mark.parametrize("case", ["leaf_link", "ancestor_link", "directory"])
def test_export_links_and_directories_are_refused(tmp_path, monkeypatch, case):
    fixture = _tiny_export(tmp_path, monkeypatch)
    root = fixture.root
    if case == "leaf_link":
        path = root / fixture.names[0]
        saved = tmp_path / "saved.safetensors"
        path.replace(saved)
        path.symlink_to(saved)
    elif case == "ancestor_link":
        root = tmp_path / "alias"
        root.symlink_to(fixture.root, target_is_directory=True)
    else:
        (root / "extra-directory").mkdir()
    with pytest.raises((ValueError, fileops.FsopsError)):
        binding.verify_beta4_component(root)


@pytest.mark.parametrize("case", ["rewrite_shard", "replace_document", "extra_at_exit"])
def test_mutation_at_final_source_recheck_is_refused(tmp_path, monkeypatch, case):
    fixture = _tiny_export(tmp_path, monkeypatch)
    original = binding.SafeTensorSources.verify_unchanged

    def mutate(sources):
        original(sources)
        if case == "rewrite_shard":
            path = fixture.root / fixture.names[0]
            raw = bytearray(path.read_bytes())
            raw[-1] ^= 1
            path.write_bytes(raw)
        elif case == "replace_document":
            path = fixture.root / "manifest.json"
            replacement = tmp_path / "replacement"
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
        else:
            (fixture.root / "late-extra").write_bytes(b"x")

    monkeypatch.setattr(binding.SafeTensorSources, "verify_unchanged", mutate)
    with pytest.raises(ContractError):
        binding.verify_beta4_component(fixture.root)


@pytest.mark.parametrize("case", ["replace_shard", "extra_file"])
def test_binding_recheck_before_load_keeps_the_closed_tree(tmp_path, monkeypatch, case):
    fixture = _tiny_export(tmp_path, monkeypatch)
    verified = binding.verify_beta4_component(fixture.root)
    if case == "replace_shard":
        path = fixture.root / fixture.names[0]
        replacement = tmp_path / "replacement"
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
    else:
        (fixture.root / "late-extra").write_bytes(b"x")
    with pytest.raises(ContractError):
        binding.verify_beta4_binding_unchanged(verified)


def test_attempted_beta4_does_not_fall_back_on_bad_contract(tmp_path, monkeypatch):
    fixture = _tiny_export(tmp_path, monkeypatch)
    fixture.plan["target"]["schema_sha256"] = "0" * 64
    fixture.manifest["target"] = copy.deepcopy(fixture.plan["target"])
    _publish(fixture)
    with pytest.raises(ContractError):
        binding.optional_beta4_binding(fixture.root)
    empty = tmp_path / "ordinary"
    empty.mkdir()
    assert binding.optional_beta4_binding(empty) is None


def test_binding_module_imports_without_torch_or_host():
    script = (
        "import sys; import comfy_omni.runtime.h3.beta4_binding; "
        "assert not any(name in ('torch', 'vllm', 'vllm_omni') or name.startswith(('torch.', 'vllm.', 'vllm_omni.')) "
        "for name in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
