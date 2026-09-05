"""Host-free binding of the accepted beta4 dense export to its exact 534 slots.

Uses ComfyOmni's accepted E3 artifact/schema contracts. Runtime verification
binds the exported bytes; it does not repeat the offline numerical oracle or
claim that a producer declaration proves a fresh source-checkpoint scan.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from math import prod
from pathlib import Path
from types import MappingProxyType

from comfy_omni.artifacts import fileops
from comfy_omni.artifacts.sources import SafeTensorSources
from comfy_omni.contracts.beta4 import (
    BETA4_SOURCE_BYTES,
    BETA4_SOURCE_INVENTORY,
    BETA4_SOURCE_NAME,
    BETA4_SOURCE_SCHEMA_SHA256,
    BETA4_SOURCE_SHA256,
    BETA4_SOURCE_TEMPLATE,
    BETA4_TARGET_INVENTORY,
    BETA4_TARGET_NAME,
    BETA4_TARGET_PAYLOAD_BYTES,
    BETA4_TARGET_SCHEMA_SHA256,
)
from comfy_omni.contracts.conversion import EXPORT_SCHEMA, PROFILE_BETA4_DENSE_BF16
from comfy_omni.contracts.models import ContractError
from comfy_omni.contracts.templates import template_digest
from comfy_omni.domain.normalization import ToolIdentity

_JSON_NAMES = ("manifest.json", "export.plan.json", "config.patch.json", "model.safetensors.index.json")
_DENSE_POLICY = {"required": False, "method": None, "ignored_layers": [], "checkpoint_int8_serialized": False}
BETA4_RUNTIME_ARCHITECTURE = MappingProxyType(
    {
        "num_layers": 50,
        "token_refiner_num_layers": 2,
        "hidden_size": 5376,
        "num_attention_heads": 56,
        "attention_head_dim": 128,
        "ffn_hidden_size": 14336,
        "latents_dim": 24,
        "audio_latents_dim": 32,
        "patch_size": (1, 2, 2),
        "text_dim": 5120,
        "time_embed_dim": 8,
        "adaln_out_features": 96768,
        "final_adaln_out_features": 10752,
        "rope_inv_freq_len": 16,
        "norm_eps": 1e-5,
        "qk_norm_eps": 1e-5,
        "final_norm_eps": 1e-5,
    }
)


@dataclass(frozen=True)
class Beta4FileBinding:
    name: str
    size: int
    sha256: str
    identity: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class Beta4ComponentBinding:
    component_root: Path
    manifest_sha256: str
    manifest_file_sha256: str
    plan_content_sha256: str
    producer: ToolIdentity
    files: tuple[Beta4FileBinding, ...]
    source_sha256: str = BETA4_SOURCE_SHA256
    source_schema_sha256: str = BETA4_SOURCE_SCHEMA_SHA256
    target_schema_sha256: str = BETA4_TARGET_SCHEMA_SHA256

    @property
    def shard_paths(self):
        return tuple(self.component_root / item.name for item in self.files if item.name.endswith(".safetensors"))

    def to_dict(self):
        return {
            "component_root": self.component_root.as_posix(),
            "manifest_sha256": self.manifest_sha256,
            "manifest_file_sha256": self.manifest_file_sha256,
            "plan_content_sha256": self.plan_content_sha256,
            "source_sha256": self.source_sha256,
            "source_schema_sha256": self.source_schema_sha256,
            "target_schema_sha256": self.target_schema_sha256,
            "producer": self.producer.to_dict(),
            "profile": PROFILE_BETA4_DENSE_BF16,
            "runtime_quantization": {**_DENSE_POLICY, "ignored_layers": []},
            "architecture": dict(BETA4_RUNTIME_ARCHITECTURE),
            "table_shape": [1025, 8],
            "basis_shape": [8, 2688],
            "basis_mean_disposition": "persistent_geometry_state_not_forward_inputs",
            "qkv_layout": _qkv_layout(),
            "files": [{"name": x.name, "size": x.size, "sha256": x.sha256} for x in self.files],
        }


def _require(condition, detail):
    if not condition:
        raise ContractError(detail, evidence={"stage": "beta4-component"})


def _digest(document):
    return hashlib.sha256(fileops.canonical_json(document)).hexdigest()


def _document(path):
    _require(path.stat().st_size <= 4 * 1024**2, "beta4 document exceeds its bounded size")
    payload, identity = fileops.read_file_pinned(path)
    document = fileops.parse_json_strict(payload)
    _require(
        isinstance(document, dict) and fileops.canonical_json(document) == payload, "beta4 document is not canonical"
    )
    return document, Beta4FileBinding(path.name, len(payload), hashlib.sha256(payload).hexdigest(), identity)


def _tree(root):
    entries = tuple(root.iterdir())
    _require(all(not fileops.is_link(x) and x.is_file() for x in entries), "beta4 export contains links or non-files")
    return {x.name for x in entries}


def _qkv_layout():
    groups, width = 56, 128
    runtime = [
        group * 3 * width + section * width + offset
        for section in range(3)
        for group in range(groups)
        for offset in range(width)
    ]
    inverse = [0] * len(runtime)
    for runtime_row, grouped_row in enumerate(runtime):
        inverse[grouped_row] = runtime_row
    return {
        "source_layout": "runtime-qkv",
        "target_layout": "grouped-for-official-loader",
        "num_query_groups": groups,
        "heads_per_group": 1,
        "head_dim": width,
        "row_count": len(runtime),
        "permutation_sha256": _digest(inverse),
    }


def _actions():
    result = []
    quantized = {name.removesuffix(".weight") for name, (dtype, _) in BETA4_SOURCE_INVENTORY.items() if dtype == "I8"}
    for name, (dtype, shape) in sorted(BETA4_SOURCE_INVENTORY.items()):
        prefix = next(
            (
                name.removesuffix(suffix)
                for suffix in (".weight", ".weight_scale", ".comfy_quant")
                if name.endswith(suffix)
            ),
            None,
        )
        qkv = name.endswith(".attn.qkv_proj.weight")
        target, target_dtype, target_bytes = name, dtype, prod(shape) * 2
        group, group_size = None, None
        operation = "copy-raw"
        if prefix in quantized:
            group, group_size = prefix, 256
            if dtype == "I8":
                operation = "inverse-convrot-to-bf16-runtime-qkv-to-grouped" if qkv else "inverse-convrot-to-bf16"
                target_dtype = "BF16"
            else:
                operation = "omit-comfy-quant-marker" if dtype == "U8" else "omit-source-rowwise-scale"
                target, target_dtype, target_bytes = None, None, 0
        elif qkv:
            operation, group = "copy-runtime-qkv-to-grouped", prefix
        result.append(
            {
                "source_name": name,
                "target_name": target,
                "source_dtype": dtype,
                "target_dtype": target_dtype,
                "shape": list(shape),
                "source_bytes": prod(shape) * {"I8": 1, "U8": 1, "F32": 4, "BF16": 2}[dtype],
                "target_bytes": target_bytes,
                "operation": operation,
                "group_prefix": group,
                "group_size": group_size,
            }
        )
    return result


def _check_documents(manifest, plan, patch):
    for field, value in {
        "component": "transformer",
        "output_schema": EXPORT_SCHEMA,
        "profile": PROFILE_BETA4_DENSE_BF16,
    }.items():
        _require(manifest.get(field) == plan.get(field) == value, f"beta4 {field} differs")
    _require(
        manifest.get("schema") == "comfy_omni.native_export.receipt/v1" and manifest.get("status") == "COMMITTED",
        "beta4 export is not committed",
    )
    _require(
        plan.get("schema") == "comfy_omni.native_export.plan/v2" and plan.get("status") == "AUTHORIZED_PLAN",
        "beta4 plan is not authorized",
    )
    plan_digest = plan.get("content_sha256")
    _require(
        _digest({k: v for k, v in plan.items() if k != "content_sha256"})
        == plan_digest
        == manifest.get("plan_content_sha256"),
        "beta4 plan digest differs",
    )
    _require(
        _digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}) == manifest.get("manifest_sha256"),
        "beta4 manifest digest differs",
    )
    _require(
        plan.get("source_contract")
        == {
            "name": BETA4_SOURCE_NAME,
            "origin": "compile-time",
            "schema_sha256": BETA4_SOURCE_SCHEMA_SHA256,
            "snapshot_manifest_sha256": None,
            "snapshot_file_sha256": None,
        },
        "beta4 source contract differs",
    )
    sources = plan.get("source_files")
    _require(
        isinstance(sources, list) and len(sources) == 1 and isinstance(sources[0], dict),
        "beta4 source declaration differs",
    )
    _require(
        set(sources[0]) == {"path", "size", "sha256"}
        and isinstance(sources[0]["path"], str)
        and bool(sources[0]["path"]),
        "beta4 source path declaration differs",
    )
    _require(
        sources[0]["size"] == BETA4_SOURCE_BYTES
        and sources[0]["sha256"] == BETA4_SOURCE_SHA256
        and manifest.get("source_files") == sources,
        "beta4 source identity differs",
    )
    _require(
        plan.get("architecture_template")
        == {
            "name": BETA4_SOURCE_TEMPLATE.template_name,
            "version": 1,
            "sha256": template_digest(BETA4_SOURCE_TEMPLATE),
        },
        "beta4 architecture authority differs",
    )
    target = {
        "tensor_count": len(BETA4_TARGET_INVENTORY),
        "payload_bytes": BETA4_TARGET_PAYLOAD_BYTES,
        "contract": BETA4_TARGET_NAME,
        "schema_sha256": BETA4_TARGET_SCHEMA_SHA256,
    }
    _require(plan.get("target") == manifest.get("target") == target, "beta4 target authority differs")
    _require(
        plan.get("runtime_quantization") == manifest.get("runtime_quantization") == _DENSE_POLICY,
        "beta4 dense execution policy differs",
    )
    for document in (plan, manifest):
        _require(
            all(document["runtime_quantization"][name] is False for name in ("required", "checkpoint_int8_serialized")),
            "beta4 dense policy flags must be boolean false",
        )
    _require(plan.get("qkv_layout") == manifest.get("qkv_layout") == _qkv_layout(), "beta4 QKV layout differs")
    _require(plan.get("actions") == _actions(), "beta4 planned source-to-target actions differ")
    _require(
        patch
        == {
            "_comfy_omni": {
                "output_schema": EXPORT_SCHEMA,
                "plan_content_sha256": plan_digest,
                "profile": PROFILE_BETA4_DENSE_BF16,
            },
            "quantization_config": None,
        },
        "beta4 config patch differs",
    )
    tool = manifest.get("tool")
    _require(
        isinstance(tool, dict) and set(tool) == {"distribution", "version", "source_commit", "wheel_sha256"},
        "beta4 producer identity is incomplete",
    )
    producer = ToolIdentity(**tool)
    _require(producer.distribution == "comfy-omni", "beta4 artifact producer is not ComfyOmni")
    return producer


def verify_beta4_component(component_root: Path | str) -> Beta4ComponentBinding:
    """Hash and strictly scan every export file without importing Torch or a host."""
    root = fileops.reject_linked_ancestors(Path(component_root)).resolve(strict=True)
    before = _tree(root)
    documents, files = {}, {}
    for name in _JSON_NAMES:
        documents[name], files[name] = _document(root / name)
    manifest, plan = documents["manifest.json"], documents["export.plan.json"]
    producer = _check_documents(manifest, plan, documents["config.patch.json"])
    records = manifest.get("files")
    _require(isinstance(records, list) and all(isinstance(x, dict) for x in records), "beta4 manifest census differs")
    _require(all(isinstance(x.get("name"), str) for x in records), "beta4 manifest file name is invalid")
    by_name = {x.get("name"): x for x in records}
    _require(len(by_name) == len(records) and set(by_name) == before - {"manifest.json"}, "beta4 export tree differs")
    shards = plan.get("shards")
    _require(isinstance(shards, list) and 0 < len(shards) <= 128, "beta4 shard count differs")
    names = tuple(f"model-{i:05d}-of-{len(shards):05d}.safetensors" for i in range(1, len(shards) + 1))
    _require(before == set(names) | set(_JSON_NAMES), "beta4 export has unexpected files")
    _require(
        all(isinstance(x, dict) and x.get("name") == name for x, name in zip(shards, names, strict=True)),
        "beta4 shard sequence differs",
    )
    with SafeTensorSources(tuple(root / name for name in names)) as sources:
        observed = {name: (item.descriptor.dtype, item.descriptor.shape) for name, item in sources.tensors.items()}
        _require(observed == dict(BETA4_TARGET_INVENTORY), "beta4 descriptor schema differs")
        weight_map = {}
        for i, (name, shard) in enumerate(zip(names, shards, strict=True)):
            tensor_names = sorted(key for key, item in sources.tensors.items() if item.source_index == i)
            payload_bytes = sum(prod(observed[key][1]) * 2 for key in tensor_names)
            _require(
                shard == {"name": name, "tensor_names": tensor_names, "payload_bytes": payload_bytes},
                "beta4 shard tensor census differs",
            )
            _require(by_name[name].get("tensor_count") == len(tensor_names), "beta4 manifest tensor count differs")
            files[name] = Beta4FileBinding(
                name, sources.sizes[i], sources.hashes[i], fileops.fd_identity((root / name).stat())
            )
            weight_map.update(dict.fromkeys(tensor_names, name))
        _require(
            documents["model.safetensors.index.json"]
            == {"metadata": {"total_size": BETA4_TARGET_PAYLOAD_BYTES}, "weight_map": weight_map},
            "beta4 safetensors index differs",
        )
        for name, record in by_name.items():
            _require(
                re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256"))) is not None, "beta4 file digest is invalid"
            )
            _require(
                (record.get("size"), record["sha256"]) == (files[name].size, files[name].sha256),
                "beta4 export file identity differs",
            )
        sources.verify_unchanged()
    binding = Beta4ComponentBinding(
        root,
        manifest["manifest_sha256"],
        files["manifest.json"].sha256,
        plan["content_sha256"],
        producer,
        tuple(files[name] for name in sorted(files)),
    )
    verify_beta4_binding_unchanged(binding)
    _require(_tree(root) == before, "beta4 export tree changed during validation")
    return binding


def verify_beta4_binding_unchanged(binding: Beta4ComponentBinding) -> None:
    """Reject path/descriptor drift between completed verification and host load."""
    _require(
        _tree(binding.component_root) == {item.name for item in binding.files},
        "beta4 export tree changed after verification",
    )
    for item in binding.files:
        path = fileops.reject_linked_ancestors(binding.component_root / item.name)
        _require(fileops.fd_identity(path.stat()) == item.identity, "beta4 export changed after verification")


def optional_beta4_binding(component_root: Path) -> Beta4ComponentBinding | None:
    """Select only from the receipt; an attempted beta4 policy cannot fall through."""
    attempted = False
    for name in ("manifest.json", "export.plan.json", "config.patch.json"):
        path = component_root / name
        if path.is_file() and path.stat().st_size <= 4 * 1024**2:
            payload, _ = fileops.read_file_pinned(path)
            try:
                document = fileops.parse_json_strict(payload)
            except fileops.FsopsError:
                if PROFILE_BETA4_DENSE_BF16.encode() in payload or BETA4_TARGET_NAME.encode() in payload:
                    _require(False, "malformed attempted beta4 component declaration")
                continue
            if not isinstance(document, dict):
                continue
            marker = document.get("_comfy_omni", {})
            target = document.get("target", {})
            attempted |= (
                document.get("profile") == PROFILE_BETA4_DENSE_BF16
                or (isinstance(marker, dict) and marker.get("profile") == PROFILE_BETA4_DENSE_BF16)
                or (isinstance(target, dict) and target.get("contract") == BETA4_TARGET_NAME)
            )
    return verify_beta4_component(component_root) if attempted else None
