"""Synthetic legacy artifacts; no model weights or private receipts are included."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from comfy_omni.artifacts import fileops
from comfy_omni.runtime.h3 import legacy_package, legacy_vae
from comfy_omni.runtime.h3.package_binding import (
    AUDITED_PRODUCER,
    CURVE_CACHE_NAME,
    CURVE_CACHE_SCHEMA,
    CURVE_PROFILE,
    HOST_COMMIT,
    LEGACY_COMPONENTS,
    LEGACY_TASKS,
    PROVENANCE_SCHEMA,
    legacy_quantization,
)


def digest(value) -> str:
    return hashlib.sha256(fileops.canonical_json(value)).hexdigest()


def write_json(path: Path, value) -> None:
    if path.exists():
        path.chmod(0o644)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(fileops.canonical_json(value))


def resign(path: Path, value, field="manifest_sha256") -> None:
    value.pop(field, None)
    value[field] = digest(value)
    write_json(path, value)


def freeze(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def thaw(root: Path) -> None:
    root.chmod(0o755)
    for path in root.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)


def make_vae(root: Path, component: str, monkeypatch):
    profile = next(value for value in legacy_vae.VAE_PROFILES.values() if value.component == component)
    config = {"_class_name": "SyntheticVAE", "latent_channels": 2}
    payload = b"\x00\x00\x80\x3f"
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}, separators=(",", ":")
    ).encode()
    header += b" " * (-len(header) % 8)
    weights = struct.pack("<Q", len(header)) + header + payload
    root.mkdir(parents=True)
    weight_path = root / profile.template_weight_path
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    weight_path.write_bytes(weights)
    code = b"# synthetic static template\n"
    (root / "template.py").write_bytes(code)
    write_json(root / "config.json", {**config, "latents_mean": [0.0, 0.0], "latents_std": [1.0, 1.0]})
    schema_sha = digest([{"name": "weight", "dtype": "F32", "shape": [1]}])
    tiny = replace(
        profile,
        output_tensor_count=1,
        output_schema_sha256=schema_sha,
        stats_length=2,
        template_config_static_sha256=digest(config),
        template_static_files={"template.py": hashlib.sha256(code).hexdigest()},
        weight_norm_prefixes=frozenset(),
    )
    monkeypatch.setitem(legacy_vae.VAE_PROFILES, profile.profile, tiny)
    numerical = {
        "python_version": "3.13.0",
        "torch_version": "test",
        "platform_system": "Linux",
        "platform_release": "test",
        "machine": "test",
        "processor": "",
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "float32_matmul_precision": "highest",
        "deterministic_algorithms_enabled": True,
    }
    source = {
        "path": "fixture.safetensors",
        "sha256": "a" * 64,
        "size": 4,
        "metadata_namespace": profile.metadata_namespace,
        "tensor_prefix": "",
        "tensor_count": profile.source_tensor_count,
        "schema_sha256": profile.source_schema_sha256,
    }
    identity = digest(
        {
            "source_sha256": source["sha256"],
            "profile": profile.profile,
            "converter": AUDITED_PRODUCER.to_dict(),
            "pinned_vllm_omni_commit": HOST_COMMIT,
            "official_template_revision": legacy_vae.PINNED_TEMPLATE_REVISION,
            "numerical_runtime": numerical,
        }
    )
    files = {
        path.relative_to(root).as_posix(): {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
        }
        for path in root.rglob("*")
        if path.is_file()
    }
    manifest = {
        "schema": "h3-comfy-vae-export/v2",
        "artifact_status": "STRUCTURALLY_VERIFIED_COMPONENT_REQUIRES_HOST_LOAD",
        "loadable_component": True,
        "component": component,
        "profile": profile.profile,
        "converter": AUDITED_PRODUCER.to_dict(),
        "conversion_identity": identity,
        "numerical_runtime": numerical,
        "pinned_vllm_omni_commit": HOST_COMMIT,
        "official_template_revision": legacy_vae.PINNED_TEMPLATE_REVISION,
        "source": source,
        "transform": {
            "stats_tensors_removed": 2,
            "stats_source": "payload-tensors-promoted-to-json-floats",
            "passthrough_tensor_count": 1,
            "weight_norm_module_count": 0,
            "weight_norm_bitwise_recomposition_count": 0,
        },
        "output": {
            "weight_path": profile.template_weight_path,
            "tensor_count": 1,
            "schema_sha256": schema_sha,
            "tensor_payload_catalog_sha256": digest(
                [{"name": "weight", "sha256": hashlib.sha256(payload).hexdigest()}]
            ),
            "files": files,
        },
        "hash_domains": {
            "source.sha256": "entire-source-safetensors-file-bytes",
            "output.files[*].sha256": "entire-published-file-bytes",
            "output.schema_sha256": "canonical-sorted-name-dtype-shape-json",
            "output.tensor_payload_catalog_sha256": "canonical-sorted-output-name-and-payload-sha256-json",
            "manifest_sha256": "canonical-manifest-json-excluding-manifest_sha256-with-trailing-newline",
        },
    }
    resign(root / "h3-comfy-vae-export.json", manifest)
    freeze(root)
    return manifest


@lru_cache(maxsize=8)
def zero_digest(size: int) -> str:
    value = hashlib.sha256()
    chunk = b"\0" * (1024 * 1024)
    while size:
        amount = min(size, len(chunk))
        value.update(chunk[:amount])
        size -= amount
    return value.hexdigest()


def make_curve(path: Path):
    from comfy_omni.runtime.h3.schedule import build_h3_schedule_contract

    schedule = build_h3_schedule_contract(denoise_steps=4)
    values = [value for plan in schedule.plans for value in plan.values]
    offsets = [0]
    for plan in schedule.plans:
        offsets.append(offsets[-1] + len(plan.values))
    slots = len(values)
    fold = {
        "module_count": 208,
        "quantized_module_count": 200,
        "dense_module_count": 8,
        "scale": 1.0,
        "use_adaln_cache": True,
        "profile": "h3/fl2va/dit-standard/lora-turbo-v4-full-ab/native-separate/v1",
        "alpha_policy": "PER_MODULE_RANK_FROM_ALPHA_NONE",
        "float32_matmul_precision": "highest",
        "fold_device": "cpu",
        "normalized_lora_sha256": "2" * 64,
    }
    metadata = {
        "schema": CURVE_CACHE_SCHEMA,
        "format_version": "1",
        "target_dtype": "BF16",
        "block_params_layout": "block-slot-width",
        "schedule_contract": json.dumps(schedule.to_dict()),
        "schedule_contract_sha256": schedule.contract_sha256,
        "normalized_lora_sha256": "2" * 64,
        "lora_scale": "1.0",
        "alpha_policy": "alpha-none-means-rank",
        "effective_lora_multiplier": "1.0",
        "vllm_omni_commit": HOST_COMMIT,
        "converter_commit": AUDITED_PRODUCER.h3_forge_commit,
        "converter_wheel_sha256": AUDITED_PRODUCER.wheel_sha256,
        "converter_provenance_schema": PROVENANCE_SCHEMA,
        "converter_build_context_sha256": AUDITED_PRODUCER.build_context_sha256,
        "converter_installed_payload_sha256": AUDITED_PRODUCER.installed_payload_sha256,
        "isolated_conversion_process_required": "true",
        "source_curve_sha256": "1" * 64,
        "silu_grid_sha256": "3" * 64,
        "curve_payload_sha256": "4" * 64,
        "block_params_sha256": zero_digest(50 * slots * 96768 * 2),
        "final_params_sha256": zero_digest(slots * 10752 * 2),
    }
    shapes = [
        ("plan_offsets", "I64", [len(offsets)], len(offsets) * 8),
        ("plan_timesteps", "F32", [slots], slots * 4),
        ("block_params", "BF16", [50, slots, 96768], 50 * slots * 96768 * 2),
        ("final_params", "BF16", [slots, 10752], slots * 10752 * 2),
    ]
    header = {"__metadata__": metadata}
    cursor = 0
    for name, dtype, shape, size in shapes:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [cursor, cursor + size]}
        cursor += size
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(raw)) + raw)
        stream.write(struct.pack(f"<{len(offsets)}q", *offsets))
        stream.write(struct.pack(f"<{slots}f", *values))
        stream.truncate(8 + len(raw) + cursor)
    sha, size = fileops.sha256_file_pinned(path)
    claim = {
        "schema": CURVE_CACHE_SCHEMA,
        "path": f"Ref2VA/transformer/{CURVE_CACHE_NAME}",
        "sha256": sha,
        "size": size,
        "schedule_contract_sha256": schedule.contract_sha256,
        "plan_count": len(schedule.plans),
        "plan_slot_count": slots,
        **{
            key: metadata[key]
            for key in (
                "source_curve_sha256",
                "normalized_lora_sha256",
                "silu_grid_sha256",
                "curve_payload_sha256",
                "block_params_sha256",
                "final_params_sha256",
            )
        },
    }
    return fold, claim, schedule, 8 + len(raw)


def make_package(root: Path, monkeypatch):
    root.mkdir()
    index = {
        "_class_name": "MiniMaxH3Pipeline",
        **legacy_package.INDEX_COMPONENTS,
        "_minimax_h3": {
            "partition": "ref2va",
            "schema_version": 1,
            "tasks": list(LEGACY_TASKS),
            "task_aliases": {},
            "sigma_shift_scales": {"audio": 3.0, "video": 12.0},
        },
    }
    write_json(root / "model_index.json", index)
    write_json(root / "Ref2VA/model_index.json", index)
    vaes = {
        component: make_vae(root / "Ref2VA" / component, component, monkeypatch)
        for component in ("audio_vae", "video_vae")
    }
    for component, filename in (
        ("processor", "preprocessor_config.json"),
        ("tokenizer", "tokenizer_config.json"),
        ("text_encoder", "config.json"),
    ):
        write_json(root / "Ref2VA" / component / filename, {})
    fold, claim, schedule, _ = make_curve(root / "Ref2VA/transformer" / CURVE_CACHE_NAME)
    write_json(
        root / "Ref2VA/transformer/config.json",
        {
            "_h3_forge": {
                "curve_adaln_cache_file": CURVE_CACHE_NAME,
                "curve_adaln_cache_required": True,
                "curve_adaln_cache_schema": CURVE_CACHE_SCHEMA,
                "export_profile": CURVE_PROFILE,
                "omitted_curve_adaln_tensor_count": 103,
                "runtime_quantization_required": True,
                "schedule_contract_sha256": schedule.contract_sha256,
                "strict_schedule_cache": True,
            }
        },
    )
    manifest = {
        "schema": "h3-comfy-package/v3",
        "loadable_package": True,
        "profile": CURVE_PROFILE,
        "partitions": ["Ref2VA"],
        "serving_entrypoint": "Ref2VA/",
        "routing_profile": "h3-hybrid-ref-primary-single-dit/v1",
        "supported_tasks": list(LEGACY_TASKS),
        "resident_dit_count": 1,
        "components": list(LEGACY_COMPONENTS),
        "source_model_mount": "read-only",
        "source_materialization": "exclusive-copy-from-read-only-mount/v1",
        "converter": {**AUDITED_PRODUCER.to_dict(), "vllm_omni_commit": HOST_COMMIT},
        "runtime_quantization_config": legacy_quantization().to_dict(),
        "component_exports": {
            "hybrid_transformer": "5" * 64,
            "text_encoder": "6" * 64,
            **{
                component: {"profile": value["profile"], "manifest_sha256": value["manifest_sha256"]}
                for component, value in vaes.items()
            },
        },
        "transformer_source_sha256": [claim["source_curve_sha256"]],
        "offline_lora_fold": {"Ref2VA": fold},
        "curve_adaln_cache": claim,
        "package_manifest_sha256_domain": "canonical-json-excluding-package_manifest_sha256-with-trailing-newline",
    }
    refresh_package(root, manifest)
    return manifest


def refresh_package(root: Path, manifest) -> None:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "h3-comfy-package.json":
            sha, size = fileops.sha256_file_pinned(path)
            files.append({"path": path.relative_to(root).as_posix(), "sha256": sha, "size": size})
    manifest["files"] = files
    manifest["file_count"] = len(files)
    resign(root / "h3-comfy-package.json", manifest, "package_manifest_sha256")
