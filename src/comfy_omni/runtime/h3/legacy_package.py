"""The complete read-only e9cb011 curve-cache v3 package validator.

Derived from h3-forge package_assembler.py at e9cb011 (Apache-2.0), blob
e64558f1d3bb6e1ee6f714b70e783d9df907f9ce. Only the accepted v3 profile is
ported; v4/v5 producer graphs and package publication are outside this slice.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from comfy_omni.artifacts import fileops
from comfy_omni.runtime.h3.curve_contract import verify_curve_cache
from comfy_omni.runtime.h3.legacy_profiles import VAE_PROFILES
from comfy_omni.runtime.h3.legacy_vae import verify_legacy_vae_export
from comfy_omni.runtime.h3.package_binding import (
    AUDITED_PRODUCER,
    CURVE_CACHE_NAME,
    CURVE_CACHE_SCHEMA,
    CURVE_PROFILE,
    HOST_COMMIT,
    LEGACY_COMPONENTS,
    LEGACY_TASKS,
    CurveCacheBinding,
    LegacyPackageError,
    legacy_quantization,
)

MANIFEST_NAME = "h3-comfy-package.json"
SCHEMA = "h3-comfy-package/v3"
INDEX_COMPONENTS = {
    "audio_vae": ["diffusers", "MiniMaxH3AudioVAE"],
    "processor": ["transformers", "Qwen3VLProcessor"],
    "scheduler": None,
    "text_encoder": ["transformers", "MiniMaxH3Qwen3VLHFEncoder"],
    "tokenizer": ["transformers", "Qwen2TokenizerFast"],
    "transformer": ["diffusers", "MiniMaxH3DiTModel"],
    "video_vae": ["diffusers", "MiniMaxH3VideoVAE"],
}


@dataclass(frozen=True)
class LegacyVerification:
    root: Path
    manifest_sha256: str
    model_index_sha256: str
    files_sha256: str
    file_count: int
    total_bytes: int
    components: tuple[tuple[str, int, int], ...]
    curve_cache: CurveCacheBinding


def _require(condition: bool, message: str, stage: str = "legacy-package") -> None:
    if not condition:
        raise LegacyPackageError(message, evidence={"stage": stage})


def _json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw, _ = fileops.read_file_pinned(path)
    value = fileops.parse_json_strict(raw)
    _require(isinstance(value, dict), "legacy package JSON must be an object")
    return value, raw


def _digest(value: object) -> str:
    return hashlib.sha256(fileops.canonical_json(value)).hexdigest()


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _product(manifest: dict[str, Any]) -> None:
    expected = {
        "schema": SCHEMA,
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
        "runtime_quantization_config": legacy_quantization().to_dict(),
        "converter": {**AUDITED_PRODUCER.to_dict(), "vllm_omni_commit": HOST_COMMIT},
        "package_manifest_sha256_domain": "canonical-json-excluding-package_manifest_sha256-with-trailing-newline",
    }
    _require(
        all(manifest.get(key) == value for key, value in expected.items()),
        "legacy package product/host/producer contract is invalid",
    )
    _require(
        manifest.get("loadable_package") is True and type(manifest.get("resident_dit_count")) is int,
        "legacy product flags are invalid",
    )
    forbidden = {
        "routing",
        "plan_content_sha256",
        "model_index_sha256",
        "tool",
        "host",
        "assembler",
        "component_producers",
        "offline_lora_composition",
        "legacy_primary_attestation",
    }
    _require(not forbidden.intersection(manifest), "legacy package mixes incompatible layout/version fields")


def _index(path: Path, expected_class_name: str) -> bytes:
    index, raw = _json(path)
    _require(
        index.get("_class_name") == expected_class_name == "MiniMaxH3Pipeline",
        "legacy pipeline class differs",
        "model-index",
    )
    _require(
        all(index.get(key) == value for key, value in INDEX_COMPONENTS.items()),
        "legacy index component contract differs",
        "model-index",
    )
    release = index.get("_minimax_h3")
    _require(isinstance(release, dict), "legacy index release is missing", "model-index")
    expected = {
        "partition": "ref2va",
        "schema_version": 1,
        "tasks": list(LEGACY_TASKS),
        "task_aliases": {},
        "sigma_shift_scales": {"audio": 3.0, "video": 12.0},
    }
    _require(
        all(release.get(key) == value for key, value in expected.items()) and "base_schedule" not in release,
        "legacy index task/schedule contract differs",
        "model-index",
    )
    return raw


def _files(root: Path, manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], tuple[tuple[str, int, int], ...]]:
    files = manifest.get("files")
    _require(isinstance(files, list) and bool(files), "legacy file census is missing", "manifest")
    _require(
        type(manifest.get("file_count")) is int and manifest["file_count"] == len(files),
        "legacy file count differs",
        "manifest",
    )
    names: set[str] = set()
    totals = {component: (0, 0) for component in LEGACY_COMPONENTS}
    for record in files:
        _require(
            isinstance(record, dict) and set(record) == {"path", "sha256", "size"},
            "legacy file record is malformed",
            "manifest",
        )
        name = record["path"]
        _require(isinstance(name, str) and bool(name) and "\\" not in name, "legacy file path is invalid", "manifest")
        parts = PurePosixPath(name).parts
        _require(
            not PurePosixPath(name).is_absolute()
            and all(part not in {".", ".."} and ":" not in part for part in parts)
            and PurePosixPath(name).as_posix() == name,
            "legacy file path is not canonical",
            "manifest",
        )
        _require(
            name not in names and _is_digest(record["sha256"]) and type(record["size"]) is int and record["size"] >= 0,
            "legacy file record is duplicate or invalid",
            "manifest",
        )
        names.add(name)
        if name not in {"model_index.json", "Ref2VA/model_index.json"}:
            _require(
                len(parts) >= 3 and parts[0] == "Ref2VA" and parts[1] in totals,
                "legacy file is outside the component tree",
                "components",
            )
            count, size = totals[parts[1]]
            totals[parts[1]] = count + 1, size + record["size"]
    _require(
        {"model_index.json", "Ref2VA/model_index.json"} <= names,
        "legacy package requires both index files",
        "tree-census",
    )
    _require(all(count > 0 for count, _ in totals.values()), "legacy component census is incomplete", "components")
    actual = set()
    for directory, dirs, filenames in os.walk(root, followlinks=False):
        for name in [*dirs, *filenames]:
            path = Path(directory) / name
            _require(not fileops.is_link(path), "legacy tree contains a link", "tree-census")
            _require(path.is_dir() or path.is_file(), "legacy tree contains a special entry", "tree-census")
        actual.update((Path(directory) / name).relative_to(root).as_posix() for name in filenames)
    _require(actual == names | {MANIFEST_NAME}, "legacy tree differs from committed census", "tree-census")
    for record in files:
        _require(
            fileops.sha256_file_pinned(root / record["path"]) == (record["sha256"], record["size"]),
            "legacy artifact SHA256 or size differs",
            "file-verification",
        )
    return files, tuple((component, *totals[component]) for component in LEGACY_COMPONENTS)


def _component_exports(root: Path, manifest: dict[str, Any]) -> None:
    exports = manifest.get("component_exports")
    _require(
        isinstance(exports, dict) and set(exports) == {"hybrid_transformer", "text_encoder", "video_vae", "audio_vae"},
        "legacy component export identities are missing",
    )
    _require(
        _is_digest(exports["hybrid_transformer"]) and _is_digest(exports["text_encoder"]),
        "legacy converted component identity is invalid",
    )
    for component in ("video_vae", "audio_vae"):
        record = exports[component]
        _require(
            isinstance(record, dict) and set(record) == {"profile", "manifest_sha256"},
            "legacy VAE export identity is malformed",
        )
        profile = VAE_PROFILES.get(record["profile"]) if isinstance(record["profile"], str) else None
        _require(
            profile is not None and profile.component == component and _is_digest(record["manifest_sha256"]),
            "legacy VAE export profile is invalid",
        )
        observed = verify_legacy_vae_export(root / "Ref2VA" / component, expected_converter=AUDITED_PRODUCER.to_dict())
        _require(
            observed["profile"] == record["profile"] and observed["manifest_sha256"] == record["manifest_sha256"],
            "legacy VAE export differs from package identity",
        )
    contracts = manifest.get("source_contracts")
    if contracts is not None:
        _require(
            isinstance(contracts, dict) and bool(contracts) and not set(contracts) - {"transformer", "text_encoder"},
            "legacy source-contract census is invalid",
        )
        for identity in contracts.values():
            _require(
                isinstance(identity, dict)
                and set(identity) == {"name", "manifest_sha256"}
                and isinstance(identity["name"], str)
                and bool(identity["name"])
                and _is_digest(identity["manifest_sha256"]),
                "legacy source-contract identity is invalid",
            )


def verify_legacy_package(root: Path, *, expected_class_name: str = "MiniMaxH3Pipeline") -> LegacyVerification:
    """Bind all package bytes and nested VAE/curve contracts without writes."""
    root = fileops.reject_linked_ancestors(root).resolve(strict=True)
    manifest, manifest_raw = _json(root / MANIFEST_NAME)
    _product(manifest)
    digest = _digest({key: value for key, value in manifest.items() if key != "package_manifest_sha256"})
    _require(manifest.get("package_manifest_sha256") == digest, "legacy manifest self-digest does not bind", "manifest")
    files, components = _files(root, manifest)
    index_raw = _index(root / "model_index.json", expected_class_name)
    _index(root / "Ref2VA" / "model_index.json", expected_class_name)
    configs = {
        "audio_vae": "config.json",
        "processor": "preprocessor_config.json",
        "text_encoder": "config.json",
        "tokenizer": "tokenizer_config.json",
        "transformer": "config.json",
        "video_vae": "config.json",
    }
    for component, name in configs.items():
        _json(root / "Ref2VA" / component / name)
    _component_exports(root, manifest)
    folds = manifest.get("offline_lora_fold")
    _require(isinstance(folds, dict) and set(folds) == {"Ref2VA"}, "legacy LoRA fold map is invalid")
    curve = verify_curve_cache(
        root / "Ref2VA" / "transformer" / CURVE_CACHE_NAME,
        folds["Ref2VA"],
        manifest.get("curve_adaln_cache"),
        AUDITED_PRODUCER,
    )
    _require(
        manifest.get("transformer_source_sha256") == [curve.source_curve_sha256],
        "legacy curve and transformer source identities differ",
    )
    config, _ = _json(root / "Ref2VA" / "transformer" / "config.json")
    expected = {
        "curve_adaln_cache_file": CURVE_CACHE_NAME,
        "curve_adaln_cache_required": True,
        "curve_adaln_cache_schema": CURVE_CACHE_SCHEMA,
        "export_profile": CURVE_PROFILE,
        "omitted_curve_adaln_tensor_count": 103,
        "runtime_quantization_required": True,
        "schedule_contract_sha256": curve.schedule.contract_sha256,
        "strict_schedule_cache": True,
    }
    _require(config.get("_h3_forge") == expected, "legacy transformer config does not bind curve cache")
    _require(
        fileops.read_file_pinned(root / MANIFEST_NAME)[0] == manifest_raw, "legacy manifest changed during verification"
    )
    return LegacyVerification(
        root,
        digest,
        hashlib.sha256(index_raw).hexdigest(),
        _digest(files),
        len(files),
        sum(record["size"] for record in files),
        components,
        curve,
    )
