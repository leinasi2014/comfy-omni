"""Read-only verification of the audited legacy VAE exports (Apache-2.0).

Derived from h3-forge e9cb011 vae_export.py verifier and its pure helpers,
blob 531a63b91354a38214db5a07ce72815427e1d6d5. Conversion/publication code and
executing h3_forge wheel discovery are deliberately outside this adapter.
"""

from __future__ import annotations

import hashlib
import math
import re
import stat
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from comfy_omni.artifacts import fileops
from comfy_omni.artifacts.safetensors import read_safetensors_header
from comfy_omni.artifacts.sources import SafeTensorSources
from comfy_omni.runtime.h3.legacy_profiles import (
    PINNED_TEMPLATE_REVISION,
    PINNED_VLLM_OMNI_COMMIT,
    VAE_EXPORT_SCHEMA,
    VAE_MANIFEST_NAME,
    VAE_PROFILES,
)
from comfy_omni.runtime.h3.package_binding import PROVENANCE_SCHEMA, LegacyPackageError


class VaeExportContractError(LegacyPackageError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail, evidence={"stage": "legacy-vae"})


_canonical_json = fileops.canonical_json
_reject_linked_ancestors = fileops.reject_linked_ancestors
_is_link = fileops.is_link


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return fileops.sha256_file_pinned(path)[0]


def _parse_unique_json(raw: str | bytes, *, label: str) -> Any:
    try:
        return fileops.parse_json_strict(raw)
    except fileops.FsopsError as exc:
        raise VaeExportContractError(f"invalid {label} JSON") from exc


def _schema_sha256(records: Iterable[tuple[str, str, Sequence[int]]]) -> str:
    payload = [{"name": name, "dtype": dtype, "shape": list(shape)} for name, dtype, shape in sorted(records)]
    return _sha256_bytes(_canonical_json(payload))


def _require_regular(path: Path, *, label: str) -> Path:
    path = _reject_linked_ancestors(path)
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise VaeExportContractError(f"{label} is missing: {path}") from exc
    if _is_link(path) or not stat.S_ISREG(info.st_mode):
        raise VaeExportContractError(f"{label} must be a regular non-linked file: {path}")
    return path.resolve(strict=True)


def _payload_hashes(sources: SafeTensorSources) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, tensor in sorted(sources.tensors.items()):
        digest = hashlib.sha256()
        for chunk in sources.iter_raw(tensor):
            digest.update(chunk)
        result[name] = digest.hexdigest()
    return result


_NUMERICAL_RUNTIME_KEYS = frozenset(
    {
        "python_version",
        "torch_version",
        "platform_system",
        "platform_release",
        "machine",
        "processor",
        "torch_num_threads",
        "torch_num_interop_threads",
        "float32_matmul_precision",
        "deterministic_algorithms_enabled",
    }
)


def _validate_recorded_numerical_runtime(value: object) -> dict[str, object]:
    """Validate the producer receipt without binding verification to this host."""

    if not isinstance(value, dict) or set(value) != _NUMERICAL_RUNTIME_KEYS:
        raise VaeExportContractError("VAE export numerical runtime receipt is malformed")
    for field in (
        "python_version",
        "torch_version",
        "platform_system",
        "platform_release",
        "machine",
    ):
        if not isinstance(value[field], str) or not value[field]:
            raise VaeExportContractError(f"VAE export numerical runtime field is invalid: {field}")
    if not isinstance(value["processor"], str):
        raise VaeExportContractError("VAE export numerical runtime field is invalid: processor")
    for field in ("torch_num_threads", "torch_num_interop_threads"):
        if type(value[field]) is not int or value[field] <= 0:
            raise VaeExportContractError(f"VAE export numerical runtime field is invalid: {field}")
    if value["float32_matmul_precision"] not in {"highest", "high", "medium"}:
        raise VaeExportContractError("VAE export numerical runtime field is invalid: float32_matmul_precision")
    if type(value["deterministic_algorithms_enabled"]) is not bool:
        raise VaeExportContractError("VAE export numerical runtime field is invalid: deterministic_algorithms_enabled")
    return dict(value)


def _conversion_identity(
    *,
    source_sha256: str,
    profile: str,
    converter: Mapping[str, object],
    numerical_runtime: Mapping[str, object],
) -> str:
    return _sha256_bytes(
        _canonical_json(
            {
                "source_sha256": source_sha256,
                "profile": profile,
                "converter": dict(converter),
                "pinned_vllm_omni_commit": PINNED_VLLM_OMNI_COMMIT,
                "official_template_revision": PINNED_TEMPLATE_REVISION,
                "numerical_runtime": dict(numerical_runtime),
            }
        )
    )


def verify_legacy_vae_export(
    output_dir: Path,
    *,
    expected_converter: Mapping[str, Any],
) -> dict[str, object]:
    """Rehash and structurally verify a committed VAE component export."""

    root = _reject_linked_ancestors(Path(output_dir)).resolve(strict=True)
    if not root.is_dir():
        raise VaeExportContractError("VAE export must be a non-linked directory")
    all_entries = tuple(root.rglob("*"))
    if any(_is_link(path) for path in all_entries):
        raise VaeExportContractError("VAE export contains a linked or reparse entry")
    if any(not path.is_dir() and not path.is_file() for path in all_entries):
        raise VaeExportContractError("VAE export contains a non-regular filesystem entry")
    if root.lstat().st_mode & 0o222 or any(path.lstat().st_mode & 0o222 for path in all_entries):
        raise VaeExportContractError("VAE export tree is not frozen read-only")
    manifest_path = _require_regular(root / VAE_MANIFEST_NAME, label="VAE export manifest")
    manifest = _parse_unique_json(fileops.read_file_pinned(manifest_path)[0], label="VAE export manifest")
    if isinstance(manifest, dict) and manifest.get("schema") == "h3-comfy-vae-export/v1":
        raise VaeExportContractError("legacy VAE export v1 requires its candidate-bound v1 runtime; rebuild as v2")
    if not isinstance(manifest, dict) or manifest.get("schema") != VAE_EXPORT_SCHEMA:
        raise VaeExportContractError("VAE export manifest schema mismatch")
    if set(manifest) != {
        "schema",
        "artifact_status",
        "loadable_component",
        "component",
        "profile",
        "converter",
        "conversion_identity",
        "numerical_runtime",
        "pinned_vllm_omni_commit",
        "official_template_revision",
        "source",
        "transform",
        "output",
        "hash_domains",
        "manifest_sha256",
    }:
        raise VaeExportContractError("VAE export manifest key set differs from v2")
    if (
        manifest.get("artifact_status") != "STRUCTURALLY_VERIFIED_COMPONENT_REQUIRES_HOST_LOAD"
        or manifest.get("loadable_component") is not True
    ):
        raise VaeExportContractError("VAE export artifact-status contract mismatch")
    claimed_manifest_sha256 = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if claimed_manifest_sha256 != _sha256_bytes(_canonical_json(unsigned)):
        raise VaeExportContractError("VAE export manifest hash mismatch")
    profile = manifest.get("profile")
    if not isinstance(profile, str) or profile not in VAE_PROFILES:
        raise VaeExportContractError("VAE export manifest has an unknown profile")
    contract = VAE_PROFILES[profile]
    if manifest.get("component") != contract.component:
        raise VaeExportContractError("VAE export component/profile mismatch")
    if manifest.get("pinned_vllm_omni_commit") != PINNED_VLLM_OMNI_COMMIT:
        raise VaeExportContractError("VAE export vLLM-Omni identity mismatch")
    if manifest.get("official_template_revision") != PINNED_TEMPLATE_REVISION:
        raise VaeExportContractError("VAE export official-template identity mismatch")
    converter = manifest.get("converter")
    if not isinstance(converter, dict) or set(converter) != {
        "schema",
        "h3_forge_commit",
        "wheel_sha256",
        "build_context_sha256",
        "installed_payload_sha256",
    }:
        raise VaeExportContractError("VAE export converter receipt is malformed")
    if converter.get("schema") != PROVENANCE_SCHEMA:
        raise VaeExportContractError("VAE export converter provenance schema mismatch")
    for field, pattern in (
        ("h3_forge_commit", r"[0-9a-f]{40}"),
        ("wheel_sha256", r"[0-9a-f]{64}"),
        ("build_context_sha256", r"[0-9a-f]{64}"),
        ("installed_payload_sha256", r"[0-9a-f]{64}"),
    ):
        if re.fullmatch(pattern, str(converter.get(field))) is None:
            raise VaeExportContractError(f"VAE export converter field is invalid: {field}")
    if converter != dict(expected_converter):
        raise VaeExportContractError("VAE export producer identity differs from the expected producer")
    numerical_runtime = _validate_recorded_numerical_runtime(manifest.get("numerical_runtime"))
    conversion_identity = manifest.get("conversion_identity")
    if re.fullmatch(r"[0-9a-f]{64}", str(conversion_identity)) is None:
        raise VaeExportContractError("VAE export conversion identity is invalid")
    source = manifest.get("source")
    if not isinstance(source, dict) or set(source) != {
        "path",
        "sha256",
        "size",
        "metadata_namespace",
        "tensor_prefix",
        "tensor_count",
        "schema_sha256",
    }:
        raise VaeExportContractError("VAE export source receipt is malformed")
    if (
        not isinstance(source.get("path"), str)
        or re.fullmatch(r"[0-9a-f]{64}", str(source.get("sha256"))) is None
        or type(source.get("size")) is not int
        or source["size"] <= 0
        or source.get("metadata_namespace") != contract.metadata_namespace
        or source.get("tensor_prefix") not in {"", "vae."}
        or source.get("tensor_count") != contract.source_tensor_count
        or source.get("schema_sha256") != contract.source_schema_sha256
    ):
        raise VaeExportContractError("VAE export source contract mismatch")
    if conversion_identity != _conversion_identity(
        source_sha256=str(source["sha256"]),
        profile=contract.profile,
        converter=converter,
        numerical_runtime=numerical_runtime,
    ):
        raise VaeExportContractError("VAE export conversion identity does not match its inputs")
    transform = manifest.get("transform")
    expected_transform = {
        "stats_tensors_removed": 2,
        "stats_source": "payload-tensors-promoted-to-json-floats",
        "passthrough_tensor_count": (contract.output_tensor_count - 2 * len(contract.weight_norm_prefixes)),
        "weight_norm_module_count": len(contract.weight_norm_prefixes),
        "weight_norm_bitwise_recomposition_count": len(contract.weight_norm_prefixes),
    }
    if transform != expected_transform:
        raise VaeExportContractError("VAE export transformation census mismatch")
    expected_hash_domains = {
        "source.sha256": "entire-source-safetensors-file-bytes",
        "output.files[*].sha256": "entire-published-file-bytes",
        "output.schema_sha256": "canonical-sorted-name-dtype-shape-json",
        "output.tensor_payload_catalog_sha256": ("canonical-sorted-output-name-and-payload-sha256-json"),
        "manifest_sha256": ("canonical-manifest-json-excluding-manifest_sha256-with-trailing-newline"),
    }
    if manifest.get("hash_domains") != expected_hash_domains:
        raise VaeExportContractError("VAE export hash-domain contract mismatch")
    output = manifest.get("output")
    if not isinstance(output, dict) or set(output) != {
        "weight_path",
        "tensor_count",
        "schema_sha256",
        "tensor_payload_catalog_sha256",
        "files",
    }:
        raise VaeExportContractError("VAE export output file census is missing")
    if (
        output.get("weight_path") != contract.template_weight_path
        or output.get("tensor_count") != contract.output_tensor_count
        or output.get("schema_sha256") != contract.output_schema_sha256
        or re.fullmatch(r"[0-9a-f]{64}", str(output.get("tensor_payload_catalog_sha256"))) is None
        or not isinstance(output.get("files"), dict)
    ):
        raise VaeExportContractError("VAE export output contract mismatch")
    file_records = output["files"]
    expected_payload_files = {
        "config.json",
        contract.template_weight_path,
        *contract.template_static_files,
    }
    if set(file_records) != expected_payload_files:
        raise VaeExportContractError("VAE export template/output file set mismatch")
    for relative, expected_sha256 in contract.template_static_files.items():
        record = file_records.get(relative)
        if not isinstance(record, dict) or record.get("sha256") != expected_sha256:
            raise VaeExportContractError(f"VAE export template hash drift: {relative}")
    actual_files = {path.relative_to(root).as_posix() for path in all_entries if path.is_file()}
    expected_files = {*file_records, VAE_MANIFEST_NAME}
    if actual_files != expected_files:
        raise VaeExportContractError(
            f"VAE export file census mismatch: missing={sorted(expected_files - actual_files)} "
            f"extra={sorted(actual_files - expected_files)}"
        )
    expected_directories: set[str] = set()
    for relative in expected_files:
        for parent in PurePosixPath(relative).parents:
            if parent.as_posix() != ".":
                expected_directories.add(parent.as_posix())
    actual_directories = {path.relative_to(root).as_posix() for path in all_entries if path.is_dir()}
    if actual_directories != expected_directories:
        raise VaeExportContractError("VAE export directory census mismatch")
    for relative, record in file_records.items():
        if (
            not isinstance(record, dict)
            or set(record) != {"sha256", "size"}
            or re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256"))) is None
            or type(record.get("size")) is not int
            or record["size"] < 0
        ):
            raise VaeExportContractError(f"invalid VAE file record: {relative}")
        path = _require_regular(root / relative, label=f"VAE export file {relative}")
        if path.stat().st_size != record.get("size") or _sha256_file(path) != record.get("sha256"):
            raise VaeExportContractError(f"VAE export file hash/size mismatch: {relative}")

    model_path = root / contract.template_weight_path
    metadata, descriptors = read_safetensors_header(model_path)
    schema_sha256 = _schema_sha256((tensor.name, tensor.dtype, tensor.shape) for tensor in descriptors)
    if metadata or len(descriptors) != contract.output_tensor_count:
        raise VaeExportContractError("VAE export weight header contract mismatch")
    if schema_sha256 != contract.output_schema_sha256 or output.get("schema_sha256") != schema_sha256:
        raise VaeExportContractError("VAE export weight schema digest mismatch")
    with SafeTensorSources([model_path]) as weights:
        hashes = _payload_hashes(weights)
        weights.verify_unchanged()
    payload_catalog = [{"name": name, "sha256": digest} for name, digest in sorted(hashes.items())]
    if output.get("tensor_payload_catalog_sha256") != _sha256_bytes(_canonical_json(payload_catalog)):
        raise VaeExportContractError("VAE export tensor-payload catalog mismatch")
    config = _parse_unique_json(fileops.read_file_pinned(root / "config.json")[0], label="VAE export config")
    if not isinstance(config, dict):
        raise VaeExportContractError("VAE export config must be an object")
    mean = config.get("latents_mean")
    std = config.get("latents_std")
    if not isinstance(mean, list) or not isinstance(std, list):
        raise VaeExportContractError("VAE export config has no latent statistics")
    if len(mean) != contract.stats_length or len(std) != contract.stats_length:
        raise VaeExportContractError("VAE export config latent-statistics length mismatch")
    if not all(type(value) in {int, float} and math.isfinite(float(value)) for value in mean):
        raise VaeExportContractError("VAE export config mean contains a non-finite value")
    if not all(type(value) in {int, float} and math.isfinite(float(value)) and float(value) > 0 for value in std):
        raise VaeExportContractError("VAE export config std is non-finite or non-positive")
    static_config = dict(config)
    static_config.pop("latents_mean")
    static_config.pop("latents_std")
    if _sha256_bytes(_canonical_json(static_config)) != contract.template_config_static_sha256:
        raise VaeExportContractError("VAE export config/auto_map template contract mismatch")
    return manifest
