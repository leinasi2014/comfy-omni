# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright h3-forge contributors
#
# Provenance: wholesale migration from h3-forge@e9cb011d00b028c149db3978de246c54f6e34acc
#   source path: src/h3_forge/lora_hotswap/comfy_oracle.py
#   source blob: edb52dc3d30b1b1dbe2c393f7aa5dd439ef33cde
#   license: Apache-2.0
#   attribution: h3-forge contributors
# Migrated byte-preserving except this provenance header, import retargeting, and
# mechanical line wrapping to satisfy the repository line-length (120).
"""Fail-closed pinned-Comfy reference-fold micro oracle.

The local runner in this module proves only five-tensor reference math and the
candidate's grouped/runtime layout round trip.  It deliberately has no switch
that can claim a real vLLM TP4 loader run or full 259-operation coverage.
Consequently a locally produced receipt is always ``INCOMPLETE`` unless a
tensor mismatch makes it ``REFERENCE_FOLD_PARITY_MISMATCH``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import stat
import struct
import tempfile
import time
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO

from comfy_omni.artifacts.safetensors import read_safetensors_header_stream
from comfy_omni.domain.qkv import grouped_to_qkv_row_indices, qkv_to_grouped_row_indices

from .bake_audit import _SafeTensorReader
from .bake_plan import (
    BAKE_PLAN_SCHEMA,
    COMFY_BAKED_NATIVE_FOLD_MODULE_COUNT,
    COMFY_BAKED_NATIVE_USE_ADALN_CACHE,
    OFFICIAL_FL2VA_BASE_CONTRACT,
    OFFICIAL_FL2VA_BF16_BAKE_PROFILE,
    BakeOperation,
    LoraBakePlan,
    _catalog_sha256,
    _validate_config,
    _validate_normalized_lora,
    _validate_official_base_contract,
    load_lora_bake_plan_json,
)

ORACLE_SCHEMA = "h3-comfy.comfy-reference-fold-oracle/v1"
PRODUCT_ORACLE = "pinned-comfy-default-eager-reference-fold"
REFERENCE_COMMIT = "099aa38c122cea030ce45a51eb1d83208b16a363"
REFERENCE_FILES = (
    ("models/fold.py", "d17b937533f0f9b69316b4f40ab745131f2504f20251ab7b244120f819f172e0"),
    ("models/lora.py", "b84b050d2bfeb7db3b01b5431cfc7fe03d58c2575918dacdcd2630cff893e41b"),
    ("models/adaln.py", "61dc641b2e07f3b06f44e9d96ba817a8620574e3889ab56606aeeaa4948325c5"),
    ("models/model.py", "d5e14e4e5487db2db7ed063ae54172c22e8b686e1877ecd5a01f8c6c655b4c6e"),
    ("utils/lifecycle.py", "a660066042e21ba26d5a39935725bb4872305b2f2456bb2a180c1226c9c08eea"),
    ("utils/blockswap.py", "2e8897f62987af19b38309b9222f95b7477552627d6572973c40c54db8510684"),
    ("nodes/sampler.py", "735632fc650482f77ccd1b5e94d4e68a548a577a907e2dd71f3d77111fe585e7"),
)
MICRO_INDICES = (1, 2, 3, 4, 252)
MICRO_TARGETS = (
    (1, "blocks.0.attn.out_proj", (5376, 7168)),
    (2, "blocks.0.attn.qkv_proj", (21504, 5376)),
    (3, "blocks.0.mlp.fc1", (28672, 5376)),
    (4, "blocks.0.mlp.fc2", (5376, 14336)),
    (252, "token_refiner.blocks.0.attn.qkv_proj", (21504, 5376)),
)
MICRO_PAYLOAD_BYTES = 1_001_914_368
MICRO_CANDIDATE_FILE_BYTES = 1_001_915_512
MICRO_ESTIMATED_PEAK_WORKING_SET_BYTES = 2_338_488_320
HARD_OUTPUT_LIMIT_BYTES = 1_280 * 1024 * 1024
RECEIPT_RESERVATION_BYTES = 1024 * 1024
DEFAULT_MAX_WORKING_SET_BYTES = 3 * 1024 * 1024 * 1024
# This is source behavior, not a tunable memory hint: pinned fold.py defaults
# to 8192, and a different GEMM M dimension could choose a different kernel.
DEFAULT_CHUNK_ROWS = 8192
RECEIPT_NAME = "comfy-reference-fold-oracle.json"
RETAINED_CLEANUP_STATUS = "RETAINED_CONDITIONAL_INODE_UNLINK_UNAVAILABLE"
EVIDENCE_ASSURANCE = {
    "receipt_self_hash": "UNKEYED_SHA256_UNATTESTED",
    "receipt_authentication": "NONE",
    "validator_artifact_readback": "NOT_PERFORMED",
    "artifact_hash_authority": "RUNNER_RECORDED_UNATTESTED",
    "same_privilege_concurrent_writer_precondition": "MUST_BE_ABSENT",
    "conditional_inode_unlink": "UNAVAILABLE",
    "publication_staging_cleanup": RETAINED_CLEANUP_STATUS,
    "promotion_capable": False,
}
_EVIDENCE_ASSURANCE_CANONICAL_JSON = json.dumps(
    EVIDENCE_ASSURANCE, sort_keys=True, separators=(",", ":"), allow_nan=False
)
HASH_DOMAINS = {
    "plan_sha256": {
        "algorithm": "SHA-256",
        "scope": "lora-bake-plan-semantic-identity/v1",
        "serialization": "canonical-json:utf8:sorted-keys:compact-separators",
    },
    "plan_file_sha256": {
        "algorithm": "SHA-256",
        "scope": "plan-json-file/v1",
        "serialization": "entire-file-bytes",
    },
    "base_catalog_sha256": {
        "algorithm": "SHA-256",
        "scope": "official-fl2va-base-catalog/v1",
        "serialization": "json-array:[tensor-name,shard,dtype,shape]:sorted-by-name:utf8:compact",
    },
    "base_shards[*].sha256": {
        "algorithm": "SHA-256",
        "scope": "official-fl2va-base-shard-file/v1",
        "serialization": "entire-safetensors-file-bytes",
    },
    "normalized_lora_sha256": {
        "algorithm": "SHA-256",
        "scope": "normalized-lora-file/v1",
        "serialization": "entire-safetensors-file-bytes",
    },
    "reference.files[*].sha256": {
        "algorithm": "SHA-256",
        "scope": "pinned-comfy-reference-source-file/v1",
        "serialization": "entire-file-bytes",
    },
    "reference.executor.sha256": {
        "algorithm": "SHA-256",
        "scope": "pinned-comfy-fold-executor-source-file/v1",
        "serialization": "entire-models/fold.py-file-bytes",
    },
    "runtime.input_contract.config_sha256": {
        "algorithm": "SHA-256",
        "scope": "official-fl2va-config-file/v1",
        "serialization": "entire-config.json-file-bytes",
    },
    "runtime.input_contract.index_sha256": {
        "algorithm": "SHA-256",
        "scope": "official-fl2va-index-file/v1",
        "serialization": "entire-model.safetensors.index.json-file-bytes",
    },
    "runtime.input_contract.base_catalog_sha256": {
        "algorithm": "SHA-256",
        "scope": "official-fl2va-base-catalog/v1",
        "serialization": "json-array:[tensor-name,shard,dtype,shape]:sorted-by-name:utf8:compact",
    },
    "micro_targets[*].reference_runtime_sha256": {
        "algorithm": "SHA-256",
        "scope": "pinned-comfy-reference-runtime-tensor/v1",
        "serialization": "raw-contiguous-bf16-runtime-layout-payload-bytes",
    },
    "micro_targets[*].candidate_safetensors_file_sha256": {
        "algorithm": "SHA-256",
        "scope": "candidate-single-tensor-safetensors-file/v1",
        "serialization": "entire-safetensors-file-bytes",
    },
    "micro_targets[*].candidate_runtime_sha256": {
        "algorithm": "SHA-256",
        "scope": "candidate-reloaded-runtime-tensor/v1",
        "serialization": "raw-contiguous-bf16-runtime-layout-payload-bytes",
    },
    "receipt_sha256": {
        "algorithm": "SHA-256",
        "scope": "comfy-reference-fold-oracle-receipt/v1",
        "serialization": ("canonical-json-excluding-receipt_sha256:utf8:sorted-keys:compact-separators:nan-forbidden"),
    },
}
_HASH_DOMAINS_CANONICAL_JSON = json.dumps(HASH_DOMAINS, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _expected_hash_domains() -> dict[str, dict[str, str]]:
    """Return the immutable-at-import SHA domain authority as a fresh object."""

    return json.loads(_HASH_DOMAINS_CANONICAL_JSON)


def _expected_evidence_assurance() -> dict[str, Any]:
    return json.loads(_EVIDENCE_ASSURANCE_CANONICAL_JSON)


class OracleContractError(ValueError):
    """Pinned source, profile, hash, shape, or layout contract failed."""


class OracleIsolationError(ValueError):
    """Output isolation or the hard resource limit failed."""


class OracleRuntimeError(RuntimeError):
    """Local math, serialization, or loader execution failed internally."""


@dataclass(frozen=True)
class MicroTargetResult:
    plan_operation_index: int
    module: str
    shape: tuple[int, int]
    layout: str
    payload_bytes: int
    reference_runtime_sha256: str
    candidate_safetensors_file_sha256: str
    candidate_safetensors_file_bytes: int
    candidate_runtime_sha256: str
    runtime_tensor_equal: bool
    tp4_rank_status: str


@dataclass(frozen=True)
class ComfyReferenceFoldReceipt:
    schema: str
    decision: str
    profile: str
    product_oracle: str
    use_adaln_cache: bool
    plan_sha256: str
    plan_file_sha256: str
    operation_count: int
    selected_micro_indices: tuple[int, ...]
    base_catalog_sha256: str
    base_shards: tuple[dict[str, Any], ...]
    normalized_lora_sha256: str
    reference: dict[str, Any]
    candidate: dict[str, Any]
    runtime: dict[str, Any]
    gpu_workers: tuple[dict[str, Any], ...]
    micro_targets: tuple[MicroTargetResult, ...]
    full_coverage: dict[str, Any]
    resource_usage: dict[str, Any]
    evidence_assurance: dict[str, Any]
    hash_domains: dict[str, dict[str, str]]
    receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateArtifact:
    path: str
    file_sha256: str
    size: int
    identity: tuple[int, int, int, int]
    directory_fsync: str
    staging_cleanup: str


@dataclass
class _DirectoryBinding:
    path: Path
    resolved: Path
    identity: tuple[int, int]
    descriptor: int | None
    mode: str
    error_type: type[ValueError]

    def verify(self, *, label: str) -> None:
        _reject_linked_path_components(self.path, label=label, require_exists=True, error_type=self.error_type)
        if self.path.resolve(strict=True) != self.resolved:
            raise self.error_type(f"{label} directory resolution changed")
        current = _directory_identity(self.path.stat())
        if current != self.identity:
            raise self.error_type(f"{label} directory identity changed")
        if self.descriptor is not None and _directory_identity(os.fstat(self.descriptor)) != self.identity:
            raise self.error_type(f"{label} directory handle identity changed")

    def close(self) -> None:
        if self.descriptor is not None:
            os.close(self.descriptor)
            self.descriptor = None


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _sha256_field_paths(payload: Any) -> tuple[str, ...]:
    paths: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else key
                if child_path == "hash_domains":
                    continue
                if key == "sha256" or key.endswith("_sha256"):
                    paths.append(child_path)
                visit(child, child_path)
        elif isinstance(value, list) or isinstance(value, tuple):
            item_path = f"{path}[*]"
            for child in value:
                visit(child, item_path)

    visit(payload, "")
    return tuple(paths)


def _validate_hash_domain_coverage(payload: dict[str, Any]) -> None:
    actual = set(_sha256_field_paths(payload))
    domains = _expected_hash_domains()
    expected = set(domains)
    if actual != expected:
        missing = sorted(actual - expected)
        unused = sorted(expected - actual)
        raise OracleContractError(f"receipt SHA256 domain coverage mismatch; missing={missing!r}; unused={unused!r}")
    for path, definition in domains.items():
        if (
            not isinstance(definition, dict)
            or set(definition) != {"algorithm", "scope", "serialization"}
            or definition.get("algorithm") != "SHA-256"
            or not isinstance(definition.get("scope"), str)
            or not definition["scope"]
            or not isinstance(definition.get("serialization"), str)
            or not definition["serialization"]
        ):
            raise OracleContractError(f"invalid SHA256 domain definition for {path!r}")


def _is_reparse_point(path: Path) -> bool:
    """Return true for symlinks, junctions, and all Windows reparse points."""

    if path.is_symlink():
        return True
    junction = getattr(path, "is_junction", None)
    if junction is not None and junction():
        return True
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_attribute)


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_linked_path_components(
    path: Path,
    *,
    label: str,
    require_exists: bool,
    error_type: type[ValueError] = OracleIsolationError,
) -> Path:
    """Reject every lexical symlink/junction/reparse component before resolve()."""

    absolute = _lexical_absolute(path)
    components = [*reversed(absolute.parents), absolute]
    for index, component in enumerate(components):
        is_final = index == len(components) - 1
        try:
            if _is_reparse_point(component):
                raise error_type(f"{label} contains a linked or reparse path: {component}")
        except FileNotFoundError:
            if require_exists or not is_final:
                raise error_type(f"{label} path component is missing: {component}") from None
        except OSError as exc:
            raise error_type(f"{label} path component could not be inspected: {component}: {exc}") from exc
    if require_exists and not absolute.exists():
        raise error_type(f"{label} path is missing: {absolute}")
    return absolute


def _directory_identity(descriptor: os.stat_result) -> tuple[int, int]:
    return descriptor.st_dev, descriptor.st_ino


def _bind_directory(
    path: Path,
    *,
    label: str,
    error_type: type[ValueError] = OracleIsolationError,
) -> _DirectoryBinding:
    absolute = _reject_linked_path_components(path, label=label, require_exists=True, error_type=error_type)
    if not absolute.is_dir():
        raise error_type(f"{label} must be a regular directory")
    resolved = absolute.resolve(strict=True)
    snapshot = absolute.stat()
    identity = _directory_identity(snapshot)
    descriptor: int | None = None
    mode = "WINDOWS_REPARSE_AND_PRE_POST_IDENTITY_GUARD"
    if os.name != "nt":
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(absolute, flags)
        except OSError as exc:
            raise error_type(f"{label} directory could not be handle-bound: {exc}") from exc
        if _directory_identity(os.fstat(descriptor)) != identity:
            os.close(descriptor)
            raise error_type(f"{label} directory changed while it was handle-bound")
        mode = "POSIX_DIRECTORY_FD_AND_PRE_POST_IDENTITY_GUARD"
    binding = _DirectoryBinding(absolute, resolved, identity, descriptor, mode, error_type)
    try:
        binding.verify(label=label)
    except Exception:
        binding.close()
        raise
    return binding


def _validate_regular_path(
    path: Path,
    *,
    label: str,
    root: Path | None = None,
    error_type: type[ValueError] = OracleContractError,
) -> Path:
    absolute = _reject_linked_path_components(path, label=label, require_exists=True, error_type=error_type)
    try:
        mode = absolute.lstat().st_mode
    except OSError as exc:
        raise error_type(f"{label} could not be inspected: {exc}") from exc
    if not stat.S_ISREG(mode):
        raise error_type(f"{label} must be a regular, non-linked file: {absolute}")
    resolved = absolute.resolve(strict=True)
    if root is not None:
        resolved_root = root.resolve(strict=True)
        if resolved.parent != resolved_root and resolved_root not in resolved.parents:
            raise error_type(f"{label} escapes its bound root: {absolute}")
    return absolute


def _regular_file_sha256(path: Path, *, root: Path | None = None) -> tuple[str, int]:
    path = _validate_regular_path(path, label="required file", root=root)
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        snapshot = os.fstat(stream.fileno())
        remaining = snapshot.st_size
        while remaining:
            chunk = stream.read(min(remaining, 8 * 1024 * 1024))
            if not chunk:
                raise OracleContractError(f"required file was truncated while hashing: {path}")
            digest.update(chunk)
            remaining -= len(chunk)
        if stream.read(1):
            raise OracleContractError(f"required file grew while hashing: {path}")
        after = os.fstat(stream.fileno())
    _validate_regular_path(path, label="required file", root=root)
    if _stat_identity(snapshot) != _stat_identity(after) or _stat_identity(snapshot) != _stat_identity(path.stat()):
        raise OracleContractError(f"required file changed while hashing: {path}")
    return digest.hexdigest(), snapshot.st_size


def _read_small_regular_file(
    path: Path, *, root: Path | None = None, maximum: int = 16 * 1024 * 1024
) -> tuple[bytes, str]:
    path = _validate_regular_path(path, label="required file", root=root)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        snapshot = os.fstat(stream.fileno())
        if snapshot.st_size <= 0 or snapshot.st_size > maximum:
            raise OracleContractError(f"required file has unsafe size: {path}")
        payload = stream.read(snapshot.st_size)
        if len(payload) != snapshot.st_size or stream.read(1):
            raise OracleContractError(f"required file changed while it was read: {path}")
        after = os.fstat(stream.fileno())
    _validate_regular_path(path, label="required file", root=root)
    if _stat_identity(snapshot) != _stat_identity(after) or _stat_identity(snapshot) != _stat_identity(path.stat()):
        raise OracleContractError(f"required file changed while it was read: {path}")
    return payload, hashlib.sha256(payload).hexdigest()


def _strict_json_object(payload: bytes, path: Path) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise OracleContractError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise OracleContractError(f"non-standard JSON constant {value!r} in {path}")

    try:
        decoded = json.loads(
            payload,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OracleContractError(f"invalid JSON input {path}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise OracleContractError(f"JSON input must be an object: {path}")
    return decoded


def validate_reference_source(reference_directory: Path | str) -> dict[str, Any]:
    """Validate every decision-bearing pinned Comfy source file."""

    binding = _bind_directory(Path(reference_directory), label="reference root", error_type=OracleContractError)
    root = binding.resolved
    files: list[dict[str, Any]] = []
    try:
        for relative, expected in REFERENCE_FILES:
            actual, size = _regular_file_sha256(root / relative, root=root)
            if actual != expected:
                raise OracleContractError(f"pinned Comfy reference hash mismatch for {relative}: {actual}")
            files.append({"path": relative, "sha256": actual, "size": size})
        commit_marker = root / "GIT_COMMIT"
        marker_present = commit_marker.exists() or commit_marker.is_symlink()
        if marker_present:
            marker, _ = _read_small_regular_file(commit_marker, root=root, maximum=4096)
            checkout_commit = marker.decode("ascii").strip()
            if checkout_commit != REFERENCE_COMMIT:
                raise OracleContractError("pinned Comfy GIT_COMMIT marker mismatch")
            checkout_status = "VERIFIED_BY_GIT_COMMIT_MARKER"
        else:
            checkout_commit = None
            checkout_status = "NOT_VERIFIED"
        binding.verify(label="reference root")
    finally:
        binding.close()
    return {
        "repository": "xiaolibai-sys/ComfyUI-MiniMaxH3",
        "expected_commit": REFERENCE_COMMIT,
        "checkout_commit": checkout_commit,
        "checkout_commit_status": checkout_status,
        "files": files,
        "mode": "default-eager",
        "use_adaln_cache": False,
        "status": "PASS",
    }


def load_pinned_comfy_fold_entries(reference_directory: Path | str, torch: Any) -> Any:
    """Load the independently authored reference function from hash-bound bytes."""

    binding = _bind_directory(
        Path(reference_directory), label="reference executor root", error_type=OracleContractError
    )
    root = binding.resolved
    source_path = root / "models/fold.py"
    try:
        source, digest = _read_small_regular_file(source_path, root=root)
        binding.verify(label="reference executor root")
    finally:
        binding.close()
    expected = dict(REFERENCE_FILES)["models/fold.py"]
    if digest != expected:
        raise OracleContractError("pinned Comfy fold.py changed before executor loading")
    namespace: dict[str, Any] = {
        "__name__": "_h3_forge_pinned_fold_reference",
        "__file__": str(source_path),
    }
    try:
        exec(compile(source, str(source_path), "exec"), namespace)
    except Exception as exc:
        raise OracleRuntimeError(f"pinned Comfy fold.py executor failed to load: {exc}") from exc
    function = namespace.get("fold_entries")
    if not callable(function) or namespace.get("torch") is not torch:
        raise OracleContractError("pinned Comfy fold.py did not expose the expected executor")
    return function


def validate_oracle_mode(*, use_adaln_cache: bool) -> None:
    """Reject the runtime-sidecar mode before any tensor work or output."""

    if use_adaln_cache is not False:
        raise OracleContractError("the pinned product oracle requires use_adaln_cache=false")


def validate_micro_operations(plan: LoraBakePlan) -> tuple[tuple[int, BakeOperation], ...]:
    """Bind the five fixed targets to their exact plan positions and layouts."""

    if (
        plan.schema != BAKE_PLAN_SCHEMA
        or plan.profile != OFFICIAL_FL2VA_BF16_BAKE_PROFILE
        or plan.target_dtype != "BF16"
        or plan.operation_count != COMFY_BAKED_NATIVE_FOLD_MODULE_COUNT
        or len(plan.operations) != COMFY_BAKED_NATIVE_FOLD_MODULE_COUNT
        or COMFY_BAKED_NATIVE_USE_ADALN_CACHE is not False
    ):
        raise OracleContractError("oracle requires the 259-operation official BF16 bake plan")
    selected: list[tuple[int, BakeOperation]] = []
    for index, module, shape in MICRO_TARGETS:
        operation = plan.operations[index]
        qkv = module.endswith("attn.qkv_proj")
        expected_layout = "grouped-qkv-to-qkv;merge-qkv;qkv-to-grouped-qkv" if qkv else "direct-runtime-layout-merge"
        expected_qkv = (56, 1, 128) if qkv else (None, None, None)
        actual_qkv = (
            operation.qkv_num_query_groups,
            operation.qkv_heads_per_group,
            operation.qkv_head_dim,
        )
        if (
            operation.module != module
            or operation.shape != shape
            or operation.layout_operation != expected_layout
            or actual_qkv != expected_qkv
        ):
            raise OracleContractError(f"micro target contract mismatch at plan index {index}")
        selected.append((index, operation))
    payload_bytes = sum(operation.shape[0] * operation.shape[1] * 2 for _, operation in selected)
    if payload_bytes != MICRO_PAYLOAD_BYTES:
        raise OracleContractError("micro target payload byte contract changed")
    return tuple(selected)


def _expected_micro_result_layout(module: str) -> str:
    return "base-grouped/B-direct/runtime-roundtrip" if module.endswith("attn.qkv_proj") else "direct"


def _qkv_indices(operation: BakeOperation, *, inverse: bool) -> tuple[int, ...] | None:
    if not operation.module.endswith("attn.qkv_proj"):
        return None
    dimensions = (
        operation.qkv_num_query_groups,
        operation.qkv_heads_per_group,
        operation.qkv_head_dim,
    )
    if any(value is None for value in dimensions):
        raise OracleContractError(f"QKV dimensions are missing for {operation.module!r}")
    function = qkv_to_grouped_row_indices if inverse else grouped_to_qkv_row_indices
    return function(
        num_query_groups=int(dimensions[0]),
        heads_per_group=int(dimensions[1]),
        head_dim=int(dimensions[2]),
    )


def base_grouped_to_runtime(base: Any, operation: BakeOperation, torch: Any) -> Any:
    """Map only the base QKV rows; supplied LoRA B is already runtime Q|K|V."""

    indices = _qkv_indices(operation, inverse=False)
    if indices is None:
        return base
    return base.index_select(0, torch.tensor(indices, dtype=torch.long, device=base.device))


def runtime_to_serialized(runtime: Any, operation: BakeOperation, torch: Any) -> Any:
    """Restore official grouped rows before candidate checkpoint serialization."""

    indices = _qkv_indices(operation, inverse=True)
    if indices is None:
        return runtime
    return runtime.index_select(0, torch.tensor(indices, dtype=torch.long, device=runtime.device))


def candidate_cleanroom_fold(
    base_grouped: Any,
    lora_a: Any,
    lora_b_runtime: Any,
    operation: BakeOperation,
    torch: Any,
    *,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> Any:
    """Candidate implementation of eager plain-A/B fold for one tensor."""

    if chunk_rows <= 0:
        raise OracleContractError("chunk_rows must be positive")
    if tuple(base_grouped.shape) != operation.shape:
        raise OracleContractError(f"base shape mismatch for {operation.module!r}")
    if tuple(lora_a.shape) != (operation.rank, operation.shape[1]):
        raise OracleContractError(f"LoRA A shape mismatch for {operation.module!r}")
    if tuple(lora_b_runtime.shape) != (operation.shape[0], operation.rank):
        raise OracleContractError(f"LoRA B shape mismatch for {operation.module!r}")
    # The asymmetric contract is intentional: base is mapped; B is never mapped.
    runtime = base_grouped_to_runtime(base_grouped, operation, torch).float()
    a = lora_a.float()
    with torch.no_grad():
        for start in range(0, operation.shape[0], chunk_rows):
            end = min(start + chunk_rows, operation.shape[0])
            delta = torch.matmul(lora_b_runtime[start:end].float(), a)
            runtime[start:end].add_(delta.mul_(operation.multiplier))
    return runtime.to(torch.bfloat16)


def pinned_comfy_reference_fold(
    base_grouped: Any,
    lora_a: Any,
    lora_b_runtime: Any,
    operation: BakeOperation,
    pinned_fold_entries: Any,
    torch: Any,
) -> Any:
    """Execute the independently authored, hash-bound Comfy fold implementation."""

    if tuple(base_grouped.shape) != operation.shape:
        raise OracleContractError(f"base shape mismatch for {operation.module!r}")
    if tuple(lora_a.shape) != (operation.rank, operation.shape[1]):
        raise OracleContractError(f"LoRA A shape mismatch for {operation.module!r}")
    if tuple(lora_b_runtime.shape) != (operation.shape[0], operation.rank):
        raise OracleContractError(f"LoRA B shape mismatch for {operation.module!r}")
    entry = SimpleNamespace(
        a=lora_a,
        b=lora_b_runtime,
        alpha=None,
        strength=operation.multiplier,
        diff=None,
        diff_b=None,
    )
    runtime_base = base_grouped_to_runtime(base_grouped, operation, torch)
    with torch.no_grad():
        folded = pinned_fold_entries(runtime_base, [entry])
    if tuple(folded.shape) != operation.shape or folded.dtype != torch.float32:
        raise OracleContractError("pinned Comfy fold executor returned an invalid tensor")
    return folded.to(torch.bfloat16)


def _tensor_sha256(tensor: Any, torch: Any, *, row_chunk: int = 256) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    for start in range(0, value.shape[0], row_chunk):
        raw = value[start : start + row_chunk].view(torch.uint8).numpy().tobytes()
        digest.update(raw)
    return digest.hexdigest()


def _encode_single_tensor_header(name: str, shape: tuple[int, int]) -> bytes:
    byte_count = shape[0] * shape[1] * 2
    payload = {
        "__metadata__": {
            "h3_comfy.oracle": ORACLE_SCHEMA,
            "h3_comfy.layout": "official-serialized",
        },
        name: {"dtype": "BF16", "shape": list(shape), "data_offsets": [0, byte_count]},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return encoded + b" " * (-len(encoded) % 8)


def _write_all(stream: BinaryIO, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        count = stream.write(view)
        if count is None or count <= 0:
            raise OracleRuntimeError("candidate output stream made no progress")
        view = view[count:]


def _stat_identity(stat: os.stat_result) -> tuple[int, int, int, int]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _fsync_directory(path: Path, binding: _DirectoryBinding | None = None) -> str:
    if os.name == "nt":
        return "NOT_SUPPORTED_ON_WINDOWS"
    descriptor = binding.descriptor if binding is not None else None
    owned = descriptor is None
    if descriptor is None:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        if owned:
            os.close(descriptor)
    return "PASS"


def _safe_unlink_identity(
    path: Path,
    identity: tuple[int, int, int, int],
    binding: _DirectoryBinding | None = None,
) -> str:
    """Retain a published path because stat-then-unlink cannot be conditional.

    Neither POSIX dirfd APIs nor ordinary Windows file APIs expose an unlink
    operation conditional on the inode/file identity observed by this process.
    Deleting after an identity check can therefore delete an attacker-swapped
    replacement. The oracle fails safely by retaining the isolated artifact.
    The arguments remain explicit so interleaving tests can prove this function
    never attempts deletion regardless of a matching or replaced identity.
    """

    del path, identity, binding
    return RETAINED_CLEANUP_STATUS


def _safe_unlink_inode(
    path: Path,
    identity: tuple[int, int],
    binding: _DirectoryBinding,
) -> str:
    """Retain a staging path rather than risk deleting a replacement."""

    del path, identity, binding
    return RETAINED_CLEANUP_STATUS


def _bind_publication_parent(
    path: Path,
    *,
    label: str,
    expected: _DirectoryBinding | None,
) -> tuple[Path, _DirectoryBinding]:
    absolute = _lexical_absolute(path)
    parent = _bind_directory(absolute.parent, label=f"{label} directory")
    try:
        if expected is not None and (parent.resolved != expected.resolved or parent.identity != expected.identity):
            raise OracleIsolationError(f"{label} directory no longer matches the run root")
        _reject_linked_path_components(absolute, label=label, require_exists=False, error_type=OracleIsolationError)
        parent.verify(label=f"{label} directory")
        return absolute, parent
    except Exception:
        parent.close()
        raise


def _link_no_replace(source: Path, destination: Path, binding: _DirectoryBinding) -> None:
    binding.verify(label="publication directory")
    if binding.descriptor is not None and os.link in os.supports_dir_fd:
        os.link(
            source.name,
            destination.name,
            src_dir_fd=binding.descriptor,
            dst_dir_fd=binding.descriptor,
            follow_symlinks=False,
        )
    else:
        # Python exposes no handle-relative hard-link API on Windows. The
        # reparse/component and directory identity guards immediately before
        # and after this call fail closed on every detectable parent swap; a
        # sub-check swap by an equally privileged directory writer remains a
        # platform limitation and is never presented as a stronger guarantee.
        os.link(source, destination)
    binding.verify(label="publication directory")


def write_candidate_tensor_atomic(
    path: Path,
    name: str,
    serialized_tensor: Any,
    torch: Any,
    *,
    row_chunk: int = 256,
    directory_binding: _DirectoryBinding | None = None,
) -> CandidateArtifact:
    """Write one BF16 safetensor without replacing any existing artifact."""

    tensor = serialized_tensor.detach().to(device="cpu").contiguous()
    if tensor.dtype != torch.bfloat16 or tensor.ndim != 2:
        raise OracleContractError("candidate tensor must be rank-two BF16")
    header = _encode_single_tensor_header(name, tuple(tensor.shape))
    path, active_binding = _bind_publication_parent(path, label="candidate artifact", expected=directory_binding)
    if path.exists() or path.is_symlink():
        active_binding.close()
        raise OracleIsolationError(f"refusing to replace candidate artifact: {path}")
    digest = hashlib.sha256()
    temporary: Path | None = None
    temporary_inode: tuple[int, int] | None = None
    published_identity: tuple[int, int, int, int] | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="x+b", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            temporary_inode = _directory_identity(os.fstat(stream.fileno()))
            preamble = struct.pack("<Q", len(header)) + header
            _write_all(stream, preamble)
            digest.update(preamble)
            for start in range(0, tensor.shape[0], row_chunk):
                payload = tensor[start : start + row_chunk].view(torch.uint8).numpy().tobytes()
                _write_all(stream, payload)
                digest.update(payload)
            stream.flush()
            os.fsync(stream.fileno())
            snapshot = os.fstat(stream.fileno())
            published_identity = _stat_identity(snapshot)
            _link_no_replace(temporary, path, active_binding)
            if _is_reparse_point(path) or _stat_identity(path.stat()) != published_identity:
                raise OracleIsolationError("candidate publication identity changed")
            stream.seek(0)
            readback = hashlib.sha256()
            remaining = snapshot.st_size
            while remaining:
                chunk = stream.read(min(remaining, 8 * 1024 * 1024))
                if not chunk:
                    raise OracleIsolationError("candidate publication readback was truncated")
                readback.update(chunk)
                remaining -= len(chunk)
            if stream.read(1) or readback.hexdigest() != digest.hexdigest():
                raise OracleIsolationError("candidate publication readback hash mismatch")
            directory_fsync = _fsync_directory(path.parent, active_binding)
        staging_cleanup = _safe_unlink_inode(temporary, temporary_inode, active_binding)
        temporary = None
        return CandidateArtifact(
            path=str(path.resolve()),
            file_sha256=digest.hexdigest(),
            size=published_identity[2],
            identity=published_identity,
            directory_fsync=directory_fsync,
            staging_cleanup=staging_cleanup,
        )
    except Exception as exc:
        if temporary is not None and temporary_inode is not None:
            _safe_unlink_inode(temporary, temporary_inode, active_binding)
        if published_identity is not None:
            _safe_unlink_identity(path, published_identity, active_binding)
        if isinstance(exc, OSError):
            raise OracleIsolationError(f"candidate publication failed: {exc}") from exc
        raise
    finally:
        active_binding.close()


def read_candidate_runtime(artifact: CandidateArtifact, operation: BakeOperation, torch: Any) -> Any:
    """Re-read exactly the published inode and verify its full-file hash."""

    path = _validate_regular_path(
        Path(artifact.path),
        label="published candidate",
        error_type=OracleIsolationError,
    )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise OracleIsolationError(f"published candidate could not be opened safely: {exc}") from exc
    with os.fdopen(descriptor, "rb") as stream:
        snapshot = os.fstat(stream.fileno())
        if (
            _stat_identity(snapshot) != artifact.identity
            or _stat_identity(path.stat()) != artifact.identity
            or snapshot.st_size != artifact.size
        ):
            raise OracleIsolationError("published candidate identity changed before readback")
        _, tensors, header_length = read_safetensors_header_stream(stream, path, snapshot.st_size)
        by_name = {tensor.name: tensor for tensor in tensors}
        tensor = by_name.get(operation.base_tensor)
        if len(tensors) != 1 or tensor is None or tensor.dtype != "BF16" or tensor.shape != operation.shape:
            raise OracleContractError("published candidate safetensors contract mismatch")
        stream.seek(0)
        file_digest = hashlib.sha256()
        remaining = snapshot.st_size
        while remaining:
            chunk = stream.read(min(remaining, 8 * 1024 * 1024))
            if not chunk:
                raise OracleIsolationError("published candidate was truncated during hash readback")
            file_digest.update(chunk)
            remaining -= len(chunk)
        if stream.read(1) or file_digest.hexdigest() != artifact.file_sha256:
            raise OracleIsolationError("published candidate full-file SHA256 changed")
        payload_start = 8 + header_length + tensor.data_offsets[0]
        byte_count = operation.shape[0] * operation.shape[1] * 2
        stream.seek(payload_start)
        payload = stream.read(byte_count)
        if len(payload) != byte_count:
            raise OracleIsolationError("published candidate tensor payload was truncated")
        if (
            _stat_identity(os.fstat(stream.fileno())) != artifact.identity
            or _stat_identity(path.stat()) != artifact.identity
        ):
            raise OracleIsolationError("published candidate identity changed during readback")
        _validate_regular_path(path, label="published candidate", error_type=OracleIsolationError)
    serialized = torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).reshape(operation.shape)
    return base_grouped_to_runtime(serialized, operation, torch)


def _decision(micro_equal: bool) -> str:
    if not micro_equal:
        return "REFERENCE_FOLD_PARITY_MISMATCH"
    return "INCOMPLETE"


def _planned_candidate_size(operation: BakeOperation) -> int:
    return (
        8
        + len(_encode_single_tensor_header(operation.base_tensor, operation.shape))
        + (operation.shape[0] * operation.shape[1] * 2)
    )


def _estimated_peak_working_set(operation: BakeOperation) -> int:
    payload = operation.shape[0] * operation.shape[1] * 2
    lora = (operation.rank * operation.shape[1] + operation.shape[0] * operation.rank) * 2
    chunk_delta = min(DEFAULT_CHUNK_ROWS, operation.shape[0]) * operation.shape[1] * 4
    # Base, mapped base, independent reference, candidate FP32 work, serialized
    # candidate, and bound-file reload. This intentionally overestimates aliases.
    return 7 * payload + lora + chunk_delta


def _host_available_bytes() -> int | None:
    if os.name == "nt":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.available_physical)
        except (AttributeError, OSError):
            return None
    elif Path("/proc/meminfo").is_file():
        try:
            for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        except (OSError, UnicodeDecodeError, ValueError, IndexError):
            return None
    return None


def _validate_reference_binding(reference: dict[str, Any]) -> None:
    if not isinstance(reference, dict):
        raise OracleContractError("receipt pinned Comfy reference binding is invalid")
    expected_reference_files = dict(REFERENCE_FILES)
    reference_files = reference.get("files")
    checkout_status = reference.get("checkout_commit_status")
    checkout_commit = reference.get("checkout_commit")
    if (
        reference.get("repository") != "xiaolibai-sys/ComfyUI-MiniMaxH3"
        or reference.get("expected_commit") != REFERENCE_COMMIT
        or checkout_status not in {"NOT_VERIFIED", "VERIFIED_BY_GIT_COMMIT_MARKER"}
        or (checkout_status == "VERIFIED_BY_GIT_COMMIT_MARKER" and checkout_commit != REFERENCE_COMMIT)
        or (checkout_status == "NOT_VERIFIED" and checkout_commit is not None)
        or reference.get("mode") != "default-eager"
        or reference.get("use_adaln_cache") is not False
        or reference.get("status") != "PASS"
        or reference.get("executor")
        != {
            "kind": "hash-bound-python-source",
            "path": "models/fold.py",
            "sha256": dict(REFERENCE_FILES)["models/fold.py"],
            "symbol": "fold_entries",
            "status": "LOADED",
        }
        or not isinstance(reference_files, list)
        or len(reference_files) != len(REFERENCE_FILES)
        or any(
            not isinstance(item, dict)
            or item.get("path") not in expected_reference_files
            or item.get("sha256") != expected_reference_files.get(item.get("path"))
            or not isinstance(item.get("size"), int)
            or isinstance(item.get("size"), bool)
            or item["size"] <= 0
            for item in reference_files
        )
        or {item["path"] for item in reference_files} != set(expected_reference_files)
    ):
        raise OracleContractError("receipt pinned Comfy reference binding is invalid")


def _validate_input_contract(input_contract: Any, *, plan: LoraBakePlan | None = None) -> None:
    if (
        not isinstance(input_contract, dict)
        or set(input_contract)
        != {
            "status",
            "config_sha256",
            "index_sha256",
            "base_tensor_count",
            "base_dtype_counts",
            "base_catalog_sha256",
            "normalized_lora_tensor_count",
            "normalized_lora_module_count",
            "normalized_lora_metadata_status",
        }
        or input_contract.get("status") != "PASS"
        or input_contract.get("base_tensor_count") != OFFICIAL_FL2VA_BASE_CONTRACT.tensor_count
        or input_contract.get("base_dtype_counts") != {"BF16": 522, "F32": 13}
        or input_contract.get("base_catalog_sha256") != OFFICIAL_FL2VA_BASE_CONTRACT.catalog_sha256
        or input_contract.get("normalized_lora_tensor_count") != 518
        or input_contract.get("normalized_lora_module_count") != COMFY_BAKED_NATIVE_FOLD_MODULE_COUNT
        or input_contract.get("normalized_lora_metadata_status") != "PASS"
        or not _is_sha256(input_contract.get("config_sha256"))
        or not _is_sha256(input_contract.get("index_sha256"))
        or (
            plan is not None
            and (
                input_contract["config_sha256"] != plan.config_sha256
                or input_contract["index_sha256"] != plan.index_sha256
            )
        )
    ):
        raise OracleContractError("runtime input tensor contract is invalid")


def build_local_receipt(
    *,
    plan: LoraBakePlan,
    plan_file_sha256: str,
    reference: dict[str, Any],
    runtime: dict[str, Any],
    micro_targets: Iterable[MicroTargetResult],
    candidate_bytes: int,
    elapsed_seconds: float,
    max_working_set_bytes: int = DEFAULT_MAX_WORKING_SET_BYTES,
) -> ComfyReferenceFoldReceipt:
    """Build a self-hashed receipt that cannot promote local-only evidence."""

    validate_micro_operations(plan)
    if (
        plan.base_catalog_sha256 != OFFICIAL_FL2VA_BASE_CONTRACT.catalog_sha256
        or len(plan.base_shards) != OFFICIAL_FL2VA_BASE_CONTRACT.shard_count
        or not _is_sha256(plan_file_sha256)
    ):
        raise OracleContractError("receipt plan source binding is invalid")
    reference = deepcopy(reference)
    runtime = deepcopy(runtime)
    _validate_reference_binding(reference)
    if runtime.get("actual_vllm_loader") != "NOT_RUN":
        raise OracleContractError("local receipt cannot bind an actual vLLM loader result")
    _validate_input_contract(runtime.get("input_contract"), plan=plan)
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        raise OracleContractError("receipt elapsed_seconds must be finite and non-negative")
    targets = tuple(micro_targets)
    selected = validate_micro_operations(plan)
    planned_candidate_bytes = sum(_planned_candidate_size(operation) for _, operation in selected)
    estimated_peak_bytes = max(_estimated_peak_working_set(operation) for _, operation in selected)
    if (
        planned_candidate_bytes != MICRO_CANDIDATE_FILE_BYTES
        or estimated_peak_bytes != MICRO_ESTIMATED_PEAK_WORKING_SET_BYTES
        or candidate_bytes != planned_candidate_bytes
        or candidate_bytes + RECEIPT_RESERVATION_BYTES > HARD_OUTPUT_LIMIT_BYTES
        or max_working_set_bytes < estimated_peak_bytes
    ):
        raise OracleIsolationError("oracle output or working-set resource contract failed")
    expected = tuple((index, module, shape) for index, module, shape in MICRO_TARGETS)
    actual = tuple((item.plan_operation_index, item.module, item.shape) for item in targets)
    if actual != expected:
        raise OracleContractError("receipt micro targets are incomplete or out of canonical order")
    for item, (_, _, shape), (_, operation) in zip(targets, MICRO_TARGETS, selected, strict=True):
        if (
            item.payload_bytes != shape[0] * shape[1] * 2
            or item.layout != _expected_micro_result_layout(item.module)
            or item.tp4_rank_status != "NOT_RUN"
            or item.candidate_safetensors_file_bytes != _planned_candidate_size(operation)
            or not all(
                _is_sha256(digest)
                for digest in (
                    item.reference_runtime_sha256,
                    item.candidate_safetensors_file_sha256,
                    item.candidate_runtime_sha256,
                )
            )
            or item.runtime_tensor_equal != (item.reference_runtime_sha256 == item.candidate_runtime_sha256)
        ):
            raise OracleContractError(f"invalid micro target result for {item.module!r}")
    micro_equal = all(item.runtime_tensor_equal for item in targets)
    decision = _decision(micro_equal)
    candidate = {
        "implementation": "clean-room-candidate-fold",
        "serialization": "BF16 official grouped QKV; direct non-QKV",
        "roundtrip_read_status": "PASS" if micro_equal else "FAIL",
        "micro_tensor_parity_status": "PASS" if micro_equal else "FAIL",
        "actual_vllm_tp4_loader_status": "NOT_RUN",
        "promotion_capable": False,
    }
    full_coverage = {
        "status": "NOT_RUN",
        "verified_operation_count": 0,
        "required_operation_count": COMFY_BAKED_NATIVE_FOLD_MODULE_COUNT,
    }
    gpu_workers = tuple({"tp_rank": rank, "status": "NOT_RUN", "device_uuid": None} for rank in range(4))
    resource_usage = {
        "hard_output_limit_bytes": HARD_OUTPUT_LIMIT_BYTES,
        "expected_micro_payload_bytes": MICRO_PAYLOAD_BYTES,
        "candidate_artifact_bytes": candidate_bytes,
        "receipt_reservation_bytes": RECEIPT_RESERVATION_BYTES,
        "accounted_upper_bound_bytes": candidate_bytes + RECEIPT_RESERVATION_BYTES,
        "planned_candidate_artifact_bytes": planned_candidate_bytes,
        "planned_candidate_files": [
            {
                "plan_operation_index": index,
                "module": operation.module,
                "bytes": _planned_candidate_size(operation),
            }
            for index, operation in selected
        ],
        "max_working_set_bytes": max_working_set_bytes,
        "estimated_peak_working_set_bytes": estimated_peak_bytes,
        "elapsed_seconds": elapsed_seconds,
        "within_limit": candidate_bytes + RECEIPT_RESERVATION_BYTES <= HARD_OUTPUT_LIMIT_BYTES,
    }
    payload: dict[str, Any] = {
        "schema": ORACLE_SCHEMA,
        "decision": decision,
        "profile": plan.profile,
        "product_oracle": PRODUCT_ORACLE,
        "use_adaln_cache": False,
        "plan_sha256": plan.plan_sha256,
        "plan_file_sha256": plan_file_sha256,
        "operation_count": plan.operation_count,
        "selected_micro_indices": MICRO_INDICES,
        "base_catalog_sha256": plan.base_catalog_sha256,
        "base_shards": [asdict(shard) for shard in plan.base_shards],
        "normalized_lora_sha256": plan.lora_sha256,
        "reference": reference,
        "candidate": candidate,
        "runtime": runtime,
        "gpu_workers": gpu_workers,
        "micro_targets": [asdict(target) for target in targets],
        "full_coverage": full_coverage,
        "resource_usage": resource_usage,
        "evidence_assurance": _expected_evidence_assurance(),
        "hash_domains": _expected_hash_domains(),
    }
    _validate_hash_domain_coverage({**payload, "receipt_sha256": "0" * 64})
    return ComfyReferenceFoldReceipt(
        schema=ORACLE_SCHEMA,
        decision=decision,
        profile=plan.profile,
        product_oracle=PRODUCT_ORACLE,
        use_adaln_cache=False,
        plan_sha256=plan.plan_sha256,
        plan_file_sha256=plan_file_sha256,
        operation_count=plan.operation_count,
        selected_micro_indices=MICRO_INDICES,
        base_catalog_sha256=plan.base_catalog_sha256,
        base_shards=tuple(asdict(shard) for shard in plan.base_shards),
        normalized_lora_sha256=plan.lora_sha256,
        reference=reference,
        candidate=candidate,
        runtime=runtime,
        gpu_workers=gpu_workers,
        micro_targets=targets,
        full_coverage=full_coverage,
        resource_usage=resource_usage,
        evidence_assurance=_expected_evidence_assurance(),
        hash_domains=_expected_hash_domains(),
        receipt_sha256=_canonical_sha256(payload),
    )


def validate_local_receipt(
    receipt: ComfyReferenceFoldReceipt | dict[str, Any],
) -> dict[str, Any]:
    """Strictly reject any local receipt that claims product promotion."""

    raw = receipt.to_dict() if isinstance(receipt, ComfyReferenceFoldReceipt) else deepcopy(receipt)
    try:
        payload = json.loads(json.dumps(raw, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise OracleContractError(f"local oracle receipt is not canonical JSON: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != set(ComfyReferenceFoldReceipt.__dataclass_fields__):
        raise OracleContractError("local oracle receipt has an invalid top-level schema")
    _validate_hash_domain_coverage(payload)
    self_hash = payload.get("receipt_sha256")
    canonical = dict(payload)
    canonical.pop("receipt_sha256")
    if not _is_sha256(self_hash) or _canonical_sha256(canonical) != self_hash:
        raise OracleContractError("local oracle receipt canonical self-hash failed")
    if (
        payload.get("schema") != ORACLE_SCHEMA
        or payload.get("decision") not in {"INCOMPLETE", "REFERENCE_FOLD_PARITY_MISMATCH"}
        or payload.get("profile") != OFFICIAL_FL2VA_BF16_BAKE_PROFILE
        or payload.get("product_oracle") != PRODUCT_ORACLE
        or payload.get("use_adaln_cache") is not False
        or payload.get("operation_count") != COMFY_BAKED_NATIVE_FOLD_MODULE_COUNT
        or tuple(payload.get("selected_micro_indices", ())) != MICRO_INDICES
        or payload.get("base_catalog_sha256") != OFFICIAL_FL2VA_BASE_CONTRACT.catalog_sha256
        or not _is_sha256(payload.get("plan_sha256"))
        or not _is_sha256(payload.get("plan_file_sha256"))
        or not _is_sha256(payload.get("normalized_lora_sha256"))
        or payload.get("evidence_assurance") != _expected_evidence_assurance()
        or payload.get("hash_domains") != _expected_hash_domains()
    ):
        raise OracleContractError("local oracle receipt product contract is invalid")
    shards = payload.get("base_shards")
    if (
        not isinstance(shards, list)
        or len(shards) != OFFICIAL_FL2VA_BASE_CONTRACT.shard_count
        or any(
            not isinstance(shard, dict)
            or set(shard) != {"name", "size", "sha256"}
            or not isinstance(shard["size"], int)
            or isinstance(shard["size"], bool)
            or shard["size"] <= 0
            or not _is_sha256(shard["sha256"])
            for shard in shards
        )
    ):
        raise OracleContractError("local oracle receipt base shard binding is invalid")
    _validate_reference_binding(payload.get("reference"))
    candidate = payload.get("candidate")
    full = payload.get("full_coverage")
    workers = payload.get("gpu_workers")
    runtime = payload.get("runtime")
    if (
        candidate
        != {
            "implementation": "clean-room-candidate-fold",
            "serialization": "BF16 official grouped QKV; direct non-QKV",
            "roundtrip_read_status": (candidate.get("roundtrip_read_status") if isinstance(candidate, dict) else None),
            "micro_tensor_parity_status": (
                candidate.get("micro_tensor_parity_status") if isinstance(candidate, dict) else None
            ),
            "actual_vllm_tp4_loader_status": "NOT_RUN",
            "promotion_capable": False,
        }
        or full
        != {
            "status": "NOT_RUN",
            "verified_operation_count": 0,
            "required_operation_count": COMFY_BAKED_NATIVE_FOLD_MODULE_COUNT,
        }
        or workers != [{"tp_rank": rank, "status": "NOT_RUN", "device_uuid": None} for rank in range(4)]
        or not isinstance(runtime, dict)
        or runtime.get("actual_vllm_loader") != "NOT_RUN"
        or runtime.get("execution") != "hash-bound-pinned-comfy-reference-vs-clean-room-candidate"
        or runtime.get("float32_matmul_precision") != "highest"
        or not isinstance(runtime.get("candidate_directory_fsync"), list)
        or len(runtime.get("candidate_directory_fsync")) != len(MICRO_TARGETS)
        or any(
            status not in {"PASS", "NOT_SUPPORTED_ON_WINDOWS"} for status in runtime.get("candidate_directory_fsync")
        )
        or runtime.get("candidate_staging_cleanup") != [RETAINED_CLEANUP_STATUS] * len(MICRO_TARGETS)
        or runtime.get("output_directory_binding")
        != {
            "evidence": (
                "WINDOWS_REPARSE_AND_PRE_POST_IDENTITY_GUARD"
                if os.name == "nt"
                else "POSIX_DIRECTORY_FD_AND_PRE_POST_IDENTITY_GUARD"
            ),
            "temp": (
                "WINDOWS_REPARSE_AND_PRE_POST_IDENTITY_GUARD"
                if os.name == "nt"
                else "POSIX_DIRECTORY_FD_AND_PRE_POST_IDENTITY_GUARD"
            ),
            "windows_handle_relative_publication": (
                "UNAVAILABLE_PRE_POST_IDENTITY_GUARD_ONLY" if os.name == "nt" else "NOT_APPLICABLE"
            ),
        }
        or (
            runtime.get("host_available_bytes_before") is not None
            and (
                not isinstance(runtime.get("host_available_bytes_before"), int)
                or isinstance(runtime.get("host_available_bytes_before"), bool)
                or runtime.get("host_available_bytes_before") <= 0
            )
        )
    ):
        raise OracleContractError("local oracle receipt attempts an unsupported promotion state")
    _validate_input_contract(runtime.get("input_contract"))
    targets = payload.get("micro_targets")
    expected = tuple((index, module, list(shape)) for index, module, shape in MICRO_TARGETS)
    if (
        not isinstance(targets, list)
        or tuple(
            (item.get("plan_operation_index"), item.get("module"), item.get("shape"))
            for item in targets
            if isinstance(item, dict)
        )
        != expected
    ):
        raise OracleContractError("local oracle receipt micro target coverage is invalid")
    equal_values: list[bool] = []
    expected_fields = set(MicroTargetResult.__dataclass_fields__)
    for item, (_, module, shape) in zip(targets, MICRO_TARGETS, strict=True):
        if not isinstance(item, dict) or set(item) != expected_fields:
            raise OracleContractError("local oracle receipt micro target schema is invalid")
        equal = item.get("runtime_tensor_equal")
        if (
            not isinstance(equal, bool)
            or item.get("layout") != _expected_micro_result_layout(module)
            or item.get("payload_bytes") != shape[0] * shape[1] * 2
            or item.get("candidate_safetensors_file_bytes")
            != 8 + len(_encode_single_tensor_header(f"{module}.weight", shape)) + shape[0] * shape[1] * 2
            or item.get("tp4_rank_status") != "NOT_RUN"
            or not all(
                _is_sha256(item.get(field))
                for field in (
                    "reference_runtime_sha256",
                    "candidate_safetensors_file_sha256",
                    "candidate_runtime_sha256",
                )
            )
            or equal != (item["reference_runtime_sha256"] == item["candidate_runtime_sha256"])
        ):
            raise OracleContractError("local oracle receipt micro target result is invalid")
        equal_values.append(equal)
    expected_decision = _decision(all(equal_values))
    expected_status = "PASS" if all(equal_values) else "FAIL"
    if (
        payload["decision"] != expected_decision
        or candidate["roundtrip_read_status"] != expected_status
        or candidate["micro_tensor_parity_status"] != expected_status
    ):
        raise OracleContractError("local oracle receipt decision is inconsistent")
    resource = payload.get("resource_usage")
    expected_planned_files = [
        {
            "plan_operation_index": index,
            "module": module,
            "bytes": 8 + len(_encode_single_tensor_header(f"{module}.weight", shape)) + shape[0] * shape[1] * 2,
        }
        for index, module, shape in MICRO_TARGETS
    ]
    if (
        not isinstance(resource, dict)
        or not isinstance(resource.get("candidate_artifact_bytes"), int)
        or isinstance(resource.get("candidate_artifact_bytes"), bool)
        or not isinstance(resource.get("planned_candidate_artifact_bytes"), int)
        or isinstance(resource.get("planned_candidate_artifact_bytes"), bool)
        or not isinstance(resource.get("accounted_upper_bound_bytes"), int)
        or isinstance(resource.get("accounted_upper_bound_bytes"), bool)
        or not isinstance(resource.get("estimated_peak_working_set_bytes"), int)
        or isinstance(resource.get("estimated_peak_working_set_bytes"), bool)
        or not isinstance(resource.get("max_working_set_bytes"), int)
        or isinstance(resource.get("max_working_set_bytes"), bool)
        or resource.get("hard_output_limit_bytes") != HARD_OUTPUT_LIMIT_BYTES
        or resource.get("expected_micro_payload_bytes") != MICRO_PAYLOAD_BYTES
        or resource.get("receipt_reservation_bytes") != RECEIPT_RESERVATION_BYTES
        or resource.get("candidate_artifact_bytes") != resource.get("planned_candidate_artifact_bytes")
        or resource.get("candidate_artifact_bytes") != MICRO_CANDIDATE_FILE_BYTES
        or resource.get("planned_candidate_files") != expected_planned_files
        or resource.get("accounted_upper_bound_bytes")
        != resource.get("candidate_artifact_bytes") + RECEIPT_RESERVATION_BYTES
        or resource.get("accounted_upper_bound_bytes") > HARD_OUTPUT_LIMIT_BYTES
        or resource.get("estimated_peak_working_set_bytes") > resource.get("max_working_set_bytes")
        or resource.get("estimated_peak_working_set_bytes") != MICRO_ESTIMATED_PEAK_WORKING_SET_BYTES
        or not isinstance(resource.get("elapsed_seconds"), (int, float))
        or isinstance(resource.get("elapsed_seconds"), bool)
        or not math.isfinite(resource.get("elapsed_seconds"))
        or resource.get("elapsed_seconds") < 0
        or resource.get("within_limit") is not True
    ):
        raise OracleContractError("local oracle receipt resource accounting is invalid")
    return payload


def write_receipt_atomic(
    receipt: ComfyReferenceFoldReceipt,
    path: Path | str,
    *,
    directory_binding: _DirectoryBinding | None = None,
) -> None:
    """Publish a receipt atomically and refuse replacement."""

    payload = validate_local_receipt(receipt)
    output, active_binding = _bind_publication_parent(Path(path), label="oracle receipt", expected=directory_binding)
    if output.exists() or output.is_symlink():
        active_binding.close()
        raise OracleIsolationError(f"refusing to replace oracle receipt: {output}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    if len(encoded) > RECEIPT_RESERVATION_BYTES:
        active_binding.close()
        raise OracleIsolationError("oracle receipt exceeds its reserved evidence budget")
    temporary: Path | None = None
    temporary_inode: tuple[int, int] | None = None
    published_identity: tuple[int, int, int, int] | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="x+b", prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            temporary_inode = _directory_identity(os.fstat(stream.fileno()))
            _write_all(stream, encoded)
            stream.flush()
            os.fsync(stream.fileno())
            published_identity = _stat_identity(os.fstat(stream.fileno()))
            _link_no_replace(temporary, output, active_binding)
            if _is_reparse_point(output) or _stat_identity(output.stat()) != published_identity:
                raise OracleIsolationError("receipt publication identity changed")
            stream.seek(0)
            if hashlib.sha256(stream.read()).hexdigest() != hashlib.sha256(encoded).hexdigest():
                raise OracleIsolationError("receipt publication readback hash mismatch")
            _fsync_directory(output.parent, active_binding)
        _safe_unlink_inode(temporary, temporary_inode, active_binding)
        temporary = None
    except Exception as exc:
        if temporary is not None and temporary_inode is not None:
            _safe_unlink_inode(temporary, temporary_inode, active_binding)
        if published_identity is not None:
            _safe_unlink_identity(output, published_identity, active_binding)
        if isinstance(exc, OSError):
            raise OracleIsolationError(f"receipt publication failed: {exc}") from exc
        raise
    finally:
        active_binding.close()


def _validate_output_directories(
    evidence_directory: Path, temp_directory: Path
) -> tuple[_DirectoryBinding, _DirectoryBinding]:
    evidence = _bind_directory(evidence_directory, label="evidence root")
    try:
        staging = _bind_directory(temp_directory, label="temp root")
    except Exception:
        evidence.close()
        raise
    try:
        if evidence.resolved == staging.resolved:
            raise OracleIsolationError("evidence and temp directories must be distinct")
        if any(staging.path.iterdir()):
            raise OracleIsolationError("temp directory must be empty for an isolated oracle run")
        evidence.verify(label="evidence root")
        staging.verify(label="temp root")
        # Preserve the immutable run-root identity while releasing preflight
        # descriptors. Each publication acquires a fresh handle and must match
        # this identity before it can create a destination.
        evidence.close()
        staging.close()
        return evidence, staging
    except Exception:
        evidence.close()
        staging.close()
        raise


def _verify_plan_sources(
    plan: LoraBakePlan,
) -> tuple[
    dict[str, _SafeTensorReader],
    _SafeTensorReader,
    dict[str, Any],
    tuple[_DirectoryBinding, ...],
]:
    base_binding = _bind_directory(Path(plan.base_directory), label="plan base root", error_type=OracleContractError)
    base = base_binding.path
    root = base_binding.resolved
    try:
        config_bytes, config_sha256 = _read_small_regular_file(base / "config.json", root=root)
        index_bytes, index_sha256 = _read_small_regular_file(base / "model.safetensors.index.json", root=root)
        if config_sha256 != plan.config_sha256 or index_sha256 != plan.index_sha256:
            raise OracleContractError("plan config or index source hash mismatch")
        config = _strict_json_object(config_bytes, base / "config.json")
        index = _strict_json_object(index_bytes, base / "model.safetensors.index.json")
        try:
            _validate_config(config)
        except ValueError as exc:
            raise OracleContractError(f"official FL2VA config contract failed: {exc}") from exc
    except Exception:
        base_binding.close()
        raise
    readers: dict[str, _SafeTensorReader] = {}
    lora_reader: _SafeTensorReader | None = None
    source_bindings: list[_DirectoryBinding] = [base_binding]
    try:
        catalog: dict[str, tuple[str, Any]] = {}
        for shard in plan.base_shards:
            shard_path = _validate_regular_path(
                base / shard.name,
                label=f"base shard {shard.name}",
                root=root,
                error_type=OracleContractError,
            )
            reader = _SafeTensorReader(shard_path)
            readers[shard.name] = reader
            if reader.snapshot.st_size != shard.size or reader.sha256() != shard.sha256:
                raise OracleContractError(f"plan source hash mismatch for {shard.name}")
            _validate_regular_path(
                shard_path,
                label=f"base shard {shard.name}",
                root=root,
                error_type=OracleContractError,
            )
            for descriptor in reader.tensors.values():
                if descriptor.name in catalog:
                    raise OracleContractError(f"base tensor appears in multiple shards: {descriptor.name!r}")
                catalog[descriptor.name] = (shard.name, descriptor)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or any(
            not isinstance(name, str) or not isinstance(shard, str) for name, shard in weight_map.items()
        ):
            raise OracleContractError("official FL2VA index weight_map is invalid")
        if set(weight_map) != set(catalog) or any(catalog[name][0] != shard for name, shard in weight_map.items()):
            raise OracleContractError("official FL2VA index/header tensor mapping mismatch")
        try:
            catalog_sha256 = _validate_official_base_contract(index, catalog)
        except ValueError as exc:
            raise OracleContractError(f"official FL2VA base contract failed: {exc}") from exc
        if catalog_sha256 != plan.base_catalog_sha256 or _catalog_sha256(catalog) != catalog_sha256:
            raise OracleContractError("runtime base catalog does not match the bound plan")
        lora_path = _validate_regular_path(
            Path(plan.normalized_lora),
            label="normalized LoRA",
            error_type=OracleContractError,
        )
        lora_binding = _bind_directory(
            lora_path.parent,
            label="normalized LoRA parent",
            error_type=OracleContractError,
        )
        source_bindings.append(lora_binding)
        lora_reader = _SafeTensorReader(lora_path)
        if lora_reader.sha256() != plan.lora_sha256:
            raise OracleContractError("normalized LoRA source hash mismatch")
        _validate_regular_path(lora_path, label="normalized LoRA", error_type=OracleContractError)
        try:
            modules = _validate_normalized_lora(lora_reader.metadata, tuple(lora_reader.tensors.values()))
        except ValueError as exc:
            raise OracleContractError(f"normalized LoRA runtime contract failed: {exc}") from exc
        dtype_counts: dict[str, int] = {}
        for _, descriptor in catalog.values():
            dtype_counts[descriptor.dtype] = dtype_counts.get(descriptor.dtype, 0) + 1
        input_contract = {
            "status": "PASS",
            "config_sha256": config_sha256,
            "index_sha256": index_sha256,
            "base_tensor_count": len(catalog),
            "base_dtype_counts": dict(sorted(dtype_counts.items())),
            "base_catalog_sha256": catalog_sha256,
            "normalized_lora_tensor_count": len(lora_reader.tensors),
            "normalized_lora_module_count": len(modules),
            "normalized_lora_metadata_status": "PASS",
        }
        for binding in source_bindings:
            binding.verify(label="bound oracle input root")
        return readers, lora_reader, input_contract, tuple(source_bindings)
    except Exception:
        for reader in readers.values():
            reader.close()
        if lora_reader is not None:
            lora_reader.close()
        for binding in source_bindings:
            binding.close()
        raise


def run_local_comfy_reference_fold_oracle(
    plan_json: Path | str,
    reference_directory: Path | str,
    evidence_directory: Path | str,
    temp_directory: Path | str,
    *,
    device: str = "cpu",
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    use_adaln_cache: bool = False,
    max_working_set_bytes: int = DEFAULT_MAX_WORKING_SET_BYTES,
) -> ComfyReferenceFoldReceipt:
    """Run the non-promotable five-tensor local oracle and write its receipt."""

    started = time.monotonic()
    validate_oracle_mode(use_adaln_cache=use_adaln_cache)
    if chunk_rows != DEFAULT_CHUNK_ROWS:
        raise OracleContractError(f"pinned Comfy reference fold requires chunk_rows={DEFAULT_CHUNK_ROWS}")
    evidence = Path(evidence_directory)
    staging = Path(temp_directory)
    evidence_binding, staging_binding = _validate_output_directories(evidence, staging)
    evidence = evidence_binding.path
    staging = staging_binding.path
    receipt_path = evidence / RECEIPT_NAME
    if receipt_path.exists() or receipt_path.is_symlink():
        raise OracleIsolationError(f"refusing to replace oracle receipt: {receipt_path}")
    try:
        plan_path = _validate_regular_path(Path(plan_json), label="oracle plan JSON", error_type=OracleContractError)
        plan, plan_file_sha256 = load_lora_bake_plan_json(plan_path)
        verified_plan_file_sha256, _ = _regular_file_sha256(plan_path)
        if verified_plan_file_sha256 != plan_file_sha256:
            raise OracleContractError("oracle plan JSON changed while it was loaded")
    except (OSError, ValueError) as exc:
        raise OracleContractError(f"invalid oracle plan JSON: {exc}") from exc
    selected = validate_micro_operations(plan)
    planned_candidate_bytes = sum(_planned_candidate_size(operation) for _, operation in selected)
    estimated_peak_bytes = max(_estimated_peak_working_set(operation) for _, operation in selected)
    if (
        planned_candidate_bytes + RECEIPT_RESERVATION_BYTES > HARD_OUTPUT_LIMIT_BYTES
        or max_working_set_bytes < estimated_peak_bytes
    ):
        raise OracleIsolationError("oracle preflight resource budget failed")
    host_available_bytes = _host_available_bytes()
    if host_available_bytes is not None and host_available_bytes < estimated_peak_bytes:
        raise OracleIsolationError("insufficient available host memory for the oracle working set")
    try:
        reference = validate_reference_source(reference_directory)
    except (OSError, ValueError) as exc:
        if isinstance(exc, OracleContractError):
            raise
        raise OracleContractError(f"invalid pinned Comfy reference: {exc}") from exc
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise OracleRuntimeError("torch is required for the reference-fold oracle") from exc
    if device != "cpu" and not device.startswith("cuda"):
        raise OracleContractError("device must be cpu or an explicit CUDA device")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise OracleRuntimeError("requested CUDA device is unavailable")
    pinned_fold_entries = load_pinned_comfy_fold_entries(reference_directory, torch)
    reference["executor"] = {
        "kind": "hash-bound-python-source",
        "path": "models/fold.py",
        "sha256": dict(REFERENCE_FILES)["models/fold.py"],
        "symbol": "fold_entries",
        "status": "LOADED",
    }
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    readers: dict[str, _SafeTensorReader] = {}
    lora_reader: _SafeTensorReader | None = None
    results: list[MicroTargetResult] = []
    candidate_bytes = 0
    input_contract: dict[str, Any] = {}
    source_bindings: tuple[_DirectoryBinding, ...] = ()
    directory_fsync_statuses: list[str] = []
    staging_cleanup_statuses: list[str] = []
    device_free_samples: list[int] = []
    try:
        readers, lora_reader, input_contract, source_bindings = _verify_plan_sources(plan)
        with torch.no_grad():
            for index, operation in selected:
                if device.startswith("cuda"):
                    free_bytes, _ = torch.cuda.mem_get_info(device)
                    device_free_samples.append(int(free_bytes))
                    if free_bytes < _estimated_peak_working_set(operation):
                        raise OracleIsolationError(f"insufficient free VRAM for micro target {operation.module!r}")
                base = (
                    readers[operation.shard]
                    .read_bf16_rows(operation.base_tensor, 0, operation.shape[0], torch)
                    .to(device)
                )
                a = lora_reader.read_bf16_rows(operation.lora_a, 0, operation.rank, torch).to(device)
                b = lora_reader.read_bf16_rows(operation.lora_b, 0, operation.shape[0], torch).to(device)
                reference_runtime = pinned_comfy_reference_fold(base, a, b, operation, pinned_fold_entries, torch)
                candidate_runtime = candidate_cleanroom_fold(base, a, b, operation, torch, chunk_rows=chunk_rows)
                serialized = runtime_to_serialized(candidate_runtime, operation, torch)
                safe_name = operation.module.replace(".", "_")
                candidate_path = staging / f"{index:03d}-{safe_name}.safetensors"
                artifact = write_candidate_tensor_atomic(
                    candidate_path,
                    operation.base_tensor,
                    serialized,
                    torch,
                    directory_binding=staging_binding,
                )
                candidate_bytes += artifact.size
                directory_fsync_statuses.append(artifact.directory_fsync)
                staging_cleanup_statuses.append(artifact.staging_cleanup)
                if candidate_bytes > planned_candidate_bytes:
                    raise OracleIsolationError("oracle staging output exceeded the 1.25 GiB hard limit")
                del base, a, b, candidate_runtime, serialized
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                reloaded_runtime = read_candidate_runtime(artifact, operation, torch).to(device)
                equal = bool(torch.equal(reference_runtime, reloaded_runtime))
                results.append(
                    MicroTargetResult(
                        plan_operation_index=index,
                        module=operation.module,
                        shape=operation.shape,
                        layout=(_expected_micro_result_layout(operation.module)),
                        payload_bytes=operation.shape[0] * operation.shape[1] * 2,
                        reference_runtime_sha256=_tensor_sha256(reference_runtime, torch),
                        candidate_safetensors_file_sha256=artifact.file_sha256,
                        candidate_safetensors_file_bytes=artifact.size,
                        candidate_runtime_sha256=_tensor_sha256(reloaded_runtime, torch),
                        runtime_tensor_equal=equal,
                        tp4_rank_status="NOT_RUN",
                    )
                )
                del reference_runtime, reloaded_runtime
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
    except (OracleContractError, OracleIsolationError):
        raise
    except Exception as exc:
        raise OracleRuntimeError(f"reference-fold local execution failed: {exc}") from exc
    finally:
        try:
            if lora_reader is not None:
                lora_reader.close()
            for reader in readers.values():
                reader.close()
            for binding in source_bindings:
                binding.verify(label="bound oracle input root")
        finally:
            for binding in source_bindings:
                binding.close()
            torch.set_float32_matmul_precision(previous_precision)

    if device == "cpu":
        device_name = "cpu"
        compute_capability = None
        device_uuid = None
    else:
        properties = torch.cuda.get_device_properties(device)
        device_name = properties.name
        compute_capability = f"{properties.major}.{properties.minor}"
        device_uuid = str(getattr(properties, "uuid", "")) or None
    runtime = {
        "execution": "hash-bound-pinned-comfy-reference-vs-clean-room-candidate",
        "device": device,
        "device_name": device_name,
        "compute_capability": compute_capability,
        "device_uuid": device_uuid,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "float32_matmul_precision": "highest",
        "actual_vllm_loader": "NOT_RUN",
        "input_contract": input_contract,
        "candidate_directory_fsync": directory_fsync_statuses,
        "candidate_staging_cleanup": staging_cleanup_statuses,
        "minimum_observed_free_device_bytes": (min(device_free_samples) if device_free_samples else None),
        "host_available_bytes_before": host_available_bytes,
        "output_directory_binding": {
            "evidence": evidence_binding.mode,
            "temp": staging_binding.mode,
            "windows_handle_relative_publication": (
                "UNAVAILABLE_PRE_POST_IDENTITY_GUARD_ONLY" if os.name == "nt" else "NOT_APPLICABLE"
            ),
        },
    }
    receipt = build_local_receipt(
        plan=plan,
        plan_file_sha256=plan_file_sha256,
        reference=reference,
        runtime=runtime,
        micro_targets=results,
        candidate_bytes=candidate_bytes,
        elapsed_seconds=time.monotonic() - started,
        max_working_set_bytes=max_working_set_bytes,
    )
    validate_local_receipt(receipt)
    write_receipt_atomic(receipt, receipt_path, directory_binding=evidence_binding)
    return receipt


def oracle_exit_code(decision: str) -> int:
    """Stable fail-closed CLI exit contract."""

    if decision == "REFERENCE_FOLD_PARITY_MISMATCH":
        return 2
    if decision == "INCOMPLETE":
        return 5
    raise OracleContractError(f"unsupported oracle decision: {decision!r}")
