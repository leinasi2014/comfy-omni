"""Fail-closed LoRA compatibility preflight for issue #12 slice 1.

``preflight_candidate`` maps one pinned LoRA candidate against a published runtime
package and returns an immutable :class:`LoRACompatibilityVerdict`.  Every import of
the migrated legacy oracle modules (:mod:`comfy_omni.conversion.lora`) is deferred
into function bodies so that importing this module never loads Torch, vLLM, or
FastAPI and never touches the migrated code at import time.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from comfy_omni.artifacts.safetensors import read_safetensors_header_stream
from comfy_omni.conversion.oracle.contract import (
    BASE_REPRESENTATION_UNBINDABLE,
    CANDIDATE_PIN_MISMATCH,
    CANDIDATE_PROFILE_UNSUPPORTED,
    CANDIDATE_TENSOR_CENSUS_MISMATCH,
    COMFY_OMNI_LORA_COMPATIBILITY_SCHEMA,
    OFFLINE_FOLD_ORACLE_NOT_PASSED,
    ORACLE_BASE_CONTRACT_NOT_BINDING,
    SUPPORTED,
    TARGET_MODULE_MAPPING_UNRESOLVED,
    UNSUPPORTED,
    LoRACompatibilityVerdict,
)

__all__ = [
    "COMFY_OMNI_LORA_COMPATIBILITY_SCHEMA",
    "SUPPORTED",
    "UNSUPPORTED",
    "CANDIDATE_PIN_MISMATCH",
    "CANDIDATE_PROFILE_UNSUPPORTED",
    "CANDIDATE_TENSOR_CENSUS_MISMATCH",
    "TARGET_MODULE_MAPPING_UNRESOLVED",
    "BASE_REPRESENTATION_UNBINDABLE",
    "ORACLE_BASE_CONTRACT_NOT_BINDING",
    "OFFLINE_FOLD_ORACLE_NOT_PASSED",
    "LoRACompatibilityVerdict",
    "preflight_candidate",
]

_TURBO_V4_PROFILE_METADATA = {
    "application": "W_eff = W + lora_B @ lora_A",
    "base_model": "MiniMax-H3",
    "dtype": "bfloat16",
    "sampler_steps": "4",
}
_COMBAT_V2_PROFILE_METADATA = {
    "ss_base_model_version": "minimax_h3",
    "ss_output_name": "doV2_copy",
    "format": "pt",
}


def _unsupported(
    candidate_id: str,
    reason_code: str,
    *,
    stage: str,
    **evidence: Any,
) -> LoRACompatibilityVerdict:
    return LoRACompatibilityVerdict(
        candidate_id,
        UNSUPPORTED,
        reason_code,
        {"stage": stage, **evidence},
    )


def _supported(candidate_id: str, **evidence: Any) -> LoRACompatibilityVerdict:
    return LoRACompatibilityVerdict(candidate_id, SUPPORTED, None, evidence)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _pins_match(path: Path, pinned_sha256: str, pinned_bytes: int) -> bool:
    if path.stat().st_size != pinned_bytes:
        return False
    return _sha256_file(path) == pinned_sha256


def _read_candidate_census(path: Path) -> tuple[dict[str, str], tuple[Any, ...], int]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        metadata, tensors, header_length = read_safetensors_header_stream(stream, path, size)
    return metadata, tensors, header_length


def _bind_profile(metadata: dict[str, str]) -> str | None:
    if all(metadata.get(key) == value for key, value in _TURBO_V4_PROFILE_METADATA.items()):
        return "TURBO_V4"
    if all(metadata.get(key) == value for key, value in _COMBAT_V2_PROFILE_METADATA.items()):
        return "COMBAT_V2"
    return None


def _bind_target_module_mapping(tensors: tuple[Any, ...]) -> list[tuple[Any, str, str]] | None:
    """Bind every candidate tensor to a known Turbo V4/Combat V2 module + side.

    Returns a list of ``(tensor, module, side)`` triples, or ``None`` if any tensor
    cannot be mapped (which the caller surfaces as ``TARGET_MODULE_MAPPING_UNRESOLVED``).
    """
    from comfy_omni.conversion.lora.normalize import _expected_modules, _module_and_side, _target_name

    expected_modules = _expected_modules()
    mapping: list[tuple[Any, str, str]] = []
    for tensor in tensors:
        try:
            target = _target_name(tensor.name)
            module, side = _module_and_side(target)
        except ValueError:
            return None
        expected = expected_modules.get(module)
        if expected is None:
            return None
        expected_shape = expected[0 if side == "A" else 1]
        if tensor.shape != expected_shape:
            return None
        mapping.append((tensor, module, side))
    return mapping


def _resolve_runtime_contract(
    package_root: Path | str,
    resolver: Any,
) -> Any | None:
    """Bind the published runtime package contract, or ``None`` on refusal.

    The resolver is injected by the caller (the conversion layer must not import
    the integrations layer where ``validate_runtime_package`` lives); any
    resolver refusal means the base contract is not binding.
    """
    try:
        return resolver(package_root)
    except Exception:
        return None


def _bind_official_base_catalog(transformer_dir: Path) -> dict[str, Any] | None:
    """Bind the official BF16 13-shard base census; ``None`` means the base is unbindable.

    The E4 package transformer is int8-convrot and does not present the official
    BF16 ``535``-tensor census, so this returns ``None`` and the caller reports
    :data:`BASE_REPRESENTATION_UNBINDABLE`.
    """
    from comfy_omni.conversion.lora.bake_plan import OFFICIAL_FL2VA_BASE_CONTRACT

    safe_files = sorted(transformer_dir.glob("*.safetensors"))
    if not safe_files:
        return None
    tensors: list[Any] = []
    dtype_counts: dict[str, int] = {}
    total_payload = 0
    for file in safe_files:
        size = file.stat().st_size
        with file.open("rb") as stream:
            metadata, shard_tensors, _ = read_safetensors_header_stream(stream, file, size)
        for tensor in shard_tensors:
            tensors.append(tensor)
            dtype_counts[tensor.dtype] = dtype_counts.get(tensor.dtype, 0) + 1
            total_payload += tensor.data_offsets[1] - tensor.data_offsets[0]

    contract = OFFICIAL_FL2VA_BASE_CONTRACT
    if len(tensors) != contract.tensor_count:
        return None
    if len(safe_files) != contract.shard_count:
        return None
    if tuple(sorted(dtype_counts.items())) != contract.dtype_counts:
        return None
    if total_payload != contract.total_size:
        return None
    return {
        "files": tuple(safe_files),
        "tensors": tuple(tensors),
        "total_payload": total_payload,
        "tensor_count": contract.tensor_count,
        "shard_count": contract.shard_count,
        "dtype_counts": contract.dtype_counts,
        "catalog_sha256": contract.catalog_sha256,
    }


def _run_offline_fold_oracle(candidate_id: str, catalog: dict[str, Any], mapping: list[tuple[Any, str, str]]) -> Any:
    """Seam over the migrated offline fold oracle.

    The real reference-fold oracle and residual diagnostic are characterized by a
    later GPU host run; until a BF16 baseline base binds, this seam is fail-closed.
    The NEW-code unit matrix monkeypatches this function.
    """
    # Fail-closed until the GPU-host oracle is characterized: never claim SUPPORTED
    # from the preflight gate alone.
    return SimpleNamespace(passed=False)


def preflight_candidate(
    candidate_id: str,
    package_root: Path | str,
    candidate_path: Path | str,
    *,
    pinned_sha256: str,
    pinned_bytes: int,
    runtime_contract_resolver: Any = None,
) -> LoRACompatibilityVerdict:
    """Produce a fail-closed LoRA compatibility verdict for one pinned candidate.

    The caller supplies ``pinned_sha256`` / ``pinned_bytes`` from
    ``docs/testing/model-baseline.v1.json`` and injects
    ``runtime_contract_resolver`` (e.g. ``validate_runtime_package`` from the
    integrations layer; dependency direction forbids importing it here).  The
    verdict is decided in a strict fail-closed order and never mutates the
    candidate or the published package.
    """
    candidate = Path(candidate_path)

    # (1) Pin the candidate identity (size + SHA256).
    try:
        if not _pins_match(candidate, pinned_sha256, pinned_bytes):
            return _unsupported(
                candidate_id,
                CANDIDATE_PIN_MISMATCH,
                stage="pin",
                path=str(candidate),
            )
    except OSError as exc:
        return _unsupported(candidate_id, CANDIDATE_PIN_MISMATCH, stage="pin", path=str(candidate), cause=str(exc))

    # (2) Read the header-only census and bind the normalization profile.
    try:
        metadata, tensors, _ = _read_candidate_census(candidate)
    except (OSError, ValueError) as exc:
        return _unsupported(
            candidate_id,
            CANDIDATE_TENSOR_CENSUS_MISMATCH,
            stage="census",
            path=str(candidate),
            cause=str(exc),
        )
    if not tensors:
        return _unsupported(candidate_id, CANDIDATE_TENSOR_CENSUS_MISMATCH, stage="census", path=str(candidate))
    if _bind_profile(metadata) is None:
        return _unsupported(
            candidate_id,
            CANDIDATE_PROFILE_UNSUPPORTED,
            stage="profile",
            path=str(candidate),
            metadata=dict(metadata),
        )
    if any(tensor.dtype != "BF16" for tensor in tensors):
        return _unsupported(candidate_id, CANDIDATE_TENSOR_CENSUS_MISMATCH, stage="census", path=str(candidate))

    # (3) Bind every candidate tensor to a known target module + side.
    mapping = _bind_target_module_mapping(tensors)
    if mapping is None:
        return _unsupported(candidate_id, TARGET_MODULE_MAPPING_UNRESOLVED, stage="target-module", path=str(candidate))

    # (4) Bind the runtime package and the official base census.
    runtime = (
        _resolve_runtime_contract(package_root, runtime_contract_resolver)
        if runtime_contract_resolver is not None
        else None
    )
    if runtime is None:
        return _unsupported(candidate_id, ORACLE_BASE_CONTRACT_NOT_BINDING, stage="runtime-package")
    transformer_dir = runtime.component_paths["transformer"]
    catalog = _bind_official_base_catalog(transformer_dir)
    if catalog is None:
        return _unsupported(
            candidate_id,
            BASE_REPRESENTATION_UNBINDABLE,
            stage="base-representation",
            transformer=str(transformer_dir),
        )

    # (5) Run the migrated offline fold / cache oracle.
    outcome = _run_offline_fold_oracle(candidate_id, catalog, mapping)
    if outcome is not None and getattr(outcome, "passed", False):
        return _supported(candidate_id, stage="offline-fold")
    return _unsupported(candidate_id, OFFLINE_FOLD_ORACLE_NOT_PASSED, stage="offline-fold")
