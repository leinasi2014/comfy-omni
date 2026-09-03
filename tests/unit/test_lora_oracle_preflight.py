"""Unit matrix for the fail-closed LoRA compatibility preflight.

The migrated legacy oracle modules are characterized by a later GPU run; this module
exercises only the NEW ``conversion.oracle`` contract and preflight code with synthetic
miniature safetensors files.  Every verdict is produced by a defer-import of
:func:`comfy_omni.conversion.oracle.preflight.preflight_candidate` so the RED state
fails exactly because the module does not yet exist.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from comfy_omni.conversion.packaging.materialization import materialize_package
from comfy_omni.conversion.packaging.models import ComponentFile, ComponentReceipt
from comfy_omni.conversion.packaging.planning import (
    PACKAGE_COMPONENTS,
    PINNED_VLLM_OMNI_COMMIT,
    plan_native_package,
)
from comfy_omni.conversion.packaging.publication import publish_package
from comfy_omni.domain.normalization import ToolIdentity

# --------------------------------------------------------------------------- #
# Synthetic safetensors construction
# --------------------------------------------------------------------------- #

_TURBO_V4_METADATA = {
    "application": "W_eff = W + lora_B @ lora_A",
    "base_model": "MiniMax-H3",
    "dtype": "bfloat16",
    "sampler_steps": "4",
}


def _write_safetensors(
    path: Path,
    tensors: list[tuple[str, str, list[int], bytes]],
    *,
    metadata: dict[str, str] | None = None,
) -> None:
    """Write a valid, header-readable safetensors file without reading payload bytes."""
    offset = 0
    header: dict[str, object] = {}
    payload = bytearray()
    if metadata:
        header["__metadata__"] = metadata
    for name, dtype, shape, raw in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        payload.extend(raw)
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(payload))


def _safetensors_bytes(tensors: list[tuple[str, str, list[int], bytes]]) -> bytes:
    offset = 0
    header: dict[str, object] = {}
    payload = bytearray()
    for name, dtype, shape, raw in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        payload.extend(raw)
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded + bytes(payload)


def _bf16(elements: int) -> bytes:
    return b"\x00" * (elements * 2)


def _minimal_candidate_tensors() -> list[tuple[str, str, list[int], bytes]]:
    """One real Turbo V4 module pair with exact expected shapes (small payloads)."""
    return [
        (
            "diffusion_model.final_layer.adaln_proj.linear.lora_A.weight",
            "BF16",
            [16, 2688],
            _bf16(16 * 2688),
        ),
        (
            "diffusion_model.final_layer.adaln_proj.linear.lora_B.weight",
            "BF16",
            [10752, 16],
            _bf16(10752 * 16),
        ),
    ]


def _candidate(path: Path, *, metadata: dict[str, str] | None = None) -> tuple[str, int]:
    _write_safetensors(path, _minimal_candidate_tensors(), metadata=metadata or _TURBO_V4_METADATA)
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size


def _non_bf16_tensors() -> list[tuple[str, str, list[int], bytes]]:
    return [
        ("diffusion_model.final_layer.adaln_proj.linear.lora_A.weight", "F32", [16, 2688], b"\x00" * (16 * 2688 * 4)),
        ("diffusion_model.final_layer.adaln_proj.linear.lora_B.weight", "BF16", [10752, 16], _bf16(10752 * 16)),
    ]


def _unresolvable_tensors() -> list[tuple[str, str, list[int], bytes]]:
    # Valid metadata and BF16, but the module does not exist in the 259-module table.
    return [
        ("diffusion_model.not.a.module.lora_A.weight", "BF16", [1, 8], _bf16(1 * 8)),
        ("diffusion_model.not.a.module.lora_B.weight", "BF16", [8, 1], _bf16(8 * 1)),
    ]


def _convrot_marker() -> bytes:
    return json.dumps(
        {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
        separators=(",", ":"),
    ).encode()


def _convrot_tensors() -> list[tuple[str, str, list[int], bytes]]:
    marker = _convrot_marker()
    return [
        ("blocks.0.linear.weight", "I8", [2, 256], b"\x00" * (2 * 256)),
        ("blocks.0.linear.weight_scale", "F32", [2, 1], b"\x00" * (2 * 1 * 4)),
        ("blocks.0.linear.comfy_quant", "U8", [len(marker)], marker),
    ]


def _convrot_safetensors_bytes() -> bytes:
    return _safetensors_bytes(_convrot_tensors())


# --------------------------------------------------------------------------- #
# Published-package fixture
# --------------------------------------------------------------------------- #
def _published_package(tmp_path: Path, *, transformer_payload: bytes) -> Path:
    tool = ToolIdentity("comfy-omni", "0.2.0a1", "a" * 40, "b" * 64)
    receipts: list[ComponentReceipt] = []
    for component in PACKAGE_COMPONENTS:
        source = tmp_path / "sources" / component
        source.mkdir(parents=True)
        if component == "transformer":
            relative = "model.safetensors"
            payload = transformer_payload
        else:
            relative = "nested/artifact.bin"
            payload = f"{component}:payload".encode()
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        receipts.append(
            ComponentReceipt(
                component=component,
                source_dir=source.as_posix(),
                receipt_schema="test.component.receipt/v1",
                receipt_sha256=hashlib.sha256(f"{component}:receipt".encode()).hexdigest(),
                tool=tool,
                files=(ComponentFile(relative, len(payload), digest),),
            )
        )
    plan = plan_native_package(tuple(receipts), vllm_omni_commit=PINNED_VLLM_OMNI_COMMIT)
    output = tmp_path / "native-package"
    materialized = materialize_package(plan, output)
    publish_package(plan, materialized)
    return output


def _convrot_package(tmp_path: Path) -> Path:
    return _published_package(tmp_path, transformer_payload=_convrot_safetensors_bytes())


def _fingerprint(path: Path) -> tuple[int, str]:
    return (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())


def _passing_outcome():
    return SimpleNamespace(passed=True)


def _failing_outcome():
    return SimpleNamespace(passed=False)


# --------------------------------------------------------------------------- #
# Verdict helper
# --------------------------------------------------------------------------- #
def _run_preflight(
    candidate_id: str,
    package_root: Path | str,
    candidate_path: Path,
    pin_sha256: str,
    pin_bytes: int,
):
    # Deferred import so the RED state fails on the missing module, not on collection.
    from comfy_omni.conversion.oracle.preflight import preflight_candidate
    from comfy_omni.integrations.vllm_omni.package_contract import validate_runtime_package

    return preflight_candidate(
        candidate_id,
        package_root,
        candidate_path,
        pinned_sha256=pin_sha256,
        pinned_bytes=pin_bytes,
        runtime_contract_resolver=validate_runtime_package,
    )


# --------------------------------------------------------------------------- #
# Green path: SUPPORTED
# --------------------------------------------------------------------------- #
def test_preflight_reports_supported_when_pinned_candidate_and_oracle_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from comfy_omni.conversion.oracle import preflight as preflight_mod

    candidate = tmp_path / "candidate.safetensors"
    sha, size = _candidate(candidate)
    package = _convrot_package(tmp_path)

    monkeypatch.setattr(preflight_mod, "_bind_official_base_catalog", lambda transformer_dir: {"bound": True})
    monkeypatch.setattr(preflight_mod, "_run_offline_fold_oracle", lambda *args, **kwargs: _passing_outcome())

    verdict = _run_preflight("candidate-a", package, candidate, sha, size)
    assert verdict.verdict == preflight_mod.SUPPORTED
    assert verdict.reason_code is None
    assert verdict.candidate_id == "candidate-a"
    assert verdict.to_dict()["status"] == preflight_mod.SUPPORTED
    assert verdict.to_dict()["schema"] == preflight_mod.COMFY_OMNI_LORA_COMPATIBILITY_SCHEMA


# --------------------------------------------------------------------------- #
# Fail-closed refusals
# --------------------------------------------------------------------------- #
def test_preflight_rejects_pin_mismatch(tmp_path: Path) -> None:
    from comfy_omni.conversion.oracle.contract import CANDIDATE_PIN_MISMATCH

    candidate = tmp_path / "candidate.safetensors"
    sha, size = _candidate(candidate)
    verdict = _run_preflight("candidate-a", tmp_path / "nonexistent-package", candidate, "0" * 64, size)
    assert verdict.verdict == "UNSUPPORTED"
    assert verdict.reason_code == CANDIDATE_PIN_MISMATCH


def test_preflight_rejects_census_and_profile_mismatch(tmp_path: Path) -> None:
    from comfy_omni.conversion.oracle.contract import (
        CANDIDATE_PROFILE_UNSUPPORTED,
        CANDIDATE_TENSOR_CENSUS_MISMATCH,
    )

    bad_dtype = tmp_path / "bad-dtype.safetensors"
    _write_safetensors(bad_dtype, _non_bf16_tensors(), metadata=_TURBO_V4_METADATA)
    dsha, dsize = hashlib.sha256(bad_dtype.read_bytes()).hexdigest(), bad_dtype.stat().st_size
    verdict = _run_preflight("candidate-a", tmp_path / "nonexistent-package", bad_dtype, dsha, dsize)
    assert verdict.reason_code == CANDIDATE_TENSOR_CENSUS_MISMATCH

    empty = tmp_path / "empty.safetensors"
    _write_safetensors(empty, [], metadata=_TURBO_V4_METADATA)
    esha, esize = hashlib.sha256(empty.read_bytes()).hexdigest(), empty.stat().st_size
    empty_verdict = _run_preflight("candidate-a", tmp_path / "nonexistent-package", empty, esha, esize)
    assert empty_verdict.reason_code == CANDIDATE_TENSOR_CENSUS_MISMATCH

    bad_profile = tmp_path / "bad-profile.safetensors"
    _write_safetensors(bad_profile, _minimal_candidate_tensors(), metadata={"output": "not-a-profile"})
    bsha, bsize = hashlib.sha256(bad_profile.read_bytes()).hexdigest(), bad_profile.stat().st_size
    profile_verdict = _run_preflight("candidate-a", tmp_path / "nonexistent-package", bad_profile, bsha, bsize)
    assert profile_verdict.reason_code == CANDIDATE_PROFILE_UNSUPPORTED


def test_preflight_rejects_unresolvable_target_module_mapping(tmp_path: Path) -> None:
    from comfy_omni.conversion.oracle.contract import TARGET_MODULE_MAPPING_UNRESOLVED

    candidate = tmp_path / "candidate.safetensors"
    _write_safetensors(candidate, _unresolvable_tensors(), metadata=_TURBO_V4_METADATA)
    sha, size = hashlib.sha256(candidate.read_bytes()).hexdigest(), candidate.stat().st_size
    verdict = _run_preflight("candidate-a", tmp_path / "nonexistent-package", candidate, sha, size)
    assert verdict.verdict == "UNSUPPORTED"
    assert verdict.reason_code == TARGET_MODULE_MAPPING_UNRESOLVED


def test_preflight_rejects_int8_convrot_base_representation(tmp_path: Path) -> None:
    from comfy_omni.conversion.oracle.contract import BASE_REPRESENTATION_UNBINDABLE

    candidate = tmp_path / "candidate.safetensors"
    sha, size = _candidate(candidate)
    package = _convrot_package(tmp_path)

    verdict = _run_preflight("candidate-a", package, candidate, sha, size)
    assert verdict.verdict == "UNSUPPORTED"
    assert verdict.reason_code == BASE_REPRESENTATION_UNBINDABLE
    assert verdict.evidence["stage"] == "base-representation"


def test_preflight_reports_offline_fold_oracle_not_passed_when_oracle_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from comfy_omni.conversion.oracle import preflight as preflight_mod

    candidate = tmp_path / "candidate.safetensors"
    sha, size = _candidate(candidate)
    package = _convrot_package(tmp_path)

    monkeypatch.setattr(preflight_mod, "_bind_official_base_catalog", lambda transformer_dir: {"bound": True})
    monkeypatch.setattr(preflight_mod, "_run_offline_fold_oracle", lambda *args, **kwargs: _failing_outcome())

    verdict = _run_preflight("candidate-a", package, candidate, sha, size)
    assert verdict.verdict == "UNSUPPORTED"
    assert verdict.reason_code == preflight_mod.OFFLINE_FOLD_ORACLE_NOT_PASSED


def test_preflight_is_fail_closed_in_stage_order(tmp_path: Path) -> None:
    from comfy_omni.conversion.oracle.contract import (
        CANDIDATE_PIN_MISMATCH,
        CANDIDATE_TENSOR_CENSUS_MISMATCH,
        TARGET_MODULE_MAPPING_UNRESOLVED,
    )

    candidate = tmp_path / "candidate.safetensors"
    sha, size = _candidate(candidate)
    verdict = _run_preflight("candidate-a", tmp_path / "nope", candidate, sha, size + 1)
    assert verdict.reason_code == CANDIDATE_PIN_MISMATCH

    bad_dtype = tmp_path / "bad-dtype.safetensors"
    _write_safetensors(bad_dtype, _non_bf16_tensors(), metadata=_TURBO_V4_METADATA)
    dsha, dsize = hashlib.sha256(bad_dtype.read_bytes()).hexdigest(), bad_dtype.stat().st_size
    dtype_verdict = _run_preflight("candidate-a", tmp_path / "nope", bad_dtype, dsha, dsize)
    assert dtype_verdict.reason_code == CANDIDATE_TENSOR_CENSUS_MISMATCH

    unresolved = tmp_path / "unresolved.safetensors"
    _write_safetensors(unresolved, _unresolvable_tensors(), metadata=_TURBO_V4_METADATA)
    usha, usize = hashlib.sha256(unresolved.read_bytes()).hexdigest(), unresolved.stat().st_size
    mapping_verdict = _run_preflight("candidate-a", tmp_path / "nope", unresolved, usha, usize)
    assert mapping_verdict.reason_code == TARGET_MODULE_MAPPING_UNRESOLVED


def test_preflight_refusing_does_not_mutate_candidate_or_package(tmp_path: Path) -> None:
    from comfy_omni.conversion.oracle.contract import BASE_REPRESENTATION_UNBINDABLE

    candidate = tmp_path / "candidate.safetensors"
    sha, size = _candidate(candidate)
    package = _convrot_package(tmp_path)

    candidate_before = _fingerprint(candidate)
    package_files_before = {
        path.relative_to(package): _fingerprint(path) for path in sorted(p for p in package.rglob("*") if p.is_file())
    }

    verdict = _run_preflight("candidate-a", package, candidate, sha, size)
    assert verdict.reason_code == BASE_REPRESENTATION_UNBINDABLE

    assert _fingerprint(candidate) == candidate_before
    package_files_after = {
        path.relative_to(package): _fingerprint(path) for path in sorted(p for p in package.rglob("*") if p.is_file())
    }
    assert package_files_after == package_files_before
