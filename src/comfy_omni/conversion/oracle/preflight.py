"""LoRA candidate compatibility preflight (fail-closed verdicts).

Stage order (each failure returns immediately):

1. ``candidate-pin``       -- candidate SHA256 + byte size must match the pinned asset
2. ``candidate-census``    -- every tensor is BF16 ``lora_A/lora_B`` with matched ranks
3. ``candidate-profile``   -- metadata must carry the pinned Comfy Turbo-V4 fold profile
4. ``target-mapping``      -- every target module must resolve to the H3 module table
5. ``base-representation`` -- the transformer base must be a bindable marker-free dense form
6. ``offline-fold-oracle`` -- the pinned reference-fold micro oracle must pass

A ``SUPPORTED`` verdict requires every stage to pass; anything else is
``UNSUPPORTED`` with an exact ``reason_code`` and evidence, and the candidate
and package are never mutated.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from comfy_omni.conversion.oracle import contract

COMFY_OMNI_LORA_COMPATIBILITY_SCHEMA = "comfy-omni.lora-compatibility/v1"
SUPPORTED = "SUPPORTED"
UNSUPPORTED = "UNSUPPORTED"

_PROFILE_APPLICATION = "W_eff = W + lora_B @ lora_A"
_PROFILE_BASE_MODEL = "MiniMax-H3"
_PROFILE_DTYPE = "bfloat16"
_PROFILE_SAMPLER_STEPS = "4"
_PROFILE_OUTPUTS = frozenset({None, "turbo-v4-s12-a3"})

#: Known H3 LoRA target modules (prefix after ``diffusion_model.``).
_KNOWN_MODULE_PREFIXES = (
    "video_patch_proj",
    "audio_patch_proj",
    "condition_proj",
    "final_layer.adaln_proj.linear",
    "final_layer.video_out",
    "final_layer.audio_out",
    "token_refiner.blocks.0.attn.qkv_proj",
    "token_refiner.blocks.0.attn.out_proj",
    "token_refiner.blocks.0.mlp.fc1",
    "token_refiner.blocks.0.mlp.fc2",
    "token_refiner.blocks.1.attn.qkv_proj",
    "token_refiner.blocks.1.attn.out_proj",
    "token_refiner.blocks.1.mlp.fc1",
    "token_refiner.blocks.1.mlp.fc2",
)
for _i in range(50):
    _KNOWN_MODULE_PREFIXES += (
        f"blocks.{_i}.attn.qkv_proj",
        f"blocks.{_i}.attn.out_proj",
        f"blocks.{_i}.mlp.fc1",
        f"blocks.{_i}.mlp.fc2",
        f"blocks.{_i}.adaln_proj.linear",
    )


class PreflightError(RuntimeError):
    """Internal fail-closed error; callers convert to verdicts."""


@dataclass
class OracleOutcome:
    passed: bool
    detail: str = ""


@dataclass
class LoRACompatibilityVerdict:
    candidate_id: str
    verdict: str
    reason_code: str | None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": COMFY_OMNI_LORA_COMPATIBILITY_SCHEMA,
            "candidate_id": self.candidate_id,
            "status": self.verdict,
            "reason_code": self.reason_code,
            "evidence": self.evidence,
        }


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if len(raw) < 8:
        raise PreflightError("candidate is too small to be a safetensors file")
    (header_len,) = struct.unpack("<Q", raw[:8])
    if header_len <= 0 or 8 + header_len > len(raw):
        raise PreflightError("candidate safetensors header length is invalid")
    try:
        header = json.loads(raw[8 : 8 + header_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError("candidate safetensors header is not valid JSON") from exc
    if not isinstance(header, dict):
        raise PreflightError("candidate safetensors header must be an object")
    return header


def _candidate_census(header: dict[str, Any]) -> dict[str, tuple[str, tuple[int, ...]]]:
    census: dict[str, tuple[str, tuple[int, ...]]] = {}
    for name, record in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(record, dict) or "dtype" not in record or "shape" not in record:
            raise PreflightError(f"candidate tensor {name!r} header record is malformed")
        census[name] = (record["dtype"], tuple(int(v) for v in record["shape"]))
    return census


def _validate_census(census: dict[str, tuple[str, tuple[int, ...]]]) -> None:
    if not census:
        raise PreflightError("candidate carries no tensors")
    a_names: set[str] = set()
    for name, (dtype, shape) in census.items():
        if dtype != "BF16":
            raise PreflightError(f"candidate tensor {name!r} is {dtype}, expected BF16")
        if not name.endswith(".lora_A.weight") and not name.endswith(".lora_B.weight"):
            raise PreflightError(f"candidate tensor {name!r} is not a lora_A/lora_B weight")
        if len(shape) != 2:
            raise PreflightError(f"candidate tensor {name!r} must be 2-D")
        if name.endswith(".lora_A.weight"):
            a_names.add(name)
    for name in a_names:
        b_name = name[: -len("lora_A.weight")] + "lora_B.weight"
        if b_name not in census:
            raise PreflightError(f"candidate tensor {name!r} has no matching lora_B")
        a_shape = census[name][1]
        b_shape = census[b_name][1]
        if a_shape[1] != b_shape[0]:
            raise PreflightError(f"candidate {name!r} rank mismatch: A{a_shape} vs B{b_shape}")


def _validate_profile(header: dict[str, Any]) -> None:
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict):
        raise PreflightError("candidate metadata is missing")
    checks = {
        "application": metadata.get("application") == _PROFILE_APPLICATION,
        "base_model": metadata.get("base_model") == _PROFILE_BASE_MODEL,
        "dtype": metadata.get("dtype") == _PROFILE_DTYPE,
        "sampler_steps": metadata.get("sampler_steps") == _PROFILE_SAMPLER_STEPS,
        "output": metadata.get("output") in _PROFILE_OUTPUTS,
    }
    if not all(checks.values()):
        raise PreflightError("candidate metadata does not carry the pinned Comfy Turbo-V4 fold profile")


def _resolve_target(module: str) -> bool:
    return module in _KNOWN_MODULE_PREFIXES


def _validate_target_mapping(census: dict[str, tuple[str, tuple[int, ...]]]) -> None:
    for name in census:
        if name.endswith(".lora_A.weight"):
            module = name[: -len(".lora_A.weight")]
        elif name.endswith(".lora_B.weight"):
            module = name[: -len(".lora_B.weight")]
        else:
            continue
        prefix = "diffusion_model."
        if not module.startswith(prefix):
            raise PreflightError(f"candidate module {module!r} is outside diffusion_model")
        target = module[len(prefix) :]
        if not _resolve_target(target):
            raise PreflightError(f"candidate target module {target!r} is not in the H3 table")


def _bind_official_base_catalog(transformer_dir: Path | str) -> dict[str, Any]:
    """Bind the official dense base catalog; reject serialized INT8 ConvRot forms.

    A marker-free dense BF16 transformer tree is bindable; a source form
    carrying ``comfy_quant`` markers (the INT8 ConvRot representation) is
    refused, because the runtime serves the dequantized dense form only.
    """
    root = Path(transformer_dir)
    if not root.is_dir():
        raise PreflightError("transformer directory does not exist")
    shards = sorted(root.glob("model-*.safetensors")) or sorted(root.glob("*.safetensors"))
    if not shards:
        raise PreflightError("transformer directory carries no safetensors shards")
    markers: list[str] = []
    for shard in shards:
        header = _read_safetensors_header(shard)
        markers.extend(name for name in header if name != "__metadata__" and ".comfy_quant" in name)
    if markers:
        raise PreflightError(
            "transformer base carries serialized comfy_quant markers; serve the dequantized dense BF16 representation"
        )
    return {"bound": True, "representation": "dense-bf16", "shards": len(shards)}


def _run_offline_fold_oracle(
    candidate_census: dict[str, tuple[str, tuple[int, ...]]],
    base_catalog: dict[str, Any],
    runtime_contract: Any,
) -> OracleOutcome:
    """Pinned reference-fold micro oracle (default: fail closed).

    The real oracle requires the pinned Comfy reference fold sources and a
    torch-backed five-tensor reference run; until those sources are vendored
    and verified, no SUPPORTED claim is produced (OFFLINE_FOLD_ORACLE_NOT_PASSED).
    """
    return OracleOutcome(
        passed=False,
        detail="pinned Comfy reference fold oracle is not bound; no SUPPORTED claim",
    )


def _refuse(candidate_id: str, reason_code: str, stage: str, detail: str) -> LoRACompatibilityVerdict:
    return LoRACompatibilityVerdict(
        candidate_id=candidate_id,
        verdict=UNSUPPORTED,
        reason_code=reason_code,
        evidence={"stage": stage, "detail": detail},
    )


def preflight_candidate(
    candidate_id: str,
    package_root: Path | str,
    candidate_path: Path | str,
    *,
    pinned_sha256: str,
    pinned_bytes: int,
    runtime_contract_resolver: Callable[..., Any] | None = None,
) -> LoRACompatibilityVerdict:
    """Run the fail-closed LoRA compatibility preflight (read-only).

    ``runtime_contract_resolver`` defaults to the integration package
    contract validator and is injected for host-free tests.
    """
    candidate = Path(candidate_path)
    try:
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        size = candidate.stat().st_size
        if digest != pinned_sha256 or size != pinned_bytes:
            return _refuse(
                candidate_id,
                contract.CANDIDATE_PIN_MISMATCH,
                contract.STAGE_CANDIDATE_PIN,
                f"sha256 {digest} size {size} differ from pin",
            )
        header = _read_safetensors_header(candidate)
        census = _candidate_census(header)
        try:
            _validate_census(census)
        except PreflightError as exc:
            return _refuse(
                candidate_id,
                contract.CANDIDATE_TENSOR_CENSUS_MISMATCH,
                contract.STAGE_CANDIDATE_CENSUS,
                str(exc),
            )
        try:
            _validate_profile(header)
        except PreflightError as exc:
            return _refuse(
                candidate_id,
                contract.CANDIDATE_PROFILE_UNSUPPORTED,
                contract.STAGE_CANDIDATE_PROFILE,
                str(exc),
            )
        try:
            _validate_target_mapping(census)
        except PreflightError as exc:
            return _refuse(
                candidate_id,
                contract.TARGET_MODULE_MAPPING_UNRESOLVED,
                contract.STAGE_TARGET_MAPPING,
                str(exc),
            )
        package = Path(package_root)
        transformer_dir = package / "Ref2VA" / "transformer"
        if not transformer_dir.is_dir():
            transformer_dir = package / "transformer"
        try:
            base_catalog = _bind_official_base_catalog(transformer_dir)
        except PreflightError as exc:
            return _refuse(
                candidate_id,
                contract.BASE_REPRESENTATION_UNBINDABLE,
                contract.STAGE_BASE_REPRESENTATION,
                str(exc),
            )
        resolver = runtime_contract_resolver
        if resolver is None:
            from comfy_omni.integrations.vllm_omni.package_contract import validate_runtime_package

            resolver = validate_runtime_package
        runtime_contract = resolver(package)
        outcome = _run_offline_fold_oracle(census, base_catalog, runtime_contract)
        if not outcome.passed:
            return _refuse(
                candidate_id,
                contract.OFFLINE_FOLD_ORACLE_NOT_PASSED,
                contract.STAGE_OFFLINE_ORACLE,
                outcome.detail or "reference-fold oracle did not pass",
            )
        return LoRACompatibilityVerdict(
            candidate_id=candidate_id,
            verdict=SUPPORTED,
            reason_code=None,
            evidence={"stage": "complete", "detail": "all preflight stages passed"},
        )
    except (OSError, PreflightError) as exc:
        return _refuse(
            candidate_id,
            contract.CANDIDATE_PIN_MISMATCH,
            contract.STAGE_CANDIDATE_PIN,
            f"candidate unreadable: {exc}",
        )


__all__ = [
    "COMFY_OMNI_LORA_COMPATIBILITY_SCHEMA",
    "SUPPORTED",
    "UNSUPPORTED",
    "OracleOutcome",
    "LoRACompatibilityVerdict",
    "preflight_candidate",
    "_bind_official_base_catalog",
    "_run_offline_fold_oracle",
]
