"""Stable schema, verdict, and reason-code registry for the LoRA compatibility preflight.

This is NEW contract code for issue #12 slice 1.  It owns the fail-closed verdict
shape and the frozen reason codes.  It must never import the migrated legacy oracle
modules nor Torch, vLLM, FastAPI, or any conversion/runtime primitive.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from comfy_omni.contracts.models import ContractError

COMFY_OMNI_LORA_COMPATIBILITY_SCHEMA = "comfy_omni.lora-compatibility/v1"

# Verdict outcomes.
SUPPORTED = "SUPPORTED"
UNSUPPORTED = "UNSUPPORTED"

# Fail-closed reason codes frozen in issue #12.
CANDIDATE_PIN_MISMATCH = "CANDIDATE_PIN_MISMATCH"
CANDIDATE_PROFILE_UNSUPPORTED = "CANDIDATE_PROFILE_UNSUPPORTED"
CANDIDATE_TENSOR_CENSUS_MISMATCH = "CANDIDATE_TENSOR_CENSUS_MISMATCH"
TARGET_MODULE_MAPPING_UNRESOLVED = "TARGET_MODULE_MAPPING_UNRESOLVED"
QUANT_LAYOUT_INCOMPATIBLE = "QUANT_LAYOUT_INCOMPATIBLE"
BASE_REPRESENTATION_UNBINDABLE = "BASE_REPRESENTATION_UNBINDABLE"
ORACLE_BASE_CONTRACT_NOT_BINDING = "ORACLE_BASE_CONTRACT_NOT_BINDING"
OFFLINE_FOLD_ORACLE_NOT_PASSED = "OFFLINE_FOLD_ORACLE_NOT_PASSED"


@dataclass(frozen=True)
class LoRACompatibilityVerdict:
    """One immutable, evidence-bound LoRA compatibility verdict."""

    candidate_id: str
    verdict: Literal["SUPPORTED", "UNSUPPORTED"]
    reason_code: str | None
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Freeze caller-owned evidence so the verdict cannot be mutated after creation."""
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready, status-bearing representation."""
        return {
            "schema": COMFY_OMNI_LORA_COMPATIBILITY_SCHEMA,
            "candidate_id": self.candidate_id,
            "verdict": self.verdict,
            "status": self.verdict,
            "reason_code": self.reason_code,
            "evidence": dict(self.evidence),
        }


class LoRAOracleError(ContractError):
    """A stable exception shape for LoRA-oracle failures that need to raise."""


__all__ = [
    "COMFY_OMNI_LORA_COMPATIBILITY_SCHEMA",
    "SUPPORTED",
    "UNSUPPORTED",
    "CANDIDATE_PIN_MISMATCH",
    "CANDIDATE_PROFILE_UNSUPPORTED",
    "CANDIDATE_TENSOR_CENSUS_MISMATCH",
    "TARGET_MODULE_MAPPING_UNRESOLVED",
    "QUANT_LAYOUT_INCOMPATIBLE",
    "BASE_REPRESENTATION_UNBINDABLE",
    "ORACLE_BASE_CONTRACT_NOT_BINDING",
    "OFFLINE_FOLD_ORACLE_NOT_PASSED",
    "LoRACompatibilityVerdict",
    "LoRAOracleError",
]
