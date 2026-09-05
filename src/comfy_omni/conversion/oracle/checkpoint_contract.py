"""Checkpoint-only observations; none authorize LoRA activation or conversion."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from comfy_omni.contracts.models import ContractError

CHECKPOINT_PREFLIGHT_SCHEMA = "comfy_omni.lora-checkpoint-preflight/v1"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


class CheckpointInputError(ContractError):
    """Invalid request/input identity or structure, not a compatibility verdict."""

    def __init__(self, role: str, reason_code: str, **evidence: Any) -> None:
        super().__init__(f"{role}: {reason_code}")
        self.role = role
        self.reason_code = reason_code
        self.evidence = _freeze(evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CHECKPOINT_PREFLIGHT_SCHEMA,
            "status": "INVALID_INPUT",
            "role": self.role,
            "reason_code": self.reason_code,
            "evidence": _thaw(self.evidence),
        }


@dataclass(frozen=True)
class CheckpointPin:
    sha256: str
    size: int

    def __post_init__(self) -> None:
        if not isinstance(self.sha256, str) or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise CheckpointInputError("request", "CHECKPOINT_PIN_INVALID")
        if type(self.size) is not int or self.size < 8:
            raise CheckpointInputError("request", "CHECKPOINT_PIN_INVALID")


@dataclass(frozen=True)
class CheckpointPreflightReceipt:
    """A deeply immutable refusal with independently serializable observations."""

    candidate_id: str
    reason_code: str
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", _freeze(self.evidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CHECKPOINT_PREFLIGHT_SCHEMA,
            "candidate_id": self.candidate_id,
            "status": "UNSUPPORTED",
            "reason_code": self.reason_code,
            "evidence": _thaw(self.evidence),
        }
