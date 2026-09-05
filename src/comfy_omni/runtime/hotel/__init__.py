"""Bounded multi-engine orchestration and residency lifecycle.

This module intentionally has no Torch or vLLM imports.  The descriptors are
safe to use from package discovery and control-plane code; tensor access stays
behind the worker-local trusted pipeline hooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class H3ResidencyError(RuntimeError):
    """A rejected or failed H3 DiT residency transition."""


class H3ResidencyPhase(str, Enum):
    """Worker-local phase of one controller-coordinated transaction."""

    IDLE = "idle"
    PREPARED = "prepared"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    POISONED = "poisoned"


@dataclass(frozen=True, slots=True)
class H3TensorDescriptor:
    """Payload-free identity of one logical tensor accepted by the DiT loader."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    kind: str = "parameter"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tensor descriptor name must not be empty")
        if not self.dtype:
            raise ValueError("tensor descriptor dtype must not be empty")
        if not self.kind:
            raise ValueError("tensor descriptor kind must not be empty")
        if any(not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in self.shape):
            raise ValueError("tensor descriptor shape must contain non-negative integers")


@dataclass(frozen=True, slots=True)
class PreparedH3DitSelection:
    """Validated registered selection without any tensor payload."""

    selection: str
    identity: str
    execution_profile: str
    tensors: tuple[H3TensorDescriptor, ...]
    logical_bytes: int

    def __post_init__(self) -> None:
        if not self.selection:
            raise ValueError("selection must not be empty")
        if not self.identity:
            raise ValueError("selection identity must not be empty")
        if not self.execution_profile:
            raise ValueError("execution profile must not be empty")
        if not self.tensors:
            raise ValueError("selection must describe at least one tensor")
        names = [tensor.name for tensor in self.tensors]
        if len(names) != len(set(names)):
            raise ValueError("selection tensor descriptor names must be unique")
        if not isinstance(self.logical_bytes, int) or isinstance(self.logical_bytes, bool) or self.logical_bytes < 0:
            raise ValueError("selection logical_bytes must be a non-negative integer")

    def tensor_signature(self) -> tuple[tuple[str, tuple[int, ...], str, str], ...]:
        """Return an order-independent signature for in-place load compatibility."""
        return tuple(sorted((item.name, item.shape, item.dtype, item.kind) for item in self.tensors))


@dataclass(slots=True)
class H3DitResidency:
    """Payload-free state machine for one live H3 DiT module."""

    active: PreparedH3DitSelection
    phase: H3ResidencyPhase = H3ResidencyPhase.IDLE
    transaction_id: str | None = None
    target: PreparedH3DitSelection | None = None
    cpu_cache_budget_bytes: int = 0
    poison_reason: str | None = None
    _rolled_back_transaction_id: str | None = field(default=None, repr=False)

    def prepare(
        self,
        transaction_id: str,
        target: PreparedH3DitSelection,
        *,
        cpu_cache_budget_bytes: int,
    ) -> None:
        """Validate and record a target without changing model memory."""
        self._require_healthy()
        if self.phase not in (H3ResidencyPhase.IDLE, H3ResidencyPhase.ROLLED_BACK):
            raise H3ResidencyError(f"cannot prepare while residency phase is {self.phase.value}")
        if not isinstance(transaction_id, str) or not transaction_id:
            raise H3ResidencyError("transaction_id must be a non-empty string")
        if not isinstance(target, PreparedH3DitSelection):
            raise H3ResidencyError("trusted prepare hook returned an invalid selection descriptor")
        if not isinstance(cpu_cache_budget_bytes, int) or isinstance(cpu_cache_budget_bytes, bool):
            raise H3ResidencyError("CPU cache budget must be an integer")
        if cpu_cache_budget_bytes < 0:
            raise H3ResidencyError("CPU cache budget must not be negative")
        if target.execution_profile != self.active.execution_profile:
            raise H3ResidencyError(
                "execution profile is incompatible with the active H3 DiT: "
                f"{target.execution_profile!r} != {self.active.execution_profile!r}"
            )
        if target.tensor_signature() != self.active.tensor_signature():
            raise H3ResidencyError("tensor descriptor is incompatible with the active H3 DiT")
        if target.logical_bytes != self.active.logical_bytes:
            raise H3ResidencyError("logical byte size is incompatible with the active H3 DiT")
        if 0 < cpu_cache_budget_bytes < target.logical_bytes:
            raise H3ResidencyError(
                f"CPU cache budget {cpu_cache_budget_bytes} cannot hold the complete "
                f"{target.logical_bytes}-byte selection"
            )

        self.phase = H3ResidencyPhase.PREPARED
        self.transaction_id = transaction_id
        self.target = target
        self.cpu_cache_budget_bytes = cpu_cache_budget_bytes
        self._rolled_back_transaction_id = None

    def mark_committed(self, transaction_id: str) -> None:
        self._require_transaction(transaction_id, H3ResidencyPhase.PREPARED)
        self.phase = H3ResidencyPhase.COMMITTED

    def mark_rolled_back(self, transaction_id: str) -> None:
        self._require_healthy()
        if self.phase is H3ResidencyPhase.ROLLED_BACK and transaction_id == self._rolled_back_transaction_id:
            return
        self._require_transaction(
            transaction_id,
            H3ResidencyPhase.PREPARED,
            H3ResidencyPhase.COMMITTED,
        )
        self.phase = H3ResidencyPhase.ROLLED_BACK
        self._rolled_back_transaction_id = transaction_id
        self.transaction_id = None
        self.target = None
        self.cpu_cache_budget_bytes = 0

    def finalize(self, transaction_id: str) -> None:
        self._require_transaction(transaction_id, H3ResidencyPhase.COMMITTED)
        assert self.target is not None
        self.active = self.target
        self.phase = H3ResidencyPhase.IDLE
        self.transaction_id = None
        self.target = None
        self.cpu_cache_budget_bytes = 0
        self._rolled_back_transaction_id = None

    def poison(self, reason: str) -> None:
        self.phase = H3ResidencyPhase.POISONED
        self.poison_reason = reason or "unknown H3 residency failure"

    def status(self) -> dict[str, object]:
        return {
            "phase": self.phase.value,
            "active_selection": self.active.selection,
            "active_identity": self.active.identity,
            "execution_profile": self.active.execution_profile,
            "transaction_id": self.transaction_id,
            "target_selection": self.target.selection if self.target is not None else None,
            "target_identity": self.target.identity if self.target is not None else None,
            "cpu_cache_budget_bytes": self.cpu_cache_budget_bytes,
            "poison_reason": self.poison_reason,
        }

    def _require_healthy(self) -> None:
        if self.phase is H3ResidencyPhase.POISONED:
            raise H3ResidencyError(f"H3 residency worker is poisoned: {self.poison_reason}")

    def _require_transaction(self, transaction_id: str, *phases: H3ResidencyPhase) -> None:
        self._require_healthy()
        if self.phase not in phases:
            expected = ", ".join(phase.value for phase in phases)
            raise H3ResidencyError(f"transaction requires phase {expected}; current phase is {self.phase.value}")
        if transaction_id != self.transaction_id:
            raise H3ResidencyError(
                f"transaction_id {transaction_id!r} does not match active transaction {self.transaction_id!r}"
            )


__all__ = [
    "H3DitResidency",
    "H3ResidencyError",
    "H3ResidencyPhase",
    "H3TensorDescriptor",
    "PreparedH3DitSelection",
]
