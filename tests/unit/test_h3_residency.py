from __future__ import annotations

import pytest

from comfy_omni.runtime.hotel import (
    H3DitResidency,
    H3ResidencyError,
    H3ResidencyPhase,
    H3TensorDescriptor,
    PreparedH3DitSelection,
)


def _selection(name: str, *, profile: str = "h3-beta4") -> PreparedH3DitSelection:
    return PreparedH3DitSelection(
        selection=name,
        identity=f"sha256:{name}",
        execution_profile=profile,
        tensors=(
            H3TensorDescriptor(name="block.weight", shape=(2, 2), dtype="torch.float32"),
            H3TensorDescriptor(name="block.bias", shape=(2,), dtype="torch.float32"),
        ),
        logical_bytes=24,
    )


def test_active_identity_changes_only_after_finalize() -> None:
    state = H3DitResidency(active=_selection("a"))

    state.prepare("tx-1", _selection("b"), cpu_cache_budget_bytes=0)
    assert state.phase is H3ResidencyPhase.PREPARED
    assert state.active.selection == "a"

    state.mark_committed("tx-1")
    assert state.phase is H3ResidencyPhase.COMMITTED
    assert state.active.selection == "a"

    state.finalize("tx-1")
    assert state.phase is H3ResidencyPhase.IDLE
    assert state.active.selection == "b"
    assert state.transaction_id is None


def test_execution_profile_is_part_of_compatibility() -> None:
    state = H3DitResidency(active=_selection("a"))

    with pytest.raises(H3ResidencyError, match="execution profile"):
        state.prepare("tx-1", _selection("b", profile="h3-beta4-different-forward"), cpu_cache_budget_bytes=0)

    assert state.phase is H3ResidencyPhase.IDLE
    assert state.active.selection == "a"


def test_descriptor_mismatch_is_rejected_before_transaction_starts() -> None:
    target = PreparedH3DitSelection(
        selection="b",
        identity="sha256:b",
        execution_profile="h3-beta4",
        tensors=(H3TensorDescriptor(name="block.weight", shape=(4, 1), dtype="torch.float32"),),
        logical_bytes=16,
    )
    state = H3DitResidency(active=_selection("a"))

    with pytest.raises(H3ResidencyError, match="tensor descriptor"):
        state.prepare("tx-1", target, cpu_cache_budget_bytes=0)

    assert state.phase is H3ResidencyPhase.IDLE


def test_positive_cache_budget_must_cover_the_complete_selection() -> None:
    state = H3DitResidency(active=_selection("a"))

    with pytest.raises(H3ResidencyError, match="CPU cache budget"):
        state.prepare("tx-1", _selection("b"), cpu_cache_budget_bytes=23)

    assert state.phase is H3ResidencyPhase.IDLE


def test_rollback_is_idempotent_and_allows_a_new_transaction() -> None:
    state = H3DitResidency(active=_selection("a"))
    state.prepare("tx-1", _selection("b"), cpu_cache_budget_bytes=0)
    state.mark_committed("tx-1")

    state.mark_rolled_back("tx-1")
    state.mark_rolled_back("tx-1")
    assert state.phase is H3ResidencyPhase.ROLLED_BACK
    assert state.active.selection == "a"

    state.prepare("tx-2", _selection("b"), cpu_cache_budget_bytes=0)
    assert state.transaction_id == "tx-2"
    assert state.phase is H3ResidencyPhase.PREPARED
