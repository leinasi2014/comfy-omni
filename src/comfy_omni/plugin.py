"""Lightweight vLLM-Omni general-plugin entry point.

This walking-skeleton entry owns registration lifecycle only. It intentionally registers no
profiles, routes, runtime architectures, or host patches until those capabilities migrate through
their own compatibility-reviewed slices.
"""

from __future__ import annotations

from threading import RLock

_registration_lock = RLock()
_NEW = 0
_REGISTERING = 1
_REGISTERED = 2
_registration_state = _NEW


def register() -> None:
    """Mark the empty bootstrap complete exactly once.

    The function is thread-safe, re-entry-safe, and performs no model, checkpoint, filesystem,
    network, host-runtime, or optional-dependency I/O.
    """

    global _registration_state
    if _registration_state != _NEW:
        return
    with _registration_lock:
        if _registration_state != _NEW:
            return
        _registration_state = _REGISTERING
        try:
            # Capability contributions will be delegated here by later migration slices.
            _registration_state = _REGISTERED
        except BaseException:
            _registration_state = _NEW
            raise


__all__ = ["register"]
