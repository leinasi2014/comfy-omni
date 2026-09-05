"""One worker lock and success latches for scoped H3 host class substitutions.

Characterized from h3-forge e9cb011 runtime_pipeline.py blob
fa94f86da746ff9a11105584081464c1162d07b6 (Apache-2.0).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock

CONSTRUCTION_LOCK = RLock()


@dataclass
class WorkerConstructionState:
    model_constructed: bool = False
    pipeline_constructed: bool = False
    model_pending: bool = False
    pipeline_pending: bool = False


state = WorkerConstructionState()


@contextmanager
def construction(kind: str):
    if kind not in {"model", "pipeline"}:
        raise ValueError("unknown H3 construction scope")
    with CONSTRUCTION_LOCK:
        if getattr(state, f"{kind}_constructed"):
            raise RuntimeError(f"h3-forge permits exactly one H3 {kind} per worker process")
        if getattr(state, f"{kind}_pending"):
            raise RuntimeError(f"nested H3 {kind} construction is forbidden")
        setattr(state, f"{kind}_pending", True)
        try:
            yield
        except BaseException:
            # A completed inner model remains latched if its pipeline fails.
            raise
        else:
            setattr(state, f"{kind}_constructed", True)
        finally:
            setattr(state, f"{kind}_pending", False)


@contextmanager
def substitute(module, **replacements):
    """Restore the exact original objects even when host construction raises."""
    with CONSTRUCTION_LOCK:
        originals = {name: getattr(module, name) for name in replacements}
        try:
            for name, value in replacements.items():
                setattr(module, name, value)
            yield
        finally:
            for name, value in originals.items():
                setattr(module, name, value)
