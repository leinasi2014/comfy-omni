"""Frozen vLLM-Omni bootstrap coordinator.

This module owns the single-process registration state machine for the vLLM-Omni host adapter.
It is intended to be imported without pulling heavy runtime dependencies: every module-level
import here is stdlib only, and no ``vllm_omni``, ``torch``, ``vllm``, or ``fastapi`` object is
ever imported by this coordinator.

Provenance: characterized from ``h3-forge@e9cb011d00b028c149db3978de246c54f6e34acc``
``src/h3_forge/plugin.py`` blob ``304a776bf4daf1f7a28b1bc6192d320da30421fd`` (resident-host rule,
silent deferral/retry, latch semantics, lazy-string registration, ``_is_root_process`` gating,
every-process loading). The legacy ``after_import`` API hook, REST routes, and pipeline
implementations are deliberately NOT migrated here.
"""

from __future__ import annotations

import importlib
import multiprocessing
import sys
import types
from threading import RLock

_NEW = 0
_REGISTERING = 1
_REGISTERED = 2

_registration_lock = RLock()
_registration_state = _NEW

_HOST_MODULE_NAME = "vllm_omni"
_REGISTRY_MODULE_NAME = "vllm_omni.diffusion.registry"

# Declarative, lazy-string architecture contributions. The pipeline modules are intentionally NOT
# imported here; the host resolves them at model-load time from the fully-qualified string paths.
_ARCHITECTURE_CONTRIBUTIONS = (
    (
        "MiniMaxH3Pipeline",
        "comfy_omni.integrations.vllm_omni.pipelines.runtime_pipeline",
        "H3ComfyMiniMaxH3Pipeline",
        "get_minimax_h3_post_process_func",
    ),
    (
        "MiniMaxH3DensePipeline",
        "comfy_omni.integrations.vllm_omni.pipelines.dense_pipeline",
        "MiniMaxH3DensePipeline",
        "get_minimax_h3_post_process_func",
    ),
)


def _is_root_process() -> bool:
    """Signal whether this is the root process.

    Documented hook for future API-server-only wiring. This slice arms nothing.
    """
    try:
        return multiprocessing.parent_process() is None
    except Exception:
        return True


def _resolve_registry() -> types.ModuleType | None:
    """Return the resident host registry module, or ``None`` to defer silently.

    A host is used only when already resident in ``sys.modules``; this coordinator never imports
    ``vllm_omni`` itself. The registry submodule is resolved from ``sys.modules`` first, falling
    back to an import only when the host package is importable. A resident but partial host
    (non-package, missing registry) defers by returning ``None``.
    """
    if _HOST_MODULE_NAME not in sys.modules:
        return None
    existing = sys.modules.get(_REGISTRY_MODULE_NAME)
    if existing is not None:
        return existing
    try:
        return importlib.import_module(_REGISTRY_MODULE_NAME)
    except ModuleNotFoundError:
        return None


def register() -> None:
    """Register the frozen architecture contributions exactly once.

    Thread-safe and re-entry-safe. Deferral paths (no resident host, or a resident but partial
    host) return without latching and without raising so a later call retries. An exception during
    registration resets state to ``NEW`` and re-raises.
    """
    global _registration_state

    if _registration_state != _NEW:
        return
    with _registration_lock:
        if _registration_state != _NEW:
            return
        _registration_state = _REGISTERING
        try:
            registry = _resolve_registry()
            if registry is None:
                _registration_state = _NEW
                return
            for architecture_id, module_path, class_name, post_process_func_name in _ARCHITECTURE_CONTRIBUTIONS:
                registry.register_diffusion_model(
                    architecture_id,
                    module_path,
                    class_name,
                    post_process_func_name=post_process_func_name,
                )
            _registration_state = _REGISTERED
        except BaseException:
            _registration_state = _NEW
            raise


__all__ = ["register"]
