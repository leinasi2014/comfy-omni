"""One coordinator for lazy runtime registration and deferred API wiring.

This module owns the single-process registration state machine for the vLLM-Omni host adapter.
It is intended to be imported without pulling heavy runtime dependencies: every module-level
import here is stdlib only. Host APIs are resolved only by the deferred root-process
phase; worker registration never imports API routes or model implementations.

Provenance: characterized from ``h3-forge@e9cb011d00b028c149db3978de246c54f6e34acc``
``src/h3_forge/plugin.py`` blob ``304a776bf4daf1f7a28b1bc6192d320da30421fd`` (resident-host rule,
silent deferral/retry, latch semantics, lazy-string registration, ``_is_root_process`` gating,
every-process loading). The deferred helper has separate exact-source attribution.
No legacy sub-plugin registration chain is retained.
"""

from __future__ import annotations

import importlib
import multiprocessing
import os
import sys
import types
from threading import RLock

_NEW = 0
_REGISTERING = 1
_REGISTERED = 2
_WAITING = 3

_registration_lock = RLock()
_registration_state = _NEW
_api_state = _NEW

_HOST_MODULE_NAME = "vllm_omni"
_REGISTRY_MODULE_NAME = "vllm_omni.diffusion.registry"
API_SERVER_MODULE = "vllm_omni.entrypoints.openai.api_server"

# Additional API owners can contribute here; no route is a placeholder.
_API_CONTRIBUTIONS = (
    ("comfy_omni.integrations.vllm_omni.api_phase", "mount_components"),
    ("comfy_omni.integrations.vllm_omni.api_phase", "mount_runtime"),
)


def _reset_after_fork() -> None:
    """A worker inherits registered strings, never a pending root API phase."""
    global _registration_lock, _registration_state, _api_state
    _registration_lock = RLock()
    if _registration_state == _REGISTERING:
        _registration_state = _NEW
    _api_state = _NEW


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)

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
        "comfy_omni.integrations.vllm_omni.pipelines.runtime_pipeline",
        "H3ComfyMiniMaxH3Pipeline",
        "get_minimax_h3_post_process_func",
    ),
)


def _is_root_process() -> bool:
    """Only the root host process can arm its API-server import callback."""
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


def _register_apis(module: types.ModuleType) -> None:
    """Complete the sole API phase, retaining runtime success on retry."""
    global _api_state
    with _registration_lock:
        if _api_state in {_REGISTERING, _REGISTERED}:
            return
        if not _is_root_process():
            _api_state = _NEW
            return
        _api_state = _REGISTERING
        try:
            for module_name, function_name in _API_CONTRIBUTIONS:
                owner = importlib.import_module(module_name)
                getattr(owner, function_name)(module)
            _api_state = _REGISTERED
        except BaseException:
            _api_state = _NEW
            raise


def _arm_api_phase() -> None:
    global _api_state
    if _api_state != _NEW or _HOST_MODULE_NAME not in sys.modules or not _is_root_process():
        return
    _api_state = _WAITING
    try:
        from .deferred_import import after_import

        after_import(API_SERVER_MODULE, _register_apis)
    except BaseException:
        _api_state = _NEW
        raise


def register() -> None:
    """Advance resident-host runtime and root-only API phases under one lock.

    Thread-safe and re-entry-safe. Deferral paths (no resident host, or a resident but partial
    host) return without latching and without raising so a later call retries. An exception during
    Runtime re-entry is a no-op while registration is active. A deferred API
    phase is retried even after runtime registration has already succeeded.
    """
    global _registration_state

    with _registration_lock:
        if _registration_state == _REGISTERING:
            return
        if _registration_state != _REGISTERED:
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
        _arm_api_phase()


__all__ = ["API_SERVER_MODULE", "register"]
