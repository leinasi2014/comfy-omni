# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: h3-forge contributors
"""One stdlib deferred-import helper for the root host API bootstrap phase.

Derived from h3-forge e9cb011d00b028c149db3978de246c54f6e34acc,
src/h3_forge/_import_hook.py blob 935e30d22558a3e9b9065421423e36cd101e35df.
It observes completed imports without importing the target; the coordinator
owns process gating, contribution retries and registration state.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import logging
import os
import sys
import threading
import time
from collections.abc import Callable
from types import ModuleType
from typing import Any

_LOG = logging.getLogger(__name__)
_LOCK = threading.RLock()
_FINDERS: dict[str, _AfterImportFinder] = {}
#: Watcher poll granularity while a target module's body is still executing.
_WATCH_POLL = 0.005


def _reset_after_fork() -> None:
    """Discard the parent's pending callbacks and possibly held thread lock."""
    global _LOCK
    _LOCK = threading.RLock()
    for finder in tuple(_FINDERS.values()):
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
        finder.callbacks.clear()
    _FINDERS.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def _module_ready(module: ModuleType) -> bool:
    """True when a resident module's body has finished executing successfully.

    The authoritative signal is ``module.__spec__._initializing``: importlib
    sets it before running the loader's ``exec_module`` and clears it in its
    ``finally`` (verified against the running interpreter), so it is exactly
    the "body completed" edge. A failed import also clears this flag after
    removing the module: recheck resident identity after reading the flag.
    Modules without a usable spec (synthetic
    ``ModuleType`` stubs, namespace edge cases) count as ready -- there is
    no initialization window to wait out.
    """
    spec = getattr(module, "__spec__", None)
    finished = spec is None or not getattr(spec, "_initializing", False)
    return finished and sys.modules.get(module.__name__) is module


def _await_initialization(module_name: str) -> None:
    """Deliver a pending registration once an in-flight import's body finishes.

    Runs on a short-lived daemon thread (at most one per target module):
    polls the completion signal and completes the armed finder -- firing
    every pending callback exactly once -- the moment the module body has
    finished.  Exits silently when the module vanished from ``sys.modules``
    (its import failed or aborted): the armed finder then owns delivery for
    the next import attempt through the wrapped loader.
    """
    while True:
        module = sys.modules.get(module_name)
        if module is None:
            return  # the in-flight import aborted: the armed finder owns the retry
        if _module_ready(module):
            finder = _FINDERS.get(module_name)
            if finder is not None:
                finder.complete(module)
            return
        time.sleep(_WATCH_POLL)


def _log_callback_failure(module_name: str) -> None:
    """Report an opportunistic-retirement callback failure without raising.

    This path runs inside ``find_spec`` of an **unrelated** import; a raising
    callback would corrupt that import, so the failure is logged instead and
    the retirement still stands (the dominant delivery paths -- the wrapped
    loader and the registration-time checks -- propagate callback errors
    normally).
    """
    _LOG.exception("after_import callback for %r failed during opportunistic retirement", module_name)


class _AfterImportLoader(importlib.abc.Loader):
    def __init__(self, loader: importlib.abc.Loader, finder: _AfterImportFinder) -> None:
        self._loader = loader
        self._finder = finder

    def create_module(self, spec: Any) -> ModuleType | None:
        creator = getattr(self._loader, "create_module", None)
        return None if creator is None else creator(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._loader.exec_module(module)
        # Delivered by construction after the body: exec_module has returned,
        # so the module is fully initialized even though the interpreter has
        # not cleared spec._initializing yet (that happens in _load's finally).
        self._finder.complete(module)


class _AfterImportFinder(importlib.abc.MetaPathFinder):
    def __init__(self, module_name: str) -> None:
        self.module_name = module_name
        self.callbacks: list[Callable[[ModuleType], None]] = []
        self.watch_armed = False  # an initialization watcher thread was started

    def find_spec(self, fullname: str, path: Any, target: ModuleType | None = None) -> Any:
        if fullname != self.module_name:
            # Opportunistic retirement (S2-1): a target import that resolved
            # its spec before this finder was installed never comes back
            # through find_spec, so this registration would be stranded.
            # Every import in the process passes through here, so the very
            # next one retires us -- but only when the target's body has
            # actually finished (a mid-body module is not delivered, the
            # watcher owns it).  Callback failures are logged, never raised
            # into the unrelated import being served.
            loaded = sys.modules.get(self.module_name)
            if loaded is not None and _module_ready(loaded):
                try:
                    self.complete(loaded)
                except Exception:  # noqa: BLE001 - never break an unrelated import
                    _log_callback_failure(self.module_name)
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _AfterImportLoader(spec.loader, self)
        return spec

    def complete(self, module: ModuleType) -> None:
        with _LOCK:
            # A watcher may hold an object whose import failed meanwhile.
            # Keep the finder armed for the next real import in that case.
            if sys.modules.get(self.module_name) is not module:
                return
            if self in sys.meta_path:
                sys.meta_path.remove(self)
            _FINDERS.pop(self.module_name, None)
            callbacks = tuple(self.callbacks)
            self.callbacks.clear()
        for callback in callbacks:
            callback(module)


def after_import(module_name: str, callback: Callable[[ModuleType], None]) -> None:
    """Run ``callback(module)`` now, or exactly once after ``module_name`` imports.

    "Imported" means the module's **body has finished executing** -- not
    merely "present in ``sys.modules``" (the object is inserted before the
    body runs).  Delivery decisions are therefore gated on the spec's
    ``_initializing`` completion signal:

    * a fully-imported module fires immediately (absorbing and retiring any
      armed finder, which closes the S2-1 race);
    * an absent module arms the wrapped-loader finder for the next import;
    * a present-but-initializing module (another thread's import is
      mid-body, and that import resolved its spec without our loader, so
      nothing on the import path can deliver) arms the finder plus a one-shot
      watcher thread that fires the callbacks the moment the body finishes;
      a body that raises makes the module vanish and leaves the finder armed
      for the retry import.

    The whole check-and-install sequence runs under one lock critical
    section, and the registry is re-checked **after** the finder is
    installed: a concurrent import that completed between the first check
    and the install is detected right away (retiring the finder if the
    module is ready, arming the watcher if it is still initializing), so no
    interleaving can strand a pending callback or leak a resident finder.

    Registration installs at most one finder per target module (repeated
    calls append their callback to the existing finder).  Because a finder
    only ever retires when its target imports, callers that arm hooks for a
    module some processes never import must gate the arming on those
    processes themselves (see :func:`comfy_omni.integrations.vllm_omni.bootstrap._is_root_process`, the
    single in-tree user): the deferred finder is only armed in the process
    that can actually import the target (the API server), never in workers
    or engine cores, so no process carries a finder residue.
    """
    fire: ModuleType | None = None
    complete: _AfterImportFinder | None = None
    with _LOCK:
        loaded = sys.modules.get(module_name)
        if loaded is not None and _module_ready(loaded):
            finder = _FINDERS.get(module_name)
            if finder is not None:
                # Already loaded while a finder is armed: the import finished
                # without ever consulting the finder (the S2-1 race window),
                # so absorb this callback and retire the registration in one
                # exactly-once completion below.
                if callback not in finder.callbacks:
                    finder.callbacks.append(callback)
                complete = finder
            else:
                fire = loaded  # fully imported: fire now, no finder ever armed
        else:
            # Absent (the wrapped loader will deliver) or present but still
            # executing its body (the watcher thread will deliver).
            finder = _FINDERS.get(module_name)
            if finder is None:
                finder = _AfterImportFinder(module_name)
                _FINDERS[module_name] = finder
                sys.meta_path.insert(0, finder)
            if callback not in finder.callbacks:
                finder.callbacks.append(callback)
            if loaded is None:
                # Post-install recheck (same critical section): the module may
                # have been imported concurrently between the first check and
                # the install -- ready means retire now, still-initializing
                # means the watcher below takes over.
                loaded = sys.modules.get(module_name)
                if loaded is not None and _module_ready(loaded):
                    complete = finder
            if loaded is not None and complete is None and not finder.watch_armed:
                finder.watch_armed = True
                threading.Thread(
                    target=_await_initialization,
                    args=(module_name,),
                    daemon=True,
                    name=f"comfy-omni-after-import-{module_name}",
                ).start()
    if complete is not None:
        complete.complete(loaded)  # type: ignore[arg-type] - loaded is not None here
        return
    if fire is not None:
        if sys.modules.get(module_name) is fire:
            callback(fire)  # already imported: fire now, no finder ever armed
        else:
            # The ready object was replaced/removed after the lock was released.
            after_import(module_name, callback)


__all__ = ["after_import"]
