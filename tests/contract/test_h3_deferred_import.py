# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: h3-forge contributors
# Source: h3-forge e9cb011d00b028c149db3978de246c54f6e34acc
# tests/test_import_hook.py blob d07d30caa47426614ba2726944ed37c120d0fa62.
"""Tests for the one-shot after_import hook (SERVABLE step 4, deliverable A).

The hook is exercised against **real imports**: each test writes a uniquely
named module file into a per-test temp directory, puts the directory on
``sys.path`` and imports it through the normal machinery, so the
MetaPathFinder/loader-wrapper path (not a simulation of it) is what runs.
Every test restores ``sys.path`` / ``sys.modules`` / ``sys.meta_path`` in
cleanup, so the one global finder registry never leaks between tests.

The S2-1 regression class locks the race the old unlocked check allowed
(a concurrent import completing between the ``sys.modules`` check and the
finder install stranded the callback and permanently parked the finder):
under concurrent import + registration the callback must fire exactly once
and no finder may survive.
"""

from __future__ import annotations

import importlib
import itertools
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from comfy_omni.integrations.vllm_omni import deferred_import as _import_hook


class _HookTestCase(unittest.TestCase):
    """Base: per-test temp dir on sys.path + full import-state restoration."""

    _counter = itertools.count()

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self._old_path = list(sys.path)
        sys.path.insert(0, str(self.tmp))
        self._armed_finders = list(sys.meta_path)
        self._tracked_modules: list[str] = []

        def _restore() -> None:
            for name in self._tracked_modules:
                sys.modules.pop(name, None)
            sys.path[:] = self._old_path
            # drop any finder the test left armed (a passing test arms none)
            for finder in list(sys.meta_path):
                if finder not in self._armed_finders:
                    sys.meta_path.remove(finder)
                    _import_hook._FINDERS.pop(finder.module_name, None)  # type: ignore[attr-defined]

        self.addCleanup(_restore)

    def _module_name(self) -> str:
        name = f"h3_forge_hook_stub_{next(_HookTestCase._counter)}"
        self._tracked_modules.append(name)
        return name

    def _write_module(self, name: str, body: str = "VALUE = 42\n") -> Path:
        path = self.tmp / f"{name}.py"
        path.write_text(body, encoding="utf-8")
        return path


class AfterImportTests(_HookTestCase):
    def test_fires_immediately_when_module_already_loaded(self) -> None:
        name = self._module_name()
        module = ModuleType(name)
        sys.modules[name] = module
        seen: list[ModuleType] = []
        _import_hook.after_import(name, seen.append)
        self.assertEqual(seen, [module])

    def test_defers_until_real_import_then_fires_exactly_once(self) -> None:
        name = self._module_name()
        self._write_module(name)
        seen: list[ModuleType] = []
        _import_hook.after_import(name, seen.append)
        self.assertEqual(seen, [])  # deferred: nothing fired yet

        module = importlib.import_module(name)
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0], module)
        self.assertEqual(module.VALUE, 42)  # module body ran before the callback

        importlib.import_module(name)  # cached re-import: no second fire
        self.assertEqual(len(seen), 1)

    def test_duplicate_registration_fires_the_callback_once(self) -> None:
        name = self._module_name()
        self._write_module(name)
        seen: list[ModuleType] = []
        _import_hook.after_import(name, seen.append)
        _import_hook.after_import(name, seen.append)  # same callable: deduplicated
        importlib.import_module(name)
        self.assertEqual(len(seen), 1)

    def test_multiple_callbacks_fire_in_registration_order(self) -> None:
        name = self._module_name()
        self._write_module(name)
        order: list[str] = []
        _import_hook.after_import(name, lambda _m: order.append("first"))
        _import_hook.after_import(name, lambda _m: order.append("second"))
        importlib.import_module(name)
        self.assertEqual(order, ["first", "second"])

    def test_finder_removes_itself_after_firing(self) -> None:
        name = self._module_name()
        self._write_module(name)
        _import_hook.after_import(name, lambda _m: None)
        armed = [finder for finder in sys.meta_path if finder not in self._armed_finders]
        self.assertEqual(len(armed), 1)
        importlib.import_module(name)
        self.assertNotIn(armed[0], sys.meta_path)
        self.assertNotIn(name, _import_hook._FINDERS)

    def test_only_the_target_module_is_observed(self) -> None:
        observed = self._module_name()
        other = self._module_name()
        self._write_module(observed)
        self._write_module(other, body="VALUE = 7\n")
        seen: list[ModuleType] = []
        _import_hook.after_import(observed, seen.append)
        importlib.import_module(other)  # a different module importing is a no-op
        self.assertEqual(seen, [])
        importlib.import_module(observed)
        self.assertEqual(len(seen), 1)

    def test_failed_import_keeps_the_callback_pending(self) -> None:
        name = self._module_name()
        self._write_module(name, body="raise RuntimeError('stub import failure')\n")
        seen: list[ModuleType] = []
        _import_hook.after_import(name, seen.append)
        with self.assertRaises(RuntimeError):
            importlib.import_module(name)
        self.assertEqual(seen, [])  # module body raised: no fire, no removal

        self._write_module(name)  # repair the module, re-import: fires once
        importlib.import_module(name)
        self.assertEqual(len(seen), 1)

    def test_module_imported_between_check_and_install_fires_anyway(self) -> None:
        """S2-1 deterministic core: a module that lands in ``sys.modules``
        without ever consulting the armed finder (another thread's import
        completed inside the check-to-install gap) must still fire every
        pending callback and retire the finder -- no stranded registration,
        no meta-path residue."""
        name = self._module_name()
        self._write_module(name)
        _import_hook.after_import(name, lambda _m: None)  # armed; module not yet imported
        armed = [f for f in sys.meta_path if f not in self._armed_finders]
        self.assertEqual(len(armed), 1)
        # the racing thread's import finished without the finder: the module
        # simply appears fully-initialized under its canonical name
        module = ModuleType(name)
        module.VALUE = 42
        sys.modules[name] = module
        self.assertIn(armed[0], sys.meta_path)  # the finder never saw that import ...
        seen: list[ModuleType] = []
        _import_hook.after_import(name, seen.append)  # any later registration re-checks
        self.assertEqual(seen, [module])  # ... and the re-check fired everything anyway
        self.assertNotIn(armed[0], sys.meta_path)  # finder retired, no residue
        self.assertNotIn(name, _import_hook._FINDERS)


class AfterImportConcurrencyRegressionTests(_HookTestCase):
    """S2-1: the concurrent import + late-registration race.

    The old implementation read ``sys.modules`` outside the lock, so an
    import completing between the check and the install produced exactly the
    poisoned state the QA observed (callback_count=0, finder_registered=True,
    meta_path_added=1, forever).  The lock-spanning check + install +
    post-install re-check must make every interleaving fire the callback
    exactly once and leave no finder behind.
    """

    def test_concurrent_import_and_registration_never_strands_the_callback(self) -> None:
        for round_index in range(30):
            with self.subTest(round=round_index):
                name = self._module_name()
                self._write_module(name, body=f"VALUE = {round_index}\n")
                importlib.invalidate_caches()  # fresh file: drop PathFinder's dir cache
                seen: list[ModuleType] = []

                def registrar(target: str = name, sink: list = seen) -> None:
                    # arrive while the import thread is in flight
                    _import_hook.after_import(target, sink.append)

                importer_started = threading.Event()

                def importer(target: str = name, started: threading.Event = importer_started) -> None:
                    started.set()
                    importlib.import_module(target)

                registrar_thread = threading.Thread(target=registrar)
                import_thread = threading.Thread(target=importer)
                import_thread.start()
                self.assertTrue(importer_started.wait(timeout=10))
                registrar_thread.start()  # race: check vs the in-flight import
                import_thread.join(timeout=10)
                registrar_thread.join(timeout=10)

                if not seen:
                    # The interleaving where the import resolved its spec
                    # before the finder was installed strands the
                    # registration until the NEXT import in the process
                    # passes through meta_path -- drive one and the stranded
                    # registration must retire and fire (never outlive it).
                    bystander = self._module_name()
                    self._write_module(bystander, body="VALUE = 0\n")
                    importlib.invalidate_caches()  # fresh file: drop PathFinder's dir cache
                    importlib.import_module(bystander)

                self.assertEqual(len(seen), 1, f"callback fired {len(seen)} times (round {round_index})")
                self.assertEqual(seen[0].VALUE, round_index)  # module body ran first
                # no residue: the finder retired however the race resolved
                # (scoped to this round's target: other suites may legally
                # keep their own long-lived registrations)
                self.assertNotIn(name, _import_hook._FINDERS)
                strays = [
                    f for f in sys.meta_path if isinstance(f, _import_hook._AfterImportFinder) and f.module_name == name
                ]
                self.assertEqual(strays, [], f"finder leaked (round {round_index})")

    def test_repeated_registration_while_import_in_flight_fires_exactly_once(self) -> None:
        name = self._module_name()
        self._write_module(name, body="import time\ntime.sleep(0.05)\nVALUE = 1\n")
        seen: list[ModuleType] = []
        import_done = threading.Event()

        def importer() -> None:
            importlib.import_module(name)
            import_done.set()

        thread = threading.Thread(target=importer)
        thread.start()
        # hammer the registration path while the module body sleeps
        for _ in range(50):
            _import_hook.after_import(name, seen.append)
            if import_done.is_set():
                break
        thread.join(timeout=10)
        import_done.wait(timeout=10)
        if not seen:  # all registrations landed before the import finished
            _import_hook.after_import(name, seen.append)
        self.assertEqual(len(seen), 1)
        self.assertNotIn(name, _import_hook._FINDERS)


class AfterImportInitializationRegressionTests(_HookTestCase):
    """S2 (QA2 finding 4): present in ``sys.modules`` is not imported.

    The module object is inserted into ``sys.modules`` *before* its body
    executes, so a registration landing while another thread's import is
    mid-body used to fire the callback on a half-initialized module (the
    module's READY marker absent, the import thread still alive).  Delivery
    must now wait for the authoritative completion signal
    (``spec._initializing`` clearing) -- pinned here with event barriers
    locking "the callback ran" against "the body finished"."""

    _BODY = """
import threading
import time

BODY_STARTED = threading.Event()
BODY_DONE = False
BODY_STARTED.set()
time.sleep(0.5)
BODY_DONE = True
"""

    def _import_in_thread(self, name: str) -> tuple[threading.Thread, list[BaseException], ModuleType]:
        errors: list[BaseException] = []

        def importer() -> None:
            try:
                importlib.import_module(name)
            except BaseException as exc:  # delivered to the test below
                errors.append(exc)

        thread = threading.Thread(target=importer)
        thread.start()
        # the module object lands in sys.modules before the body runs, and
        # BODY_STARTED appears a few statements into it: poll for the marker
        # (not merely presence) so the barrier is the body, not bookkeeping
        deadline = time.monotonic() + 10
        module: ModuleType | None = None
        while time.monotonic() < deadline:
            candidate = sys.modules.get(name)
            if candidate is not None and hasattr(candidate, "BODY_STARTED"):
                module = candidate
                break
            time.sleep(0.002)
        assert module is not None
        module.BODY_STARTED.wait(timeout=10)  # inside the body sleep
        return thread, errors, module

    def test_registration_during_module_body_fires_only_after_the_body_finished(self) -> None:
        name = self._module_name()
        self._write_module(name, body=self._BODY)
        thread, _errors, module = self._import_in_thread(name)

        delivered: list[ModuleType] = []
        second: list[ModuleType] = []
        _import_hook.after_import(name, delivered.append)
        _import_hook.after_import(name, second.append)  # repeated mid-body: still once
        # the half-initialized module was NOT delivered: the old behaviour
        # fired here immediately with BODY_DONE still False
        self.assertEqual(delivered, [])
        self.assertEqual(second, [])
        self.assertFalse(module.BODY_DONE)

        thread.join(timeout=10)
        deadline = time.monotonic() + 10
        while not delivered and time.monotonic() < deadline:
            time.sleep(0.002)
        self.assertEqual(len(delivered), 1)
        self.assertEqual(len(second), 1)  # every pending callback fired exactly once
        self.assertIs(delivered[0], module)
        self.assertIs(second[0], module)
        self.assertTrue(delivered[0].BODY_DONE)  # the body had finished at delivery
        # and the registration retired cleanly: no finder or registry residue
        self.assertNotIn(name, _import_hook._FINDERS)
        strays = [f for f in sys.meta_path if isinstance(f, _import_hook._AfterImportFinder) and f.module_name == name]
        self.assertEqual(strays, [])

    def test_mid_body_registration_on_a_failing_import_fires_on_the_retry_import(self) -> None:
        name = self._module_name()
        self._write_module(name, body=self._BODY + "\nraise RuntimeError('body failure')\n")
        thread, errors, _mid_body_module = self._import_in_thread(name)

        seen: list[ModuleType] = []
        _import_hook.after_import(name, seen.append)
        self.assertEqual(seen, [])  # nothing delivered on the half-initialized module

        thread.join(timeout=10)
        self.assertEqual(len(errors), 1)  # the body raised; the import failed
        self.assertIsInstance(errors[0], RuntimeError)
        time.sleep(0.05)  # the watcher observes the vanished module and exits
        self.assertEqual(seen, [])  # an aborted import never delivers

        # the armed finder owns the retry: repairing and re-importing fires
        # exactly once, after the (now clean) body finished
        self._write_module(name, body=self._BODY)
        importlib.invalidate_caches()
        importlib.import_module(name)
        self.assertEqual(len(seen), 1)
        self.assertTrue(seen[0].BODY_DONE)
        self.assertNotIn(name, _import_hook._FINDERS)

    def test_opportunistic_retirement_waits_for_the_body_too(self) -> None:
        """An unrelated import passing through find_spec during the target's
        body must not retire the finder onto a half-initialized module --
        the delivery still waits for the body to finish."""
        name = self._module_name()
        self._write_module(name, body=self._BODY)
        thread, _errors, module = self._import_in_thread(name)
        _import_hook.after_import(name, lambda _m: None)  # armed while the body runs

        # an unrelated import while the target's body is still executing: its
        # find_spec must observe the mid-body module and NOT retire onto it
        bystander = self._module_name()
        self._write_module(bystander, body="VALUE = 0\n")
        importlib.invalidate_caches()
        importlib.import_module(bystander)
        self.assertFalse(module.BODY_DONE)  # the body is provably still running
        self.assertIn(name, _import_hook._FINDERS)  # ... so nothing retired early

        thread.join(timeout=10)
        deadline = time.monotonic() + 10
        while name in _import_hook._FINDERS and time.monotonic() < deadline:
            time.sleep(0.002)
        self.assertNotIn(name, _import_hook._FINDERS)  # retired only after the body finished
        strays = [f for f in sys.meta_path if isinstance(f, _import_hook._AfterImportFinder) and f.module_name == name]
        self.assertEqual(strays, [])


class AfterImportStaleModuleRegressionTests(_HookTestCase):
    """A failed real import clears _initializing on an object it has removed.

    Event barriers place failure between taking the module reference and
    checking readiness. The target module and import failure are real; the
    patched readiness call only controls scheduling and delegates its result.
    """

    def _check_failed_import_window(self, *, watcher: bool) -> None:
        name, control_name = self._module_name(), self._module_name()
        control = ModuleType(control_name)
        control.started = threading.Event()
        control.release = threading.Event()
        sys.modules[control_name] = control
        self._write_module(
            name,
            body=(
                f"from {control_name} import started, release\n"
                "started.set()\n"
                "if not release.wait(10):\n"
                "    raise TimeoutError('import body was not released')\n"
                "raise RuntimeError('controlled stale import failure')\n"
            ),
        )
        import_errors: list[BaseException] = []
        registration_errors: list[BaseException] = []
        seen: list[ModuleType] = []
        captured, release_check = threading.Event(), threading.Event()

        def importer() -> None:
            try:
                importlib.import_module(name)
            except BaseException as exc:
                import_errors.append(exc)

        import_thread = threading.Thread(target=importer, daemon=True)
        checked_thread: threading.Thread | None = None
        import_thread.start()
        try:
            self.assertTrue(control.started.wait(timeout=10))
            target = sys.modules[name]
            self.assertTrue(target.__spec__._initializing)
            ready = _import_hook._module_ready
            checked_name = f"comfy-omni-after-import-{name}" if watcher else f"stale-registrar-{name}"

            def scheduled_ready(module: ModuleType) -> bool:
                if module is target and threading.current_thread().name == checked_name and not captured.is_set():
                    captured.set()  # caller already holds the soon-to-be-stale object
                    if not release_check.wait(timeout=10):
                        raise TimeoutError("readiness check was not released")
                return ready(module)

            def registrar() -> None:
                try:
                    _import_hook.after_import(name, seen.append)
                except BaseException as exc:
                    registration_errors.append(exc)

            with patch.object(_import_hook, "_module_ready", side_effect=scheduled_ready):
                try:
                    if watcher:
                        _import_hook.after_import(name, seen.append)
                        checked_thread = next(thread for thread in threading.enumerate() if thread.name == checked_name)
                    else:
                        checked_thread = threading.Thread(target=registrar, name=checked_name, daemon=True)
                        checked_thread.start()
                    self.assertTrue(captured.wait(timeout=10), "the readiness check must capture the live target")
                    control.release.set()
                    import_thread.join(timeout=10)
                    self.assertFalse(import_thread.is_alive())
                    self.assertEqual(len(import_errors), 1)
                    self.assertIsInstance(import_errors[0], RuntimeError)
                    self.assertEqual(str(import_errors[0]), "controlled stale import failure")
                    self.assertNotIn(name, sys.modules)
                    self.assertFalse(target.__spec__._initializing)
                    release_check.set()
                    checked_thread.join(timeout=10)
                    self.assertFalse(checked_thread.is_alive())
                finally:
                    control.release.set()
                    release_check.set()
                    import_thread.join(timeout=10)
                    if checked_thread is not None:
                        checked_thread.join(timeout=10)
            self.assertEqual(registration_errors, [])
            self.assertEqual(seen, [], "a removed module from a failed import must never be delivered")
            self.assertIn(name, _import_hook._FINDERS, "failed import must retain pending delivery for retry")
            self._write_module(name, body="VALUE = 'repaired'\n")
            importlib.invalidate_caches()
            repaired = importlib.import_module(name)
            self.assertEqual(seen, [repaired])
            self.assertIsNot(repaired, target)
            self.assertEqual(repaired.VALUE, "repaired")
            self.assertNotIn(name, _import_hook._FINDERS)
        finally:
            control.release.set()
            release_check.set()
            import_thread.join(timeout=10)
            if checked_thread is not None:
                checked_thread.join(timeout=10)

    def test_watcher_rejects_removed_module_when_import_fails_during_ready_check(self) -> None:
        self._check_failed_import_window(watcher=True)

    def test_immediate_callback_rejects_removed_module_when_import_fails_during_ready_check(self) -> None:
        self._check_failed_import_window(watcher=False)


if __name__ == "__main__":
    unittest.main()
