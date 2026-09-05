"""Verified route selection and the common cache/beta4 construction boundary."""

from __future__ import annotations

import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_curve_cache_runtime import _package, _router

from comfy_omni.integrations.vllm_omni.pipelines import scoped_construction as construction

BETA4 = "comfy_omni.integrations.vllm_omni.pipelines.beta4_pipeline"


def test_verified_beta4_binding_selects_the_dedicated_pipeline(monkeypatch, tmp_path):
    router, official = _router(monkeypatch)
    root, partition = _package(tmp_path)
    package = types.SimpleNamespace(package_root=root, partition_path=partition, beta4=object(), curve_cache=None)
    monkeypatch.setattr(router, "validate_runtime_package", lambda path: package)
    module = types.ModuleType(BETA4)

    class H3Beta4Pipeline:
        def __init__(self, *, od_config, package, prefix=""):
            self.package = package

    module.H3Beta4Pipeline = H3Beta4Pipeline
    monkeypatch.setitem(sys.modules, BETA4, module)
    selected = router.H3ComfyMiniMaxH3Pipeline(od_config=types.SimpleNamespace(model=str(root)))
    assert isinstance(selected, H3Beta4Pipeline)
    assert selected.package is package and not official.calls


def test_profile_string_without_a_verified_binding_does_not_select_beta4(monkeypatch, tmp_path):
    router, official = _router(monkeypatch)
    root, partition = _package(tmp_path)
    package = types.SimpleNamespace(package_root=root, partition_path=partition, profile="beta4-dense-bf16")
    monkeypatch.setattr(router, "validate_runtime_package", lambda path: package)
    router.H3ComfyMiniMaxH3Pipeline(od_config=types.SimpleNamespace(model=str(root)))
    assert len(official.calls) == 1


def test_competing_bindings_are_rejected_before_host_construction(monkeypatch, tmp_path):
    router, official = _router(monkeypatch)
    root, _ = _package(tmp_path)
    package = types.SimpleNamespace(beta4=object(), curve_cache=object())
    monkeypatch.setattr(router, "validate_runtime_package", lambda path: package)
    with pytest.raises(ValueError, match="competing DiT routes"):
        router.H3ComfyMiniMaxH3Pipeline(od_config=types.SimpleNamespace(model=str(root)))
    assert not official.calls


def test_substitution_and_failed_pipeline_preserve_the_completed_model_latch(monkeypatch):
    monkeypatch.setattr(construction, "state", construction.WorkerConstructionState())
    original = object()
    host = types.SimpleNamespace(model_class=original)
    with pytest.raises(RuntimeError, match="pipeline failed"):
        with construction.construction("pipeline"):
            with construction.substitute(host, model_class=object()):
                with construction.construction("model"):
                    pass
                raise RuntimeError("pipeline failed")
    assert host.model_class is original
    assert construction.state.model_constructed
    assert not construction.state.pipeline_constructed
    assert not construction.state.model_pending and not construction.state.pipeline_pending
    with pytest.raises(RuntimeError, match="exactly one H3 model"):
        with construction.construction("model"):
            pass


def test_competing_cache_and_beta4_scopes_wait_and_cannot_swap_classes(monkeypatch):
    monkeypatch.setattr(construction, "state", construction.WorkerConstructionState())
    entered, attempted, release = threading.Event(), threading.Event(), threading.Event()
    original, cache, beta4 = object(), object(), object()
    host = types.SimpleNamespace(model_class=original)
    observed = []

    def cache_route():
        with construction.construction("pipeline"):
            with construction.substitute(host, model_class=cache):
                with construction.construction("model"):
                    entered.set()
                    assert release.wait(10)
                    observed.append(host.model_class)

    def beta4_route():
        assert entered.wait(10)
        attempted.set()
        with construction.construction("pipeline"):
            with construction.substitute(host, model_class=beta4):
                observed.append(host.model_class)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.submit(cache_route), pool.submit(beta4_route)
        try:
            assert attempted.wait(10)
            assert host.model_class is cache and not second.done()
        finally:
            release.set()
        first.result(timeout=10)
        with pytest.raises(RuntimeError, match="exactly one H3 pipeline"):
            second.result(timeout=10)
    assert observed == [cache] and host.model_class is original
