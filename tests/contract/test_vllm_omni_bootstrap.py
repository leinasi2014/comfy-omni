from __future__ import annotations

import ast
import sys
import threading
import types
from pathlib import Path
from unittest import mock

import comfy_omni.plugin

_HEAVY_MODULE_NAMES = ("torch", "vllm", "vllm_omni", "fastapi")
_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "comfy_omni"
_EXPECTED_ARCHITECTURE_CALLS = [
    mock.call(
        "MiniMaxH3Pipeline",
        "comfy_omni.integrations.vllm_omni.pipelines.runtime_pipeline",
        "H3ComfyMiniMaxH3Pipeline",
        post_process_func_name="get_minimax_h3_post_process_func",
    ),
    mock.call(
        "MiniMaxH3DensePipeline",
        "comfy_omni.integrations.vllm_omni.pipelines.dense_pipeline",
        "MiniMaxH3DensePipeline",
        post_process_func_name="get_minimax_h3_post_process_func",
    ),
]


def _install_host(monkeypatch, register_diffusion_model) -> None:
    host = types.ModuleType("vllm_omni")
    host.__path__ = []
    diffusion = types.ModuleType("vllm_omni.diffusion")
    diffusion.__path__ = []
    registry_module = types.ModuleType("vllm_omni.diffusion.registry")
    registry_module.register_diffusion_model = register_diffusion_model
    host.diffusion = diffusion
    diffusion.registry = registry_module
    monkeypatch.setitem(sys.modules, "vllm_omni", host)
    monkeypatch.setitem(sys.modules, "vllm_omni.diffusion", diffusion)
    monkeypatch.setitem(sys.modules, "vllm_omni.diffusion.registry", registry_module)


def _ensure_no_host(monkeypatch) -> None:
    for name in ("vllm_omni.diffusion.registry", "vllm_omni.diffusion", "vllm_omni"):
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_bootstrap_registers_frozen_architectures_once_into_a_resident_host(monkeypatch) -> None:
    from comfy_omni.integrations.vllm_omni import bootstrap

    registry = mock.Mock()
    host = types.ModuleType("vllm_omni")
    host.__path__ = []
    diffusion = types.ModuleType("vllm_omni.diffusion")
    diffusion.__path__ = []
    registry_module = types.ModuleType("vllm_omni.diffusion.registry")
    registry_module.register_diffusion_model = registry
    host.diffusion = diffusion
    diffusion.registry = registry_module

    monkeypatch.setitem(sys.modules, "vllm_omni", host)
    monkeypatch.setitem(sys.modules, "vllm_omni.diffusion", diffusion)
    monkeypatch.setitem(sys.modules, "vllm_omni.diffusion.registry", registry_module)

    bootstrap._registration_state = 0

    comfy_omni.plugin.register()

    expected = [
        mock.call(
            "MiniMaxH3Pipeline",
            "comfy_omni.integrations.vllm_omni.pipelines.runtime_pipeline",
            "H3ComfyMiniMaxH3Pipeline",
            post_process_func_name="get_minimax_h3_post_process_func",
        ),
        mock.call(
            "MiniMaxH3DensePipeline",
            "comfy_omni.integrations.vllm_omni.pipelines.dense_pipeline",
            "MiniMaxH3DensePipeline",
            post_process_func_name="get_minimax_h3_post_process_func",
        ),
    ]
    assert registry.mock_calls == expected

    comfy_omni.plugin.register()

    assert registry.mock_calls == expected


def test_bootstrap_defers_silently_without_a_resident_host_and_retries(monkeypatch) -> None:
    from comfy_omni.integrations.vllm_omni import bootstrap

    _ensure_no_host(monkeypatch)
    bootstrap._registration_state = 0

    bootstrap.register()

    assert bootstrap._registration_state == 0

    registry = mock.Mock()
    _install_host(monkeypatch, registry)

    bootstrap.register()

    assert bootstrap._registration_state == 2
    assert registry.mock_calls == _EXPECTED_ARCHITECTURE_CALLS


def test_bootstrap_defers_for_a_resident_but_partial_host(monkeypatch) -> None:
    from comfy_omni.integrations.vllm_omni import bootstrap

    partial_host = types.ModuleType("vllm_omni")
    monkeypatch.setitem(sys.modules, "vllm_omni", partial_host)
    monkeypatch.delitem(sys.modules, "vllm_omni.diffusion", raising=False)
    monkeypatch.delitem(sys.modules, "vllm_omni.diffusion.registry", raising=False)
    bootstrap._registration_state = 0

    bootstrap.register()

    assert bootstrap._registration_state == 0


def test_register_loads_no_heavy_modules_without_a_host(monkeypatch) -> None:
    _ensure_no_host(monkeypatch)

    before = set(sys.modules)
    comfy_omni.plugin.register()
    added = set(sys.modules) - before

    assert all(name not in added for name in _HEAVY_MODULE_NAMES)


def test_concurrent_registration_registers_exactly_once(monkeypatch) -> None:
    from comfy_omni.integrations.vllm_omni import bootstrap

    registry = mock.Mock()
    _install_host(monkeypatch, registry)
    bootstrap._registration_state = 0

    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def _register() -> None:
        barrier.wait()
        try:
            bootstrap.register()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_register) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert registry.mock_calls == _EXPECTED_ARCHITECTURE_CALLS


def test_plugin_register_delegates_to_bootstrap(monkeypatch) -> None:
    from comfy_omni.integrations.vllm_omni import bootstrap

    registry = mock.Mock()
    _install_host(monkeypatch, registry)
    bootstrap._registration_state = 0

    comfy_omni.plugin.register()

    assert registry.mock_calls == _EXPECTED_ARCHITECTURE_CALLS

    bootstrap.register()

    assert registry.mock_calls == _EXPECTED_ARCHITECTURE_CALLS


def test_vllm_imports_are_confined_to_the_integration_boundary() -> None:
    boundary = _PACKAGE_ROOT / "integrations" / "vllm_omni"
    forbidden_roots = {"vllm", "vllm_omni", "torch", "fastapi"}
    offenders = []
    for path in _PACKAGE_ROOT.rglob("*.py"):
        if boundary in path.parents:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module_level = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        for node in module_level:
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
            else:
                roots = {(node.module or "").split(".")[0]}
            if roots & forbidden_roots:
                offenders.append(path.relative_to(_PACKAGE_ROOT).as_posix())
                break

    assert offenders == []
