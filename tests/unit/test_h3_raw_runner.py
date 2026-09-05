"""Plugin-boundary tests for the raw H3 GPU acceptance runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "acceptance" / "h3_raw_runtime.py"


def _runner_module():
    spec = importlib.util.spec_from_file_location("h3_raw_runtime_test", _PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plugin_preflight_rejects_missing_comfy_omni_before_host_import(monkeypatch):
    runner = _runner_module()

    def missing_distribution(name):
        raise runner.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(runner.metadata, "distribution", missing_distribution)
    with pytest.raises(RuntimeError, match="comfy-omni distribution"):
        runner._plugin_preflight()


def _install_valid_plugin_metadata(runner, monkeypatch):
    def distribution(name):
        if name == "comfy-omni":
            return object()
        raise runner.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(runner.metadata, "distribution", distribution)
    monkeypatch.setattr(
        runner.metadata,
        "entry_points",
        lambda: SimpleNamespace(
            select=lambda *, group: (
                (SimpleNamespace(name="comfy_omni", value="comfy_omni.plugin:register"),)
                if group == "vllm_omni.general_plugins"
                else ()
            )
        ),
    )


def test_plugin_preflight_accepts_the_exact_registered_plugin(monkeypatch):
    runner = _runner_module()
    _install_valid_plugin_metadata(runner, monkeypatch)
    monkeypatch.delenv("VLLM_PLUGINS", raising=False)

    runner._plugin_preflight()


def test_plugin_preflight_rejects_an_explicit_plugin_filter_without_comfy_omni(monkeypatch):
    runner = _runner_module()
    _install_valid_plugin_metadata(runner, monkeypatch)
    monkeypatch.setenv("VLLM_PLUGINS", "other_plugin")

    with pytest.raises(RuntimeError, match="VLLM_PLUGINS"):
        runner._plugin_preflight()
