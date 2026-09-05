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


def test_thumbnail_pixels_converts_unit_interval_float32_rgb_without_mutating_the_frame():
    np = pytest.importorskip("numpy")
    runner = _runner_module()
    frame = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)

    thumbnail = runner._thumbnail_pixels(frame, np)

    assert thumbnail.dtype == np.uint8
    assert thumbnail.tolist() == [[[0, 128, 255]]]
    assert frame.dtype == np.float32 and frame.tolist() == [[[0.0, 0.5, 1.0]]]


def test_release_receipt_requires_zero_resident_weight_bytes_and_a_weight_sized_allocation_drop():
    runner = _runner_module()
    before = {
        "device_weight_bytes": 200 * 1024 * 1024,
        "cpu_weight_bytes": 0,
        "resident_weight_bytes": 200 * 1024 * 1024,
        "cuda_memory_allocated_bytes": 600 * 1024 * 1024,
    }
    after = {
        "device_weight_bytes": 0,
        "cpu_weight_bytes": 0,
        "resident_weight_bytes": 0,
        "cuda_memory_allocated_bytes": 390 * 1024 * 1024,
    }

    receipt = runner._release_receipt(before, after)

    assert receipt["allocated_drop_bytes"] == 210 * 1024 * 1024
    assert receipt["reported_device_weight_bytes"] == before["device_weight_bytes"]


def test_release_receipt_rejects_a_release_that_retains_weight_bytes():
    runner = _runner_module()
    before = {
        "device_weight_bytes": 100,
        "cpu_weight_bytes": 0,
        "resident_weight_bytes": 100,
        "cuda_memory_allocated_bytes": 1_000,
    }
    after = {
        "device_weight_bytes": 1,
        "cpu_weight_bytes": 0,
        "resident_weight_bytes": 1,
        "cuda_memory_allocated_bytes": 800,
    }

    with pytest.raises(RuntimeError, match="retained H3 weight bytes"):
        runner._release_receipt(before, after)
