"""Plugin-boundary tests for the raw H3 GPU acceptance runner."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
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


def test_full_media_mismatch_still_runs_release_and_reload_before_failing(monkeypatch, tmp_path):
    runner = _runner_module()
    monkeypatch.setattr(runner, "_plugin_preflight", lambda: None)
    calls: list[tuple[str, str | None]] = []

    class Engine:
        def __init__(self, **_kwargs):
            calls.append(("engine", None))

        def shutdown(self):
            calls.append(("shutdown", None))

    def status(selection="a", residency="loaded", *, device=100, allocated=1_000):
        return {
            "active_selection": selection,
            "weight_residency": residency,
            "device_weight_bytes": device,
            "cpu_weight_bytes": 0,
            "resident_weight_bytes": device,
            "cuda_memory_allocated_bytes": allocated,
            "worker_pid_scope": "parent-owned-all-ranks",
            "worker_pids_by_replica": {"0": [101, 102]},
            "routes": [
                {
                    "worker_pid": 101,
                    "pipeline_id": 11,
                    "transformer_id": 12,
                    "shared_object_ids": {"vae": 13},
                }
            ],
        }

    class Coordinator:
        instance = None

        def __init__(self, _engine, **_kwargs):
            type(self).instance = self
            self.statuses = iter((status(), status("b"), status(), status(), status()))

        async def status(self):
            return next(self.statuses)

        async def switch(self, selection):
            calls.append(("switch", selection))
            return {"selection": selection}

        async def unload(self, *, mode):
            calls.append(("unload", mode))
            return status(residency="released", device=0, allocated=900)

        async def load(self):
            calls.append(("load", None))
            return status()

    monkeypatch.setitem(sys.modules, "vllm_omni.entrypoints.async_omni", SimpleNamespace(AsyncOmni=Engine))
    monkeypatch.setitem(
        sys.modules,
        "comfy_omni.integrations.vllm_omni.residency_control",
        SimpleNamespace(H3ResidencyCoordinator=Coordinator),
    )
    media = {
        "a-initial": {"frame_sha256": "a-frame", "audio_sha256": "a-audio"},
        "b": {"frame_sha256": "b-frame", "audio_sha256": "b-audio"},
        "a-restored": {"frame_sha256": "a-frame", "audio_sha256": "changed-audio"},
        "a-reloaded": {"frame_sha256": "a-frame", "audio_sha256": "a-audio"},
    }

    async def generate(_engine, _args, _output, label):
        return {"label": label, **media[label]}

    monkeypatch.setattr(runner, "_generate", generate)
    components = tmp_path / "components"
    components.mkdir()
    source_a = tmp_path / "a.safetensors"
    source_b = tmp_path / "b.safetensors"
    prompt = tmp_path / "prompt.txt"
    reference = tmp_path / "reference.png"
    for path in (source_a, source_b, prompt, reference):
        path.write_bytes(b"fixture")
    output = tmp_path / "out"
    output.mkdir()
    args = SimpleNamespace(
        stage="full",
        component_root=components,
        source_a=source_a,
        source_b=source_b,
        prompt=prompt,
        reference=reference,
        seed=0,
        width=1,
        height=1,
        fps=1,
        frame_count=1,
        duration=1.0,
        steps=1,
        init_timeout=1.0,
        rpc_timeout=1.0,
    )

    with pytest.raises(RuntimeError, match="A→B→A"):
        asyncio.run(runner._run(args, output))

    assert ("unload", "release") in calls
    assert ("load", None) in calls
    receipt = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "FAILED"
    event_phases = {event["phase"] for event in receipt["events"]}
    assert event_phases >= {"forward-a-restored-media-check", "unload-release", "reload"}
    assert receipt["failed_phase"] == "media-verification"
    assert receipt["media_assertion_failures"][0]["phase"] == "forward-a-restored"
