"""Contract checks for the standalone native-package acceptance harness."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from comfy_omni.artifacts.build_identity import ToolIdentity

ROOT = Path(__file__).resolve().parents[2]
ASSEMBLY_SCRIPT = ROOT / "scripts" / "acceptance" / "native_package_assembly.py"
RUNTIME_COMPONENT_CONFIGS = ROOT / "docs" / "testing" / "e4-component-configs.v1.json"


def _load_assembly_harness():
    spec = importlib.util.spec_from_file_location("native_package_assembly", ASSEMBLY_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_package_assembly_pins_every_runtime_component_file() -> None:
    document = json.loads(RUNTIME_COMPONENT_CONFIGS.read_text(encoding="utf-8"))
    expected: dict[str, dict[str, tuple[int, str]]] = {}
    for record in document["files"]:
        expected.setdefault(record["component"], {})[record["file"]] = (record["bytes"], record["sha256"])
    for record in document["runtime_derivations"]:
        component, name = record["component"], record["file"]
        assert expected[component][name] == (record["source_bytes"], record["source_sha256"])
        expected[component][name] = (record["bytes"], record["sha256"])

    harness = _load_assembly_harness()
    assert harness.RUNTIME_COMPONENT_FILES == expected


def test_runtime_video_configuration_is_bound_to_the_selected_payload() -> None:
    document = json.loads(RUNTIME_COMPONENT_CONFIGS.read_text(encoding="utf-8"))
    (derived,) = document["runtime_derivations"]
    assert (derived["component"], derived["file"]) == ("video_vae", "config.json")
    harness = _load_assembly_harness()
    assert harness.RUNTIME_COMPONENT_FILES["video_vae"]["config.json"] == (
        2906,
        "5d1163e8fb4030f3c927714611335840a6e500071cdf5d75ea9c13fccf9f5abc",
    )
    assert derived["runtime_payload_sha256"] == harness.SINGLE_PAYLOAD_SHA256["video_vae"][1]
    assert derived["source_payload_sha256"] == "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522"


def _small_component(harness, tmp_path, monkeypatch):
    source = tmp_path / "video_vae"
    source.mkdir()
    contents = {"model.safetensors": b"payload", "config.json": b"good", "vae.py": b"code"}
    for name, data in contents.items():
        (source / name).write_bytes(data)
    pins = {name: (len(data), hashlib.sha256(data).hexdigest()) for name, data in contents.items()}
    monkeypatch.setattr(harness, "COMPONENT_CENSUS", {"video_vae": (3, 15)})
    monkeypatch.setattr(
        harness, "SINGLE_PAYLOAD_SHA256", {"video_vae": ("model.safetensors", pins["model.safetensors"][1])}
    )
    monkeypatch.setattr(
        harness,
        "RUNTIME_COMPONENT_FILES",
        {"video_vae": {name: pin for name, pin in pins.items() if name != "model.safetensors"}},
        raising=False,
    )
    tool = ToolIdentity("comfy-omni", "0.2.0a1", "a" * 40, "b" * 64)
    return source, tool


@pytest.mark.parametrize("changed", ["config.json", "vae.py"])
def test_assembly_rejects_same_size_runtime_file_substitution(tmp_path, monkeypatch, changed) -> None:
    harness = _load_assembly_harness()
    source, tool = _small_component(harness, tmp_path, monkeypatch)
    (source / changed).write_bytes(b"evil")
    with pytest.raises(RuntimeError, match="runtime component digest drifted"):
        harness._receipt("video_vae", source, tool)


def test_assembly_accepts_matching_payload_config_and_code(tmp_path, monkeypatch) -> None:
    harness = _load_assembly_harness()
    source, tool = _small_component(harness, tmp_path, monkeypatch)
    receipt, result = harness._receipt("video_vae", source, tool)
    assert len(receipt.files) == result["file_count"] == 3
