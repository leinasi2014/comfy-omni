from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from test_legacy_fixtures import freeze, make_vae, resign, thaw

from comfy_omni.artifacts import fileops
from comfy_omni.runtime.h3.legacy_vae import VaeExportContractError, verify_legacy_vae_export
from comfy_omni.runtime.h3.package_binding import AUDITED_PRODUCER


@pytest.fixture
def vae(tmp_path: Path, monkeypatch):
    root = tmp_path / "video_vae"
    manifest = make_vae(root, "video_vae", monkeypatch)
    yield root, manifest
    thaw(root)


def test_legacy_vae_verifier_binds_frozen_payload_and_producer(vae):
    root, manifest = vae
    assert verify_legacy_vae_export(root, expected_converter=AUDITED_PRODUCER.to_dict()) == manifest


@pytest.mark.parametrize(
    "mutation", ["producer", "runtime", "conversion-identity", "template", "tensor-catalog", "stats"]
)
def test_legacy_vae_refuses_self_resigned_semantic_tamper(vae, mutation):
    root, manifest = vae
    thaw(root)
    if mutation == "producer":
        manifest["converter"]["wheel_sha256"] = "0" * 64
    elif mutation == "runtime":
        manifest["numerical_runtime"]["torch_num_threads"] = True
    elif mutation == "conversion-identity":
        manifest["numerical_runtime"]["torch_num_threads"] = 17
    elif mutation == "template":
        path = root / "template.py"
        path.write_bytes(b"# changed static template\n")
        manifest["output"]["files"]["template.py"] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
        }
    elif mutation == "tensor-catalog":
        manifest["output"]["tensor_payload_catalog_sha256"] = "0" * 64
    else:
        path = root / "config.json"
        value = fileops.parse_json_strict(path.read_bytes())
        value["latents_std"][0] = 0.0
        path.write_bytes(fileops.canonical_json(value))
        manifest["output"]["files"]["config.json"] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
        }
    resign(root / "h3-comfy-vae-export.json", manifest)
    freeze(root)
    with pytest.raises(VaeExportContractError):
        verify_legacy_vae_export(root, expected_converter=AUDITED_PRODUCER.to_dict())


def test_legacy_vae_refuses_nonfrozen_source(vae):
    root, _ = vae
    root.chmod(0o755)
    with pytest.raises(VaeExportContractError, match="not frozen"):
        verify_legacy_vae_export(root, expected_converter=AUDITED_PRODUCER.to_dict())
