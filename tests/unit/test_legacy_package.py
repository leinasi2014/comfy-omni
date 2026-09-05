from __future__ import annotations

from pathlib import Path

import pytest
from test_legacy_fixtures import make_package, refresh_package, resign, thaw, write_json

from comfy_omni.artifacts import fileops
from comfy_omni.integrations.vllm_omni.package_contract import RuntimePackageContractError, validate_runtime_package
from comfy_omni.integrations.vllm_omni.serving import prepare_serving_layout
from comfy_omni.runtime.h3.package_binding import AUDITED_PRODUCER, CURVE_CACHE_NAME, legacy_quantization


@pytest.fixture
def package(tmp_path: Path, monkeypatch):
    pytest.importorskip("torch")
    root = tmp_path / "legacy"
    manifest = make_package(root, monkeypatch)
    yield root, manifest
    thaw(root)


def test_complete_legacy_v3_validation_and_native_serving_leave_source_unchanged(package, tmp_path):
    root, manifest = package
    original = (root / "h3-comfy-package.json").read_bytes()
    contract = validate_runtime_package(root)
    assert contract.layout == "h3-forge-native-v3"
    assert contract.plan_content_sha256 is None
    assert contract.partition_path == root / "Ref2VA"
    assert contract.curve_cache.producer == AUDITED_PRODUCER
    assert contract.runtime_quantization_config.to_dict() == legacy_quantization().to_dict()
    assert contract.file_count == len(manifest["files"])
    assert prepare_serving_layout(root, tmp_path / "unused-view") == root / "Ref2VA"
    assert not (tmp_path / "unused-view").exists()
    assert (root / "h3-comfy-package.json").read_bytes() == original


@pytest.mark.parametrize(
    "mutation",
    ["mixed-layout", "producer", "quantization", "duplicate", "traversal", "index", "config", "source", "extra"],
)
def test_legacy_v3_refuses_self_resigned_inconsistent_package(package, mutation):
    root, manifest = package
    if mutation == "mixed-layout":
        manifest["routing"] = {"serving_entrypoint": "Ref2VA/"}
    elif mutation == "producer":
        manifest["converter"]["wheel_sha256"] = "0" * 64
    elif mutation == "quantization":
        manifest["runtime_quantization_config"]["transformer"]["ignored_layers"] = []
    elif mutation == "duplicate":
        manifest["files"].append(dict(manifest["files"][0]))
        manifest["file_count"] += 1
    elif mutation == "traversal":
        manifest["files"][0]["path"] = "Ref2VA/../outside.bin"
    elif mutation == "index":
        path = root / "Ref2VA/model_index.json"
        index = fileops.parse_json_strict(path.read_bytes())
        index["_minimax_h3"]["tasks"] = ["ref2va"]
        write_json(path, index)
        refresh_package(root, manifest)
    elif mutation == "config":
        path = root / "Ref2VA/transformer/config.json"
        config = fileops.parse_json_strict(path.read_bytes())
        config["_h3_forge"]["strict_schedule_cache"] = False
        write_json(path, config)
        refresh_package(root, manifest)
    elif mutation == "source":
        manifest["transformer_source_sha256"] = ["0" * 64]
    else:
        (root / "uncommitted.txt").write_text("extra")
    resign(root / "h3-comfy-package.json", manifest, "package_manifest_sha256")
    with pytest.raises(RuntimePackageContractError):
        validate_runtime_package(root)


def test_legacy_v3_rejects_cache_symlink_before_payload_read(package):
    root, _ = package
    path = root / "Ref2VA/transformer" / CURVE_CACHE_NAME
    path.unlink()
    path.symlink_to(root / "model_index.json")
    with pytest.raises(RuntimePackageContractError, match="link"):
        validate_runtime_package(root)
