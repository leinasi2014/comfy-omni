"""The two shipped v3 layouts have distinct validation paths (issue #11)."""

from pathlib import Path

import pytest

from comfy_omni.artifacts import fileops
from comfy_omni.integrations.vllm_omni import package_contract


def test_legacy_top_level_v3_reaches_its_complete_validator(tmp_path: Path, monkeypatch) -> None:
    """Routing characterization only; the separate verifier tests exercise payloads."""
    index = {
        "_class_name": "MiniMaxH3Pipeline",
        "_minimax_h3": {"partition": "ref2va", "tasks": ["ref2va", "t2va", "fl2va"]},
    }
    (tmp_path / "model_index.json").write_bytes(fileops.canonical_json(index))
    manifest = {
        "schema": "h3-comfy-package/v3",
        "serving_entrypoint": "Ref2VA/",
        "routing_profile": "h3-hybrid-ref-primary-single-dit/v1",
        "loadable_package": True,
    }
    (tmp_path / "h3-comfy-package.json").write_bytes(fileops.canonical_json(manifest))
    expected = object()
    calls = []

    def validate(root: Path, *, expected_class_name: str):
        calls.append((root, expected_class_name))
        return expected

    monkeypatch.setattr(package_contract, "_validate_legacy_runtime_package", validate, raising=False)
    assert package_contract.validate_runtime_package(tmp_path) is expected
    assert calls == [(tmp_path.resolve(), "MiniMaxH3Pipeline")]


@pytest.mark.parametrize("schema", ["h3-comfy-package/v2", "h3-comfy-package/v4", "h3-comfy-package/v5"])
def test_other_legacy_versions_never_reach_v3_validator(tmp_path: Path, monkeypatch, schema: str) -> None:
    (tmp_path / "h3-comfy-package.json").write_bytes(
        fileops.canonical_json({"schema": schema, "serving_entrypoint": "Ref2VA/"})
    )

    def unexpected(*args, **kwargs):
        pytest.fail("unsupported legacy version reached the v3 validator")

    monkeypatch.setattr(package_contract, "_validate_legacy_runtime_package", unexpected, raising=False)
    with pytest.raises(package_contract.RuntimePackageContractError):
        package_contract.validate_runtime_package(tmp_path)
