"""Contract tests for the digest-pinned external validation model set."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = ROOT / "docs" / "testing" / "model-baseline.v1.json"
BASELINE_DOCUMENT_PATH = ROOT / "docs" / "testing" / "model-validation-baseline.md"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
PRIVATE_IPV4_RE = re.compile(
    r"\b(?:10(?:\.\d+){3}|127(?:\.\d+){3}|169\.254(?:\.\d+){2}|"
    r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d+){2}|192\.168(?:\.\d+){2})\b"
)
EXPECTED_ASSETS = {
    "primary-dit": (
        "diffusion-checkpoint",
        20_967_637_320,
        "54d56b15c65923b54c9ca16b494dae641bfe9455cfcb1c19c49b1008e270bbc1",
    ),
    "text-encoder": (
        "text-encoder",
        15_683_129_659,
        "47babbb3e4b7e43c097351ca39cfb7f326d014ae53a584f8559dc8121abca94c",
    ),
    "audio-vae": (
        "audio-vae",
        605_254_808,
        "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
    ),
    "video-vae": (
        "video-vae",
        5_207_808_496,
        "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522",
    ),
    "spatial-physics-lora": (
        "lora-candidate",
        155_109_672,
        "7d14f3701560068e7004159c8b2a7278bd2dbfc9e5e3b60d0bc9aef6c049919d",
    ),
    "realism-people-lora": (
        "lora-candidate",
        131_229_656,
        "acc529601d2da117fb81179e76c56e488a3beab1171659d305f04fa3655b787e",
    ),
    "hot-swap-dit": (
        "diffusion-checkpoint",
        20_970_379_680,
        "71b8085ac4221ee036708c230a007d617dccca1b0028b95bb4ee106cb2a385c5",
    ),
}
EXPECTED_SCENARIOS = {
    "artifact-identity",
    "primary-runtime-smoke",
    "lora-compatibility",
    "full-dit-hot-swap",
}


def _load_baseline() -> dict[str, object]:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def test_model_baseline_has_the_frozen_asset_identities() -> None:
    baseline = _load_baseline()
    assert baseline["schema_id"] == "comfy-omni.test-model-baseline/v1"
    assert baseline["status"] == "pinned-for-validation"
    assert baseline["download_policy"] == "maintainer-managed-external-only"
    assert baseline["redistribute_with_comfy_omni"] is False

    assets = baseline["assets"]
    assert isinstance(assets, list)
    by_id = {asset["id"]: asset for asset in assets}
    assert set(by_id) == set(EXPECTED_ASSETS)

    for asset_id, (expected_role, expected_bytes, expected_sha256) in EXPECTED_ASSETS.items():
        asset = by_id[asset_id]
        assert asset["role"] == expected_role
        assert asset["bytes"] == expected_bytes
        assert asset["sha256"] == expected_sha256


def test_model_baseline_sources_are_explicit_and_content_bound() -> None:
    assets = _load_baseline()["assets"]
    assert isinstance(assets, list)

    observed_hashes: set[str] = set()
    observed_filenames: set[str] = set()
    for asset in assets:
        assert isinstance(asset, dict)
        sha256 = asset["sha256"]
        filename = asset["filename"]
        source_url = asset["source_url"]
        assert isinstance(sha256, str) and SHA256_RE.fullmatch(sha256)
        assert isinstance(filename, str) and filename.endswith(".safetensors")
        assert isinstance(source_url, str)
        parsed_url = urlparse(source_url)
        assert parsed_url.scheme == "https" and parsed_url.netloc
        assert asset["bytes"] > 0
        assert asset["repository"]
        assert asset["license"]
        assert asset["validation_uses"]

        revision = asset["revision"]
        if asset["provider"] == "huggingface":
            assert isinstance(revision, str) and REVISION_RE.fullmatch(revision)
            assert revision in source_url
        else:
            assert asset["provider"] == "modelscope"
            assert revision == "master"
            assert asset["source_binding"] == "modelscope-x-linked-etag-sha256"

        assert sha256 not in observed_hashes
        assert filename not in observed_filenames
        observed_hashes.add(sha256)
        observed_filenames.add(filename)


def test_model_baseline_scenarios_reference_only_pinned_assets() -> None:
    baseline = _load_baseline()
    assets = baseline["assets"]
    scenarios = baseline["scenarios"]
    assert isinstance(assets, list)
    assert isinstance(scenarios, list)

    asset_ids = {asset["id"] for asset in assets}
    scenario_ids = {scenario["id"] for scenario in scenarios}
    assert scenario_ids == EXPECTED_SCENARIOS
    for scenario in scenarios:
        assert scenario["asset_ids"]
        assert set(scenario["asset_ids"]) <= asset_ids
        assert scenario["required_outcome"]


def test_model_baseline_document_lists_every_pinned_payload_without_private_paths() -> None:
    baseline = _load_baseline()
    assets = baseline["assets"]
    document = BASELINE_DOCUMENT_PATH.read_text(encoding="utf-8")
    assert isinstance(assets, list)

    for asset in assets:
        assert asset["filename"] in document
        assert asset["sha256"] in document

    public_contract = BASELINE_PATH.read_text(encoding="utf-8") + document
    assert not PRIVATE_IPV4_RE.search(public_contract)
    assert "/data/models/" not in public_contract
