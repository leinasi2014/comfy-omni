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
EXPECTED_CONFIG_TOTALS = {
    "processor-config": (7, 11_498_352),
    "tokenizer-config": (4, 11_492_078),
}
EXPECTED_CONFIG_FILES = {
    "tokenizer-config": {
        "merges.txt": (
            1_671_839,
            "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
            "20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0",
        ),
        "tokenizer.json": (
            7_032_403,
            "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
            "c6cc1014128b19d1fc46b1d30a23e3b1d35db421",
        ),
        "tokenizer_config.json": (
            11_003,
            "a07e942ac874baa13758de8d1fbdb186683cc03416b5589e1b6671c6b3057c68",
            "204d76f78dac6dedc820418c30bf01145de78a21",
        ),
        "vocab.json": (
            2_776_833,
            "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
            "4783fe10ac3adce15ac8f358ef5462739852c569",
        ),
    },
    "processor-config": {
        "chat_template.json": (
            5_499,
            "5c72a170d2a4a1a3bc5adad2e689ae28138a9700e5b8c96c0266331e86c0acce",
            "1081bacf1af7c7c6de4a585ce02cd0fd34e382da",
        ),
        "merges.txt": (
            1_671_839,
            "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
            "20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0",
        ),
        "preprocessor_config.json": (
            390,
            "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
            "2ea84a437d448ff71b08df68fdd949d5cc4ebb64",
        ),
        "tokenizer.json": (
            7_032_403,
            "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
            "c6cc1014128b19d1fc46b1d30a23e3b1d35db421",
        ),
        "tokenizer_config.json": (
            11_003,
            "a07e942ac874baa13758de8d1fbdb186683cc03416b5589e1b6671c6b3057c68",
            "204d76f78dac6dedc820418c30bf01145de78a21",
        ),
        "video_preprocessor_config.json": (
            385,
            "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
            "3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe",
        ),
        "vocab.json": (
            2_776_833,
            "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
            "4783fe10ac3adce15ac8f358ef5462739852c569",
        ),
    },
}
EXPECTED_SCENARIOS = {
    "artifact-identity",
    "primary-runtime-smoke",
    "lora-compatibility",
    "full-dit-hot-swap",
    "package-assembly",
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
    assert set(by_id) == set(EXPECTED_ASSETS) | set(EXPECTED_CONFIG_TOTALS)

    for asset_id, (expected_role, expected_bytes, expected_sha256) in EXPECTED_ASSETS.items():
        asset = by_id[asset_id]
        assert asset["role"] == expected_role
        assert asset["bytes"] == expected_bytes
        assert asset["sha256"] == expected_sha256

    for asset_id, (expected_count, expected_total) in EXPECTED_CONFIG_TOTALS.items():
        asset = by_id[asset_id]
        assert asset["role"] == "package-component-config"
        observed = {item["path"]: (item["bytes"], item["sha256"], item["git_blob_sha1"]) for item in asset["files"]}
        assert len(observed) == expected_count
        assert sum(item[0] for item in observed.values()) == expected_total
        assert observed == EXPECTED_CONFIG_FILES[asset_id]
        assert asset["source_prefix"] == f"Ref2VA/{asset_id.removesuffix('-config')}"


def test_model_baseline_sources_are_explicit_and_content_bound() -> None:
    assets = _load_baseline()["assets"]
    assert isinstance(assets, list)

    observed_hashes: set[str] = set()
    observed_filenames: set[str] = set()
    for asset in assets:
        assert isinstance(asset, dict)
        source_url = asset["source_url"]
        assert isinstance(source_url, str)
        parsed_url = urlparse(source_url)
        assert parsed_url.scheme == "https" and parsed_url.netloc
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

        if asset["role"] == "package-component-config":
            assert asset["source_prefix"]
            for item in asset["files"]:
                assert item["path"] and item["bytes"] > 0
                assert isinstance(item["sha256"], str) and SHA256_RE.fullmatch(item["sha256"])
                assert isinstance(item["git_blob_sha1"], str) and REVISION_RE.fullmatch(item["git_blob_sha1"])
            continue

        sha256 = asset["sha256"]
        filename = asset["filename"]
        assert isinstance(sha256, str) and SHA256_RE.fullmatch(sha256)
        assert isinstance(filename, str) and filename.endswith(".safetensors")
        assert asset["bytes"] > 0
        assert sha256 not in observed_hashes
        assert filename not in observed_filenames
        observed_hashes.add(sha256)
        observed_filenames.add(filename)


def test_nonconformant_text_encoder_requires_digest_bound_normalization() -> None:
    assets = _load_baseline()["assets"]
    assert isinstance(assets, list)
    text_encoder = next(asset for asset in assets if asset["id"] == "text-encoder")

    conformance = text_encoder["format_conformance"]
    assert conformance == {
        "status": "nonconformant",
        "reason_code": "safetensors-unindexed-trailing-bytes",
        "observed_header_bytes": 217_976,
        "indexed_payload_bytes": 15_682_911_603,
        "trailing_bytes": 72,
        "trailing_sha256": "8bbc743f1fdc67acb6b09c977485e7d8bed7ff073a12d70865e0e4b793ed8e75",
    }

    staging = text_encoder["staging_policy"]
    assert staging == {
        "action": "derive-strict-safetensors-copy",
        "in_place_mutation": False,
        "generic_trailing_tolerance": False,
        "expected_derived_bytes": 15_683_129_587,
        "expected_derived_sha256": "a166c7bbbe66a22065159e478335fee4a633c4a3e3bb34c8e8ac4cc91bf4996f",
    }


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
        if asset["role"] == "package-component-config":
            for item in asset["files"]:
                assert item["sha256"] in document
            continue
        assert asset["filename"] in document
        assert asset["sha256"] in document

    public_contract = BASELINE_PATH.read_text(encoding="utf-8") + document
    assert not PRIVATE_IPV4_RE.search(public_contract)
    assert "/data/models/" not in public_contract
