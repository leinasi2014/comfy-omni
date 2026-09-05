# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: h3-forge contributors
"""Legacy directory behavior, using synthetic files without checkpoint payloads.

Characterizes h3-forge e9cb011d00b028c149db3978de246c54f6e34acc:
catalog.py blob 322dd5b5e37722d82675d9d6c547901b296b759f;
test_component_catalog.py blob 77a6b54d1e72c1de9e9000d671c42ab5a4ebc40c.
"""

from __future__ import annotations

import json
import threading
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from comfy_omni.domain.components import SINGLE_SELECT_KINDS, Component, ComponentKind, ComponentZone
from comfy_omni.runtime.components import catalog as catalog_module
from comfy_omni.runtime.components.catalog import COMPONENT_ROOTS_ENV, CatalogError, ComponentCatalog


def _put(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"directory fixture; not a checkpoint")
    return path


@pytest.fixture
def roots(tmp_path):
    comfy = tmp_path / "comfy"
    official = tmp_path / "upstream"
    servable = tmp_path / "ready"
    for relative in (
        "diffusion_models/A.bin",
        "diffusion_models/nested/B.weird",
        "text_encoders/TE.bin",
        "vae/Video.bin",
        "vae/CAPS_AUDIO.weird",
        "loras/Z.bin",
        "loras/nested/L.json",
        "latent_upscale_models/U.bin",
        "diffusion_models/.hidden/X.bin",
        "diffusion_models/.ignored.bin",
        "checkpoints/ignored.bin",
    ):
        _put(comfy / relative)
    for relative in (
        "FL2VA/transformer/config.json",
        "Ref2VA/transformer/model.bin",
        "Ref2VA/text_encoder/config.json",
    ):
        _put(official / relative)
    (official / "Ref2VA/video_vae").mkdir()
    _put(servable / "Ref2VA/transformer/config.json")
    _put(servable / "Ref2VA/video_vae/config.json")
    return {"comfy": comfy, "official": official, "servable": servable}


def test_component_value_is_immutable_and_selection_is_only_a_candidate():
    entry = Component(ComponentKind.DIT, ComponentZone.COMFY, "A", "/fixture/A", selection="A")
    assert entry.contract_digest is None
    assert entry.locked is False
    with pytest.raises(FrozenInstanceError):
        entry.path = "/other"
    assert replace(entry, locked=True, selection=None).selection is None
    assert replace(entry, contract_digest="opaque-legacy-value").contract_digest == "opaque-legacy-value"
    for kind in (ComponentKind.LORA, ComponentKind.TOOL):
        assert Component(kind, ComponentZone.COMFY, "A", "/fixture/A").selection is None
    schedule = Component(ComponentKind.SCHEDULE, None, "turbo_4step", "", selection="turbo_4step")
    assert schedule.zone is None and schedule.path == ""


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "dit"},
        {"zone": "comfy"},
        {"zone": None},
        {"id": ""},
        {"id": 3},
        {"path": Path("A")},
        {"locked": 1},
        {"contract_digest": ""},
        {"contract_digest": 1},
        {"selection": "B"},
        {"locked": True, "selection": "A"},
        {"kind": ComponentKind.LORA, "selection": "A"},
        {"kind": ComponentKind.TOOL, "selection": "A"},
        {"kind": ComponentKind.SCHEDULE, "path": "", "zone": ComponentZone.COMFY},
        {"kind": ComponentKind.SCHEDULE, "path": "/fixture/A", "zone": None},
    ],
)
def test_component_rejects_invalid_value_combinations(changes):
    values = {"kind": ComponentKind.DIT, "zone": ComponentZone.COMFY, "id": "A", "path": "/fixture/A"}
    with pytest.raises(ValueError):
        Component(**(values | changes))


@pytest.mark.parametrize("configuration", [None, "", "  ", "{}", '{"comfy": []}', "; ;"])
def test_no_roots_still_has_only_static_schedules(configuration):
    env = {} if configuration is None else {COMPONENT_ROOTS_ENV: configuration}
    catalog = ComponentCatalog(env=env)
    assert not catalog.roots_configured()
    assert [entry.id for entry in catalog.list()] == ["turbo_4step", "turbo_8step"]
    catalog.scan()
    assert catalog.selection_candidates(ComponentKind.SCHEDULE) == ["turbo_4step", "turbo_8step"]


@pytest.mark.parametrize(
    "configuration",
    [
        "comfy",
        "comfy=",
        "=path",
        "COMFY=path",
        "unknown=path",
        "{broken",
        '{"comfy": 2}',
        '{"comfy": [null]}',
        '{"comfy": [""]}',
        '{"unknown": []}',
    ],
)
def test_bad_environment_roots_are_configuration_errors(configuration):
    with pytest.raises(CatalogError):
        ComponentCatalog(env={COMPONENT_ROOTS_ENV: configuration})


def test_root_forms_repeated_zones_and_explicit_empty_override(tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    _put(first / "diffusion_models/A.bin")
    _put(second / "diffusion_models/B.bin")
    configurations = [
        ComponentCatalog(env={COMPONENT_ROOTS_ENV: f" comfy={first}; ;comfy={second} "}),
        ComponentCatalog(env={COMPONENT_ROOTS_ENV: json.dumps({"comfy": [str(first), str(second)]})}),
        ComponentCatalog({ComponentZone.COMFY: [first, second]}, env={}),
    ]
    for catalog in configurations:
        assert catalog.list(ComponentKind.DIT) == []
        catalog.scan()
        assert catalog.selection_candidates(ComponentKind.DIT) == ["A", "B"]
    explicit = ComponentCatalog({}, env={COMPONENT_ROOTS_ENV: "broken"})
    assert not explicit.roots_configured()
    with pytest.raises(CatalogError, match="path string"):
        ComponentCatalog({"comfy": b"invalid"})


def test_directory_census_and_legacy_candidate_policy(roots):
    catalog = ComponentCatalog(roots)
    catalog.scan()
    expected = {
        ComponentKind.DIT: ["A", "B", "ready", "upstream/FL2VA", "upstream/Ref2VA"],
        ComponentKind.TEXT_ENCODER: ["TE", "upstream/Ref2VA/text_encoder"],
        ComponentKind.VIDEO_VAE: ["Video", "ready/Ref2VA/video_vae"],
        ComponentKind.AUDIO_VAE: ["CAPS_AUDIO"],
        ComponentKind.SCHEDULE: ["turbo_4step", "turbo_8step"],
        ComponentKind.LORA: ["L", "Z"],
        ComponentKind.TOOL: ["U"],
    }
    for kind, ids in expected.items():
        entries, candidates = catalog.snapshot(kind)
        assert [entry.id for entry in entries] == ids
        assert candidates == [entry.id for entry in entries if entry.zone is not ComponentZone.OFFICIAL]
        assert catalog.selection_candidates(kind) == candidates
        for entry in entries:
            assert catalog.get(kind, entry.id) == entry
            assert entry.contract_digest is None
            assert entry.locked is (entry.zone is ComponentZone.OFFICIAL)
            expected_selection = entry.id if kind in SINGLE_SELECT_KINDS and not entry.locked else None
            assert entry.selection == expected_selection
            if kind is ComponentKind.SCHEDULE:
                assert entry.path == "" and entry.zone is None
    assert [entry.id for entry in catalog.list()] == sorted(entry.id for entry in catalog.list())


def test_scanning_never_opens_payloads_or_changes_input_files(roots, monkeypatch):
    files = [path for root in roots.values() for path in root.rglob("*") if path.is_file()]
    before = {path: path.read_bytes() for path in files}

    def forbidden(*args, **kwargs):
        raise AssertionError("catalog must not open a payload")

    with monkeypatch.context() as scoped:
        scoped.setattr("builtins.open", forbidden)
        scoped.setattr(Path, "open", forbidden)
        scoped.setattr("os.open", forbidden)
        catalog = ComponentCatalog(roots)
        catalog.scan()
        assert catalog.get(ComponentKind.DIT, "A").contract_digest is None
    assert {path: path.read_bytes() for path in files} == before


def test_package_depth_and_hidden_payload_discovery(tmp_path):
    root = tmp_path / "package"
    _put(root / "transformer/.metadata")
    _put(root / "Ref2VA/text_encoder/nested/anything.txt")
    _put(root / "too/deep/video_vae/ignored.bin")
    (root / "audio_vae").mkdir()
    catalog = ComponentCatalog({"servable": root})
    catalog.scan()
    assert catalog.selection_candidates(ComponentKind.DIT) == ["package"]
    assert catalog.selection_candidates(ComponentKind.TEXT_ENCODER) == ["package/Ref2VA/text_encoder"]
    assert catalog.list(ComponentKind.VIDEO_VAE) == []
    assert catalog.list(ComponentKind.AUDIO_VAE) == []


def test_multiple_direct_transformers_have_distinct_legacy_ids(tmp_path):
    _put(tmp_path / "transformer/config.json")
    _put(tmp_path / "transformer_ref/config.json")
    catalog = ComponentCatalog({"official": tmp_path})
    catalog.scan()
    assert [entry.id for entry in catalog.list(ComponentKind.DIT)] == [
        f"{tmp_path.name}/transformer",
        f"{tmp_path.name}/transformer_ref",
    ]
    assert catalog.selection_candidates(ComponentKind.DIT) == []


def test_duplicate_id_scan_keeps_previous_complete_index(tmp_path):
    root = tmp_path / "comfy"
    _put(root / "diffusion_models/A.bin")
    catalog = ComponentCatalog({"comfy": [root, root]})
    catalog.scan()
    assert catalog.selection_candidates(ComponentKind.DIT) == ["A"]
    before = catalog.list()
    clash = _put(root / "diffusion_models/sub/A.other")
    with pytest.raises(CatalogError, match="duplicate 'dit' id 'A'"):
        catalog.scan()
    assert catalog.list() == before
    clash.unlink()
    _put(root / "diffusion_models/B.bin")
    catalog.scan()
    assert catalog.selection_candidates(ComponentKind.DIT) == ["A", "B"]


def test_missing_or_file_root_fails_at_scan_not_construction(tmp_path):
    file = _put(tmp_path / "plain-file")
    for root in (tmp_path / "missing", file):
        catalog = ComponentCatalog({"comfy": root})
        assert catalog.roots_configured()
        with pytest.raises(CatalogError, match="is not a directory"):
            catalog.scan()
        assert len(catalog.list()) == 2


def test_rescan_tracks_files_but_keeps_original_roots(tmp_path):
    root, other = tmp_path / "original", tmp_path / "replacement"
    first = _put(root / "diffusion_models/A.bin")
    _put(other / "diffusion_models/C.bin")
    env = {COMPONENT_ROOTS_ENV: f"comfy={root}"}
    catalog = ComponentCatalog(env=env)
    catalog.scan()
    first.unlink()
    _put(root / "diffusion_models/B.bin")
    env[COMPONENT_ROOTS_ENV] = f"comfy={other}"
    assert catalog.selection_candidates(ComponentKind.DIT) == ["A"]
    catalog.scan()
    assert catalog.selection_candidates(ComponentKind.DIT) == ["B"]


def test_unknown_lookups_are_catalog_errors(roots):
    catalog = ComponentCatalog(roots)
    catalog.scan()
    with pytest.raises(CatalogError, match="cataloged lora ids: 'L', 'Z'"):
        catalog.get(ComponentKind.LORA, "missing")
    for invalid in ("dit", None, 2):
        with pytest.raises(CatalogError):
            catalog.snapshot(invalid)
        with pytest.raises(CatalogError):
            catalog.selection_candidates(invalid)
    for invalid in (None, "", 2):
        with pytest.raises(CatalogError):
            catalog.get(ComponentKind.DIT, invalid)
    with pytest.raises(CatalogError):
        catalog.list("dit")


def test_snapshots_never_mix_entries_and_candidates_across_scans(tmp_path, monkeypatch):
    catalog = ComponentCatalog({"comfy": tmp_path})
    generation = 0
    failures = []
    started = threading.Barrier(2)

    def scan_generation(root, zone):
        nonlocal generation
        generation += 1
        name = f"generation-{generation}"
        return [Component(ComponentKind.DIT, zone, name, str(root / name), selection=name)]

    monkeypatch.setattr(catalog_module, "_scan_root", scan_generation)

    def writer():
        started.wait(timeout=5)
        for _ in range(400):
            catalog.scan()

    thread = threading.Thread(target=writer)
    thread.start()
    started.wait(timeout=5)
    try:
        for _ in range(400):
            entries, candidates = catalog.snapshot(ComponentKind.DIT)
            if candidates != [entry.id for entry in entries]:
                failures.append((entries, candidates))
    finally:
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert not failures
    assert generation == 400
