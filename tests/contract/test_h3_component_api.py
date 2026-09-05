# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: h3-forge contributors
"""Actual HTTP characterization of the legacy component directory API.

Derived from h3-forge e9cb011d00b028c149db3978de246c54f6e34acc:
api.py blob 03597ba2952a6d7933fa174cdfe5b1073b234d9d;
test_component_api.py blob 434f944bf5b581a7b8571595012285de8a85adb8.
Fixtures contain arbitrary bytes and establish discovery, not model support.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from comfy_omni.api.routes import components as api
from comfy_omni.runtime.components.catalog import CatalogError, ComponentCatalog


def _put(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"directory fixture; not a checkpoint")
    return path


@pytest.fixture(autouse=True)
def isolated_catalog(monkeypatch):
    api.reset_catalog()
    monkeypatch.delenv(api.ENV_ROOTS, raising=False)
    yield
    api.reset_catalog()


@pytest.fixture
def client():
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    app = fastapi.FastAPI()

    @app.get("/unrelated")
    def unrelated(count: int):
        return {"count": count}

    app.include_router(api.build_router())
    assert api._CATALOG is None
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def configured(tmp_path, monkeypatch):
    comfy, official = tmp_path / "comfy", tmp_path / "official"
    _put(comfy / "diffusion_models/B.bin")
    _put(comfy / "diffusion_models/sub/A.unchecked")
    _put(comfy / "loras/style.json")
    _put(official / "Ref2VA/transformer/config.json")
    monkeypatch.setenv(api.ENV_ROOTS, json.dumps({"comfy": str(comfy), "official": str(official)}))
    return comfy, official


def _assert_error(response, status, kind):
    assert response.status_code == status
    body = response.json()
    assert set(body) == {"schema", "error"}
    assert body["schema"] == "h3_forge.error/v1"
    assert set(body["error"]) == {"kind", "message"}
    assert body["error"]["kind"] == kind
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]
    return body["error"]["message"]


@pytest.mark.parametrize("method,suffix", [("get", ""), ("get", "/dit"), ("post", "/scan"), ("get", "/unknown")])
def test_unconfigured_http_is_503_before_kind_resolution(client, method, suffix):
    response = getattr(client, method)(api.API_PREFIX + suffix)
    message = _assert_error(response, 503, "components_not_configured")
    assert "H3_FORGE_COMPONENT_ROOTS" in message
    assert "comfy=/data/comfy;official=/data/MiniMax-H3-official" in message
    assert api._CATALOG is None


def test_first_get_scans_once_and_emits_exact_legacy_json(client, configured, monkeypatch):
    calls = []
    original = ComponentCatalog.scan

    def count_scan(catalog):
        calls.append(catalog)
        return original(catalog)

    monkeypatch.setattr(ComponentCatalog, "scan", count_scan)
    response = client.get(api.API_PREFIX)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"schema", "kinds"}
    assert body["schema"] == "h3_forge.components.list/v1"
    assert list(body["kinds"]) == ["dit", "schedule", "lora"]
    assert [entry["id"] for entry in body["kinds"]["dit"]] == ["A", "B", "official"]
    assert body["kinds"]["dit"][0] == {
        "kind": "dit",
        "zone": "comfy",
        "id": "A",
        "path": str(configured[0] / "diffusion_models/sub/A.unchecked"),
        "locked": False,
        "selection": "A",
        "contract_digest": None,
    }
    official = body["kinds"]["dit"][-1]
    assert official["locked"] is True and official["selection"] is None
    assert official["path"] == str(configured[1] / "Ref2VA/transformer")
    assert body["kinds"]["schedule"] == [
        {
            "kind": "schedule",
            "zone": None,
            "id": name,
            "path": "",
            "locked": False,
            "selection": name,
            "contract_digest": None,
        }
        for name in ("turbo_4step", "turbo_8step")
    ]
    assert body["kinds"]["lora"][0]["selection"] is None
    for entries in body["kinds"].values():
        for entry in entries:
            assert set(entry) == {"kind", "zone", "id", "path", "locked", "selection", "contract_digest"}
            assert entry["contract_digest"] is None
    assert client.get(api.API_PREFIX).json() == body
    assert client.get(api.API_PREFIX + "/dit").status_code == 200
    assert len(calls) == 1


def test_kind_response_case_candidates_and_empty_kind(client, configured):
    response = client.get(api.API_PREFIX + "/DiT")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"schema", "kind", "selection_candidates", "components"}
    assert body["schema"] == "h3_forge.components.list/v1"
    assert body["kind"] == "dit"
    assert body["selection_candidates"] == ["A", "B"]
    assert [entry["id"] for entry in body["components"]] == ["A", "B", "official"]
    assert client.get(api.API_PREFIX + "/lora").json()["selection_candidates"] == ["style"]
    assert client.get(api.API_PREFIX + "/tool").json() == {
        "schema": "h3_forge.components.list/v1",
        "kind": "tool",
        "selection_candidates": [],
        "components": [],
    }
    message = _assert_error(client.get(api.API_PREFIX + "/scan"), 404, "H3_COMPONENT_KIND_UNKNOWN")
    assert "dit, text_encoder, video_vae, audio_vae, schedule, lora, tool" in message


@pytest.mark.parametrize(
    "bad,status,kind",
    [
        ("", 503, "components_not_configured"),
        ("{bad", 500, "components_misconfigured"),
        ("comfy={missing}", 500, "components_scan_failed"),
    ],
)
def test_configuration_failure_is_retryable(client, tmp_path, monkeypatch, bad, status, kind):
    monkeypatch.setenv(api.ENV_ROOTS, bad.replace("{missing}", str(tmp_path / "missing")))
    _assert_error(client.get(api.API_PREFIX), status, kind)
    assert api._CATALOG is None
    root = tmp_path / "fixed"
    _put(root / "diffusion_models/recovered.bin")
    monkeypatch.setenv(api.ENV_ROOTS, f"comfy={root}")
    response = client.get(api.API_PREFIX + "/dit")
    assert response.status_code == 200
    assert response.json()["selection_candidates"] == ["recovered"]


def test_scan_refreshes_index_without_changing_roots_or_files(client, configured, tmp_path, monkeypatch):
    before = client.get(api.API_PREFIX).json()
    catalog = api._CATALOG
    comfy = configured[0]
    (comfy / "diffusion_models/B.bin").unlink()
    added = _put(comfy / "diffusion_models/C.anything")
    replacement = tmp_path / "other"
    _put(replacement / "diffusion_models/ignored.bin")
    monkeypatch.setenv(api.ENV_ROOTS, f"comfy={replacement}")
    assert client.get(api.API_PREFIX).json() == before
    response = client.post(api.API_PREFIX + "/scan")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"schema", "scanned", "kinds"}
    assert body["schema"] == "h3_forge.components.scan/v1" and body["scanned"] is True
    assert [entry["id"] for entry in body["kinds"]["dit"]] == ["A", "C", "official"]
    assert api._CATALOG is catalog
    assert added.read_bytes() == b"directory fixture; not a checkpoint"


def test_failed_rescan_preserves_last_successful_snapshot(client, configured):
    before = client.get(api.API_PREFIX).json()
    catalog = api._CATALOG
    _put(configured[0] / "diffusion_models/duplicate/A.bin")
    _assert_error(client.post(api.API_PREFIX + "/scan"), 500, "components_scan_failed")
    assert api._CATALOG is catalog
    assert client.get(api.API_PREFIX).json() == before


def test_kind_route_uses_one_atomic_snapshot(client, configured, monkeypatch):
    assert client.get(api.API_PREFIX).status_code == 200

    def forbidden(*args):
        raise AssertionError("split catalog reads can mix scan generations")

    monkeypatch.setattr(api._CATALOG, "list", forbidden)
    monkeypatch.setattr(api._CATALOG, "selection_candidates", forbidden)
    response = client.get(api.API_PREFIX + "/dit")
    assert response.status_code == 200
    assert response.json()["selection_candidates"] == ["A", "B"]


@pytest.mark.parametrize(
    "exception,status,kind",
    [
        (CatalogError("scan issue"), 500, "components_scan_failed"),
        (ValueError("invalid value"), 422, "invalid_request"),
        (TypeError("invalid type"), 422, "invalid_request"),
        (RuntimeError("unexpected"), 500, "internal_error"),
    ],
)
@pytest.mark.parametrize(
    "method,suffix,operation", [("get", "", "list"), ("get", "/dit", "snapshot"), ("post", "/scan", "scan")]
)
def test_catalog_operation_error_envelopes(
    client, configured, monkeypatch, exception, status, kind, method, suffix, operation
):
    assert client.get(api.API_PREFIX).status_code == 200

    def fail(*args):
        raise exception

    monkeypatch.setattr(api._CATALOG, operation, fail)
    _assert_error(getattr(client, method)(api.API_PREFIX + suffix), status, kind)


def test_component_routes_do_not_replace_host_validation_or_add_mutations(client, configured):
    unrelated = client.get("/unrelated?count=bad")
    assert unrelated.status_code == 422
    assert "detail" in unrelated.json() and "schema" not in unrelated.json()
    for method, suffix in (("put", "/dit"), ("delete", "/dit"), ("post", ""), ("post", "/dit")):
        assert getattr(client, method)(api.API_PREFIX + suffix).status_code == 405
