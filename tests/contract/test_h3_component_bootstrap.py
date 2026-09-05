"""Missing HTTP routes and unresolved lazy targets through the existing entry.

Characterization: Apache-2.0 h3-forge e9cb011 plugin.py blob
304a776bf4daf1f7a28b1bc6192d320da30421fd, component_catalog/api.py blob
03597ba2952a6d7933fa174cdfe5b1073b234d9d.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest import mock

import pytest

from comfy_omni.integrations.vllm_omni import bootstrap

API_SERVER = "vllm_omni.entrypoints.openai.api_server"


@pytest.fixture
def resident_host(monkeypatch):
    host = types.ModuleType("vllm_omni")
    host.__path__ = []
    registry = types.ModuleType("vllm_omni.diffusion.registry")
    registry.register_diffusion_model = mock.Mock()
    monkeypatch.setitem(sys.modules, "vllm_omni", host)
    monkeypatch.setitem(sys.modules, "vllm_omni.diffusion.registry", registry)
    monkeypatch.setattr(bootstrap, "_registration_state", 0)
    monkeypatch.setattr(bootstrap, "_is_root_process", lambda: True)
    monkeypatch.delenv("H3_FORGE_COMPONENT_ROOTS", raising=False)
    # Deferred finders must not escape a synthetic host test into other imports.
    before = list(sys.meta_path)
    yield registry.register_diffusion_model
    sys.meta_path[:] = before


@pytest.mark.parametrize("method,path", [("get", ""), ("get", "/dit"), ("post", "/scan")])
def test_root_api_phase_exposes_existing_component_http_contract(monkeypatch, resident_host, method, path):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    api_server = types.ModuleType(API_SERVER)
    api_server.router = fastapi.APIRouter()
    monkeypatch.setitem(sys.modules, API_SERVER, api_server)
    bootstrap.register()
    app = fastapi.FastAPI()
    app.include_router(api_server.router)
    with TestClient(app) as client:
        response = getattr(client, method)("/v1/h3-forge/components" + path)
    assert response.status_code == 503, "root API phase must mount the configured-catalog gate, not return route 404"
    assert response.json()["schema"] == "h3_forge.error/v1"
    assert response.json()["error"]["kind"] == "components_not_configured"


def test_dense_wire_key_resolves_to_the_existing_verified_package_dispatcher(resident_host):
    bootstrap.register()
    calls = {call.args[0]: call.args for call in resident_host.call_args_list}
    dense = calls["MiniMaxH3DensePipeline"]
    source = Path(__file__).resolve().parents[2] / "src"
    module_file = source.joinpath(*dense[1].split(".")).with_suffix(".py")
    assert module_file.is_file(), "a published lazy architecture key must resolve to an existing module"
    primary = calls["MiniMaxH3Pipeline"]
    assert dense[1:] == primary[1:], "both preserved wire keys must enter the same verified package dispatcher"
