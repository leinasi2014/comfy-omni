"""Missing HTTP routes and unresolved lazy targets through the existing entry.

Characterization: Apache-2.0 h3-forge e9cb011 plugin.py blob
304a776bf4daf1f7a28b1bc6192d320da30421fd, component_catalog/api.py blob
03597ba2952a6d7933fa174cdfe5b1073b234d9d.
"""

from __future__ import annotations

import importlib
import subprocess
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
    monkeypatch.setattr(bootstrap, "_api_state", 0)
    monkeypatch.setattr(bootstrap, "_is_root_process", lambda: True)
    monkeypatch.delenv("H3_FORGE_COMPONENT_ROOTS", raising=False)
    # Deferred finders must not escape a synthetic host test into other imports.
    before = list(sys.meta_path)
    yield registry.register_diffusion_model
    sys.meta_path[:] = before
    helper = sys.modules.get("comfy_omni.integrations.vllm_omni.deferred_import")
    if helper is not None:
        helper._FINDERS.clear()


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


def test_root_defers_until_the_actual_import_body_has_created_its_router(monkeypatch, resident_host, tmp_path):
    pytest.importorskip("fastapi")
    name = "h3_component_api_import_fixture"
    monkeypatch.setattr(bootstrap, "API_SERVER_MODULE", name)
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / f"{name}.py").write_text(
        "from fastapi import APIRouter\nrouter = APIRouter()\nBODY_COMPLETED = True\n", encoding="utf-8"
    )
    bootstrap.register()
    bootstrap.register()
    assert name not in sys.modules
    assert bootstrap._api_state == bootstrap._WAITING
    try:
        module = importlib.import_module(name)
        assert module.BODY_COMPLETED
        assert module.router.h3_forge_components_mounted
        assert bootstrap._api_state == bootstrap._REGISTERED
        assert len(resident_host.call_args_list) == 2
        assert {route.path for route in module.router.routes} == {
            "/v1/h3-forge/components",
            "/v1/h3-forge/components/scan",
            "/v1/h3-forge/components/{kind}",
        }
    finally:
        sys.modules.pop(name, None)


def test_api_failure_retries_without_registering_architectures_or_mounting_twice(monkeypatch, resident_host):
    fastapi = pytest.importorskip("fastapi")
    module = types.ModuleType(API_SERVER)
    module.router = fastapi.APIRouter()
    monkeypatch.setitem(sys.modules, API_SERVER, module)
    attempts = []
    later = types.ModuleType("component_api_retry_fixture")

    def contribute(api):
        attempts.append(api)
        bootstrap.register()  # A contribution cannot recurse through the coordinator.
        if len(attempts) == 1:
            raise RuntimeError("retryable API contribution fault")

    later.contribute = contribute
    monkeypatch.setitem(sys.modules, later.__name__, later)
    monkeypatch.setattr(
        bootstrap, "_API_CONTRIBUTIONS", (*bootstrap._API_CONTRIBUTIONS, (later.__name__, "contribute"))
    )
    with pytest.raises(RuntimeError, match="retryable API"):
        bootstrap.register()
    assert bootstrap._registration_state == bootstrap._REGISTERED
    assert bootstrap._api_state == bootstrap._NEW
    assert len(module.router.routes) == 3
    bootstrap.register()
    bootstrap.register()
    assert bootstrap._api_state == bootstrap._REGISTERED
    assert len(module.router.routes) == 3 and len(attempts) == 2
    assert len(resident_host.call_args_list) == 2


def test_partial_api_target_can_retry_with_app_shaped_host(monkeypatch, resident_host):
    fastapi = pytest.importorskip("fastapi")
    module = types.ModuleType(API_SERVER)
    monkeypatch.setitem(sys.modules, API_SERVER, module)
    with pytest.raises(RuntimeError, match="no APIRouter"):
        bootstrap.register()
    assert bootstrap._api_state == bootstrap._NEW
    module.app = fastapi.FastAPI()
    bootstrap.register()
    assert module.app.h3_forge_components_mounted
    assert len(resident_host.call_args_list) == 2


def test_worker_process_registers_without_api_imports_hooks_or_catalog_io():
    code = r"""
import importlib.abc, os, sys, types
from pathlib import Path
from unittest.mock import patch
class NoHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','fastapi','vllm'} or fullname.startswith('comfy_omni.api'):
            raise AssertionError('worker attempted heavy/API import: ' + fullname)
sys.meta_path.insert(0, NoHeavy())
host = types.ModuleType('vllm_omni'); host.__path__=[]
registry = types.ModuleType('vllm_omni.diffusion.registry')
calls=[]
registry.register_diffusion_model=lambda *a, **k: calls.append(a)
sys.modules['vllm_omni']=host
sys.modules[registry.__name__]=registry
from comfy_omni.integrations.vllm_omni import bootstrap
before=list(sys.meta_path)
with (
    patch.object(bootstrap.multiprocessing,'parent_process',return_value=object()),
    patch.object(Path,'iterdir',side_effect=AssertionError('scan')),
):
    bootstrap.register(); bootstrap.register()
assert len(calls)==2 and sys.meta_path==before
assert bootstrap._api_state==bootstrap._NEW
assert 'comfy_omni.integrations.vllm_omni.deferred_import' not in sys.modules
assert not any(n.startswith('comfy_omni.runtime.components') for n in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX fork inheritance contract")
def test_fork_worker_discards_root_pending_api_hook_without_touching_parent():
    code = r"""
import multiprocessing, sys, types
host = types.ModuleType('vllm_omni'); host.__path__=[]
registry = types.ModuleType('vllm_omni.diffusion.registry')
calls=[]
registry.register_diffusion_model=lambda *a, **k: calls.append(a)
sys.modules['vllm_omni']=host
sys.modules[registry.__name__]=registry
from comfy_omni.integrations.vllm_omni import bootstrap
bootstrap.register()
from comfy_omni.integrations.vllm_omni import deferred_import
finder=deferred_import._FINDERS[bootstrap.API_SERVER_MODULE]
assert finder in sys.meta_path and bootstrap._api_state==bootstrap._WAITING
def child():
    assert multiprocessing.parent_process() is not None
    bootstrap.register()
    assert bootstrap._api_state==bootstrap._NEW
    assert finder not in sys.meta_path
    assert not deferred_import._FINDERS
    assert len(calls)==2
    assert 'fastapi' not in sys.modules and 'torch' not in sys.modules
    assert not any(n.startswith('comfy_omni.api') for n in sys.modules)
process=multiprocessing.get_context('fork').Process(target=child)
process.start()
process.join(10)
if process.is_alive():
    process.terminate(); process.join(5)
    raise AssertionError('worker registration stranded on inherited state')
assert process.exitcode==0, 'fork child retained the root API hook/state'
assert deferred_import._FINDERS[bootstrap.API_SERVER_MODULE] is finder
assert finder in sys.meta_path and bootstrap._api_state==bootstrap._WAITING
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
