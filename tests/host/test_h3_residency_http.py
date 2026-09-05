"""HTTP -> real coordinator -> pinned StagePool/inline control dispatch."""

from test_h3_residency_control import _InlineClientBoundary, _omni


def test_host_mounts_runtime_routes_and_switches_the_existing_inline_client():
    from types import SimpleNamespace

    from fastapi import APIRouter, FastAPI
    from fastapi.testclient import TestClient

    from comfy_omni.integrations.vllm_omni.api_phase import mount_components, mount_runtime

    api = SimpleNamespace(router=APIRouter())
    mount_components(api)
    mount_runtime(api)
    mount_components(api)
    mount_runtime(api)
    app = FastAPI()
    app.include_router(api.router)
    prefix = "/v1/comfy-omni/h3/runtime"
    with TestClient(app) as http:
        unavailable = http.get(prefix)
        assert unavailable.status_code == 503
        client = _InlineClientBoundary(0)
        omni = _omni(client)
        app.state.engine_client = omni
        try:
            response = http.post(prefix + "/switch", json={"selection": "b"})
            assert response.status_code == 200, response.text
            assert response.json()["active_selection"] == "b"
            assert client.active == "b" and omni._paused is False
            assert app.state.engine_client is omni
            invalid = http.post(prefix + "/switch", json={"selection": "a", "path": "/unregistered"})
            assert invalid.status_code == 422
            assert client.active == "b"
            status = http.get(prefix)
            assert status.status_code == 200 and status.json()["active_selection"] == "b"
            unloaded = http.post(prefix + "/unload", json={})
            assert unloaded.status_code == 200, unloaded.text
            assert unloaded.json()["weight_residency"] == "released" and omni._paused
            loaded = http.post(prefix + "/load", json={})
            assert loaded.status_code == 200, loaded.text
            assert loaded.json()["weight_residency"] == "loaded" and not omni._paused
            assert len([route for route in api.router.routes if route.path == prefix + "/switch"]) == 1
        finally:
            client._executor.shutdown(wait=True)
