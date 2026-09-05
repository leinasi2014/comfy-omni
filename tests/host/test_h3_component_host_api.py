"""CPU-only HTTP observation through the actual pinned host API module."""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest


@pytest.mark.skipif(importlib.util.find_spec("vllm_omni") is None, reason="requires the pinned host image")
def test_actual_host_api_import_mounts_components_without_starting_an_engine():
    code = r"""
import hashlib, importlib, os, tempfile
from pathlib import Path
with tempfile.TemporaryDirectory(prefix='h3-component-api-') as cache:
    os.environ.update(USER='h3-component-test', LOGNAME='h3-component-test', VLLM_PLUGINS='comfy_omni',
        XDG_CACHE_HOME=cache, VLLM_CACHE_ROOT=cache+'/vllm', TRITON_CACHE_DIR=cache+'/triton',
        TORCHINDUCTOR_CACHE_DIR=cache+'/torch', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    os.environ.pop('H3_FORGE_COMPONENT_ROOTS', None)
    import vllm_omni
    from comfy_omni.plugin import register
    register()
    api = importlib.import_module('vllm_omni.entrypoints.openai.api_server')
    source = Path(api.__file__).read_bytes()
    blob = hashlib.sha1(b'blob '+str(len(source)).encode()+b'\0'+source).hexdigest()
    assert blob == '57adaad08ff28160831f503e639425f250bf4313', blob
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    app.include_router(api.router)
    routes = [route for route in api.router.routes if route.path.startswith('/v1/h3-forge/components')]
    assert len(routes)==3
    register()
    assert len([route for route in api.router.routes if route.path.startswith('/v1/h3-forge/components')])==3
    with TestClient(app) as client:
        for method, suffix in [('get',''),('get','/dit'),('post','/scan')]:
            response=getattr(client,method)('/v1/h3-forge/components'+suffix)
            assert response.status_code==503, (response.status_code,response.text)
            assert response.json()['error']['kind']=='components_not_configured'
        models = Path(cache,'models','diffusion_models')
        models.mkdir(parents=True)
        first = models/'H3-test.bin'
        first.write_bytes(b'catalog fixture only')
        os.environ['H3_FORGE_COMPONENT_ROOTS']='comfy='+str(models.parent)
        response=client.get('/v1/h3-forge/components/dit')
        assert response.status_code==200, response.text
        assert response.json()['selection_candidates']==['H3-test']
        second = models/'H3-second.bin'
        second.write_bytes(b'another directory fixture')
        assert client.get('/v1/h3-forge/components/dit').json()['selection_candidates']==['H3-test']
        response=client.post('/v1/h3-forge/components/scan')
        assert response.status_code==200 and response.json()['scanned'] is True
        assert client.get('/v1/h3-forge/components/dit').json()['selection_candidates']==['H3-second','H3-test']
        assert first.read_bytes()==b'catalog fixture only' and second.read_bytes()==b'another directory fixture'
    print('ACTUAL_HOST_COMPONENT_API_PASSED')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ACTUAL_HOST_COMPONENT_API_PASSED" in result.stdout
