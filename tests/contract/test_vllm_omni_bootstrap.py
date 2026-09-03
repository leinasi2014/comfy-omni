from __future__ import annotations

import sys
import types
from unittest import mock

import comfy_omni.plugin


def test_bootstrap_registers_frozen_architectures_once_into_a_resident_host(monkeypatch) -> None:
    from comfy_omni.integrations.vllm_omni import bootstrap

    registry = mock.Mock()
    host = types.ModuleType("vllm_omni")
    host.__path__ = []
    diffusion = types.ModuleType("vllm_omni.diffusion")
    diffusion.__path__ = []
    registry_module = types.ModuleType("vllm_omni.diffusion.registry")
    registry_module.register_diffusion_model = registry
    host.diffusion = diffusion
    diffusion.registry = registry_module

    monkeypatch.setitem(sys.modules, "vllm_omni", host)
    monkeypatch.setitem(sys.modules, "vllm_omni.diffusion", diffusion)
    monkeypatch.setitem(sys.modules, "vllm_omni.diffusion.registry", registry_module)

    bootstrap._registration_state = 0

    comfy_omni.plugin.register()

    expected = [
        mock.call(
            "MiniMaxH3Pipeline",
            "comfy_omni.integrations.vllm_omni.pipelines.runtime_pipeline",
            "H3ComfyMiniMaxH3Pipeline",
            post_process_func_name="get_minimax_h3_post_process_func",
        ),
        mock.call(
            "MiniMaxH3DensePipeline",
            "comfy_omni.integrations.vllm_omni.pipelines.dense_pipeline",
            "MiniMaxH3DensePipeline",
            post_process_func_name="get_minimax_h3_post_process_func",
        ),
    ]
    assert registry.mock_calls == expected

    comfy_omni.plugin.register()

    assert registry.mock_calls == expected
