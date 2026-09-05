"""Acceptance arguments through the pinned stage parser to the worker boundary.

CUDA platform metadata and the final inline-client constructor are substituted
in the CPU container, before any model loading.
Stage resolution, replica layout, diffusion configuration and extension-class
resolution are the actual host implementations; no model payload is present.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import runpy
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("vllm_omni") is None,
    reason="requires the actual pinned host image",
)


def _assert_blob(module, expected):
    source = Path(module.__file__).read_bytes()
    assert hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source).hexdigest() == expected


def test_acceptance_arguments_reach_actual_worker_config_without_model_loading(tmp_path, monkeypatch):
    monkeypatch.setenv("USER", "h3-startup-config-test")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    import torch
    from vllm_omni.diffusion import data, inline_stage_diffusion_client
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _resolve_component_quant_config,
        _resolve_minimax_h3_text_encoder_quant_config,
    )
    from vllm_omni.diffusion.worker.diffusion_worker import WorkerWrapperBase
    from vllm_omni.engine import async_omni_engine, stage_engine_startup, stage_init_utils, stage_runtime
    from vllm_omni.entrypoints import utils as entrypoint_utils
    from vllm_omni.platforms.interface import UnspecifiedOmniPlatform

    from comfy_omni.integrations.vllm_omni.residency import H3ResidencyWorkerExtension

    for module, expected in (
        (async_omni_engine, "12236294e8593f5ca4fa13640b1a511d19180266"),
        (stage_init_utils, "a4c3e1650aa34bbdde821d3820230ba39bd111f1"),
        (stage_runtime, "dea5242d1e457696565a7960e5c4badc77daeb0f"),
        (data, "9adda521aef3baed516a448ac3efdb5eecbb85ac"),
    ):
        _assert_blob(module, expected)

    def no_cuda(*args, **kwargs):
        pytest.fail("configuration parsing attempted CUDA initialization")

    monkeypatch.setattr(torch.cuda, "_lazy_init", no_cuda)

    # The CUDA platform class imports libcuda, unavailable in this CPU-only
    # container. Its two metadata values come from pinned cuda/platform.py:38
    # and vLLM's CUDA_VISIBLE_DEVICES interface, without mocking any parser.
    class ConfigPlatform(UnspecifiedOmniPlatform):
        device_control_env_var = "CUDA_VISIBLE_DEVICES"

        @classmethod
        def get_default_stage_config_path(cls):
            return "vllm_omni/deploy"

    for module in (entrypoint_utils, stage_init_utils, stage_runtime, stage_engine_startup):
        monkeypatch.setattr(module, "current_omni_platform", ConfigPlatform())
    components = tmp_path / "existing-components"
    components.mkdir()
    (components / "model_index.json").write_text(json.dumps({"_class_name": "MiniMaxH3Pipeline"}), encoding="utf-8")
    harness = runpy.run_path(str(Path(__file__).parents[2] / "scripts/acceptance/h3_raw_runtime.py"))
    options = harness["_engine_arguments"](
        SimpleNamespace(
            component_root=components,
            source_a=tmp_path / "original-a",
            source_b=tmp_path / "original-b",
            init_timeout=120,
        )
    )
    expected_sources = options["additional_config"]["comfy_omni_h3"]
    engine = object.__new__(async_omni_engine.AsyncOmniEngine)
    engine._omni_lb_policy = "random"
    _, stages = engine._resolve_stage_configs(str(components), options.copy(), trust_remote_code=True)
    replicas, device_map = stage_init_utils.compute_replica_layout(stages)
    assert len(stages) == 1 and replicas == [1] and device_map == {}
    metadata = stage_init_utils.extract_stage_metadata(stages[0])
    assert metadata.stage_id == 0 and metadata.stage_type == "diffusion"

    observed = []

    class ConfigOnlyInlineClient:
        def __init__(self, model, od_config, metadata, batch_size=1):
            observed.append((model, od_config, metadata, batch_size))

    monkeypatch.setattr(inline_stage_diffusion_client, "InlineStageDiffusionClient", ConfigOnlyInlineClient)
    runtime = object.__new__(stage_runtime.StageRuntime)
    runtime._model = str(components)
    runtime._stage_configs = stages
    runtime._num_stages = len(stages)
    runtime._diffusion_batch_size = 1
    runtime._init_visible_devices_baseline = "0,1"
    runtime._spawn_device_lock = threading.Lock()
    plan = stage_init_utils.ReplicaInitPlan(
        replica_id=0,
        num_replicas=replicas[0],
        launch_mode="local",
        stage_cfg=stages[0],
        metadata=metadata,
        stage_connector_spec={},
        omni_kv_connector=(None, None, None),
    )
    client = runtime._initialize_local_diffusion_replica(plan, 120)
    assert isinstance(client, ConfigOnlyInlineClient) and len(observed) == 1
    model, config, worker_metadata, batch_size = observed[0]
    assert model == config.model == str(components)
    assert worker_metadata.stage_id == 0 and batch_size == 1
    assert config.additional_config["comfy_omni_h3"] == expected_sources
    assert config.worker_extension_cls == "comfy_omni.integrations.vllm_omni.residency.H3ResidencyWorkerExtension"
    assert config.parallel_config.tensor_parallel_size == 2
    assert config.parallel_config.text_encoder_tp_size == 2
    assert config.parallel_config.data_parallel_size == 1
    assert config.parallel_config.world_size == config.num_gpus == 2
    assert _resolve_component_quant_config(config.quantization_config, "transformer") is None
    assert _resolve_minimax_h3_text_encoder_quant_config(config.quantization_config).get_name() == "int8"
    assert config.cache_backend == "none"
    assert config.enable_distributed_layerwise_offload is False

    class EmptyWorker:
        pass

    wrapper = object.__new__(WorkerWrapperBase)
    wrapper.base_worker_class = EmptyWorker
    wrapper.worker_extension_cls = config.worker_extension_cls
    wrapper.custom_pipeline_args = None
    assert issubclass(wrapper._prepare_worker_class(), H3ResidencyWorkerExtension)
    assert torch.cuda.is_initialized() is False
    assert not (tmp_path / "original-a").exists() and not (tmp_path / "original-b").exists()
