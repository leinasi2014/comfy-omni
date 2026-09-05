from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest
from beta4_host_fixture import actual_cpu_host

from comfy_omni.integrations.vllm_omni.residency import H3ResidencyWorkerExtension
from comfy_omni.runtime.hotel import H3TensorDescriptor, PreparedH3DitSelection

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("vllm_omni") is None,
    reason="the actual pinned host image is required",
)


def _descriptor(selection: str) -> PreparedH3DitSelection:
    return PreparedH3DitSelection(
        selection=selection,
        identity=f"registered:{selection}",
        execution_profile="test-h3-pinned-17285",
        tensors=(
            H3TensorDescriptor(name="weight", shape=(2,), dtype="torch.float32"),
            H3TensorDescriptor(name="bias", shape=(1,), dtype="torch.float32"),
        ),
        logical_bytes=12,
    )


def _worker_wrapper(host):
    from vllm_omni.diffusion.worker.diffusion_worker import WorkerWrapperBase

    torch = host.torch

    class TinyLoadableDiT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
            self.bias = torch.nn.Parameter(torch.tensor([3.0]))
            self.post_load_count = 0
            self.beta4_ready = True

        def load_weights(self, weights):
            self.beta4_ready = False
            loaded = set()
            with torch.no_grad():
                for name, value in weights:
                    getattr(self, name).copy_(value)
                    loaded.add(name)
            self.beta4_ready = True
            return loaded

        def post_load_weights(self):
            self.post_load_count += 1

    class TrustedPipeline:
        def __init__(self):
            self.transformer = TinyLoadableDiT()
            self.text_encoder = object()
            self.video_vae = object()
            self.audio_vae = object()
            self.comfy_omni_active_h3_dit = _descriptor("a")
            self._registered = {
                "a": (torch.tensor([1.0, 2.0]), torch.tensor([3.0])),
                "b": (torch.tensor([7.0, 11.0]), torch.tensor([13.0])),
                "broken": (torch.tensor([17.0, 19.0]), RuntimeError("broken source")),
            }
            self.reads = {name: 0 for name in self._registered}

        def comfy_omni_prepare_h3_dit(self, selection):
            if selection not in self._registered:
                raise KeyError(f"selection is not registered: {selection}")
            return _descriptor(selection)

        def comfy_omni_iter_h3_dit(self, prepared):
            self.reads[prepared.selection] += 1
            weight, bias = self._registered[prepared.selection]
            yield "weight", weight.clone()
            if isinstance(bias, Exception):
                raise bias
            yield "bias", bias.clone()

    class PinnedDiffusionWorkerBoundary:
        pass

    pipeline = TrustedPipeline()
    wrapper = object.__new__(WorkerWrapperBase)
    wrapper.base_worker_class = PinnedDiffusionWorkerBoundary
    wrapper.worker_extension_cls = H3ResidencyWorkerExtension
    wrapper.custom_pipeline_args = None
    worker_class = wrapper._prepare_worker_class()
    worker = object.__new__(worker_class)
    worker.model_runner = SimpleNamespace(
        pipeline=pipeline,
        state_cache={},
        input_batch=None,
        cache_backend=None,
        offload_backend=None,
    )
    worker.lora_manager = None
    worker._step_lora_state = {}
    wrapper.worker = worker
    return wrapper, pipeline


def test_worker_rpc_switches_a_to_b_to_a_without_replacing_shared_objects() -> None:
    with actual_cpu_host() as host:
        wrapper, pipeline = _worker_wrapper(host)
        transformer = pipeline.transformer
        shared = (pipeline.text_encoder, pipeline.video_vae, pipeline.audio_vae)

        prepared = wrapper.execute_method("comfy_omni_prepare_h3_dit", "tx-b", "b")
        assert prepared["phase"] == "prepared"
        assert pipeline.reads["b"] == 0

        committed = wrapper.execute_method("comfy_omni_commit_h3_dit", "tx-b")
        assert committed["phase"] == "committed"
        assert pipeline.comfy_omni_active_h3_dit.selection == "a"
        assert pipeline.transformer is transformer
        assert (pipeline.text_encoder, pipeline.video_vae, pipeline.audio_vae) == shared
        host.torch.testing.assert_close(transformer.weight, host.torch.tensor([7.0, 11.0]))

        wrapper.execute_method("comfy_omni_finalize_h3_dit", "tx-b")
        assert pipeline.comfy_omni_active_h3_dit.selection == "b"

        wrapper.execute_method("comfy_omni_prepare_h3_dit", "tx-a", "a")
        wrapper.execute_method("comfy_omni_commit_h3_dit", "tx-a")
        wrapper.execute_method("comfy_omni_finalize_h3_dit", "tx-a")
        assert pipeline.comfy_omni_active_h3_dit.selection == "a"
        assert pipeline.transformer is transformer
        host.torch.testing.assert_close(transformer.weight, host.torch.tensor([1.0, 2.0]))


def test_failed_commit_restores_active_selection_from_its_registered_source() -> None:
    with actual_cpu_host() as host:
        wrapper, pipeline = _worker_wrapper(host)
        wrapper.execute_method("comfy_omni_prepare_h3_dit", "tx-broken", "broken")

        with pytest.raises(RuntimeError, match="commit failed.*rollback restored"):
            wrapper.execute_method("comfy_omni_commit_h3_dit", "tx-broken")

        status = wrapper.execute_method("comfy_omni_h3_residency_status")
        assert status["phase"] == "rolled_back"
        assert status["active_selection"] == "a"
        assert pipeline.comfy_omni_active_h3_dit.selection == "a"
        host.torch.testing.assert_close(pipeline.transformer.weight, host.torch.tensor([1.0, 2.0]))
        host.torch.testing.assert_close(pipeline.transformer.bias, host.torch.tensor([3.0]))


def test_positive_budget_still_opens_payload_only_at_commit_and_caches_once() -> None:
    with actual_cpu_host() as host:
        wrapper, pipeline = _worker_wrapper(host)

        wrapper.execute_method("comfy_omni_prepare_h3_dit", "tx-b", "b", cpu_cache_budget_bytes=12)
        assert pipeline.reads["b"] == 0
        wrapper.execute_method("comfy_omni_commit_h3_dit", "tx-b")
        assert pipeline.reads["b"] == 1
        status = wrapper.execute_method("comfy_omni_finalize_h3_dit", "tx-b")
        assert status["cpu_cached_bytes"] == 12
        assert status["cpu_cached_identity"] == "registered:b"


def test_busy_runner_is_rejected_before_payload_is_opened() -> None:
    with actual_cpu_host() as host:
        wrapper, pipeline = _worker_wrapper(host)
        wrapper.worker.model_runner.state_cache["request"] = object()

        with pytest.raises(RuntimeError, match="not quiescent"):
            wrapper.execute_method("comfy_omni_prepare_h3_dit", "tx-b", "b")

        assert pipeline.reads["b"] == 0
