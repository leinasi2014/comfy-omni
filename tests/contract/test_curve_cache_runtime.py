"""Exercise the registered package router before optional host dependencies load."""

from __future__ import annotations

import importlib
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pytest

RUNTIME = "comfy_omni.integrations.vllm_omni.pipelines.runtime_pipeline"
CACHE = "comfy_omni.integrations.vllm_omni.pipelines.cache_pipeline"


def _router(monkeypatch):
    parent = None
    for name in (
        "vllm_omni",
        "vllm_omni.diffusion",
        "vllm_omni.diffusion.models",
        "vllm_omni.diffusion.models.minimax_h3",
        "vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3",
    ):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        if parent is not None:
            setattr(parent, name.rsplit(".", 1)[-1], module)
        parent = module

    class OfficialPipeline:
        calls = []

        def __init__(self, **kwargs):
            self.calls.append(kwargs)

    parent.MiniMaxH3Pipeline = OfficialPipeline
    parent.get_minimax_h3_post_process_func = lambda: None
    monkeypatch.delitem(sys.modules, RUNTIME, raising=False)
    return importlib.import_module(RUNTIME), OfficialPipeline


def _package(tmp_path: Path):
    root = tmp_path / "package"
    partition = root / "Ref2VA"
    partition.mkdir(parents=True)
    (root / "h3-comfy-package.json").write_text("{}")
    (root / "model_index.json").write_text("{}")
    (partition / "model_index.json").write_text("{}")
    return root, partition


def test_registered_pipeline_routes_a_verified_curve_binding(monkeypatch, tmp_path):
    router, official = _router(monkeypatch)
    root, partition = _package(tmp_path)
    binding = types.SimpleNamespace(package_root=root, partition_path=partition, curve_cache=object())
    monkeypatch.setattr(router, "validate_runtime_package", lambda path: binding)
    cache = types.ModuleType(CACHE)

    class H3CurveCachePipeline:
        def __init__(self, *, od_config, package, prefix=""):
            self.comfy_omni_package = package
            self.prefix = prefix

        def forward(self, request):
            return "exact-cache-output"

    cache.H3CurveCachePipeline = H3CurveCachePipeline
    monkeypatch.setitem(sys.modules, CACHE, cache)
    pipeline = router.H3ComfyMiniMaxH3Pipeline(od_config=types.SimpleNamespace(model=str(root)), prefix="worker")
    assert isinstance(pipeline, H3CurveCachePipeline), "verified curve package reached the uncached official pipeline"
    assert pipeline.forward(None) == "exact-cache-output"
    assert pipeline.comfy_omni_package is binding
    assert official.calls == []


def test_registered_pipeline_resolves_legacy_native_partition(monkeypatch, tmp_path):
    router, _ = _router(monkeypatch)
    root, partition = _package(tmp_path)
    assert router._resolve_package_root(partition) == root


def test_cache_construction_failure_never_falls_back(monkeypatch, tmp_path):
    router, official = _router(monkeypatch)
    root, partition = _package(tmp_path)
    binding = types.SimpleNamespace(package_root=root, partition_path=partition, curve_cache=object())
    monkeypatch.setattr(router, "validate_runtime_package", lambda path: binding)
    cache = types.ModuleType(CACHE)

    def fail(**kwargs):
        raise RuntimeError("compiled cache unavailable")

    cache.H3CurveCachePipeline = fail
    monkeypatch.setitem(sys.modules, CACHE, cache)
    with pytest.raises(RuntimeError, match="compiled cache unavailable"):
        router.H3ComfyMiniMaxH3Pipeline(od_config=types.SimpleNamespace(model=str(root)))
    assert official.calls == []


@dataclass
class _Sampling:
    extra_args: dict = field(default_factory=lambda: {"h3_forge": {"api_version": 1, "task": "ref2va"}})
    num_inference_steps: int = 5
    height: int = 480
    width: int = 864


@dataclass
class _Request:
    sampling_params: _Sampling = field(default_factory=_Sampling)
    prompt: dict = field(default_factory=lambda: {"prompt": "ordinary scene"})


@dataclass
class _Batch:
    requests: list = field(default_factory=lambda: [_Request()])


@pytest.fixture
def cache_host(monkeypatch, tmp_path):
    """Characterized from h3-forge e9cb011 tests/test_runtime_pipeline.py.

    Source blob 64771fff9bfca02ad509b100a189dd5b8f19fd43, Apache-2.0.
    Real Torch tensors and sidecar bytes exercise the runtime; package-verifier
    tests separately establish the full production binding accepted here.
    """
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    from comfy_omni.runtime.h3.package_binding import AUDITED_PRODUCER, CurveCacheBinding
    from comfy_omni.runtime.h3.schedule import build_h3_schedule_contract

    _router(monkeypatch)
    host = types.SimpleNamespace(
        step=None,
        construct_failure=False,
        partition="ref2va",
        fail_diffuse=False,
        forward_calls=[],
        before_forward=lambda: None,
    )
    for name in (
        "vllm_omni.diffusion.forward_context",
        "vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer",
        "vllm_omni.diffusion.models.minimax_h3.denoise_loop",
        "vllm_omni.diffusion.models.minimax_h3.time_request",
        "vllm_omni.errors",
    ):
        module = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        monkeypatch.setattr(sys.modules[parent], child, module, raising=False)
    transformer = sys.modules["vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer"]
    pipeline = sys.modules["vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3"]
    denoise = sys.modules["vllm_omni.diffusion.models.minimax_h3.denoise_loop"]
    time_request = sys.modules["vllm_omni.diffusion.models.minimax_h3.time_request"]
    forward = sys.modules["vllm_omni.diffusion.forward_context"]

    class ClientError(ValueError):
        def __init__(self, message, *, status_code=400, error_type="client_error"):
            super().__init__(message)
            self.status_code = status_code
            self.error_type = error_type

    class DiT(torch.nn.Module):
        hidden_size = 8

        def __init__(self, od_config, quant_config=None):
            super().__init__()
            if host.construct_failure:
                raise RuntimeError("injected constructor failure")
            self.time_embedder = transformer.MiniMaxH3TimeEmbedder(self, prefix="time_embedder")
            self.blocks = torch.nn.ModuleList(
                [
                    transformer.MiniMaxH3AdalnProj(
                        self, 96, quant_config, expand_ratio=6, modality_num=2, prefix=f"blocks.{i}.adaln_proj"
                    )
                    for i in range(50)
                ]
            )
            self.final = transformer.MiniMaxH3AdalnProj(
                self, 32, quant_config, expand_ratio=2, modality_num=2, prefix="final_layer.adaln_proj"
            )

    class Pipeline(torch.nn.Module):
        def __init__(self, *, od_config, prefix=""):
            super().__init__()
            self.transformer = pipeline.MiniMaxH3DiTModel(od_config)
            self.partition = host.partition
            self.model_path = od_config.model
            self.device = "cpu"

        def forward(self, request):
            host.forward_calls.append(request)
            host.before_forward()
            sampling = request.requests[0].sampling_params
            return self.diffuse(
                task=sampling.extra_args["task"],
                ref_blocks=[{"kind": "image"}],
                num_steps=sampling.num_inference_steps,
                video_shift=sampling.extra_args["flow_shift"],
                audio_shift=sampling.extra_args["audio_flow_shift"],
            )

        def diffuse(self, **kwargs):
            if host.fail_diffuse:
                raise RuntimeError("injected denoise failure")
            state = self.transformer.h3_forge_curve_cache
            result = []
            for step in range(kwargs["num_steps"] - 1):
                host.step = step
                index = state.active_plan_indices[step]
                width = len(state.schedule.plans[index].values)
                token = self.transformer.time_embedder(torch.zeros(width))
                result.append((self.transformer.blocks[0](token), self.transformer.final(token)))
            return result

    def sigmas(*, num_steps, shift_scale):
        base = torch.linspace(1.0, 0.0, num_steps, dtype=torch.float32, device="cpu")
        return (shift_scale * base / (1 + (shift_scale - 1) * base)).tolist()

    transformer.MiniMaxH3TimeEmbedder = object()
    transformer.MiniMaxH3AdalnProj = object()
    transformer.MiniMaxH3DiTModel = DiT
    pipeline.MiniMaxH3DiTModel = DiT
    pipeline.MiniMaxH3Pipeline = Pipeline
    denoise.MINIMAX_H3_IMGVID_COND_TIMESTEP = 0.999
    denoise.MINIMAX_H3_AUDIO_REF_COND_TIMESTEP = 1.0
    forward.get_forward_context = lambda: types.SimpleNamespace(denoise_step_idx=host.step)
    time_request.minimax_h3_time_shift_sigmas = sigmas
    sys.modules["vllm_omni.errors"].OmniClientError = ClientError
    monkeypatch.delitem(sys.modules, CACHE, raising=False)
    runtime = importlib.import_module(CACHE)
    monkeypatch.setattr(runtime.construction, "state", runtime.construction.WorkerConstructionState())
    monkeypatch.delitem(sys.modules, RUNTIME, raising=False)
    router = importlib.import_module(RUNTIME)
    root, partition = _package(tmp_path)
    schedule = build_h3_schedule_contract(denoise_steps=4)
    offsets = [0]
    timesteps = []
    for plan in schedule.plans:
        timesteps.extend(plan.values)
        offsets.append(len(timesteps))
    slots = len(timesteps)
    blocks = torch.arange(slots * 96, dtype=torch.float32).to(torch.bfloat16).reshape(1, slots, 96)
    blocks = blocks.repeat(50, 1, 1)
    final = torch.arange(slots * 32, dtype=torch.float32).to(torch.bfloat16).reshape(slots, 32)
    cache_path = partition / "curve.safetensors"
    save_file(
        {
            "plan_offsets": torch.tensor(offsets, dtype=torch.int64),
            "plan_timesteps": torch.tensor(timesteps, dtype=torch.float32),
            "block_params": blocks,
            "final_params": final,
        },
        str(cache_path),
    )
    curve = CurveCacheBinding(cache_path, schedule, "a" * 64, cache_path.stat().st_size, "b" * 64, AUDITED_PRODUCER)
    binding = types.SimpleNamespace(package_root=root, partition_path=partition, curve_cache=curve)
    monkeypatch.setattr(router, "validate_runtime_package", lambda path: binding)
    host.build = lambda: router.H3ComfyMiniMaxH3Pipeline(od_config=types.SimpleNamespace(model=str(partition)))
    host.runtime = runtime
    host.router = router
    host.package = binding
    host.torch = torch
    host.original_dit = DiT
    host.transformer = transformer
    host.pipeline = pipeline
    host.error = ClientError
    host.blocks = blocks
    host.final = final
    yield host
    monkeypatch.delitem(sys.modules, CACHE, raising=False)
    monkeypatch.delitem(sys.modules, RUNTIME, raising=False)


def test_cache_lookup_replays_all_modes_and_returns_exact_bf16_views(cache_host):
    host = cache_host
    pipeline = host.build().eval()
    state = pipeline.transformer.h3_forge_curve_cache
    assert state.loaded
    assert set(state.block_modules) == set(range(50))
    for mode, indices in state.schedule.mode_plan_indices:
        with state.activate(mode):
            for step, index in enumerate(indices):
                host.step = step
                start, end = state._offsets[index : index + 2]
                token = state.time_embedder(host.torch.zeros(end - start))
                assert host.torch.equal(token, host.torch.tensor(state.schedule.plans[index].values))
                assert (
                    token.untyped_storage().data_ptr()
                    == state.time_embedder.plan_timesteps.untyped_storage().data_ptr()
                )
                actual = host.torch.cat(state.block_modules[0](token), dim=-1)
                expected = host.blocks[0, start:end].reshape((end - start) * 2, 48)
                assert host.torch.equal(actual, expected)
                final = host.torch.cat(state.final_module(token), dim=-1)
                assert host.torch.equal(final, host.final[start:end].reshape((end - start) * 2, 16))
    assert state.active_plan_indices is None
    with pytest.raises(RuntimeError, match="exactly one H3 pipeline"):
        host.build()
    with pytest.raises(RuntimeError, match="exactly one H3 model"):
        host.runtime.H3ComfyCacheDiTModel(types.SimpleNamespace(model="unused"))


def test_lookup_refuses_unloaded_overlapping_and_invalid_tokens(cache_host):
    host = cache_host
    pipeline = host.build()
    state = pipeline.transformer.h3_forge_curve_cache
    with pytest.raises(RuntimeError, match="not resident"):
        with state.activate("ref2va-image"):
            pass
    pipeline.eval()
    with pytest.raises(host.error, match="not compiled") as caught:
        with state.activate("unsupported"):
            pass
    assert (caught.value.status_code, caught.value.error_type) == (409, "H3_SCHEDULE_NOT_COMPILED")
    for indices in ((), (True,), (-1,), (99,)):
        with pytest.raises(ValueError, match="step_indices"):
            with state.activate("t2va", step_indices=indices):
                pass
    with state.activate("t2va"):
        with pytest.raises(RuntimeError, match="overlapping"):
            with state.activate("fl2va"):
                pass
        for step in (None, 99):
            host.step = step
            with pytest.raises(RuntimeError, match="invalid H3 denoise step"):
                state.time_embedder(host.torch.zeros(1))
        host.step = 0
        with pytest.raises(RuntimeError, match="wrong compiled width"):
            state.time_embedder(host.torch.zeros(99))
        with pytest.raises(RuntimeError, match="wrong device or dtype"):
            state.time_embedder(host.torch.zeros(1, dtype=host.torch.float64))
        with pytest.raises(RuntimeError, match="wrong storage"):
            state.block_modules[0](host.torch.zeros(1))
        source = state.time_embedder.plan_timesteps
        offset = next(index for index in range(1, source.numel()) if index not in state._span_by_offset)
        with pytest.raises(RuntimeError, match="invalid span"):
            state.block_modules[0](source.narrow(0, offset, 1))
    with pytest.raises(RuntimeError, match="no active request"):
        state.time_embedder(host.torch.zeros(1))


def test_forward_normalizes_without_mutation_and_cleans_failed_denoise(cache_host):
    host = cache_host
    pipeline = host.build().eval()
    original = _Batch()
    result = pipeline.forward(original)
    assert len(result) == 4
    assert original.requests[0].sampling_params.extra_args == {"h3_forge": {"api_version": 1, "task": "ref2va"}}
    assert host.forward_calls[0] is not original
    assert host.forward_calls[0].requests[0].sampling_params.num_inference_steps == 5
    assert host.forward_calls[0].requests[0].sampling_params.extra_args["flow_shift"] == 12.0
    host.fail_diffuse = True
    with pytest.raises(RuntimeError, match="denoise failure"):
        pipeline.forward(original)
    assert pipeline.transformer.h3_forge_curve_cache.active_plan_indices is None
    host.fail_diffuse = False
    assert len(pipeline.forward(original)) == 4


def test_bad_sampling_and_schedule_fail_before_the_host(cache_host):
    host = cache_host
    pipeline = host.build().eval()
    for sampling, expected in (
        (_Sampling(num_inference_steps=4), "H3_SCHEDULE_NOT_COMPILED"),
        (
            _Sampling(extra_args={"h3_forge": {"api_version": 1, "acceleration": {"profile": "lowres-resize-v0"}}}),
            "H3_ACCELERATION_UNSUPPORTED",
        ),
    ):
        with pytest.raises(host.error) as caught:
            pipeline.forward(_Batch([_Request(sampling)]))
        assert caught.value.error_type == expected
    with pytest.raises(host.error, match="one request at a time"):
        pipeline.forward(_Batch([_Request(), _Request()]))
    assert host.forward_calls == []
    args = dict(task="ref2va", num_steps=5, video_shift=12.0, audio_shift=3.0)
    for extra in ({"base_schedule": [1.0, 0.0]}, {"num_steps": 4}, {"video_shift": 11.0}):
        with pytest.raises(host.error) as caught:
            pipeline.diffuse(**{**args, **extra})
        assert (caught.value.status_code, caught.value.error_type) == (409, "H3_SCHEDULE_NOT_COMPILED")


def test_construction_restores_host_symbols_after_failure(cache_host):
    host = cache_host
    original_time = host.transformer.MiniMaxH3TimeEmbedder
    original_adaln = host.transformer.MiniMaxH3AdalnProj
    host.construct_failure = True
    with pytest.raises(RuntimeError, match="constructor failure"):
        host.build()
    assert host.pipeline.MiniMaxH3DiTModel is host.original_dit
    assert host.transformer.MiniMaxH3TimeEmbedder is original_time
    assert host.transformer.MiniMaxH3AdalnProj is original_adaln
    assert host.runtime._BUILD_STATE is None
    assert host.runtime._PACKAGE_BINDING is None
    host.construct_failure = False
    assert host.build().partition == "ref2va"


def test_completed_model_is_not_rebuilt_after_pipeline_failure(cache_host):
    host = cache_host
    host.partition = "fl2va"
    with pytest.raises(RuntimeError, match="one Ref2VA-primary"):
        host.build()
    host.partition = "ref2va"
    with pytest.raises(RuntimeError, match="exactly one H3 model"):
        host.build()
    assert host.pipeline.MiniMaxH3DiTModel is host.original_dit


def test_approximate_cache_backend_is_rejected_before_construction(cache_host):
    host = cache_host
    config = types.SimpleNamespace(model=str(host.package.partition_path), cache_backend="tea_cache")
    with pytest.raises(RuntimeError, match="forbids approximate"):
        host.router.H3ComfyMiniMaxH3Pipeline(od_config=config)
    assert not host.runtime.construction.state.model_constructed
    assert host.pipeline.MiniMaxH3DiTModel is host.original_dit


def test_package_root_loads_verified_partition_without_mutating_config(cache_host):
    host = cache_host
    config = types.SimpleNamespace(model=str(host.package.package_root))
    pipeline = host.router.H3ComfyMiniMaxH3Pipeline(od_config=config)
    assert pipeline.model_path == str(host.package.partition_path)
    assert config.model == str(host.package.package_root)


def test_parallel_requests_wait_for_the_request_lock(cache_host, monkeypatch):
    host = cache_host
    pipeline = host.build().eval()
    first_entered = threading.Event()
    second_attempted = threading.Event()
    release_first = threading.Event()
    original_lock = host.runtime._REQUEST_LOCK

    class ObservedLock:
        def __enter__(self):
            if first_entered.is_set():
                second_attempted.set()
            return original_lock.__enter__()

        def __exit__(self, *args):
            return original_lock.__exit__(*args)

    def before_forward():
        if not first_entered.is_set():
            first_entered.set()
            assert release_first.wait(10)

    monkeypatch.setattr(host.runtime, "_REQUEST_LOCK", ObservedLock())
    host.before_forward = before_forward
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(pipeline.forward, _Batch())
        assert first_entered.wait(10)
        second = executor.submit(pipeline.forward, _Batch())
        try:
            assert second_attempted.wait(10)
            assert len(host.forward_calls) == 1
        finally:
            release_first.set()
        assert len(first.result(timeout=10)) == len(second.result(timeout=10)) == 4
    assert len(host.forward_calls) == 2


def test_runtime_replay_detects_changed_host_scheduler(cache_host, monkeypatch):
    host = cache_host
    original = host.runtime.minimax_h3_time_shift_sigmas

    def drift(**kwargs):
        sigmas = original(**kwargs)
        sigmas[1] = 0.5
        return sigmas

    monkeypatch.setattr(host.runtime, "minimax_h3_time_shift_sigmas", drift)
    with pytest.raises(ValueError, match="runtime schedule differs"):
        host.build()
    assert not host.runtime.construction.state.model_constructed


def test_cache_mode_matrix(cache_host):
    route = cache_host.runtime._cache_mode
    for task, references, expected in (
        ("t2va", None, "t2va"),
        ("fl2va", None, "fl2va"),
        ("ref2va", None, "ref2va-image"),
        ("ref2va", [{"kind": "audio"}], "ref2va-audio"),
        ("ref2va", [{"kind": "video_audio"}], "ref2va-mixed"),
        ("ref2va", [{"kind": "image"}, {"kind": "audio"}], "ref2va-mixed"),
    ):
        assert route(task, references) == expected
