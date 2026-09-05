"""Small raw bindings through real H3 DiT reloads and pinned worker RPCs.

The source provider substitutes BF16 fixture reads for INT8 mathematics. Real
source identity checks, H3 constructors/loaders, 534 slots and WorkerWrapperBase
dispatch remain in use; this does not prove GPU or full-model hot switching.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from beta4_host_fixture import actual_cpu_host, tiny_arch_config, tiny_inventory, tiny_tensor_stream
from test_h3_raw_source_routing import _assert_host_blob, _raw_fixture, _tree_identity

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("vllm_omni") is None,
    reason="the actual pinned host image is required",
)


def _assert_slots(torch, model, expected):
    slots = model.state_dict()
    assert len(slots) == 534 and model.beta4_ready
    for name, values in expected.items():
        target = "time_embedder.adaln_t_table" if name == "adaln_t_table" else name
        if name.endswith(".attn.qkv_proj.weight"):
            rows = [
                head * 384 + section * 128 + offset
                for section in range(3)
                for head in range(2)
                for offset in range(128)
            ]
            values = values[rows]
        assert torch.equal(slots[target], values.to(slots[target].dtype)), name
    assert model.beta4_loaded_sources == frozenset(expected)


def _storage(model):
    return {name: tensor.untyped_storage().data_ptr() for name, tensor in model.state_dict().items()}


@contextmanager
def _headers_only(bindings):
    """Allow metadata reads while rejecting payload access during prepare."""
    original_open = Path.open
    limits = {}
    for binding in bindings:
        with original_open(binding.source_path, "rb") as stream:
            limits[binding.source_path] = 8 + int.from_bytes(stream.read(8), "little")

    class HeaderReader:
        def __init__(self, stream, limit):
            self.stream, self.limit = stream, limit

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def read(self, size=-1):
            assert size >= 0 and self.stream.tell() + size <= self.limit, "prepare must not read tensor payload"
            return self.stream.read(size)

    def guarded_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return HeaderReader(stream, limits[path]) if path in limits else stream

    with patch.object(Path, "open", guarded_open):
        yield


@pytest.fixture
def raw_runtime(tmp_path):
    with actual_cpu_host() as host, ExitStack() as stack:
        torch = host.torch
        from safetensors import safe_open
        from safetensors.torch import save_file
        from vllm.config.load import LoadConfig
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader
        from vllm_omni.diffusion.distributed import parallel_state as diffusion_parallel
        from vllm_omni.diffusion.model_loader import diffusers_loader

        assert diffusion_parallel._WORLD is None
        stack.enter_context(patch.object(diffusion_parallel, "_WORLD", None))
        diffusion_parallel.init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
        stack.callback(diffusion_parallel.get_world_group().destroy)

        adapter = importlib.import_module("comfy_omni.integrations.vllm_omni.pipelines.beta4_pipeline")
        official_pipeline = adapter.pipeline
        _assert_host_blob(diffusers_loader, "62bc0525f1053af56db8a6ee112f5612473c359a")
        _assert_host_blob(official_pipeline, "62d10488ce0a82e9d3f72b3d0b23550565ff4db9")
        stack.enter_context(patch.object(adapter.construction, "state", adapter.construction.WorkerConstructionState()))

        class TinyDiT(adapter.H3Beta4DiTModel):
            _inventory = tiny_inventory()
            _architecture = tiny_arch_config()

        components = tmp_path / "existing" / "Ref2VA"
        components.mkdir(parents=True)
        (components / "model_index.json").write_text(
            json.dumps({"_minimax_h3": {"partition": "ref2va", "tasks": ["ref2va"]}}), encoding="utf-8"
        )
        for name in ("text_encoder", "video_vae", "audio_vae", "tokenizer", "processor"):
            (components / name).mkdir()
        shared_values = torch.tensor([1, 2], dtype=torch.bfloat16)
        shared_file = components / "text_encoder" / "model.safetensors"
        save_file({"weight": shared_values}, str(shared_file))
        shared_file.chmod(0o444)
        raw_root = tmp_path / "comfy_models"
        raw_root.mkdir()
        values_a = dict(tiny_tensor_stream())
        values_b = {name: values + 0.25 for name, values in values_a.items()}
        bindings = {
            "a": _raw_fixture(raw_root / "a.safetensors", values_a),
            "b": _raw_fixture(raw_root / "b.safetensors", values_b),
        }
        before = _tree_identity(tmp_path)
        control = SimpleNamespace(reads=[], fail_b=False, saw_partial_write=False, pipeline=None)

        class TinyEncoder(torch.nn.Module):
            def __init__(self, model_path, **kwargs):
                super().__init__()
                assert Path(model_path) == components / "text_encoder"
                self.weight = torch.nn.Parameter(torch.empty_like(shared_values))

            def load_weights(self, weights):
                loaded = set()
                for name, tensor in weights:
                    assert name == "weight" and name not in loaded
                    default_weight_loader(self.weight, tensor)
                    loaded.add(name)
                return loaded

        class TinyVAE(torch.nn.Module):
            def __init__(self, model_path, **kwargs):
                super().__init__()
                assert Path(model_path) in {components / "video_vae", components / "audio_vae"}
                self.weight = torch.nn.Parameter(torch.ones(1))

        def open_fixture(binding):
            name = "a" if binding is bindings["a"] else "b"
            assert binding is bindings[name]
            binding.verify_unchanged()
            control.reads.append(name)
            with safe_open(str(binding.source_path), framework="pt", device="cpu") as source:
                for index, key in enumerate(source.keys()):
                    if name == "b" and control.fail_b and index == 50:
                        control.fail_b = False
                        observed = control.pipeline.transformer.state_dict()["adaln_basis"]
                        control.saw_partial_write = torch.equal(observed, values_b["adaln_basis"])
                        assert control.saw_partial_write, "failure must follow actual mutation of the live DiT"
                        raise RuntimeError("injected source failure after 50 real loads")
                    yield key, source.get_tensor(key)
            binding.verify_unchanged()

        stack.enter_context(patch.object(type(bindings["a"]), "open_weights", open_fixture))
        stack.enter_context(patch.object(adapter, "H3Beta4DiTModel", TinyDiT))
        stack.enter_context(patch.object(official_pipeline, "get_local_device", lambda: torch.device("cpu")))
        stack.enter_context(patch.object(official_pipeline, "MiniMaxH3Qwen3VLEncoder", TinyEncoder))
        stack.enter_context(patch.object(official_pipeline, "MiniMaxH3VideoVAE", TinyVAE))
        stack.enter_context(patch.object(official_pipeline, "MiniMaxH3AudioVAE", TinyVAE))
        for tokenizer in (official_pipeline.Qwen2TokenizerFast, official_pipeline.Qwen3VLProcessor):
            stack.enter_context(patch.object(tokenizer, "from_pretrained", return_value=SimpleNamespace()))
        host.od_config.model = str(components)
        host.od_config.task_type = "ref2va"
        host.od_config.cache_backend = "none"
        host.od_config.enable_multithread_weight_load = False
        model = adapter.H3Beta4Pipeline(od_config=host.od_config, raw_binding=bindings["a"], package=None)
        control.pipeline = model
        loader = diffusers_loader.DiffusersPipelineLoader(
            LoadConfig(load_format="auto", safetensors_load_strategy="lazy", use_tqdm_on_load=False), host.od_config
        )
        loader.load_weights(model)
        _assert_slots(torch, model.transformer, values_a)
        yield SimpleNamespace(
            torch=torch,
            pipeline=model,
            bindings=bindings,
            values_a=values_a,
            values_b=values_b,
            control=control,
            before=before,
            root=tmp_path,
        )


def test_raw_dit_reloads_complete_weights_in_the_existing_storage(raw_runtime):
    fixture = raw_runtime
    model = fixture.pipeline.transformer
    storage = _storage(model)
    model.beta4_binding = fixture.bindings["b"]
    model.load_weights(fixture.bindings["b"].open_weights())
    model.post_load_weights()
    _assert_slots(fixture.torch, model, fixture.values_b)
    assert _storage(model) == storage
    model.beta4_binding = fixture.bindings["a"]
    model.load_weights(fixture.bindings["a"].open_weights())
    model.post_load_weights()
    _assert_slots(fixture.torch, model, fixture.values_a)
    assert _storage(model) == storage
    assert _tree_identity(fixture.root) == fixture.before


def test_pinned_worker_switches_registered_raw_files_and_restores_after_partial_failure(raw_runtime):
    from vllm_omni.diffusion.worker.diffusion_worker import WorkerWrapperBase

    from comfy_omni.integrations.vllm_omni.residency import H3ResidencyWorkerExtension
    from comfy_omni.runtime.hotel import PreparedH3DitSelection

    fixture = raw_runtime
    pipeline = fixture.pipeline
    model = pipeline.transformer
    storage = _storage(model)
    shared = (pipeline.text_encoder, pipeline.video_vae, pipeline.audio_vae, pipeline.tokenizer, pipeline.processor)
    with _headers_only(fixture.bindings.values()):
        registered = pipeline.comfy_omni_register_h3_dit("b", fixture.bindings["b"])
    assert isinstance(registered, PreparedH3DitSelection)
    assert registered.selection == "b" and len(registered.tensors) == 534
    assert pipeline.comfy_omni_active_h3_dit.selection == "initial"
    assert fixture.control.reads == ["a"]

    class PinnedWorkerBoundary:
        pass

    wrapper = object.__new__(WorkerWrapperBase)
    wrapper.base_worker_class = PinnedWorkerBoundary
    wrapper.worker_extension_cls = H3ResidencyWorkerExtension
    wrapper.custom_pipeline_args = None
    worker = object.__new__(wrapper._prepare_worker_class())
    worker.model_runner = SimpleNamespace(
        pipeline=pipeline, state_cache={}, input_batch=None, cache_backend=None, offload_backend=None
    )
    worker.lora_manager = None
    worker._step_lora_state = {}
    wrapper.worker = worker

    for transaction, selection, expected in (("to-b", "b", fixture.values_b), ("back-a", "initial", fixture.values_a)):
        reads_before = list(fixture.control.reads)
        previous = pipeline.comfy_omni_active_h3_dit
        with _headers_only(fixture.bindings.values()):
            prepared = wrapper.execute_method("comfy_omni_prepare_h3_dit", transaction, selection)
        assert prepared["phase"] == "prepared" and prepared["cpu_cached_bytes"] == 0
        assert fixture.control.reads == reads_before
        committed = wrapper.execute_method("comfy_omni_commit_h3_dit", transaction)
        assert committed["phase"] == "committed"
        assert pipeline.comfy_omni_active_h3_dit is previous
        _assert_slots(fixture.torch, model, expected)
        finished = wrapper.execute_method("comfy_omni_finalize_h3_dit", transaction)
        assert finished["active_selection"] == selection and finished["cpu_cached_bytes"] == 0
        assert pipeline.transformer is model and wrapper.worker is worker
        assert _storage(model) == storage
        assert all(
            current is old
            for current, old in zip(
                (pipeline.text_encoder, pipeline.video_vae, pipeline.audio_vae, pipeline.tokenizer, pipeline.processor),
                shared,
                strict=True,
            )
        )

    fixture.control.fail_b = True
    with _headers_only(fixture.bindings.values()):
        wrapper.execute_method("comfy_omni_prepare_h3_dit", "broken-b", "b")
    with pytest.raises(RuntimeError, match="commit failed.*rollback restored"):
        wrapper.execute_method("comfy_omni_commit_h3_dit", "broken-b")
    status = wrapper.execute_method("comfy_omni_h3_residency_status")
    assert status["phase"] == "rolled_back" and status["active_selection"] == "initial"
    assert status["cpu_cached_bytes"] == 0 and fixture.control.saw_partial_write
    assert fixture.control.reads == ["a", "b", "a", "b", "a"]
    assert pipeline.comfy_omni_active_h3_dit.selection == "initial"
    assert pipeline.transformer is model and wrapper.worker is worker
    _assert_slots(fixture.torch, model, fixture.values_a)
    assert _storage(model) == storage
    assert _tree_identity(fixture.root) == fixture.before
