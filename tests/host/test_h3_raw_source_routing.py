"""Raw-source routing through pinned H3 constructors and DiffusersPipelineLoader.

Only expensive tokenizer/TE/VAE construction and the raw numerical provider are
small substitutes. Host source selection, safetensors reading, pipeline dispatch,
the 534-slot DiT loader and raw identity checks are real. This is CPU routing
evidence, not INT8 conversion, full-component loading or GPU generation evidence.
The INT8 cases prove configuration routing, not execution of an INT8 TE kernel.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from beta4_host_fixture import actual_cpu_host, tiny_arch_config, tiny_inventory, tiny_tensor_stream

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("vllm_omni") is None,
    reason="the actual pinned host image is required",
)


def _assert_host_blob(module, expected):
    payload = Path(module.__file__).read_bytes()
    observed = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
    assert observed == expected, "routing evidence requires the exact 17285 host source"


def _tree_identity(root):
    return {
        str(path.relative_to(root)): (
            path.stat().st_ino,
            path.stat().st_size,
            path.stat().st_mtime_ns,
            path.stat().st_ctime_ns,
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def _raw_fixture(source_path, expected):
    from safetensors.torch import save_file

    from comfy_omni.artifacts.fileops import fd_identity
    from comfy_omni.artifacts.safetensors import read_safetensors_header_stream
    from comfy_omni.conversion.contract_workflows.census import schema_sha256
    from comfy_omni.runtime.h3.raw_beta4 import RawBeta4Binding, RawBeta4Identity

    save_file(expected, str(source_path))
    source_path.chmod(0o444)
    size = source_path.stat().st_size
    with source_path.open("rb") as stream:
        _, descriptors, _ = read_safetensors_header_stream(stream, source_path, size)
    descriptors = tuple(sorted(descriptors, key=lambda item: item.name))
    identity = RawBeta4Identity(
        source_path.name, size, hashlib.sha256(source_path.read_bytes()).hexdigest(), schema_sha256(descriptors)
    )
    # This test replaces numerical adaptation, not source/header authentication.
    return RawBeta4Binding(source_path, identity, fd_identity(source_path.stat()), descriptors, descriptors, None)


@pytest.mark.parametrize("quant_scope", ("none", "text_encoder", "transformer", "global"))
def test_existing_raw_file_routes_to_actual_dit_without_transformer_export(tmp_path, quant_scope):
    with actual_cpu_host() as host, ExitStack() as stack:
        torch = host.torch
        from safetensors import safe_open
        from safetensors.torch import save_file
        from vllm.config.load import LoadConfig
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader
        from vllm_omni.diffusion.distributed import parallel_state as diffusion_parallel
        from vllm_omni.diffusion.model_loader import diffusers_loader
        from vllm_omni.quantization import ComponentQuantizationConfig, build_quant_config

        # The pipeline uses Omni's world group in addition to vLLM's TP groups.
        # Create real Gloo groups and destroy only these owned groups before the
        # outer fixture tears down vLLM and the default process group.
        assert diffusion_parallel._WORLD is None
        stack.enter_context(patch.object(diffusion_parallel, "_WORLD", None))
        diffusion_parallel.init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
        stack.callback(diffusion_parallel.get_world_group().destroy)
        assert diffusion_parallel.get_world_group().world_size == 1

        adapter = importlib.import_module("comfy_omni.integrations.vllm_omni.pipelines.beta4_pipeline")
        pipeline = adapter.pipeline
        _assert_host_blob(diffusers_loader, "62bc0525f1053af56db8a6ee112f5612473c359a")
        _assert_host_blob(pipeline, "62d10488ce0a82e9d3f72b3d0b23550565ff4db9")
        stack.enter_context(patch.object(adapter.construction, "state", adapter.construction.WorkerConstructionState()))
        int8_config = None if quant_scope == "none" else build_quant_config("int8")
        expected_encoder_quant = int8_config if quant_scope == "text_encoder" else None
        if quant_scope == "none":
            selected_quant = None
        elif quant_scope == "global":
            selected_quant = int8_config
        else:
            selected_quant = ComponentQuantizationConfig(
                {"transformer": None, "text_encoder": None, quant_scope: int8_config}
            )
        constructed = []

        class TinyDiT(adapter.H3Beta4DiTModel):
            _inventory = tiny_inventory()
            _architecture = tiny_arch_config()

            def __init__(self, *args, **kwargs):
                constructed.append("transformer")
                super().__init__(*args, **kwargs)

        components = tmp_path / "existing" / "Ref2VA"
        components.mkdir(parents=True)
        (components / "model_index.json").write_text(
            json.dumps({"_minimax_h3": {"partition": "ref2va", "tasks": ["ref2va"]}}), encoding="utf-8"
        )
        for name in ("text_encoder", "video_vae", "audio_vae", "tokenizer", "processor"):
            (components / name).mkdir()
        shared_values = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)
        shared_file = components / "text_encoder" / "model.safetensors"
        save_file({"weight": shared_values}, str(shared_file))
        shared_file.chmod(0o444)
        raw_root = tmp_path / "comfy_models"
        raw_root.mkdir()
        expected = dict(tiny_tensor_stream())
        raw_binding = _raw_fixture(raw_root / "original.safetensors", expected)
        before = _tree_identity(tmp_path)
        events = []

        class TinyEncoder(torch.nn.Module):
            def __init__(self, model_path, *, device, load_model, encoder_group, quant_config):
                super().__init__()
                assert Path(model_path) == components / "text_encoder"
                assert load_model and device.type == "cpu"
                assert quant_config is expected_encoder_quant, "the host must retain the selected TE quantization"
                if quant_config is not None:
                    assert quant_config.get_name() == "int8"
                self.quant_config = quant_config
                self.weight = torch.nn.Parameter(torch.empty_like(shared_values))

            def load_weights(self, weights):
                loaded = set()
                for name, tensor in weights:
                    assert name == "weight" and name not in loaded
                    default_weight_loader(self.weight, tensor)
                    loaded.add(name)
                events.append("shared-loaded")
                return loaded

        class TinyVAE(torch.nn.Module):
            def __init__(self, model_path, *, device, load_device):
                super().__init__()
                assert Path(model_path) in {components / "video_vae", components / "audio_vae"}
                assert device.type == load_device.type == "cpu"
                self.weight = torch.nn.Parameter(torch.ones(1))

        def raw_weights(binding):
            assert binding is raw_binding
            assert events == ["shared-loaded"], "shared host components must load before the raw DiT stream"
            binding.verify_unchanged()
            events.append("raw-opened")
            with safe_open(str(binding.source_path), framework="pt", device="cpu") as source:
                for name in source.keys():
                    yield name, source.get_tensor(name)
            binding.verify_unchanged()

        stack.enter_context(patch.object(type(raw_binding), "open_weights", raw_weights))
        stack.enter_context(patch.object(adapter, "H3Beta4DiTModel", TinyDiT))
        stack.enter_context(patch.object(pipeline, "get_local_device", lambda: torch.device("cpu")))
        stack.enter_context(patch.object(pipeline, "MiniMaxH3Qwen3VLEncoder", TinyEncoder))
        stack.enter_context(patch.object(pipeline, "MiniMaxH3VideoVAE", TinyVAE))
        stack.enter_context(patch.object(pipeline, "MiniMaxH3AudioVAE", TinyVAE))
        for tokenizer in (pipeline.Qwen2TokenizerFast, pipeline.Qwen3VLProcessor):
            stack.enter_context(patch.object(tokenizer, "from_pretrained", return_value=SimpleNamespace()))

        host.od_config.model = str(components)
        host.od_config.task_type = "ref2va"
        host.od_config.cache_backend = "none"
        host.od_config.enable_multithread_weight_load = False
        host.od_config.quantization_config = selected_quant
        if quant_scope in {"transformer", "global"}:
            with pytest.raises(ValueError, match="quantization|quant config"):
                adapter.H3Beta4Pipeline(od_config=host.od_config, raw_binding=raw_binding, package=None)
            assert host.od_config.quantization_config is selected_quant
            assert constructed == [] and events == []
            assert _tree_identity(tmp_path) == before
            return
        model = adapter.H3Beta4Pipeline(od_config=host.od_config, raw_binding=raw_binding, package=None)
        assert host.od_config.quantization_config is selected_quant
        assert model.od_config.quantization_config is selected_quant
        assert model.text_encoder.quant_config is expected_encoder_quant
        assert isinstance(model.transformer, TinyDiT)
        assert [source.prefix for source in model.weights_sources] == ["text_encoder."]
        assert not (components / "transformer").exists()
        loader = diffusers_loader.DiffusersPipelineLoader(
            LoadConfig(load_format="auto", safetensors_load_strategy="lazy", use_tqdm_on_load=False), host.od_config
        )
        loader.load_weights(model)

        assert events == ["shared-loaded", "raw-opened"]
        assert torch.equal(model.text_encoder.weight, shared_values)
        assert model.transformer.beta4_ready
        assert model.transformer.beta4_loaded_sources == frozenset(expected)
        actual = model.transformer.state_dict()
        assert len(actual) == 534
        for name, source in expected.items():
            runtime = actual[adapter._runtime_name(name)]
            if name.endswith(".attn.qkv_proj.weight"):
                # The real host rearranges grouped [head, q/k/v, dim] input
                # into concatenated q/k/v. Keep this oracle independent of
                # its loader implementation, as in the existing slot test.
                rows = [
                    head * 384 + section * 128 + offset
                    for section in range(3)
                    for head in range(2)
                    for offset in range(128)
                ]
                source = source[rows]
            assert torch.equal(runtime, source.to(runtime.dtype)), name
        model.transformer.post_load_weights()
        assert _tree_identity(tmp_path) == before, "loading must not write a model, manifest or shard"
