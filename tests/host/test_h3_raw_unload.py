"""Explicit H3 unload/reload against the real pinned 534-slot TinyDiT."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_h3_raw_residency import _assert_slots, _headers_only, _tree_identity
from test_h3_raw_residency import pytestmark as pytestmark
from test_h3_raw_residency import raw_runtime as raw_runtime


def _worker(fixture):
    from vllm_omni.diffusion.worker.diffusion_worker import WorkerWrapperBase

    from comfy_omni.integrations.vllm_omni.residency import H3ResidencyWorkerExtension

    class PinnedWorkerBoundary:
        pass

    wrapper = object.__new__(WorkerWrapperBase)
    wrapper.base_worker_class = PinnedWorkerBoundary
    wrapper.worker_extension_cls = H3ResidencyWorkerExtension
    wrapper.custom_pipeline_args = None
    worker = object.__new__(wrapper._prepare_worker_class())
    worker.model_runner = SimpleNamespace(
        pipeline=fixture.pipeline,
        state_cache={},
        input_batch=None,
        cache_backend=None,
        offload_backend=None,
    )
    worker.lora_manager = None
    worker._step_lora_state = {}
    wrapper.worker = worker
    return wrapper


def _slot_objects(model):
    return model.state_dict(keep_vars=True)


def _slot_bytes(model):
    return sum(tensor.numel() * tensor.element_size() for tensor in _slot_objects(model).values())


def test_release_unload_drops_all_slot_storage_and_reload_restores_original_file(raw_runtime):
    fixture = raw_runtime
    wrapper = _worker(fixture)
    model = fixture.pipeline.transformer
    slots = _slot_objects(model)
    slot_ids = {name: id(tensor) for name, tensor in slots.items()}
    loader_attrs = {name: tensor.weight_loader for name, tensor in slots.items() if hasattr(tensor, "weight_loader")}
    assert loader_attrs, "the pinned DiT must expose real per-parameter loader attributes"
    shared = (
        fixture.pipeline.text_encoder,
        fixture.pipeline.video_vae,
        fixture.pipeline.audio_vae,
        fixture.pipeline.tokenizer,
        fixture.pipeline.processor,
    )
    reads_before = list(fixture.control.reads)

    status = wrapper.execute_method("comfy_omni_unload_h3_dit")

    assert status["weight_residency"] == "released"
    assert status["cpu_weight_bytes"] == 0
    assert status["device_weight_bytes"] == 0
    assert status["resident_weight_bytes"] == 0
    assert status["cuda_memory_allocated_bytes"] == 0
    assert status["cuda_memory_reserved_bytes"] == 0
    assert status["worker_pid"] == os.getpid()
    assert status["pipeline_id"] == id(fixture.pipeline)
    assert status["transformer_id"] == id(model)
    assert status["shared_object_ids"] == {
        "text_encoder": id(fixture.pipeline.text_encoder),
        "video_vae": id(fixture.pipeline.video_vae),
        "audio_vae": id(fixture.pipeline.audio_vae),
        "tokenizer": id(fixture.pipeline.tokenizer),
        "processor": id(fixture.pipeline.processor),
    }
    assert model.beta4_ready is False
    assert fixture.control.reads == reads_before
    released = _slot_objects(model)
    assert set(released) == set(slots)
    assert all(tensor.numel() == 0 and tensor.untyped_storage().nbytes() == 0 for tensor in released.values())
    assert {name: id(tensor) for name, tensor in released.items()} == slot_ids
    assert all(released[name].weight_loader is loader for name, loader in loader_attrs.items())

    loaded = wrapper.execute_method("comfy_omni_load_h3_dit")

    assert loaded["weight_residency"] == "loaded"
    assert loaded["cpu_weight_bytes"] > 0
    assert loaded["resident_weight_bytes"] == loaded["cpu_weight_bytes"] + loaded["device_weight_bytes"]
    assert loaded["worker_pid"] == status["worker_pid"]
    assert loaded["pipeline_id"] == status["pipeline_id"]
    assert loaded["transformer_id"] == status["transformer_id"]
    assert loaded["shared_object_ids"] == status["shared_object_ids"]
    assert fixture.control.reads == [*reads_before, "a"]
    _assert_slots(fixture.torch, model, fixture.values_a)
    restored = _slot_objects(model)
    assert {name: id(tensor) for name, tensor in restored.items()} == slot_ids
    assert all(restored[name].weight_loader is loader for name, loader in loader_attrs.items())
    assert (
        fixture.pipeline.text_encoder,
        fixture.pipeline.video_vae,
        fixture.pipeline.audio_vae,
        fixture.pipeline.tokenizer,
        fixture.pipeline.processor,
    ) == shared
    assert _tree_identity(fixture.root) == fixture.before


def test_cpu_unload_requires_full_budget_and_reloads_without_opening_payload(raw_runtime):
    fixture = raw_runtime
    wrapper = _worker(fixture)
    model = fixture.pipeline.transformer
    required = _slot_bytes(model)
    reads_before = list(fixture.control.reads)

    with pytest.raises(RuntimeError, match="CPU unload budget"):
        wrapper.execute_method("comfy_omni_unload_h3_dit", mode="cpu", cpu_budget_bytes=required - 1)
    assert model.beta4_ready is True
    _assert_slots(fixture.torch, model, fixture.values_a)

    status = wrapper.execute_method("comfy_omni_unload_h3_dit", mode="cpu", cpu_budget_bytes=required)
    assert status["weight_residency"] == "cpu"
    assert status["cpu_weight_bytes"] == required
    assert model.beta4_ready is False
    assert fixture.control.reads == reads_before

    loaded = wrapper.execute_method("comfy_omni_load_h3_dit")
    assert loaded["weight_residency"] == "loaded"
    assert fixture.control.reads == reads_before
    _assert_slots(fixture.torch, model, fixture.values_a)
    assert all(Path(binding.source_path).is_file() for binding in fixture.bindings.values())
    assert _tree_identity(fixture.root) == fixture.before


def test_switch_can_load_directly_from_release_and_failed_reload_clears_partial_storage(raw_runtime):
    fixture = raw_runtime
    wrapper = _worker(fixture)
    model = fixture.pipeline.transformer
    with _headers_only(fixture.bindings.values()):
        fixture.pipeline.comfy_omni_register_h3_dit("b", fixture.bindings["b"])

    wrapper.execute_method("comfy_omni_unload_h3_dit")
    wrapper.execute_method("comfy_omni_prepare_h3_dit", "to-b", "b")
    wrapper.execute_method("comfy_omni_commit_h3_dit", "to-b")
    wrapper.execute_method("comfy_omni_finalize_h3_dit", "to-b")
    _assert_slots(fixture.torch, model, fixture.values_b)

    wrapper.execute_method("comfy_omni_unload_h3_dit")
    fixture.control.fail_b = True
    with pytest.raises(RuntimeError, match="weight load failed.*remains released"):
        wrapper.execute_method("comfy_omni_load_h3_dit")
    status = wrapper.execute_method("comfy_omni_h3_residency_status")
    assert status["weight_residency"] == "released"
    assert status["cpu_weight_bytes"] == status["device_weight_bytes"] == 0
    assert model.beta4_ready is False and fixture.control.saw_partial_write
    assert all(tensor.untyped_storage().nbytes() == 0 for tensor in _slot_objects(model).values())
    assert _tree_identity(fixture.root) == fixture.before
