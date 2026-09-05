"""Pinned-host residency of the 532-weight standard H3 raw source."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from test_h3_raw_residency import (
    _assert_slots,
    _storage,
)
from test_h3_raw_residency import (
    pytestmark as pytestmark,
)
from test_h3_raw_residency import (
    raw_runtime as raw_runtime,  # noqa: F401
)
from test_h3_raw_source_routing import _raw_fixture

_AUXILIARY_SLOTS = frozenset({"adaln_basis", "adaln_mean"})


def _standard_binding(raw_root, values):
    """Create a real small safetensors source whose header has only 532 slots."""
    from comfy_omni.runtime.h3.raw_standard import RawStandardBinding, RawStandardIdentity

    base = _raw_fixture(raw_root / "standard-b.safetensors", values)
    source_identity = base.trusted_identity
    identity = RawStandardIdentity(
        source_identity.name,
        source_identity.size,
        source_identity.sha256,
        source_identity.source_schema_sha256,
        len(base.source_descriptors),
        len(base.target_descriptors),
    )
    return RawStandardBinding(
        base.source_path,
        identity,
        base.source_file_identity,
        base.source_descriptors,
        base.target_descriptors,
        base.plan,
    )


def test_pruned_standard_raw_source_initializes_auxiliary_slots_and_switches_back(raw_runtime):  # noqa: F811
    """A 532-slot source loads through the real 534-slot host without rebuild."""
    from safetensors import safe_open
    from vllm_omni.diffusion.worker.diffusion_worker import WorkerWrapperBase

    from comfy_omni.integrations.vllm_omni.residency import H3ResidencyWorkerExtension
    from comfy_omni.runtime.h3.raw_standard import RawStandardBinding

    fixture = raw_runtime
    pipeline = fixture.pipeline
    model = pipeline.transformer
    storage = _storage(model)
    source_values = {name: values + 0.25 for name, values in fixture.values_a.items() if name not in _AUXILIARY_SLOTS}
    source_values["adaln_t_table"] = source_values["adaln_t_table"].float() + 0.00003
    for name, values in list(source_values.items()):
        if ".adaln_proj.linear." in name:
            source_values[name] = values.to(fixture.torch.float16)
    binding = _standard_binding(fixture.root / "comfy_models", source_values)
    assert len(binding.source_descriptors) == len(binding.target_descriptors) == 532
    assert set(fixture.values_a) - set(source_values) == _AUXILIARY_SLOTS

    def open_standard_weights(self):
        assert self is binding
        self.verify_unchanged()
        with safe_open(str(self.source_path), framework="pt", device="cpu") as source:
            for name in source.keys():
                yield name, source.get_tensor(name)
        self.verify_unchanged()

    with patch.object(RawStandardBinding, "open_weights", open_standard_weights):
        registered = pipeline.comfy_omni_register_h3_dit("standard-b", binding)
        assert len(registered.tensors) == 534

        initial = pipeline.comfy_omni_active_h3_dit
        materialized = dict(pipeline.comfy_omni_iter_h3_dit(registered))
        assert set(materialized) == set(fixture.values_a)
        for name, values in source_values.items():
            assert fixture.torch.equal(materialized[name], values), name
        for name in _AUXILIARY_SLOTS:
            assert materialized[name].dtype == fixture.torch.bfloat16
            assert fixture.torch.count_nonzero(materialized[name]).item() == 0

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

        wrapper.execute_method("comfy_omni_prepare_h3_dit", "to-standard", "standard-b")
        wrapper.execute_method("comfy_omni_commit_h3_dit", "to-standard")
        wrapper.execute_method("comfy_omni_finalize_h3_dit", "to-standard")
        receipt = model.loading_receipt()
        assert receipt["source_slots"] == 532
        assert receipt["runtime_initializers"] == 2
        _assert_slots(fixture.torch, model, source_values)
        for name in _AUXILIARY_SLOTS:
            assert fixture.torch.count_nonzero(model.state_dict()[name]).item() == 0
        assert pipeline.transformer is model and wrapper.worker is worker
        assert _storage(model) == storage

        wrapper.execute_method("comfy_omni_prepare_h3_dit", "back-a", initial.selection)
        wrapper.execute_method("comfy_omni_commit_h3_dit", "back-a")
        wrapper.execute_method("comfy_omni_finalize_h3_dit", "back-a")
        _assert_slots(fixture.torch, model, fixture.values_a)
        assert model.loading_receipt()["source_slots"] == 534
        assert pipeline.transformer is model and wrapper.worker is worker
        assert _storage(model) == storage
