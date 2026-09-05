"""Small tensors through real pinned host constructors and TP linears."""

from __future__ import annotations

import importlib
import importlib.util
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from beta4_host_fixture import actual_cpu_host, packed_inputs, tiny_arch_config, tiny_inventory, tiny_tensor_stream

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("vllm_omni") is None,
    reason="the actual pinned host image is required",
)

ADAPTER = "comfy_omni.integrations.vllm_omni.pipelines.beta4_pipeline"


def _implementation(name, official):
    # Keep the accepted-base RED executable through the actual official class.
    if importlib.util.find_spec(ADAPTER) is None:
        return official
    return getattr(importlib.import_module(ADAPTER), name)


def test_beta4_actual_host_adaln_is_fp32_without_silu():
    with actual_cpu_host() as host:
        torch, official = host.torch, host.official
        arch = official.MiniMaxH3DiTArchConfig.from_mapping(tiny_arch_config())
        cls = _implementation("_Beta4AdalnProj", official.MiniMaxH3AdalnProj)
        module = cls(arch, 18 * arch.hidden_size, None, expand_ratio=6, modality_num=3, prefix="blocks.0.adaln_proj")
        assert module.linear.weight.dtype == torch.float32, "beta4 rank-eight AdaLN must retain FP32 parameters"
        with torch.no_grad():
            module.linear.weight.fill_(0.125)
            module.linear.bias.fill_(0.0625)
        values = torch.full((2, 8), -1.0, dtype=torch.float32)
        observed = torch.cat(module(values), dim=-1)
        expected = torch.full((6, 6 * arch.hidden_size), -0.9375, dtype=torch.bfloat16)
        assert observed.dtype == torch.bfloat16
        assert torch.equal(observed, expected), "negative table conditioning must not pass through SiLU"


def test_beta4_actual_host_final_adaln_keeps_fp32_output():
    with actual_cpu_host() as host:
        torch, official = host.torch, host.official
        arch = official.MiniMaxH3DiTArchConfig.from_mapping(tiny_arch_config())
        cls = _implementation("_Beta4AdalnProj", official.MiniMaxH3AdalnProj)
        module = cls(arch, 2 * arch.hidden_size, None, expand_ratio=2, modality_num=1, prefix="final_layer.adaln_proj")
        with torch.no_grad():
            module.linear.weight.fill_(0.125)
            module.linear.bias.fill_(0.00001)
        values = torch.full((1, 8), 0.123456, dtype=torch.float32)
        observed = torch.cat(module(values), dim=-1)
        assert observed.dtype == torch.float32, "final modulation must not incur a BF16 intermediate rounding"
        expected = torch.nn.functional.linear(values, module.linear.weight, module.linear.bias)
        assert torch.equal(observed, expected)


@pytest.fixture
def tiny_model(tmp_path):
    with actual_cpu_host() as host:
        adapter = importlib.import_module(ADAPTER)
        with patch.object(adapter.construction, "state", adapter.construction.WorkerConstructionState()):

            class TinyDiT(adapter.H3Beta4DiTModel):
                _inventory = tiny_inventory()
                _architecture = tiny_arch_config()

            model = TinyDiT(host.od_config, binding=SimpleNamespace(component_root=tmp_path, files=()))
            yield host, adapter, model, TinyDiT


def test_all_534_slots_load_through_actual_host_and_complete_packed_forward(tiny_model):
    host, adapter, model, _ = tiny_model
    torch = host.torch
    assert not model.beta4_ready
    with pytest.raises(RuntimeError, match="534"):
        model(**packed_inputs())
    loaded = model.load_weights(tiny_tensor_stream())
    assert len(loaded) == 534
    assert model.beta4_loaded_sources == frozenset(tiny_inventory())
    assert len(dict(model.named_parameters())) == 530
    assert len(dict(model.named_buffers())) == 4
    assert sum(x.dtype == torch.float32 for x in model.parameters()) == 110
    assert sum(x.dtype == torch.bfloat16 for x in model.parameters()) == 420
    assert type(model.blocks[0]) is host.official.MiniMaxH3DiTBlock
    assert type(model.blocks[0].attn) is host.official.MiniMaxH3Attention
    model.post_load_weights()
    with torch.inference_mode():
        video, audio = model(**packed_inputs())
    assert video.shape == (4, 8) and audio.shape == (2, 4)
    assert video.dtype == audio.dtype == torch.float32
    assert torch.isfinite(video).all() and torch.isfinite(audio).all()
    assert torch.count_nonzero(video[-1]) == torch.count_nonzero(audio[-1]) == 0
    assert torch.count_nonzero(video[:-1]) > 0 and torch.count_nonzero(audio[:-1]) > 0


def test_actual_host_final_widens_before_modulation(tiny_model):
    host, adapter, model, _ = tiny_model
    torch = host.torch
    final = model.final_layer
    with torch.no_grad():
        final.norm.weight.fill_(1)
        final.adaln_proj.linear.weight.zero_()
        final.adaln_proj.linear.bias.zero_()
        final.adaln_proj.linear.bias[:32].fill_(2**-9)
        for head in (final.video_out, final.audio_out):
            head.weight.zero_()
            head.weight[:, 0] = 1
            head.bias.zero_()
        values = torch.ones((2, 32), dtype=torch.bfloat16)
        kwargs = {"t_emb": torch.zeros((1, 8)), "inverse_indices": torch.zeros(2, dtype=torch.long)}
        observed = final(values, **kwargs)
        historical_official = host.official.MiniMaxH3FinalLayer.forward(final, values, **kwargs)
    for current, old in zip(observed, historical_official, strict=True):
        assert torch.equal(current, torch.full_like(current, 1.001953125))
        assert torch.equal(old, torch.ones_like(old)), "fixture must distinguish the extra BF16 rounding"


def test_table_interpolation_and_persistent_buffer_movement(tiny_model):
    host, _, model, _ = tiny_model
    torch = host.torch
    model.load_weights(tiny_tensor_stream())
    embedder = model.time_embedder
    table = dict(tiny_tensor_stream())["adaln_t_table"].float()
    values = torch.tensor([-1.0, 0, 0.125, 0.25, 0.875, 1, 2])
    expected = torch.stack(
        [table[0], table[0], (table[0] + table[1]) / 2, table[1], (table[3] + table[4]) / 2, table[4], table[4]]
    )
    assert torch.equal(embedder(values), expected)
    assert torch.equal(embedder(torch.tensor([0, 2, 4])), table[[0, 2, 4]])
    # Module.to can replace a registered buffer; forward must use that buffer.
    embedder._apply(lambda tensor: tensor.clone())
    embedder.adaln_t_table.add_(1)
    assert torch.equal(embedder(torch.tensor([0.0])), table[:1] + 1)
    for invalid in (torch.tensor([-1]), torch.tensor([5]), torch.tensor([float("nan")]), torch.zeros((1, 1))):
        with pytest.raises(ValueError):
            embedder(invalid)


def test_all_qkv_gate_up_and_persistent_values_reach_exact_runtime_slots(tiny_model):
    host, _, model, _ = tiny_model
    torch = host.torch
    weights = dict(tiny_tensor_stream())
    model.load_weights(weights.items())
    state = model.state_dict()
    qkv_count = fc1_count = 0
    for source, values in weights.items():
        target = "time_embedder.adaln_t_table" if source == "adaln_t_table" else source
        if source.endswith(".attn.qkv_proj.weight"):
            # Independent grouped [head, q/k/v, dim] -> concatenated q/k/v.
            indices = [
                head * 384 + section * 128 + offset
                for section in range(3)
                for head in range(2)
                for offset in range(128)
            ]
            values = values[indices]
            qkv_count += 1
        elif source.endswith(".mlp.fc1.weight"):
            fc1_count += 1
        assert torch.equal(state[target], values.to(state[target].dtype)), source
    assert qkv_count == fc1_count == 52
    receipt = model.loading_receipt()
    assert receipt["source_slots"] == receipt["runtime_slots"] == 534
    assert sum(x["numerical_forward_input"] for x in receipt["ledger"]) == 532


def test_basis_mean_are_retained_state_and_table_changes_forward(tiny_model):
    host, _, model, _ = tiny_model
    torch = host.torch
    model.load_weights(tiny_tensor_stream())
    with torch.inference_mode():
        baseline = model(**packed_inputs())
        original = {name: value.clone() for name, value in model.state_dict().items()}
        model.adaln_basis.add_(3)
        model.adaln_mean.sub_(7)
        retained_only = model(**packed_inputs())
        assert all(torch.equal(a, b) for a, b in zip(baseline, retained_only, strict=True))
        model.time_embedder.adaln_t_table.add_(2)
        changed = model(**packed_inputs())
        assert any(not torch.equal(a, b) for a, b in zip(baseline, changed, strict=True))
        model.load_state_dict(original, strict=True)
        model.post_load_weights()
        restored = model(**packed_inputs())
        assert all(torch.equal(a, b) for a, b in zip(baseline, restored, strict=True))


def test_failed_actual_model_construction_restores_classes_and_allows_retry(tmp_path):
    with actual_cpu_host() as host:
        adapter = importlib.import_module(ADAPTER)
        with patch.object(adapter.construction, "state", adapter.construction.WorkerConstructionState()):

            class TinyDiT(adapter.H3Beta4DiTModel):
                _inventory = tiny_inventory()
                _architecture = tiny_arch_config()

            originals = tuple(
                getattr(host.official, name)
                for name in ("MiniMaxH3TimeEmbedder", "MiniMaxH3AdalnProj", "MiniMaxH3FinalLayer")
            )
            with patch.object(host.official, "_norm", side_effect=RuntimeError("constructor fault")):
                with pytest.raises(RuntimeError, match="constructor fault"):
                    TinyDiT(host.od_config, binding=SimpleNamespace(component_root=tmp_path, files=()))
            assert not adapter.construction.state.model_constructed
            assert not adapter.construction.state.model_pending
            assert originals == tuple(
                getattr(host.official, name)
                for name in ("MiniMaxH3TimeEmbedder", "MiniMaxH3AdalnProj", "MiniMaxH3FinalLayer")
            )
            model = TinyDiT(host.od_config, binding=SimpleNamespace(component_root=tmp_path, files=()))
            model.load_weights(tiny_tensor_stream())
            assert model.beta4_ready
            with pytest.raises(RuntimeError, match="exactly one H3 model"):
                TinyDiT(host.od_config, binding=SimpleNamespace(component_root=tmp_path, files=()))


@pytest.mark.parametrize("fault", ["missing-buffer", "missing-weight", "unknown", "duplicate", "dtype", "shape"])
def test_invalid_source_stream_never_marks_the_model_ready(tiny_model, fault):
    host, _, model, _ = tiny_model
    torch = host.torch
    weights = list(tiny_tensor_stream())
    if fault == "missing-buffer":
        weights = [(n, t) for n, t in weights if n != "adaln_basis"]
    elif fault == "missing-weight":
        weights = [(n, t) for n, t in weights if n != "blocks.49.norm1.weight"]
    elif fault == "unknown":
        weights.append(("silu_t_emb_grid", torch.zeros((5, 16), dtype=torch.bfloat16)))
    elif fault == "duplicate":
        weights.append(weights[0])
    elif fault == "dtype":
        weights[0] = (weights[0][0], weights[0][1].float())
    else:
        weights[0] = (weights[0][0], weights[0][1].reshape(-1))
    with pytest.raises(ValueError):
        model.load_weights(iter(weights))
    assert not model.beta4_ready and not model.beta4_loaded_sources
    with pytest.raises(RuntimeError, match="one complete load"):
        model.load_weights(tiny_tensor_stream())
