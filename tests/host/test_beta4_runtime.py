"""Small tensors through real pinned host constructors and TP linears."""

from __future__ import annotations

import importlib
import importlib.util

import pytest

from beta4_host_fixture import actual_cpu_host, tiny_arch_config

pytest.importorskip("torch")
pytest.importorskip("vllm_omni")

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
