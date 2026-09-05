"""Descriptor mapping is deliberately weaker than a numeric compatibility oracle."""

from __future__ import annotations

import pytest

from comfy_omni.conversion.oracle.checkpoint_mapping import observe_mapping
from comfy_omni.domain.checkpoints import TensorDescriptor


def _tensor(name, shape, dtype="BF16"):
    return TensorDescriptor(name, dtype, tuple(shape), (0, 0))


def _mapping(keys=None, *, rank=3, module="blocks.0.attn.out_proj", a_shape=None, b_shape=None, dtype="BF16"):
    keys = keys or (".lora_A.weight", ".lora_B.weight")
    base = [_tensor(module + ".weight", [8, 16])]
    adapter = [
        _tensor("diffusion_model." + module + keys[0], a_shape or [rank, 16]),
        _tensor("diffusion_model." + module + keys[1], b_shape or [8, rank], dtype),
    ]
    return observe_mapping(base, adapter, alpha_values={}, scale=0.5)


@pytest.mark.parametrize("keys", [(".lora_A.weight", ".lora_B.weight"), (".lora_down.weight", ".lora_up.weight")])
@pytest.mark.parametrize("rank", [1, 3, 32])
def test_variable_rank_pairs_are_observed_without_claiming_numeric_support(keys, rank):
    result = _mapping(keys, rank=rank)
    assert not result["failures"]
    record = result["modules"][0]
    assert record["rank"] == rank
    assert record["binding"] == "SHAPE_ONLY"
    assert record["alpha_source"] == "NOT_DECLARED"
    assert record["effective_multiplier"] is None


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"keys": (".lora_A.weight", ".lora_up.weight")}, "MIXED_PAIR_SYNTAX"),
        ({"keys": (".lora_A.weight", ".lora_A.weight")}, "INCOMPLETE_OR_DUPLICATE_PAIR"),
        ({"a_shape": [3, 16], "b_shape": [8, 4]}, "PAIR_RANK_MISMATCH"),
        ({"a_shape": [0, 16]}, "PAIR_SHAPE_INVALID"),
        ({"a_shape": [3, 32]}, "TARGET_SHAPE_MISMATCH"),
        ({"dtype": "F16"}, "PAIR_DTYPE_MISMATCH"),
        ({"module": "blocks.50.attn.out_proj"}, "UNKNOWN_TARGET_MODULE"),
    ],
)
def test_pair_refusals_are_specific(kwargs, reason):
    assert _mapping(**kwargs)["failures"][0]["reason"] == reason


def test_qkv_rows_and_adaln_basis_are_not_inferred_from_shape():
    qkv = _mapping(module="blocks.0.attn.qkv_proj")
    assert qkv["modules"][0]["binding"] == "SHAPE_ONLY_QKV_ROW_ORDER_UNPROVED"
    base = [_tensor("blocks.0.adaln_proj.linear.weight", [96768, 8])]
    adapter = [
        _tensor("blocks.0.adaln_proj.linear.lora_A.weight", [16, 2688]),
        _tensor("blocks.0.adaln_proj.linear.lora_B.weight", [96768, 16]),
    ]
    result = observe_mapping(base, adapter, alpha_values={}, scale=1.0)
    assert result["failures"][0]["reason"] == "TARGET_SHAPE_MISMATCH"


def test_missing_pairs_unknown_keys_and_alias_collisions_are_covered():
    module = "blocks.0.attn.out_proj"
    base = [_tensor(module + ".weight", [8, 16])]
    adapter = [
        _tensor(module + ".lora_A.weight", [3, 16]),
        _tensor("diffusion_model." + module + ".lora_A.weight", [3, 16]),
        _tensor(module + ".lora_B.weight", [8, 3]),
        _tensor("lora_unet_unknown.alpha", []),
    ]
    result = observe_mapping(base, adapter, alpha_values={}, scale=1.0)
    assert {item["reason"] for item in result["failures"]} == {"INCOMPLETE_OR_DUPLICATE_PAIR"}
    unknown = observe_mapping(base, [_tensor("unrecognized.weight", [3, 4])], alpha_values={}, scale=1.0)
    assert unknown["unknown_keys"] == ["unrecognized.weight"]
    assert any(item["reason"] == "NO_ADAPTER_PAIRS" for item in unknown["failures"])


@pytest.mark.parametrize("target", ["unknown", "missing"])
@pytest.mark.parametrize("suffixes", [(".lora_A.weight", ".lora_B.weight"), (".lora_down.weight", ".lora_up.weight")])
@pytest.mark.parametrize(
    "b_rank,b_dtype,reason",
    [
        (3, "BF16", "UNKNOWN_TARGET_MODULE"),
        (4, "BF16", "PAIR_RANK_MISMATCH"),
        (3, "F16", "PAIR_DTYPE_MISMATCH"),
    ],
)
def test_target_failure_does_not_hide_pair_rank_or_validation(target, suffixes, b_rank, b_dtype, reason):
    module = "unrecognized.projection" if target == "unknown" else "blocks.0.attn.out_proj"
    base = [_tensor(module + ".weight", [8, 16])] if target == "unknown" else []
    adapter = [
        _tensor("diffusion_model." + module + suffixes[0], [3, 16]),
        _tensor("diffusion_model." + module + suffixes[1], [8, b_rank], b_dtype),
    ]
    result = observe_mapping(base, adapter, alpha_values={}, scale=0.3)
    assert result["modules"][0]["rank"] == 3
    assert result["modules"][0]["binding"] == "UNRESOLVED"
    assert result["failures"] == [{"module": module, "reason": reason}]
