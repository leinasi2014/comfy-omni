from __future__ import annotations

from comfy_omni.contracts.conversion import (
    DENSE_BF16_ONLINE_INT8,
    EXPORT_SCHEMA,
    PROFILE_DENSE_BF16_ONLINE_INT8,
    QKV_SOURCE_LAYOUT,
    QKV_TARGET_LAYOUT,
)


def test_first_native_export_profile_preserves_legacy_wire_identifiers() -> None:
    assert EXPORT_SCHEMA == "h3-comfy-int8-export/v2"
    assert PROFILE_DENSE_BF16_ONLINE_INT8 == "dense-bf16-online-int8"
    assert DENSE_BF16_ONLINE_INT8.qkv.source_layout == QKV_SOURCE_LAYOUT == "runtime-qkv"
    assert DENSE_BF16_ONLINE_INT8.qkv.target_layout == QKV_TARGET_LAYOUT == "grouped-for-official-loader"


def test_transformer_runtime_quantization_policy_is_complete_and_unique() -> None:
    ignored = DENSE_BF16_ONLINE_INT8.runtime_ignored_layers

    assert DENSE_BF16_ONLINE_INT8.runtime_quant_method == "int8"
    assert len(ignored) == len(set(ignored)) == 60
    assert "condition_proj" in ignored
    assert "final_layer.adaln_proj.linear" in ignored
    assert "blocks.49.adaln_proj.linear" in ignored
    assert "token_refiner.blocks.1.mlp.fc2" in ignored
    assert "not-payload-preserving" in DENSE_BF16_ONLINE_INT8.payload_semantics
