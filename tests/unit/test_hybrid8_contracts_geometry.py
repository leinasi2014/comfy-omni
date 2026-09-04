"""Unit tests for the hybrid8 runtime contracts and geometry (pure, no Torch)."""

import pytest

from comfy_omni.runtime.h3.hybrid8 import (
    Hybrid8Geometry,
    Hybrid8StructureError,
    derive_hybrid8_geometry,
    has_hybrid8_signature,
    pinned_hybrid8_inventory,
    validate_hybrid8_census,
)
from comfy_omni.runtime.h3.hybrid8.contracts import HYBRID8_SIGNATURE_NAMES


def _bf16(shape: tuple[int, ...]) -> tuple[str, tuple[int, ...]]:
    return ("BF16", shape)


def _inventory() -> dict[str, tuple[str, tuple[int, ...]]]:
    inv: dict[str, tuple[str, tuple[int, ...]]] = {}
    for block in range(2):
        prefix = f"blocks.{block}"
        inv[f"{prefix}.adaln_proj.linear.bias"] = _bf16((108,))
        inv[f"{prefix}.adaln_proj.linear.weight"] = _bf16((108, 8))
        inv[f"{prefix}.attn.k_norm.weight"] = _bf16((6,))
        inv[f"{prefix}.attn.out_proj.weight"] = _bf16((6, 12))
        inv[f"{prefix}.attn.q_norm.weight"] = _bf16((6,))
        inv[f"{prefix}.attn.qkv_proj.weight"] = _bf16((36, 6))
        inv[f"{prefix}.mlp.fc1.weight"] = _bf16((8, 6))
        inv[f"{prefix}.mlp.fc2.weight"] = _bf16((6, 4))
        inv[f"{prefix}.norm1.weight"] = _bf16((6,))
        inv[f"{prefix}.norm2.weight"] = _bf16((6,))
        tprefix = f"token_refiner.blocks.{block}"
        inv[f"{tprefix}.attn.k_norm.weight"] = _bf16((6,))
        inv[f"{tprefix}.attn.out_proj.weight"] = _bf16((6, 12))
        inv[f"{tprefix}.attn.q_norm.weight"] = _bf16((6,))
        inv[f"{tprefix}.attn.qkv_proj.weight"] = _bf16((36, 6))
        inv[f"{tprefix}.mlp.fc1.weight"] = _bf16((8, 6))
        inv[f"{tprefix}.mlp.fc2.weight"] = _bf16((6, 4))
        inv[f"{tprefix}.norm1.weight"] = _bf16((6,))
        inv[f"{tprefix}.norm2.weight"] = _bf16((6,))
    inv["token_refiner.final_norm.weight"] = _bf16((6,))
    inv["adaln_t_table"] = _bf16((1025, 8))
    inv["adaln_basis"] = _bf16((8, 4))
    inv["adaln_mean"] = _bf16((4,))
    inv["silu_t_emb_grid"] = _bf16((1025, 4))
    inv["audio_patch_proj.bias"] = _bf16((6,))
    inv["audio_patch_proj.weight"] = _bf16((6, 2))
    inv["condition_proj.bias"] = _bf16((6,))
    inv["condition_proj.weight"] = _bf16((6, 4))
    inv["final_layer.adaln_proj.linear.bias"] = _bf16((12,))
    inv["final_layer.adaln_proj.linear.weight"] = _bf16((12, 8))
    inv["final_layer.audio_out.bias"] = _bf16((2,))
    inv["final_layer.audio_out.weight"] = _bf16((2, 6))
    inv["final_layer.norm.weight"] = _bf16((6,))
    inv["final_layer.video_out.bias"] = _bf16((6,))
    inv["final_layer.video_out.weight"] = _bf16((6, 6))
    inv["rope.inv_freq"] = _bf16((1,))
    inv["video_patch_proj.bias"] = _bf16((6,))
    inv["video_patch_proj.weight"] = _bf16((6, 6))
    return inv


def test_pinned_hybrid8_manifest_is_self_consistent() -> None:
    pinned = pinned_hybrid8_inventory()
    form = validate_hybrid8_census(dict(pinned), source="pin")
    assert form.num_blocks == 50
    assert form.cond_dim == 8
    assert len(form.inventory) == 535
    geometry = derive_hybrid8_geometry(pinned, num_blocks=50)
    assert geometry == Hybrid8Geometry(
        hidden_size=5376,
        num_heads=56,
        head_dim=128,
        rot_dim=96,
        ffn_hidden_size=14336,
        video_patch_dim=96,
        audio_patch_dim=32,
        text_dim=5120,
    )


def test_valid_census_accepts_and_reports_form() -> None:
    inventory = _inventory()
    form = validate_hybrid8_census(dict(inventory), source="unit", inventory=inventory)
    assert form.num_blocks == 2
    assert form.cond_dim == 8
    assert form.observed_schema_sha256 != ""
    assert set(form.inventory) == set(inventory)


def test_digest_defense_in_depth_is_stable() -> None:
    inventory = _inventory()
    form_a = validate_hybrid8_census(dict(inventory), source="unit-a", inventory=inventory)
    form_b = validate_hybrid8_census(dict(inventory), source="unit-b", inventory=inventory)
    assert form_a.observed_schema_sha256 == form_b.observed_schema_sha256


def test_signature_switch_requires_all_names() -> None:
    inventory = _inventory()
    assert has_hybrid8_signature(inventory) is True
    partial = {name: value for name, value in inventory.items() if name != "adaln_basis"}
    assert has_hybrid8_signature(partial) is False
    assert has_hybrid8_signature(partial, require_all=False) is True
    none_signature = {name: value for name, value in inventory.items() if name not in HYBRID8_SIGNATURE_NAMES}
    assert has_hybrid8_signature(none_signature, require_all=False) is False


def test_extra_tensor_is_rejected() -> None:
    pinned = _inventory()
    census = dict(pinned)
    census["blocks.0.not_a_tensor"] = _bf16((1,))
    with pytest.raises(Hybrid8StructureError, match="extra"):
        validate_hybrid8_census(census, source="unit", inventory=pinned)


def test_missing_tensor_is_rejected() -> None:
    pinned = _inventory()
    census = dict(pinned)
    del census["silu_t_emb_grid"]
    with pytest.raises(Hybrid8StructureError, match="missing"):
        validate_hybrid8_census(census, source="unit", inventory=pinned)


def test_dtype_or_shape_drift_is_rejected() -> None:
    pinned = _inventory()
    census = {name: value for name, value in pinned.items()}
    census["blocks.0.mlp.fc2.weight"] = ("F16", (6, 4))
    with pytest.raises(Hybrid8StructureError, match="deviates"):
        validate_hybrid8_census(census, source="unit", inventory=pinned)


def test_narrow_adaln_input_dim_is_enforced() -> None:
    pinned = _inventory()
    pinned["blocks.0.adaln_proj.linear.weight"] = _bf16((108, 2688))
    census = dict(pinned)
    with pytest.raises(Hybrid8StructureError, match="8-dim conditioning"):
        validate_hybrid8_census(census, source="unit", inventory=pinned)


def test_geometry_derivation() -> None:
    geometry = derive_hybrid8_geometry(_inventory(), num_blocks=2)
    assert geometry == Hybrid8Geometry(
        hidden_size=6,
        num_heads=2,
        head_dim=6,
        rot_dim=6,
        ffn_hidden_size=4,
        video_patch_dim=6,
        audio_patch_dim=2,
        text_dim=4,
    )


def test_geometry_rejects_bad_qkv_rows() -> None:
    pinned = _inventory()
    census = dict(pinned)
    census["blocks.0.attn.qkv_proj.weight"] = _bf16((35, 6))
    with pytest.raises(Hybrid8StructureError, match="qkv"):
        derive_hybrid8_geometry(census, num_blocks=2)


def test_geometry_rejects_rot_wider_than_head() -> None:
    pinned = _inventory()
    census = dict(pinned)
    census["rope.inv_freq"] = _bf16((2,))
    with pytest.raises(Hybrid8StructureError, match="rope width"):
        derive_hybrid8_geometry(census, num_blocks=2)
