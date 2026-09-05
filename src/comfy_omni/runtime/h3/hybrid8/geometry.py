"""Hybrid8 dense DiT forward geometry: derive and validate, fail-closed.

Provenance: ported from ``h3-forge`` at commit ``e9cb011d00b028c149db3978de246c54f6e34acc``
(Apache-2.0), file ``src/h3_forge/h3/dense_pipeline.py``, blob
``6ddd34c49d532d56f568ec0010e925dbd86e5a2a`` (``_Hybrid8Geometry`` and
``_derive_hybrid8_geometry``).  Pure module: no Torch, vLLM, or host imports.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from comfy_omni.runtime.h3.hybrid8.contracts import (
    HYBRID8_BLOCK_SUFFIXES,
    HYBRID8_TOKEN_REFINER_SUFFIXES,
    Hybrid8StructureError,
)


@dataclass(frozen=True)
class Hybrid8Geometry:
    """The forward geometry of a validated hybrid8 manifest (all derived).

    Every field is derived from the validated manifest and cross-checked for
    the official skeleton's coherence (no dimension is assumed): hidden size
    from the block norms, head layout from the q/k norms and fused qkv rows,
    rotary width from ``rope.inv_freq`` (``rot_dim = 6 * inv_freq_len``), and
    the video/audio/text input widths from the patch/condition projections.
    """

    hidden_size: int
    num_heads: int
    head_dim: int
    rot_dim: int
    ffn_hidden_size: int
    video_patch_dim: int
    audio_patch_dim: int
    text_dim: int


def derive_hybrid8_geometry(
    inventory: Mapping[str, tuple[str, tuple[int, ...]]],
    *,
    num_blocks: int,
) -> Hybrid8Geometry:
    """Derive and structurally validate the forward geometry, fail-closed."""

    def _shape(name: str) -> tuple[int, ...]:
        return inventory[name][1]

    hidden = _shape("blocks.0.norm1.weight")[0]
    head_dim = _shape("blocks.0.attn.q_norm.weight")[0]
    qkv_rows, qkv_cols = _shape("blocks.0.attn.qkv_proj.weight")
    if qkv_cols != hidden or qkv_rows % (3 * head_dim):
        raise Hybrid8StructureError(
            f"hybrid8 block 0 qkv {qkv_rows}x{qkv_cols} does not fit 3*heads*{head_dim} over hidden {hidden}"
        )
    heads = qkv_rows // (3 * head_dim)
    if heads < 1:
        raise Hybrid8StructureError("hybrid8 attention needs at least one head")
    out_rows, out_cols = _shape("blocks.0.attn.out_proj.weight")
    if out_rows != hidden or out_cols != heads * head_dim:
        raise Hybrid8StructureError(
            f"hybrid8 block 0 out_proj {out_rows}x{out_cols} != hidden x heads*head_dim ({hidden}x{heads * head_dim})"
        )
    fc1_rows, fc1_cols = _shape("blocks.0.mlp.fc1.weight")
    fc2_rows, fc2_cols = _shape("blocks.0.mlp.fc2.weight")
    ffn = fc2_cols
    if fc1_cols != hidden or fc1_rows != 2 * ffn or fc2_rows != hidden:
        raise Hybrid8StructureError(
            f"hybrid8 block 0 mlp fc1 {fc1_rows}x{fc1_cols}/fc2 {fc2_rows}x{fc2_cols} "
            f"disagrees with hidden {hidden}, ffn {ffn}"
        )
    inv_freq = _shape("rope.inv_freq")
    if len(inv_freq) != 1:
        raise Hybrid8StructureError(f"hybrid8 rope.inv_freq must be 1-D, got {inv_freq}")
    rot_dim = 6 * inv_freq[0]
    if rot_dim > head_dim:
        raise Hybrid8StructureError(f"hybrid8 rope width {rot_dim} (6*{inv_freq[0]}) exceeds head dim {head_dim}")
    video_patch_dim = _shape("video_patch_proj.weight")[1]
    audio_patch_dim = _shape("audio_patch_proj.weight")[1]
    text_dim = _shape("condition_proj.weight")[1]
    for name, dim in (
        ("video_patch_proj", video_patch_dim),
        ("audio_patch_proj", audio_patch_dim),
        ("condition_proj", text_dim),
    ):
        rows, cols = _shape(f"{name}.weight")
        if rows != hidden or cols != dim:
            raise Hybrid8StructureError(f"hybrid8 {name}.weight {rows}x{cols} disagrees with hidden {hidden}")
    if _shape("final_layer.norm.weight")[0] != hidden:
        raise Hybrid8StructureError("hybrid8 final layer norm disagrees with the block hidden size")
    if _shape("final_layer.video_out.weight") != (video_patch_dim, hidden):
        raise Hybrid8StructureError("hybrid8 final video_out disagrees with the video patch width")
    if _shape("final_layer.audio_out.weight") != (audio_patch_dim, hidden):
        raise Hybrid8StructureError("hybrid8 final audio_out disagrees with the audio patch width")
    if _shape("token_refiner.final_norm.weight")[0] != hidden:
        raise Hybrid8StructureError("hybrid8 token refiner final norm disagrees with the block hidden size")
    for index in range(num_blocks):
        for prefix, _family in (
            (f"blocks.{index}", HYBRID8_BLOCK_SUFFIXES),
            (f"token_refiner.blocks.{index}", HYBRID8_TOKEN_REFINER_SUFFIXES),
        ):
            if f"{prefix}.norm1.weight" not in inventory:
                break  # refiner family is shorter than the DiT stack
            if _shape(f"{prefix}.norm1.weight")[0] != hidden:
                raise Hybrid8StructureError(f"{prefix} hidden size deviates from block 0")
            if _shape(f"{prefix}.attn.q_norm.weight")[0] != head_dim:
                raise Hybrid8StructureError(f"{prefix} head dim deviates from block 0")
            if _shape(f"{prefix}.attn.qkv_proj.weight") != (qkv_rows, qkv_cols):
                raise Hybrid8StructureError(f"{prefix} qkv geometry deviates from block 0")
        if _shape(f"blocks.{index}.adaln_proj.linear.weight")[0] != 18 * hidden:
            raise Hybrid8StructureError(f"blocks.{index} adaln rows != 18*hidden ({18 * hidden})")
    if _shape("final_layer.adaln_proj.linear.weight")[0] != 2 * hidden:
        raise Hybrid8StructureError(f"final adaln rows != 2*hidden ({2 * hidden})")
    return Hybrid8Geometry(
        hidden_size=hidden,
        num_heads=heads,
        head_dim=head_dim,
        rot_dim=rot_dim,
        ffn_hidden_size=ffn,
        video_patch_dim=video_patch_dim,
        audio_patch_dim=audio_patch_dim,
        text_dim=text_dim,
    )


__all__ = ["Hybrid8Geometry", "derive_hybrid8_geometry"]
