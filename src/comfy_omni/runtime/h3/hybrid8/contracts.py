"""Hybrid8 dense DiT form contracts: signature discovery and fail-closed census validation.

Provenance: ported from ``h3-forge`` at commit ``e9cb011d00b028c149db3978de246c54f6e34acc``
(Apache-2.0), file ``src/h3_forge/h3/dense_pipeline.py``, blob
``6ddd34c49d532d56f568ec0010e925dbd86e5a2a`` (constants and ``_validate_hybrid8_census`` /
``_manifest_schema_sha256`` / ``_family_shapes`` / ``_block_indices``).  See
``docs/migration/dense-hybrid8-runtime-port-e9cb011.md`` for the full migration record.

This module is pure: no Torch, vLLM, or host imports.  Model construction lives in the pipeline
adapter under ``comfy_omni.integrations.vllm_omni``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from comfy_omni.contracts.templates import ARCHITECTURE_TEMPLATES
from comfy_omni.conversion.contract_workflows.census import schema_sha256
from comfy_omni.domain.checkpoints import TensorDescriptor

#: The pinned hybrid8 dense transformer template name.
HYBRID8_TEMPLATE_NAME = "h3-transformer-50l-hybrid8-bf16-plain"

#: Top-level tensors that exist ONLY in the hybrid8 dense form.  Their
#: presence in a transformer census is the route switch; none of these names
#: appears in the official-shaped form, so the switch cannot misfire.
HYBRID8_SIGNATURE_NAMES = frozenset(
    {
        "adaln_t_table",
        "adaln_basis",
        "adaln_mean",
        "silu_t_emb_grid",
    }
)

#: The top-level timestep code table: one 8-dim conditioning code per row.
HYBRID8_T_TABLE_NAME = "adaln_t_table"

#: Tensor-name suffixes of one hybrid8 transformer block (10 tensors).
HYBRID8_BLOCK_SUFFIXES = (
    "adaln_proj.linear.bias",
    "adaln_proj.linear.weight",
    "attn.k_norm.weight",
    "attn.out_proj.weight",
    "attn.q_norm.weight",
    "attn.qkv_proj.weight",
    "mlp.fc1.weight",
    "mlp.fc2.weight",
    "norm1.weight",
    "norm2.weight",
)

#: Token-refiner blocks keep the official block skeleton minus AdaLN (8 tensors).
HYBRID8_TOKEN_REFINER_SUFFIXES = (
    "attn.k_norm.weight",
    "attn.out_proj.weight",
    "attn.q_norm.weight",
    "attn.qkv_proj.weight",
    "mlp.fc1.weight",
    "mlp.fc2.weight",
    "norm1.weight",
    "norm2.weight",
)

#: Tensor-name suffixes of the final layer family (7 tensors).
HYBRID8_FINAL_LAYER_SUFFIXES = (
    "adaln_proj.linear.bias",
    "adaln_proj.linear.weight",
    "audio_out.bias",
    "audio_out.weight",
    "norm.weight",
    "video_out.bias",
    "video_out.weight",
)

#: Serialized-quantization marker suffixes rejected on the dense route.
QUANT_MARKER_SUFFIXES = (".comfy_quant", ".weight_scale")

#: The only QKV checkpoint row layout the hybrid8 route serves.
HYBRID8_QKV_GROUPED_LAYOUT = "grouped-for-official-loader"

#: AdaLN modality count of the official skeleton (token tags -1/0/1/2).
HYBRID8_ADALN_MODALITY_NUM = 3


class Hybrid8StructureError(ValueError):
    """A hybrid8-shaped checkpoint declared an unsupported structure."""


@dataclass(frozen=True)
class Hybrid8DitForm:
    """A validated hybrid8 dense DiT form.

    ``inventory`` is the pinned manifest of the hybrid8 template (name -> (dtype, shape)),
    the exact table a pinned bf16-plain contract composes as its architecture inventory.
    ``observed_schema_sha256`` is the canonical census digest.  ``qkv_layout`` is the trusted
    fused-QKV row-layout declaration; ``"grouped-for-official-loader"`` is the only servable
    value and ``None`` means no declaration (fail closed instead of guessing).
    """

    inventory: Mapping[str, tuple[str, tuple[int, ...]]]
    observed_schema_sha256: str
    num_blocks: int
    cond_dim: int
    source: str
    qkv_layout: str | None = None
    transformer_dir: str | None = None


def pinned_hybrid8_inventory() -> Mapping[str, tuple[str, tuple[int, ...]]]:
    """Return the live pinned hybrid8 manifest (read at call time for patchability)."""
    template = ARCHITECTURE_TEMPLATES.get(HYBRID8_TEMPLATE_NAME)
    if template is None:
        raise Hybrid8StructureError(f"the pinned hybrid8 template {HYBRID8_TEMPLATE_NAME!r} is not registered")
    inventory = template.non_quantized_inventory
    if not inventory:
        raise Hybrid8StructureError(
            "the pinned hybrid8 template carries no tensor manifest; dense host-load requires the complete inventory"
        )
    return inventory


def manifest_schema_sha256(inventory: Mapping[str, tuple[str, tuple[int, ...]]]) -> str:
    """Canonical name/dtype/shape digest of a hybrid8 manifest."""
    return schema_sha256(
        TensorDescriptor(name, dtype, shape, (0, 0)) for name, (dtype, shape) in sorted(inventory.items())
    )


def has_hybrid8_signature(
    census: Mapping[str, tuple[str, tuple[int, ...]]],
    *,
    require_all: bool = True,
) -> bool:
    """Whether a transformer census carries the hybrid8 dense-form signature.

    With ``require_all=True`` every signature name must be present (the strict route switch);
    with ``require_all=False`` at least one is sufficient (used by early probes).
    """
    present = [name for name in sorted(HYBRID8_SIGNATURE_NAMES) if name in census]
    if require_all:
        return len(present) == len(HYBRID8_SIGNATURE_NAMES)
    return bool(present)


def _family_shapes(
    inventory: Mapping[str, tuple[str, tuple[int, ...]]],
    prefix: str,
    suffixes: tuple[str, ...],
) -> dict[str, tuple[int, ...]]:
    """Shapes of one block family, asserting the family is complete and exact."""
    shapes: dict[str, tuple[int, ...]] = {}
    for suffix in suffixes:
        entry = inventory.get(f"{prefix}.{suffix}")
        if entry is None:
            raise Hybrid8StructureError(
                f"the pinned hybrid8 manifest is missing {prefix}.{suffix}; the dense hybrid8 "
                "form cannot be assembled from a drifted template"
            )
        shapes[suffix] = entry[1]
    unexpected = sorted(
        name for name in inventory if name.startswith(f"{prefix}.") and name[len(prefix) + 1 :] not in suffixes
    )
    if unexpected:
        raise Hybrid8StructureError(f"the pinned hybrid8 manifest carries unknown {prefix}.* tensors: {unexpected[:4]}")
    return shapes


def family_shapes(
    inventory: Mapping[str, tuple[str, tuple[int, ...]]],
    prefix: str,
    suffixes: tuple[str, ...],
) -> dict[str, tuple[int, ...]]:
    """Public alias of the strict block-family shape check."""
    return _family_shapes(inventory, prefix, suffixes)


def block_indices(names: Sequence[str] | Mapping[str, Any], prefix: str) -> tuple[int, ...]:
    """Distinct block indices under ``prefix``, asserting a contiguous run from 0."""
    indices: set[int] = set()
    for name in names:
        if not name.startswith(f"{prefix}."):
            continue
        head = name[len(prefix) + 1 :].split(".", 1)[0]
        if head.isdigit():
            indices.add(int(head))
    if not indices:
        raise Hybrid8StructureError(f"the hybrid8 manifest declares no {prefix}.* blocks")
    ordered = sorted(indices)
    if ordered != list(range(len(ordered))):
        raise Hybrid8StructureError(
            f"the hybrid8 {prefix} block indices must be a contiguous run from 0, got {ordered[:8]}"
        )
    return tuple(ordered)


def validate_hybrid8_census(
    census: Mapping[str, tuple[str, tuple[int, ...]]],
    *,
    source: str,
    inventory: Mapping[str, tuple[str, tuple[int, ...]]] | None = None,
) -> Hybrid8DitForm:
    """Fail-closed validation of a hybrid8-signature census against the pinned manifest.

    Every tensor must match the manifest on name, dtype and shape -- there is no lenient
    subset.  ``inventory`` defaults to the pinned template inventory and is injectable for
    tests; production callers must not pass a driftable inventory.
    """
    pinned = inventory if inventory is not None else pinned_hybrid8_inventory()
    if not pinned:
        raise Hybrid8StructureError(
            "the hybrid8 template carries no tensor manifest; dense host-load requires the complete inventory"
        )
    markers = sorted(name for name in census if name.endswith(QUANT_MARKER_SUFFIXES))
    if markers:
        raise Hybrid8StructureError(
            f"hybrid8 dense transformer census in {source} carries serialized-quantization "
            f"markers {markers[:4]}; serve the marker-free dense bf16 checkpoint instead"
        )
    missing = sorted(set(pinned) - set(census))
    extra = sorted(set(census) - set(pinned))
    if missing or extra:
        raise Hybrid8StructureError(
            f"transformer census in {source} does not match the pinned hybrid8 manifest: "
            f"missing={missing[:8]}{'...' if len(missing) > 8 else ''} "
            f"extra={extra[:8]}{'...' if len(extra) > 8 else ''}"
        )
    wrong = sorted(name for name, entry in census.items() if pinned[name] != entry)
    if wrong:
        details = "; ".join(f"{name}: census {census[name]} != pinned {pinned[name]}" for name in wrong[:4])
        raise Hybrid8StructureError(
            f"transformer census in {source} deviates from the pinned hybrid8 manifest on "
            f"{len(wrong)} tensors: {details}"
        )
    observed = manifest_schema_sha256(census)
    pinned_digest = manifest_schema_sha256(pinned)
    if observed != pinned_digest:
        # Defense in depth: unreachable while every item matched, kept so the
        # digest machinery itself can never silently drift from the pin.
        raise Hybrid8StructureError(
            f"transformer census digest {observed} in {source} != the pinned hybrid8 manifest digest {pinned_digest}"
        )

    def _shape(name: str) -> tuple[int, ...]:
        return census[name][1]

    table_shape = _shape(HYBRID8_T_TABLE_NAME)
    basis_shape = _shape("adaln_basis")
    grid_shape = _shape("silu_t_emb_grid")
    mean_shape = _shape("adaln_mean")
    if len(table_shape) != 2 or len(basis_shape) != 2 or len(grid_shape) != 2 or len(mean_shape) != 1:
        raise Hybrid8StructureError(
            f"hybrid8 conditioning tensors in {source} have unexpected ranks: "
            f"table {table_shape}, basis {basis_shape}, grid {grid_shape}, mean {mean_shape}"
        )
    cond_dim = table_shape[1]
    embed_dim = basis_shape[1]
    if basis_shape[0] != cond_dim or grid_shape[1] != embed_dim or mean_shape[0] != embed_dim:
        raise Hybrid8StructureError(
            f"hybrid8 conditioning geometry in {source} is inconsistent: table {table_shape}, "
            f"basis {basis_shape}, grid {grid_shape}, mean {mean_shape}"
        )
    if grid_shape[0] != table_shape[0]:
        raise Hybrid8StructureError(
            f"hybrid8 timestep grids in {source} disagree: adaln_t_table {table_shape} vs silu_t_emb_grid {grid_shape}"
        )
    for name in pinned:
        if not name.endswith(".adaln_proj.linear.weight"):
            continue
        weight_shape = _shape(name)
        bias = census.get(name[: -len(".weight")] + ".bias")
        if len(weight_shape) != 2 or weight_shape[1] != cond_dim:
            raise Hybrid8StructureError(
                f"{name} in {source} has shape {weight_shape}; every hybrid8 AdaLN projection "
                f"must be a 2-D linear fed by the shared {cond_dim}-dim conditioning"
            )
        if bias is None or bias[1] != (weight_shape[0],):
            raise Hybrid8StructureError(f"{name} in {source} has no matching bias of shape ({weight_shape[0]},)")
    _family_shapes(pinned, "blocks.0", HYBRID8_BLOCK_SUFFIXES)
    _family_shapes(pinned, "token_refiner.blocks.0", HYBRID8_TOKEN_REFINER_SUFFIXES)
    block_count = len(block_indices(pinned, "blocks"))
    block_indices(pinned, "token_refiner.blocks")
    return Hybrid8DitForm(
        inventory=MappingProxyType(dict(pinned)),
        observed_schema_sha256=observed,
        num_blocks=block_count,
        cond_dim=cond_dim,
        source=source,
    )


__all__ = [
    "HYBRID8_TEMPLATE_NAME",
    "HYBRID8_SIGNATURE_NAMES",
    "HYBRID8_T_TABLE_NAME",
    "HYBRID8_BLOCK_SUFFIXES",
    "HYBRID8_TOKEN_REFINER_SUFFIXES",
    "HYBRID8_FINAL_LAYER_SUFFIXES",
    "QUANT_MARKER_SUFFIXES",
    "HYBRID8_QKV_GROUPED_LAYOUT",
    "HYBRID8_ADALN_MODALITY_NUM",
    "Hybrid8StructureError",
    "Hybrid8DitForm",
    "pinned_hybrid8_inventory",
    "manifest_schema_sha256",
    "has_hybrid8_signature",
    "validate_hybrid8_census",
    "block_indices",
    "family_shapes",
]
