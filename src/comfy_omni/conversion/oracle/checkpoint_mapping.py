"""Pure observations of explicit H3 LoRA pairs against checkpoint descriptors.

This is new structural inspection, not the legacy Turbo normalization profile.
No row reorder, basis projection, numeric fold, or implicit scale is performed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from comfy_omni.domain.checkpoints import TensorDescriptor

_PAIR_SUFFIXES = (
    (".lora_A.weight", "AB", "A"),
    (".lora_B.weight", "AB", "B"),
    (".lora_down.weight", "down-up", "A"),
    (".lora_up.weight", "down-up", "B"),
)
_FLOAT_DTYPES = frozenset({"BF16", "F16", "F32", "F64"})
_BLOCK = re.compile(
    r"(blocks|token_refiner\.blocks)\.(0|[1-9][0-9]*)\.(attn\.(qkv_proj|out_proj)|mlp\.fc[12]|adaln_proj\.linear)"
)


def _module_name(name: str) -> str:
    return name.removeprefix("diffusion_model.")


def _known_module(module: str) -> bool:
    if module == "final_layer.adaln_proj.linear":
        return True
    match = _BLOCK.fullmatch(module)
    if match is None:
        return False
    family, index, suffix = match.group(1, 2, 3)
    return int(index) < (50 if family == "blocks" else 2) and (family == "blocks" or suffix != "adaln_proj.linear")


def observe_mapping(
    base: Sequence[TensorDescriptor],
    adapter: Sequence[TensorDescriptor],
    *,
    alpha_values: Mapping[str, float],
    scale: float,
) -> dict[str, Any]:
    """Cover every adapter descriptor without inferring unknown key conventions."""
    targets = {item.name: item for item in base}
    groups: dict[str, list[tuple[TensorDescriptor, str, str]]] = {}
    alphas: dict[str, list[str]] = {}
    unknown: list[str] = []
    for tensor in sorted(adapter, key=lambda item: item.name):
        for suffix, syntax, side in _PAIR_SUFFIXES:
            if tensor.name.endswith(suffix):
                module = _module_name(tensor.name[: -len(suffix)])
                groups.setdefault(module, []).append((tensor, syntax, side))
                break
        else:
            if tensor.name.endswith(".alpha"):
                module = _module_name(tensor.name.removesuffix(".alpha"))
                alphas.setdefault(module, []).append(tensor.name)
            else:
                unknown.append(tensor.name)

    observations: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for name in unknown:
        failures.append({"tensor": name, "reason": "UNKNOWN_ADAPTER_KEY"})
    for module in sorted(set(groups) | set(alphas)):
        members = groups.get(module, [])
        alpha_names = alphas.get(module, [])
        target_name = module + ".weight"
        target = targets.get(target_name)
        record: dict[str, Any] = {
            "module": module,
            "keys": [item.name for item, _, _ in members],
            "pair_syntax": sorted({syntax for _, syntax, _ in members}),
            "shapes": {item.name: list(item.shape) for item, _, _ in members},
            "dtypes": {item.name: item.dtype for item, _, _ in members},
            "target": target_name,
            "target_shape": list(target.shape) if target is not None else None,
            "target_dtype": target.dtype if target is not None else None,
            "rank": None,
            "alpha": {name: alpha_values[name] for name in alpha_names if name in alpha_values},
            "alpha_source": "tensor" if alpha_names else "NOT_DECLARED",
            "requested_scale": scale,
            "effective_multiplier": None,
            "scale_semantics": "NOT_PROVEN",
            "binding": "UNRESOLVED",
        }
        observations.append(record)
        reason = None
        if len(alpha_names) > 1:
            reason = "AMBIGUOUS_ALPHA"
        elif len(members) != 2 or {side for _, _, side in members} != {"A", "B"}:
            reason = "INCOMPLETE_OR_DUPLICATE_PAIR"
        elif len({syntax for _, syntax, _ in members}) != 1:
            reason = "MIXED_PAIR_SYNTAX"
        else:
            a = next(item for item, _, side in members if side == "A")
            b = next(item for item, _, side in members if side == "B")
            valid_shapes = len(a.shape) == 2 and len(b.shape) == 2 and all((*a.shape, *b.shape))
            if valid_shapes:
                record["rank"] = a.shape[0]
            if a.dtype not in _FLOAT_DTYPES or b.dtype != a.dtype:
                reason = "PAIR_DTYPE_MISMATCH"
            elif not valid_shapes:
                reason = "PAIR_SHAPE_INVALID"
            else:
                if a.shape[0] != b.shape[1]:
                    reason = "PAIR_RANK_MISMATCH"
                elif not _known_module(module) or target is None:
                    reason = "UNKNOWN_TARGET_MODULE"
                elif len(target.shape) != 2 or (b.shape[0], a.shape[1]) != target.shape:
                    reason = "TARGET_SHAPE_MISMATCH"
                else:
                    record["binding"] = "SHAPE_ONLY"
                    if module.endswith("attn.qkv_proj"):
                        record["binding"] = "SHAPE_ONLY_QKV_ROW_ORDER_UNPROVED"
                    if module.endswith("adaln_proj.linear"):
                        record["binding"] = "SHAPE_ONLY_ADALN_BASIS_UNPROVED"
        if reason is not None:
            failures.append({"module": module, "reason": reason})
    if not groups:
        failures.append({"reason": "NO_ADAPTER_PAIRS"})
    return {"modules": observations, "unknown_keys": unknown, "failures": failures}
