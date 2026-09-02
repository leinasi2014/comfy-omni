"""Pure fail-closed Comfy INT8 ConvRot marker and triplet contracts.

Derived from Apache-2.0 h3-forge convrot.py at commit
e9cb011d00b028c149db3978de246c54f6e34acc. Runtime tensor operations are intentionally excluded.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from comfy_omni.contracts.models import ContractError
from comfy_omni.domain.checkpoints import TensorDescriptor

COMFY_MARKER_SUFFIX = ".comfy_quant"
WEIGHT_SUFFIX = ".weight"
SCALE_SUFFIX = ".weight_scale"
DEFAULT_GROUP_SIZE = 256
MAX_MARKER_BYTES = 4096


class ConvRotError(ContractError):
    """A source cannot be proven to contain valid ConvRot triplets."""


@dataclass(frozen=True)
class ConvRotGroup:
    prefix: str
    weight: TensorDescriptor
    scale: TensorDescriptor
    marker: TensorDescriptor
    group_size: int = DEFAULT_GROUP_SIZE


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConvRotError(f"duplicate marker JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ConvRotError(f"non-standard JSON constant {value!r}")


def parse_convrot_marker(raw: bytes, *, expected_group_size: int = DEFAULT_GROUP_SIZE) -> None:
    """Validate the exact marker emitted by the audited Comfy H3 artifacts."""

    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_MARKER_BYTES:
        raise ConvRotError(f"comfy_quant marker size must be in 1..{MAX_MARKER_BYTES} bytes")
    try:
        payload = json.loads(raw.decode(), object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ConvRotError(f"invalid comfy_quant JSON: {exc}") from exc
    expected_keys = {"format", "convrot", "convrot_groupsize"}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        observed = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        raise ConvRotError(f"comfy_quant marker keys must be exactly {sorted(expected_keys)}, got {observed}")
    if payload["format"] != "int8_tensorwise":
        raise ConvRotError("comfy_quant format must be 'int8_tensorwise'")
    if type(payload["convrot"]) is not bool or not payload["convrot"]:
        raise ConvRotError("comfy_quant convrot must be boolean true")
    if type(payload["convrot_groupsize"]) is not int or payload["convrot_groupsize"] != expected_group_size:
        raise ConvRotError(f"comfy_quant convrot_groupsize must be {expected_group_size}")


def _validate_triplet(prefix: str, tensors: Mapping[str, TensorDescriptor], group_size: int) -> ConvRotGroup:
    names = (prefix + WEIGHT_SUFFIX, prefix + SCALE_SUFFIX, prefix + COMFY_MARKER_SUFFIX)
    try:
        weight, scale, marker = (tensors[name] for name in names)
    except KeyError as exc:
        raise ConvRotError(f"incomplete ConvRot triplet for {prefix!r}: missing {exc.args[0]!r}") from exc
    marker_bytes = marker.data_offsets[1] - marker.data_offsets[0]
    if marker.dtype != "U8" or len(marker.shape) != 1 or marker.shape[0] != marker_bytes:
        raise ConvRotError(f"{marker.name}: marker must be a one-dimensional U8 byte tensor")
    if weight.dtype != "I8" or len(weight.shape) != 2:
        raise ConvRotError(f"{weight.name}: ConvRot weight must be rank-2 I8")
    if weight.shape[0] == 0 or weight.shape[1] == 0 or weight.shape[1] % group_size:
        raise ConvRotError(f"{weight.name}: dimensions must be nonzero and input width divisible by {group_size}")
    if scale.dtype != "F32" or scale.shape != (weight.shape[0], 1):
        raise ConvRotError(f"{scale.name}: source rowwise scale must be F32 shaped [{weight.shape[0]}, 1]")
    return ConvRotGroup(prefix, weight, scale, marker, group_size)


def _reject_orphans(tensors: Mapping[str, TensorDescriptor], claimed: set[str]) -> None:
    scales = sorted(
        name
        for name in tensors
        if name.endswith(SCALE_SUFFIX)
        and name not in claimed
        and (weight := tensors.get(name[: -len(SCALE_SUFFIX)] + WEIGHT_SUFFIX)) is not None
        and weight.dtype == "I8"
    )
    weights = sorted(
        name
        for name, descriptor in tensors.items()
        if name.endswith(WEIGHT_SUFFIX)
        and descriptor.dtype == "I8"
        and len(descriptor.shape) == 2
        and name not in claimed
    )
    if scales:
        raise ConvRotError(f"orphan INT8 weight_scale tensors without markers: {scales}")
    if weights:
        raise ConvRotError(f"orphan rank-2 INT8 weights without complete ConvRot triplets: {weights}")


def discover_convrot_groups(
    descriptors: Sequence[TensorDescriptor],
    marker_payloads: Mapping[str, bytes],
    *,
    expected_groups: int,
    expected_group_sizes: Mapping[str, int] | None = None,
) -> tuple[ConvRotGroup, ...]:
    """Validate complete, non-ambiguous weight/scale/marker triplets."""

    if expected_groups <= 0:
        raise ConvRotError("expected_groups must be positive")
    tensors = {descriptor.name: descriptor for descriptor in descriptors}
    if len(tensors) != len(descriptors):
        raise ConvRotError("duplicate tensor name in ConvRot census")
    marker_names = sorted(name for name in tensors if name.endswith(COMFY_MARKER_SUFFIX))
    if len(marker_names) != expected_groups:
        raise ConvRotError(f"expected exactly {expected_groups} comfy_quant markers, found {len(marker_names)}")
    if set(marker_payloads) != set(marker_names):
        missing = sorted(set(marker_names) - set(marker_payloads))
        extra = sorted(set(marker_payloads) - set(marker_names))
        raise ConvRotError(f"marker payload coverage mismatch: missing={missing}, extra={extra}")
    sizes = dict(expected_group_sizes or {})
    prefixes = {name[: -len(COMFY_MARKER_SUFFIX)] for name in marker_names}
    if unknown := sorted(set(sizes) - prefixes):
        raise ConvRotError(f"group-size contract references absent ConvRot groups: {unknown[:4]}")
    groups: list[ConvRotGroup] = []
    claimed: set[str] = set()
    for marker_name in marker_names:
        prefix = marker_name[: -len(COMFY_MARKER_SUFFIX)]
        group = _validate_triplet(prefix, tensors, sizes.get(prefix, DEFAULT_GROUP_SIZE))
        parse_convrot_marker(marker_payloads[marker_name], expected_group_size=group.group_size)
        members = {group.weight.name, group.scale.name, group.marker.name}
        if claimed & members:
            raise ConvRotError(f"duplicate ConvRot triplet claim for {prefix!r}")
        claimed.update(members)
        groups.append(group)
    _reject_orphans(tensors, claimed)
    return tuple(groups)


__all__ = [
    "COMFY_MARKER_SUFFIX",
    "SCALE_SUFFIX",
    "ConvRotError",
    "ConvRotGroup",
    "discover_convrot_groups",
    "parse_convrot_marker",
]
