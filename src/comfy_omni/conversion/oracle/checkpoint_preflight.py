"""Digest-bound raw-checkpoint LoRA preflight without a runtime-package dependency."""

from __future__ import annotations

import hashlib
import math
import struct
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from comfy_omni.artifacts.fileops import FsopsError, canonical_json, parse_json_strict
from comfy_omni.artifacts.sources import SafeTensorSources
from comfy_omni.contracts.models import ContractError
from comfy_omni.conversion.contract_workflows.census import census_tensors, schema_sha256
from comfy_omni.conversion.contract_workflows.convrot import (
    COMFY_MARKER_SUFFIX,
    MAX_MARKER_BYTES,
    discover_convrot_groups,
)
from comfy_omni.conversion.oracle.checkpoint_contract import (
    CheckpointInputError,
    CheckpointPin,
    CheckpointPreflightReceipt,
)
from comfy_omni.conversion.oracle.checkpoint_mapping import observe_mapping
from comfy_omni.conversion.oracle.contract import (
    BASE_REPRESENTATION_UNBINDABLE,
    OFFLINE_FOLD_ORACLE_NOT_PASSED,
    QUANT_LAYOUT_INCOMPATIBLE,
    TARGET_MODULE_MAPPING_UNRESOLVED,
)


@contextmanager
def _held_source(role: str, path: Path):
    try:
        with SafeTensorSources([path]) as source:
            try:
                yield source
            finally:
                source.verify_unchanged()
    except CheckpointInputError:
        raise
    except (ValueError, OSError, ContractError, FsopsError) as exc:
        raise CheckpointInputError(role, "CHECKPOINT_INPUT_INVALID", cause_type=type(exc).__name__) from exc


def _identity(source: SafeTensorSources, pin: CheckpointPin, role: str) -> dict[str, Any]:
    descriptors = tuple(item.descriptor for item in source.tensors.values())
    result = {
        "expected_sha256": pin.sha256,
        "expected_bytes": pin.size,
        "actual_sha256": source.hashes[0],
        "actual_bytes": source.sizes[0],
        "descriptor_schema_sha256": schema_sha256(descriptors),
        "tensor_count": len(descriptors),
        "dtype_counts": dict(sorted(Counter(item.dtype for item in descriptors).items())),
        "metadata_sha256": hashlib.sha256(canonical_json(source.metadata[0])).hexdigest(),
    }
    if (source.hashes[0], source.sizes[0]) != (pin.sha256, pin.size):
        raise CheckpointInputError(role, "CHECKPOINT_PIN_MISMATCH", **result)
    if not descriptors:
        raise CheckpointInputError(role, "CHECKPOINT_EMPTY", **result)
    return result


def _base_census(source: SafeTensorSources) -> dict[str, Any]:
    markers = {}
    claimed: set[str] = set()
    known_markers = {}
    group_sizes = {}
    for name, located in source.tensors.items():
        if name.endswith(COMFY_MARKER_SUFFIX):
            descriptor = located.descriptor
            start, end = descriptor.data_offsets
            if (
                descriptor.dtype != "U8"
                or descriptor.shape != (end - start,)
                or not 0 < end - start <= MAX_MARKER_BYTES
            ):
                raise CheckpointInputError("base", "CHECKPOINT_QUANTIZATION_INVALID", tensor=name)
            raw = source.read_raw(located)
            try:
                declaration = parse_json_strict(raw)
                if (
                    not isinstance(declaration, dict)
                    or not isinstance(declaration.get("format"), str)
                    or not declaration["format"]
                ):
                    raise ValueError("invalid quantization declaration")
                if (
                    declaration["format"] == "int8_tensorwise"
                    and "convrot" in declaration
                    and type(declaration["convrot"]) is not bool
                ):
                    raise ValueError("ConvRot flag must be a boolean")
                if declaration["format"] == "int8_tensorwise" and declaration.get("convrot") is True:
                    group_size = declaration.get("convrot_groupsize")
                    if type(group_size) is not int or group_size <= 0 or group_size & (group_size - 1):
                        raise ValueError("invalid ConvRot group size")
                    known_markers[name] = raw
                    group_sizes[name.removesuffix(COMFY_MARKER_SUFFIX)] = group_size
            except (ValueError, ContractError, FsopsError, RecursionError) as exc:
                raise CheckpointInputError("base", "CHECKPOINT_QUANTIZATION_INVALID", tensor=name) from exc
            markers[name] = raw
            prefix = name.removesuffix(COMFY_MARKER_SUFFIX)
            claimed.update((name, prefix + ".weight", prefix + ".weight_scale"))
            if name not in known_markers:
                claimed.add(prefix + ".weight_scale_2")
    orphan_scales = [
        name
        for name in sorted(source.tensors)
        if name.endswith((".weight_scale", ".weight_scale_2")) and name not in claimed
    ]
    if orphan_scales:
        raise CheckpointInputError("base", "CHECKPOINT_QUANTIZATION_INVALID", orphan_scales=orphan_scales)
    # A foreign declaration cannot hide a malformed known ConvRot triplet.
    known_names = {prefix + suffix for prefix in group_sizes for suffix in (".weight", ".weight_scale", ".comfy_quant")}
    groups = ()
    if known_markers:
        try:
            groups = discover_convrot_groups(
                tuple(item.descriptor for name, item in source.tensors.items() if name in known_names),
                known_markers,
                expected_groups=len(known_markers),
                expected_group_sizes=group_sizes,
            )
        except ContractError as exc:
            raise CheckpointInputError(
                "base", "CHECKPOINT_QUANTIZATION_INVALID", cause_type=type(exc).__name__
            ) from exc
    try:
        report = census_tensors(
            tuple(item.descriptor for item in source.tensors.values()), markers, metadata=source.metadata[0]
        )
        result = report.census_summary()
        groups = report.groups
    except ContractError as exc:
        refusal = getattr(exc, "evidence", {})
        if refusal.get("reason_code") != "unsupported-comfy-quant-storage":
            raise CheckpointInputError(
                "base", "CHECKPOINT_QUANTIZATION_INVALID", cause_type=type(exc).__name__
            ) from exc
        result = {
            "storage_kind": "unsupported-comfy-quant",
            "tensor_count": len(source.tensors),
            "marker_count": len(markers),
            "convrot_group_count": len(groups),
            "refusal": refusal,
        }
    result["input_mode"] = "single-file"
    result["file_count"] = 1
    result["quantization_markers"] = [
        {"tensor": name, "sha256": hashlib.sha256(raw).hexdigest()} for name, raw in sorted(markers.items())
    ]
    result["quantized_targets"] = [
        {
            "module": group.prefix,
            "group_size": group.group_size,
            "weight_shape": list(group.weight.shape),
            "scale_shape": list(group.scale.shape),
        }
        for group in groups
    ]
    result["adaln_geometry"] = [
        {
            "tensor": name,
            "shape": list(source.tensors[name].descriptor.shape),
            "dtype": source.tensors[name].descriptor.dtype,
        }
        for name in ("adaln_basis", "adaln_mean", "adaln_t_table")
        if name in source.tensors
    ]
    return result


def _alpha_observations(source: SafeTensorSources) -> tuple[dict[str, float], list[dict[str, str]], dict[str, str]]:
    values: dict[str, float] = {}
    failures: list[dict[str, str]] = []
    for name, located in sorted(source.tensors.items()):
        if not name.endswith(".alpha"):
            continue
        descriptor = located.descriptor
        if descriptor.shape not in ((), (1,)) or descriptor.dtype not in {"F16", "BF16", "F32", "F64"}:
            failures.append({"tensor": name, "reason": "ALPHA_NOT_FLOAT_SCALAR"})
            continue
        raw = source.read_raw(located)
        if descriptor.dtype == "BF16":
            value = struct.unpack("<f", b"\x00\x00" + raw)[0]
        else:
            value = struct.unpack({"F16": "<e", "F32": "<f", "F64": "<d"}[descriptor.dtype], raw)[0]
        if not math.isfinite(value):
            failures.append({"tensor": name, "reason": "ALPHA_NOT_FINITE"})
        else:
            values[name] = value
    declarations = {
        key: value
        for key, value in source.metadata[0].items()
        if key in {"alpha", "lora_alpha", "network_alpha", "ss_network_alpha", "ss_network_dim", "rank", "application"}
    }
    for key, raw in declarations.items():
        if key == "application":
            continue
        try:
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError
        except ValueError:
            failures.append({"metadata": key, "reason": "SCALE_DECLARATION_UNSUPPORTED"})
    return values, failures, declarations


def _scale_conflicts(mapping: dict[str, Any], declarations: dict[str, str]) -> list[dict[str, str]]:
    failures = []
    parsed = {}
    for key, raw in declarations.items():
        if key == "application":
            continue
        try:
            number = float(raw)
        except ValueError:
            continue  # Already reported by _alpha_observations.
        if math.isfinite(number):
            parsed[key] = number
    alpha_keys = {"alpha", "lora_alpha", "network_alpha", "ss_network_alpha"}
    alpha_values = {value for key, value in parsed.items() if key in alpha_keys}
    rank_values = {value for key, value in parsed.items() if key in {"rank", "ss_network_dim"}}
    if len(alpha_values) > 1:
        failures.append({"reason": "ALPHA_DECLARATION_CONFLICT"})
    if len(rank_values) > 1 or any(value <= 0 or not value.is_integer() for value in rank_values):
        failures.append({"reason": "RANK_DECLARATION_INVALID"})
    for record in mapping["modules"]:
        if alpha_values and any(value not in alpha_values for value in record["alpha"].values()):
            failures.append({"module": record["module"], "reason": "ALPHA_DECLARATION_CONFLICT"})
        if rank_values and record["rank"] is not None and rank_values != {record["rank"]}:
            failures.append({"module": record["module"], "reason": "RANK_DECLARATION_MISMATCH"})
    return failures


def preflight_checkpoint_candidate(
    candidate_id: str,
    base_path: Path | str,
    candidate_path: Path | str,
    *,
    base_sha256: str,
    base_bytes: int,
    pinned_sha256: str,
    pinned_bytes: int,
    scale: float = 1.0,
) -> CheckpointPreflightReceipt:
    """Observe a fully pinned pair; successful inspection still refuses activation.

    Invalid inputs raise CheckpointInputError. Every returned receipt means the
    actual files were hashed, strictly inspected and reverified before closing.
    It never authorizes conversion, a numerical fold or a runtime mutation.
    """
    base_pin, adapter_pin = CheckpointPin(base_sha256, base_bytes), CheckpointPin(pinned_sha256, pinned_bytes)
    if not isinstance(candidate_id, str) or not candidate_id or len(candidate_id) > 128:
        raise CheckpointInputError("request", "CANDIDATE_ID_INVALID")
    try:
        valid_scale = type(scale) in (int, float) and math.isfinite(scale)
    except OverflowError:
        valid_scale = False
    if not valid_scale:
        raise CheckpointInputError("request", "REQUEST_SCALE_INVALID")
    with _held_source("base", Path(base_path)) as base, _held_source("adapter", Path(candidate_path)) as adapter:
        evidence: dict[str, Any] = {
            "scope": "checkpoint-only",
            "promotion_capable": False,
            "offline_fold": "NOT_RUN",
            "runtime_activation": "NOT_RUN",
            "base": _identity(base, base_pin, "base"),
            "adapter": _identity(adapter, adapter_pin, "adapter"),
            "requested_scale": float(scale),
        }
        evidence["adapter"]["tensors"] = [
            {"name": name, "dtype": located.descriptor.dtype, "shape": list(located.descriptor.shape)}
            for name, located in sorted(adapter.tensors.items())
        ]
        evidence["base"]["census"] = _base_census(base)
        alpha_values, alpha_failures, declarations = _alpha_observations(adapter)
        mapping = observe_mapping(
            tuple(item.descriptor for item in base.tensors.values()),
            tuple(item.descriptor for item in adapter.tensors.values()),
            alpha_values=alpha_values,
            scale=float(scale),
        )
        evidence["mapping"] = mapping
        evidence["mapping_sha256"] = hashlib.sha256(canonical_json(mapping)).hexdigest()
        evidence["scale_declarations"] = declarations
        alpha_failures.extend(_scale_conflicts(mapping, declarations))
        evidence["scale_failures"] = alpha_failures
        if evidence["base"]["census"]["storage_kind"] == "unsupported-comfy-quant":
            reason = QUANT_LAYOUT_INCOMPATIBLE
        elif alpha_failures:
            reason = "ADAPTER_SCALE_UNSUPPORTED"
        elif mapping["failures"]:
            reason = TARGET_MODULE_MAPPING_UNRESOLVED
        elif evidence["base"]["census"]["storage_kind"] == "int8-convrot":
            reason = BASE_REPRESENTATION_UNBINDABLE
        else:
            reason = OFFLINE_FOLD_ORACLE_NOT_PASSED
        evidence["route_limitation"] = (
            "No proved numeric fold/activation route for this checkpoint pair; not universal incompatibility."
        )
        receipt = CheckpointPreflightReceipt(candidate_id, reason, evidence)
    return receipt
