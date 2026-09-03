# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright h3-forge contributors
#
# Provenance: wholesale migration from h3-forge@e9cb011d00b028c149db3978de246c54f6e34acc
#   source path: src/h3_forge/lora_hotswap/bake_audit.py
#   source blob: 25b1664d206e5319a12455c352a01fb3ac9d5869
#   license: Apache-2.0
#   attribution: h3-forge contributors
# Migrated byte-preserving except this provenance header, import retargeting, and
# mechanical line wrapping to satisfy the repository line-length (120).
"""Bounded residual-survival diagnostics for official H3 LoRA bake plans.

This module deliberately stops before writing a checkpoint.  It measures whether
one final target-dtype cast preserves an ideal FP32 LoRA residual.  This is not the
product bake gate: the product oracle is parity with the reference Comfy fold.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

from comfy_omni.artifacts.safetensors import read_safetensors_header_stream
from comfy_omni.domain.checkpoints import TensorDescriptor
from comfy_omni.domain.qkv import grouped_to_qkv_row_indices

from .bake_plan import (
    COMFY_BAKED_NATIVE_PRODUCT_GATE,
    BakeOperation,
    LoraBakePlan,
    _plan_sha256,
    _read_json_object,
    _target_contract,
    _validate_normalized_lora,
    plan_official_fl2va_bf16_bake,
)
from .normalize import TURBO_V4_MODULE_COUNT

LEGACY_BF16_AUDIT_SCHEMA = "h3-comfy.bf16-lora-bake-feasibility/v1"
AUDIT_SCHEMA = "h3-comfy.bf16-lora-bake-diagnostic/v2"
AGGREGATE_AUDIT_SCHEMA = "h3-comfy.bf16-lora-bake-diagnostic-aggregate/v2"
FP16_AUDIT_SCHEMA = "h3-comfy.fp16-lora-bake-diagnostic/v1"
FP16_AGGREGATE_AUDIT_SCHEMA = "h3-comfy.fp16-lora-bake-diagnostic-aggregate/v1"
DEFAULT_MAX_WORKING_SET_BYTES = 1024 * 1024 * 1024
MIN_WORKING_SET_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class AuditThresholds:
    """Conservative diagnostic limits for residual preservation after final RNE."""

    min_nonzero_survival: float = 0.999
    max_residual_relative_l2: float = 0.01
    min_residual_cosine: float = 0.9999
    min_residual_energy_ratio: float = 0.99
    max_residual_energy_ratio: float = 1.01
    advisory_max_merged_relative_l2: float = 0.005
    max_nonfinite_count: int = 0
    max_sign_flip_count: int = 0

    def validate(self) -> None:
        values = (
            self.min_nonzero_survival,
            self.max_residual_relative_l2,
            self.min_residual_cosine,
            self.min_residual_energy_ratio,
            self.max_residual_energy_ratio,
            self.advisory_max_merged_relative_l2,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("audit thresholds must be finite")
        if not 0 <= self.min_nonzero_survival <= 1:
            raise ValueError("min_nonzero_survival must be in [0, 1]")
        if not -1 <= self.min_residual_cosine <= 1:
            raise ValueError("min_residual_cosine must be in [-1, 1]")
        if self.max_residual_relative_l2 < 0 or self.advisory_max_merged_relative_l2 < 0:
            raise ValueError("relative L2 thresholds must be non-negative")
        if self.min_residual_energy_ratio < 0:
            raise ValueError("min_residual_energy_ratio must be non-negative")
        if self.max_residual_energy_ratio < self.min_residual_energy_ratio:
            raise ValueError("residual energy ratio bounds are reversed")
        if self.max_nonfinite_count < 0:
            raise ValueError("max_nonfinite_count must be non-negative")
        if self.max_sign_flip_count != 0:
            raise ValueError("max_sign_flip_count is a non-waivable zero threshold")


DEFAULT_AUDIT_THRESHOLDS = AuditThresholds()


@dataclass(frozen=True)
class PartitionSpec:
    name: str
    kind: str
    runtime_row_start: int
    runtime_row_end: int
    base_source_rows: tuple[int, ...] | None = None

    @property
    def row_count(self) -> int:
        return self.runtime_row_end - self.runtime_row_start


@dataclass
class MetricSums:
    element_count: int = 0
    source_nonzero_count: int = 0
    effective_nonzero_count: int = 0
    survived_nonzero_count: int = 0
    annihilated_count: int = 0
    finite_nonzero_to_zero_count: int = 0
    sign_flip_count: int = 0
    nonfinite_count: int = 0
    delta_l2_squared: float = 0.0
    effective_l2_squared: float = 0.0
    residual_error_l2_squared: float = 0.0
    merged_l2_squared: float = 0.0
    merged_error_l2_squared: float = 0.0
    residual_dot: float = 0.0
    max_abs_delta: float = 0.0
    max_abs_residual_error: float = 0.0

    def add(self, other: MetricSums) -> None:
        for field in (
            "element_count",
            "source_nonzero_count",
            "effective_nonzero_count",
            "survived_nonzero_count",
            "annihilated_count",
            "finite_nonzero_to_zero_count",
            "sign_flip_count",
            "nonfinite_count",
        ):
            setattr(self, field, getattr(self, field) + getattr(other, field))
        for field in (
            "delta_l2_squared",
            "effective_l2_squared",
            "residual_error_l2_squared",
            "merged_l2_squared",
            "merged_error_l2_squared",
            "residual_dot",
        ):
            setattr(self, field, math.fsum((getattr(self, field), getattr(other, field))))
        self.max_abs_delta = max(self.max_abs_delta, other.max_abs_delta)
        self.max_abs_residual_error = max(self.max_abs_residual_error, other.max_abs_residual_error)


@dataclass(frozen=True)
class PartitionMetrics:
    name: str
    kind: str
    row_count: int
    element_count: int
    source_nonzero_count: int
    effective_nonzero_count: int
    survived_nonzero_count: int
    annihilated_count: int
    finite_nonzero_to_zero_count: int
    sign_flip_count: int
    nonzero_survival: float
    residual_relative_l2: float | None
    residual_cosine: float | None
    residual_energy_ratio: float | None
    merged_relative_l2: float | None
    delta_l2: float
    max_abs_delta: float
    max_abs_residual_error: float
    nonfinite_count: int
    passed: bool
    failures: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class OperationAudit:
    plan_operation_index: int
    module: str
    kind: str
    base_tensor: str
    shard: str
    row_chunk_size: int
    partition_count: int
    passed: bool
    partitions: tuple[PartitionMetrics, ...]


@dataclass(frozen=True)
class OperationSelection:
    strategy: str
    partition_index: int
    partition_count: int
    selected_indices: tuple[int, ...]
    selection_sha256: str


@dataclass(frozen=True)
class BakeFeasibilityReceipt:
    schema: str
    profile: str
    plan_sha256: str
    plan_json_sha256: str | None
    base_catalog_sha256: str
    lora_sha256: str
    normalized_lora_size: int
    validated_base_shard_count: int
    source_content_binding: str
    audit_role: str
    product_gate: str
    product_gate_status: str
    oracle: str
    arithmetic: str
    target_dtype: str
    thresholds: AuditThresholds
    max_working_set_bytes: int
    device: str
    device_name: str
    compute_capability: str | None
    device_identity: str
    matmul_mode: str
    float32_matmul_precision: str
    torch_version: str
    python_version: str
    plan_operation_count: int
    selection: OperationSelection
    operation_count: int
    metric_partition_count: int
    failed_operation_count: int
    failed_partition_count: int
    weight_oracle_status: str
    activation_oracle_status: str
    decision: str
    elapsed_seconds: float
    operations: tuple[OperationAudit, ...]
    receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AggregateWorker:
    path: str
    receipt_json_sha256: str
    partition_index: int
    selection_sha256: str
    device_identity: str
    operation_count: int
    weight_oracle_status: str
    decision: str


@dataclass(frozen=True)
class AggregateBakeAuditReceipt:
    schema: str
    profile: str
    plan_sha256: str
    plan_json_sha256: str | None
    base_catalog_sha256: str
    lora_sha256: str
    target_dtype: str
    audit_role: str
    product_gate: str
    product_gate_status: str
    thresholds: dict[str, Any]
    partition_count: int
    plan_operation_count: int
    covered_operation_count: int
    device_identities: tuple[str, ...]
    failed_worker_count: int
    weight_oracle_status: str
    activation_oracle_status: str
    decision: str
    workers: tuple[AggregateWorker, ...]
    receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _partition_specs(operation: BakeOperation) -> tuple[PartitionSpec, ...]:
    rows = operation.shape[0]
    module = operation.module
    if module.endswith("attn.qkv_proj"):
        groups = operation.qkv_num_query_groups
        heads = operation.qkv_heads_per_group
        head_dim = operation.qkv_head_dim
        if groups is None or heads is None or head_dim is None:
            raise ValueError(f"QKV layout dimensions are missing for {module!r}")
        q_rows = groups * heads * head_dim
        kv_rows = groups * head_dim
        if rows != q_rows + 2 * kv_rows:
            raise ValueError(f"QKV row count is inconsistent for {module!r}")
        source = grouped_to_qkv_row_indices(num_query_groups=groups, heads_per_group=heads, head_dim=head_dim)
        return tuple(
            PartitionSpec(name, "qkv", start, end, source[start:end])
            for name, start, end in (
                ("q", 0, q_rows),
                ("k", q_rows, q_rows + kv_rows),
                ("v", q_rows + kv_rows, rows),
            )
        )
    if module.endswith("mlp.fc1"):
        if rows % 2:
            raise ValueError(f"fused gate/up row count must be even for {module!r}")
        half = rows // 2
        return (
            PartitionSpec("gate", "fused-mlp", 0, half),
            PartitionSpec("up", "fused-mlp", half, rows),
        )
    if module.endswith("adaln_proj.linear"):
        if module.startswith("final_layer."):
            if rows % 2:
                raise ValueError(f"final AdaLN row count must be divisible by two for {module!r}")
            hidden = rows // 2
            return (
                PartitionSpec("shift", "adaln-final", 0, hidden),
                PartitionSpec("scale", "adaln-final", hidden, rows),
            )
        if rows % 18:
            raise ValueError(f"block AdaLN row count must be divisible by 3*6 for {module!r}")
        hidden = rows // 18
        modalities = ("video", "text", "audio")
        channels = ("shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp")
        result: list[PartitionSpec] = []
        for modality_index, modality in enumerate(modalities):
            for channel_index, channel in enumerate(channels):
                start = (modality_index * len(channels) + channel_index) * hidden
                result.append(PartitionSpec(f"{modality}.{channel}", "adaln-block", start, start + hidden))
        return tuple(result)
    return (PartitionSpec("all", "ordinary", 0, rows),)


def finalize_partition_metrics(
    name: str,
    kind: str,
    row_count: int,
    sums: MetricSums,
    thresholds: AuditThresholds,
) -> PartitionMetrics:
    """Convert additive chunk statistics into deterministic gate metrics."""

    thresholds.validate()
    delta_l2 = math.sqrt(max(0.0, sums.delta_l2_squared))
    effective_l2 = math.sqrt(max(0.0, sums.effective_l2_squared))
    residual_error_l2 = math.sqrt(max(0.0, sums.residual_error_l2_squared))
    merged_l2 = math.sqrt(max(0.0, sums.merged_l2_squared))
    merged_error_l2 = math.sqrt(max(0.0, sums.merged_error_l2_squared))
    survival = sums.survived_nonzero_count / sums.source_nonzero_count if sums.source_nonzero_count else 0.0
    relative_error = residual_error_l2 / delta_l2 if delta_l2 else None
    cosine = sums.residual_dot / (delta_l2 * effective_l2) if delta_l2 and effective_l2 else None
    energy_ratio = effective_l2 / delta_l2 if delta_l2 else None
    merged_relative = merged_error_l2 / merged_l2 if merged_l2 else (0.0 if not merged_error_l2 else None)
    failures: list[str] = []
    if sums.element_count <= 0:
        failures.append("empty_partition")
    if sums.source_nonzero_count <= 0 or not delta_l2:
        failures.append("delta_has_no_energy")
    if survival < thresholds.min_nonzero_survival:
        failures.append("nonzero_survival")
    if relative_error is not None and relative_error > thresholds.max_residual_relative_l2:
        failures.append("residual_relative_l2")
    if cosine is None or cosine < thresholds.min_residual_cosine:
        failures.append("residual_cosine")
    if energy_ratio is None or not (
        thresholds.min_residual_energy_ratio <= energy_ratio <= thresholds.max_residual_energy_ratio
    ):
        failures.append("residual_energy_ratio")
    if sums.nonfinite_count > thresholds.max_nonfinite_count:
        failures.append("nonfinite_count")
    if sums.sign_flip_count > thresholds.max_sign_flip_count:
        failures.append("sign_flip_count")
    warnings = (
        ("merged_relative_l2",)
        if merged_relative is None or merged_relative > thresholds.advisory_max_merged_relative_l2
        else ()
    )
    return PartitionMetrics(
        name=name,
        kind=kind,
        row_count=row_count,
        element_count=sums.element_count,
        source_nonzero_count=sums.source_nonzero_count,
        effective_nonzero_count=sums.effective_nonzero_count,
        survived_nonzero_count=sums.survived_nonzero_count,
        annihilated_count=sums.annihilated_count,
        finite_nonzero_to_zero_count=sums.finite_nonzero_to_zero_count,
        sign_flip_count=sums.sign_flip_count,
        nonzero_survival=survival,
        residual_relative_l2=relative_error,
        residual_cosine=cosine,
        residual_energy_ratio=energy_ratio,
        merged_relative_l2=merged_relative,
        delta_l2=delta_l2,
        max_abs_delta=sums.max_abs_delta,
        max_abs_residual_error=sums.max_abs_residual_error,
        nonfinite_count=sums.nonfinite_count,
        passed=not failures,
        failures=tuple(failures),
        warnings=warnings,
    )


class _SafeTensorReader:
    def __init__(self, path: Path) -> None:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"audit input must be a regular, non-linked file: {path}")
        self.path = path
        self.stream: BinaryIO = path.open("rb")
        try:
            self.snapshot = os.fstat(self.stream.fileno())
            metadata, tensors, header_length = read_safetensors_header_stream(self.stream, path, self.snapshot.st_size)
        except BaseException:
            self.stream.close()
            raise
        self.metadata = metadata
        self.payload_start = 8 + header_length
        self.tensors = {tensor.name: tensor for tensor in tensors}

    def close(self) -> None:
        try:
            descriptor = os.fstat(self.stream.fileno())
            path_stat = self.path.stat()

            def identity(stat):
                return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

            if identity(descriptor) != identity(self.snapshot) or identity(path_stat) != identity(self.snapshot):
                raise ValueError(f"audit input changed while it was read: {self.path}")
        finally:
            self.stream.close()

    def __enter__(self) -> _SafeTensorReader:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def descriptor(self, name: str) -> TensorDescriptor:
        try:
            return self.tensors[name]
        except KeyError as exc:
            raise ValueError(f"audit tensor is missing from {self.path}: {name!r}") from exc

    def sha256(self) -> str:
        """Hash the complete input through the same descriptor used for tensor reads."""

        digest = hashlib.sha256()
        self.stream.seek(0)
        remaining = self.snapshot.st_size
        while remaining:
            chunk = self.stream.read(min(remaining, 8 * 1024 * 1024))
            if not chunk:
                raise ValueError(f"audit input was truncated while hashing: {self.path}")
            digest.update(chunk)
            remaining -= len(chunk)
        if self.stream.read(1):
            raise ValueError(f"audit input grew while hashing: {self.path}")
        descriptor = os.fstat(self.stream.fileno())

        def identity(stat):
            return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

        if identity(descriptor) != identity(self.snapshot) or identity(self.path.stat()) != identity(self.snapshot):
            raise ValueError(f"audit input changed while hashing: {self.path}")
        return digest.hexdigest()

    def read_bf16_rows(
        self,
        name: str,
        start: int,
        end: int,
        torch: Any,
        *,
        source_rows: tuple[int, ...] | None = None,
    ) -> Any:
        tensor = self.descriptor(name)
        if tensor.dtype != "BF16" or len(tensor.shape) != 2:
            raise ValueError(f"audit requires a rank-two BF16 tensor: {name!r}")
        if start < 0 or end < start or end > tensor.shape[0]:
            raise ValueError(f"invalid row range for {name!r}: [{start}, {end})")
        width = tensor.shape[1]
        if source_rows is None:
            rows = tuple(range(start, end))
        else:
            if len(source_rows) != end - start:
                raise ValueError(f"source row mapping length mismatch for {name!r}")
            rows = source_rows
        if not rows:
            return torch.empty((0, width), dtype=torch.bfloat16)
        pieces = []
        run_start = rows[0]
        previous = rows[0]
        for row in (*rows[1:], None):
            if row is not None and row == previous + 1:
                previous = row
                continue
            pieces.append(self._read_bf16_contiguous(tensor, run_start, previous + 1, width, torch))
            if row is not None:
                run_start = previous = row
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)

    def _read_bf16_contiguous(self, tensor: TensorDescriptor, start: int, end: int, width: int, torch: Any) -> Any:
        if start < 0 or end > tensor.shape[0]:
            raise ValueError(f"source row mapping is outside {tensor.name!r}")
        byte_count = (end - start) * width * 2
        offset = self.payload_start + tensor.data_offsets[0] + start * width * 2
        self.stream.seek(offset)
        payload = self.stream.read(byte_count)
        if len(payload) != byte_count:
            raise ValueError(f"audit input was truncated at {tensor.name!r}")
        return torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).reshape(end - start, width)


def _chunk_metric_sums(
    base: Any,
    a: Any,
    b: Any,
    multiplier: float,
    torch: Any,
    target_dtype: str = "BF16",
) -> MetricSums:
    delta = torch.matmul(b.float(), a.float()).mul_(multiplier)
    base_fp32 = base.float()
    exact_merged = base_fp32 + delta
    if target_dtype == "BF16":
        cast_dtype = torch.bfloat16
    elif target_dtype == "FP16":
        cast_dtype = torch.float16
    else:
        raise ValueError(f"unsupported audit target dtype: {target_dtype!r}")
    baked = exact_merged.to(cast_dtype).float()
    effective = baked - base_fp32
    error = effective - delta
    finite = (
        torch.isfinite(delta)
        & torch.isfinite(exact_merged)
        & torch.isfinite(baked)
        & torch.isfinite(effective)
        & torch.isfinite(error)
    )
    safe_delta = torch.where(finite, delta, 0.0)
    safe_merged = torch.where(finite, exact_merged, 0.0)
    safe_effective = torch.where(finite, effective, 0.0)
    safe_error = torch.where(finite, error, 0.0)
    source_nonzero = finite & (delta != 0)
    effective_nonzero = finite & (effective != 0)
    finite_nonzero_to_zero = source_nonzero & ~effective_nonzero
    sign_flip = source_nonzero & effective_nonzero & (torch.signbit(delta) != torch.signbit(effective))
    return MetricSums(
        element_count=delta.numel(),
        source_nonzero_count=int(source_nonzero.sum().item()),
        effective_nonzero_count=int(effective_nonzero.sum().item()),
        survived_nonzero_count=int((source_nonzero & effective_nonzero).sum().item()),
        annihilated_count=int(finite_nonzero_to_zero.sum().item()),
        finite_nonzero_to_zero_count=int(finite_nonzero_to_zero.sum().item()),
        sign_flip_count=int(sign_flip.sum().item()),
        nonfinite_count=int((~finite).sum().item()),
        delta_l2_squared=float(torch.sum(safe_delta.double().square()).item()),
        effective_l2_squared=float(torch.sum(safe_effective.double().square()).item()),
        residual_error_l2_squared=float(torch.sum(safe_error.double().square()).item()),
        merged_l2_squared=float(torch.sum(safe_merged.double().square()).item()),
        merged_error_l2_squared=float(
            torch.sum(torch.where(finite, baked - exact_merged, 0.0).double().square()).item()
        ),
        residual_dot=float(torch.sum(safe_delta.double() * safe_effective.double()).item()),
        max_abs_delta=float(torch.max(torch.abs(safe_delta)).item()) if delta.numel() else 0.0,
        max_abs_residual_error=float(torch.max(torch.abs(safe_error)).item()) if error.numel() else 0.0,
    )


def _row_chunk_size(operation: BakeOperation, maximum: int) -> int:
    # Conservative peak estimate covers BF16 base/B; FP32 base, delta, merge,
    # baked, effective, error, finite-safe copies and double reduction temporaries.
    # The resident FP32 A matrix is accounted once.
    width = operation.shape[1]
    per_row = 4 * operation.rank + 64 * width
    fixed = operation.rank * width * 8
    return max(1, (maximum - min(maximum // 4, fixed)) // max(1, per_row))


def _audit_operation(
    plan_operation_index: int,
    operation: BakeOperation,
    base_reader: _SafeTensorReader,
    lora_reader: _SafeTensorReader,
    thresholds: AuditThresholds,
    maximum: int,
    device: str,
    torch: Any,
    target_dtype: str,
) -> OperationAudit:
    a_desc = lora_reader.descriptor(operation.lora_a)
    b_desc = lora_reader.descriptor(operation.lora_b)
    if a_desc.shape != (operation.rank, operation.shape[1]):
        raise ValueError(f"LoRA A shape changed after planning for {operation.module!r}")
    if b_desc.shape != (operation.shape[0], operation.rank):
        raise ValueError(f"LoRA B shape changed after planning for {operation.module!r}")
    base_desc = base_reader.descriptor(operation.base_tensor)
    if base_desc.shape != operation.shape or base_desc.dtype != operation.dtype:
        raise ValueError(f"base tensor changed after planning for {operation.module!r}")
    a = lora_reader.read_bf16_rows(operation.lora_a, 0, operation.rank, torch).to(device).float()
    chunk_rows = _row_chunk_size(operation, maximum)
    partition_results: list[PartitionMetrics] = []
    with torch.no_grad():
        for partition in _partition_specs(operation):
            sums = MetricSums()
            for start in range(partition.runtime_row_start, partition.runtime_row_end, chunk_rows):
                end = min(partition.runtime_row_end, start + chunk_rows)
                source_rows = None
                if partition.base_source_rows is not None:
                    relative_start = start - partition.runtime_row_start
                    relative_end = end - partition.runtime_row_start
                    source_rows = partition.base_source_rows[relative_start:relative_end]
                base = base_reader.read_bf16_rows(operation.base_tensor, start, end, torch, source_rows=source_rows).to(
                    device
                )
                b = lora_reader.read_bf16_rows(operation.lora_b, start, end, torch).to(device)
                sums.add(
                    _chunk_metric_sums(
                        base,
                        a,
                        b,
                        operation.multiplier,
                        torch,
                        target_dtype,
                    )
                )
                del base, b
            partition_results.append(
                finalize_partition_metrics(partition.name, partition.kind, partition.row_count, sums, thresholds)
            )
    kind = partition_results[0].kind
    return OperationAudit(
        plan_operation_index=plan_operation_index,
        module=operation.module,
        kind=kind,
        base_tensor=operation.base_tensor,
        shard=operation.shard,
        row_chunk_size=chunk_rows,
        partition_count=len(partition_results),
        passed=all(partition.passed for partition in partition_results),
        partitions=tuple(partition_results),
    )


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _audit_contract(target_dtype: str) -> tuple[str, str, str, str]:
    profile, plan_schema = _target_contract(target_dtype)
    if target_dtype == "BF16":
        return profile, plan_schema, AUDIT_SCHEMA, AGGREGATE_AUDIT_SCHEMA
    if target_dtype == "FP16":
        return profile, plan_schema, FP16_AUDIT_SCHEMA, FP16_AGGREGATE_AUDIT_SCHEMA
    raise ValueError(f"unsupported audit target dtype: {target_dtype!r}")


def _rejection_decision(target_dtype: str) -> str:
    return "IDEAL_RESIDUAL_NOT_PRESERVED" if target_dtype == "BF16" else "FP16_DIAGNOSTIC_NOT_PRESERVED"


def _complete_pass_decision(target_dtype: str) -> str:
    return "DIAGNOSTIC_PASS" if target_dtype == "BF16" else "DIAGNOSTIC_PASS_UNDEPLOYABLE"


def _device_evidence(torch: Any, device: str) -> tuple[str, str | None, str]:
    if device == "cpu":
        return "cpu", None, "cpu"
    properties = torch.cuda.get_device_properties(device)
    actual = _normalize_gpu_uuid(getattr(properties, "uuid", None))
    declared = os.environ.get("H3_FORGE_AUDIT_GPU_UUID", "")
    if declared != actual:
        raise ValueError(f"H3_FORGE_AUDIT_GPU_UUID must exactly match the actual CUDA hardware UUID {actual!r}")
    return properties.name, f"{properties.major}.{properties.minor}", actual


def _normalize_gpu_uuid(value: Any) -> str:
    if value is None:
        raise ValueError("Torch CUDA device properties do not expose a hardware UUID")
    raw = str(value)
    raw = raw.removeprefix("GPU-")
    try:
        canonical = str(uuid.UUID(raw))
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"Torch CUDA device UUID is invalid: {value!r}") from exc
    if raw.lower() != canonical:
        raise ValueError(f"Torch CUDA device UUID is not canonical: {value!r}")
    return f"GPU-{canonical}"


def _is_gpu_uuid(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("GPU-"):
        return False
    try:
        return value == _normalize_gpu_uuid(value)
    except ValueError:
        return False


def operation_selection(
    operation_count: int, *, partition_index: int = 0, partition_count: int = 1
) -> OperationSelection:
    """Select a stable, balanced and mutually exclusive subset of plan operations."""

    if operation_count < 0:
        raise ValueError("operation_count must be non-negative")
    if partition_count <= 0:
        raise ValueError("partition_count must be positive")
    if partition_count > operation_count:
        raise ValueError("partition_count must not exceed operation_count")
    if not 0 <= partition_index < partition_count:
        raise ValueError("partition_index must be in [0, partition_count)")
    selected = tuple(index for index in range(operation_count) if index % partition_count == partition_index)
    if not selected:
        raise ValueError("operation partition must not be empty")
    identity = {
        "strategy": "operation-index-modulo/v1",
        "partition_index": partition_index,
        "partition_count": partition_count,
        "selected_indices": selected,
    }
    return OperationSelection(
        **identity,
        selection_sha256=_canonical_sha256(identity),
    )


def audit_bf16_bake_plan(
    plan: LoraBakePlan,
    *,
    thresholds: AuditThresholds = DEFAULT_AUDIT_THRESHOLDS,
    max_working_set_bytes: int = DEFAULT_MAX_WORKING_SET_BYTES,
    device: str = "cpu",
    partition_index: int = 0,
    partition_count: int = 1,
    plan_json_sha256: str | None = None,
) -> BakeFeasibilityReceipt:
    """Run a bounded, diagnostic-only FP32 residual-survival audit."""

    thresholds.validate()
    expected_profile, expected_plan_schema, audit_schema, _ = _audit_contract(plan.target_dtype)
    if plan.schema != expected_plan_schema:
        raise ValueError(f"unsupported audit plan schema: {plan.schema!r}")
    if plan.profile != expected_profile:
        raise ValueError(f"unsupported audit profile: {plan.profile!r}")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("LoRA bake diagnostic requires PyTorch; install h3-forge[audit]") from exc
    if plan_json_sha256 is not None and (
        len(plan_json_sha256) != 64 or any(character not in "0123456789abcdef" for character in plan_json_sha256)
    ):
        raise ValueError("plan_json_sha256 must be a lowercase SHA256 digest")
    if max_working_set_bytes < MIN_WORKING_SET_BYTES:
        raise ValueError(f"max_working_set_bytes must be at least {MIN_WORKING_SET_BYTES}")
    if plan.operation_count != len(plan.operations):
        raise ValueError("bake plan operation_count does not match its operations")
    recomputed_plan_sha256 = _plan_sha256(
        profile=plan.profile,
        config_sha256=plan.config_sha256,
        index_sha256=plan.index_sha256,
        lora_sha256=plan.lora_sha256,
        base_catalog_sha256=plan.base_catalog_sha256,
        base_shards=plan.base_shards,
        operations=plan.operations,
        target_dtype=plan.target_dtype,
    )
    if recomputed_plan_sha256 != plan.plan_sha256:
        raise ValueError("bake plan canonical SHA256 self-check failed")
    selection = operation_selection(
        plan.operation_count,
        partition_index=partition_index,
        partition_count=partition_count,
    )
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("audit device must be 'cpu' or a CUDA device such as 'cuda:0'")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA audit device requested but CUDA is unavailable")
    device_name, compute_capability, device_identity = _device_evidence(torch, device)

    started = time.monotonic()
    base_input = Path(plan.base_directory)
    lora_input = Path(plan.normalized_lora)
    if base_input.is_symlink() or not base_input.is_dir():
        raise ValueError(f"audit base directory must be a regular, non-linked directory: {base_input}")
    if lora_input.is_symlink() or not lora_input.is_file():
        raise ValueError(f"audit LoRA must be a regular, non-linked file: {lora_input}")
    base_root = base_input.resolve()
    lora_path = lora_input.resolve()
    readers: dict[str, _SafeTensorReader] = {}
    lora_reader: _SafeTensorReader | None = None
    normalized_lora_size = 0
    previous_matmul_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    matmul_mode = "cuda-fp32-highest" if device.startswith("cuda") else "cpu-fp32-highest"
    try:
        for shard in plan.base_shards:
            reader = _SafeTensorReader(base_root / shard.name)
            readers[shard.name] = reader
            if reader.snapshot.st_size != shard.size:
                raise ValueError(f"base shard size changed after planning: {shard.name!r}")
            if reader.sha256() != shard.sha256:
                raise ValueError(f"base shard SHA256 changed after planning: {shard.name!r}")
        lora_reader = _SafeTensorReader(lora_path)
        normalized_lora_size = lora_reader.snapshot.st_size
        if lora_reader.sha256() != plan.lora_sha256:
            raise ValueError("normalized LoRA SHA256 changed after planning")
        _validate_normalized_lora(lora_reader.metadata, tuple(lora_reader.tensors.values()))
        selected_indices = set(selection.selected_indices)
        operations = tuple(
            _audit_operation(
                operation_index,
                operation,
                readers[operation.shard],
                lora_reader,
                thresholds,
                max_working_set_bytes,
                device,
                torch,
                plan.target_dtype,
            )
            for operation_index, operation in enumerate(plan.operations)
            if operation_index in selected_indices
        )
    finally:
        close_errors: list[BaseException] = []
        if lora_reader is not None:
            try:
                lora_reader.close()
            except BaseException as exc:
                close_errors.append(exc)
        for reader in readers.values():
            try:
                reader.close()
            except BaseException as exc:
                close_errors.append(exc)
        torch.set_float32_matmul_precision(previous_matmul_precision)
        if close_errors and not sys.exc_info()[0]:
            raise close_errors[0]

    failed_operations = sum(not operation.passed for operation in operations)
    metric_partition_count = sum(operation.partition_count for operation in operations)
    failed_partitions = sum(not partition.passed for operation in operations for partition in operation.partitions)
    weight_status = "PASS" if not failed_operations else "FAIL"
    if weight_status != "PASS":
        decision = _rejection_decision(plan.target_dtype)
    elif selection.partition_count == 1:
        decision = _complete_pass_decision(plan.target_dtype)
    else:
        decision = "PARTITION_DIAGNOSTIC_PASS"
    payload: dict[str, Any] = {
        "schema": audit_schema,
        "profile": plan.profile,
        "plan_sha256": plan.plan_sha256,
        "plan_json_sha256": plan_json_sha256,
        "base_catalog_sha256": plan.base_catalog_sha256,
        "lora_sha256": plan.lora_sha256,
        "normalized_lora_size": normalized_lora_size,
        "validated_base_shard_count": len(readers),
        "source_content_binding": (
            "plan-json-full-hash;worker-full-sha256-same-fd;read-only-inputs-required"
            if plan_json_sha256 is not None
            else "in-process-full-hash-plan;worker-full-sha256-same-fd"
        ),
        "audit_role": "diagnostic-only",
        "product_gate": COMFY_BAKED_NATIVE_PRODUCT_GATE,
        "product_gate_status": "NOT_RUN",
        "oracle": f"ideal-fp32-residual-plus-one-final-{plan.target_dtype.lower()}-rne",
        "arithmetic": (
            f"delta=float32(B)@float32(A)*multiplier; merged=float32(base)+delta; cast={plan.target_dtype}-RNE"
        ),
        "target_dtype": plan.target_dtype,
        "thresholds": asdict(thresholds),
        "max_working_set_bytes": max_working_set_bytes,
        "device": device,
        "device_name": device_name,
        "compute_capability": compute_capability,
        "device_identity": device_identity,
        "matmul_mode": matmul_mode,
        "float32_matmul_precision": "highest",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "plan_operation_count": plan.operation_count,
        "selection": asdict(selection),
        "operation_count": len(operations),
        "metric_partition_count": metric_partition_count,
        "failed_operation_count": failed_operations,
        "failed_partition_count": failed_partitions,
        "weight_oracle_status": weight_status,
        "activation_oracle_status": "NOT_RUN",
        "decision": decision,
        "elapsed_seconds": time.monotonic() - started,
        "operations": [asdict(operation) for operation in operations],
    }
    return BakeFeasibilityReceipt(
        schema=audit_schema,
        profile=plan.profile,
        plan_sha256=plan.plan_sha256,
        plan_json_sha256=plan_json_sha256,
        base_catalog_sha256=plan.base_catalog_sha256,
        lora_sha256=plan.lora_sha256,
        normalized_lora_size=payload["normalized_lora_size"],
        validated_base_shard_count=payload["validated_base_shard_count"],
        source_content_binding=payload["source_content_binding"],
        audit_role=payload["audit_role"],
        product_gate=payload["product_gate"],
        product_gate_status=payload["product_gate_status"],
        oracle=payload["oracle"],
        arithmetic=payload["arithmetic"],
        target_dtype=plan.target_dtype,
        thresholds=thresholds,
        max_working_set_bytes=max_working_set_bytes,
        device=device,
        device_name=device_name,
        compute_capability=compute_capability,
        device_identity=device_identity,
        matmul_mode=matmul_mode,
        float32_matmul_precision="highest",
        torch_version=torch.__version__,
        python_version=payload["python_version"],
        plan_operation_count=plan.operation_count,
        selection=selection,
        operation_count=len(operations),
        metric_partition_count=metric_partition_count,
        failed_operation_count=failed_operations,
        failed_partition_count=failed_partitions,
        weight_oracle_status=weight_status,
        activation_oracle_status="NOT_RUN",
        decision=decision,
        elapsed_seconds=payload["elapsed_seconds"],
        operations=operations,
        receipt_sha256=_canonical_sha256(payload),
    )


def _validated_worker_receipt(path: Path) -> tuple[dict[str, Any], str]:
    payload, encoded = _read_json_object(path)
    schema = payload.get("schema")
    expected_fields = set(BakeFeasibilityReceipt.__dataclass_fields__)
    if schema == LEGACY_BF16_AUDIT_SCHEMA:
        expected_fields -= {"audit_role", "product_gate", "product_gate_status"}
    if set(payload) != expected_fields:
        raise ValueError(f"worker audit receipt has an invalid top-level schema: {path}")
    if schema not in {LEGACY_BF16_AUDIT_SCHEMA, AUDIT_SCHEMA, FP16_AUDIT_SCHEMA}:
        raise ValueError(f"unsupported worker audit schema in {path}: {schema!r}")
    receipt_sha256 = payload.get("receipt_sha256")
    if not isinstance(receipt_sha256, str) or len(receipt_sha256) != 64:
        raise ValueError(f"worker audit receipt SHA256 is invalid: {path}")
    canonical = dict(payload)
    del canonical["receipt_sha256"]
    if _canonical_sha256(canonical) != receipt_sha256:
        raise ValueError(f"worker audit receipt canonical SHA256 self-check failed: {path}")
    return payload, hashlib.sha256(encoded).hexdigest()


def aggregate_bake_audit_receipts(
    receipt_paths: Iterable[Path | str],
) -> AggregateBakeAuditReceipt:
    """Validate complete, disjoint worker coverage and aggregate the weight gate."""

    paths = tuple(Path(path) for path in receipt_paths)
    if len(paths) < 2:
        raise ValueError("aggregation requires at least two worker receipts")
    loaded = tuple((*_validated_worker_receipt(path), path) for path in paths)
    first = loaded[0][0]
    target_dtype = first.get("target_dtype")
    expected_profile, _, expected_audit_schema, aggregate_schema = _audit_contract(target_dtype)
    if first.get("schema") == LEGACY_BF16_AUDIT_SCHEMA:
        raise ValueError("legacy BF16 feasibility receipts are diagnostic-only and cannot be aggregated")
    if first.get("schema") != expected_audit_schema:
        raise ValueError("worker audit schema does not bind its target dtype")
    first_device = first.get("device")
    if not isinstance(first_device, str) or (first_device != "cpu" and not first_device.startswith("cuda")):
        raise ValueError("worker audit device is invalid")
    digest_fields = ("plan_sha256", "base_catalog_sha256", "lora_sha256")
    if first.get("profile") != expected_profile or any(
        not isinstance(first.get(field), str)
        or len(first[field]) != 64
        or any(character not in "0123456789abcdef" for character in first[field])
        for field in digest_fields
    ):
        raise ValueError("worker audit profile or content digest is invalid")
    plan_json_sha256 = first.get("plan_json_sha256")
    if plan_json_sha256 is not None and (
        not isinstance(plan_json_sha256, str)
        or len(plan_json_sha256) != 64
        or any(character not in "0123456789abcdef" for character in plan_json_sha256)
    ):
        raise ValueError("worker audit plan JSON SHA256 is invalid")
    if (
        first.get("validated_base_shard_count") != 13
        or "worker-full-sha256-same-fd" not in first.get("source_content_binding", "")
        or first.get("audit_role") != "diagnostic-only"
        or first.get("product_gate") != COMFY_BAKED_NATIVE_PRODUCT_GATE
        or first.get("product_gate_status") != "NOT_RUN"
        or first.get("float32_matmul_precision") != "highest"
        or first.get("activation_oracle_status") != "NOT_RUN"
    ):
        raise ValueError("worker audit source binding or oracle state is invalid")
    threshold_values = first.get("thresholds")
    if not isinstance(threshold_values, dict) or set(threshold_values) != set(AuditThresholds.__dataclass_fields__):
        raise ValueError("worker audit threshold schema is invalid")
    try:
        AuditThresholds(**threshold_values).validate()
    except (TypeError, ValueError) as exc:
        raise ValueError("worker audit thresholds are invalid") from exc
    selection = first.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("worker audit selection must be an object")
    partition_count = selection.get("partition_count")
    plan_operation_count = first.get("plan_operation_count")
    if (
        not isinstance(partition_count, int)
        or isinstance(partition_count, bool)
        or partition_count < 2
        or not isinstance(plan_operation_count, int)
        or isinstance(plan_operation_count, bool)
        or plan_operation_count != TURBO_V4_MODULE_COUNT
    ):
        raise ValueError("worker audit partition or plan operation count is invalid")
    if len(paths) != partition_count:
        raise ValueError("worker receipt count must equal partition_count")

    common_fields = (
        "schema",
        "profile",
        "plan_sha256",
        "plan_json_sha256",
        "base_catalog_sha256",
        "lora_sha256",
        "normalized_lora_size",
        "validated_base_shard_count",
        "source_content_binding",
        "audit_role",
        "product_gate",
        "product_gate_status",
        "oracle",
        "arithmetic",
        "target_dtype",
        "thresholds",
        "plan_operation_count",
        "float32_matmul_precision",
        "device",
        "device_name",
        "compute_capability",
        "matmul_mode",
        "torch_version",
        "python_version",
        "max_working_set_bytes",
    )
    workers: list[AggregateWorker] = []
    seen_partitions: set[int] = set()
    covered_indices: set[int] = set()
    failed_workers = 0
    for payload, file_sha256, path in loaded:
        if any(payload.get(field) != first.get(field) for field in common_fields):
            raise ValueError(f"worker audit receipts do not share one contract: {path}")
        if payload.get("activation_oracle_status") != "NOT_RUN":
            raise ValueError(f"worker audit activation oracle state is invalid: {path}")
        identity = payload.get("device_identity")
        if payload["device"].startswith("cuda"):
            if not _is_gpu_uuid(identity):
                raise ValueError(f"CUDA worker audit device identity is invalid: {path}")
            if not isinstance(payload.get("device_name"), str) or not payload["device_name"]:
                raise ValueError(f"CUDA worker audit device name is invalid: {path}")
            capability = payload.get("compute_capability")
            if not isinstance(capability, str) or "." not in capability:
                raise ValueError(f"CUDA worker audit compute capability is invalid: {path}")
        elif (payload.get("device_name"), payload.get("compute_capability"), identity) != (
            "cpu",
            None,
            "cpu",
        ):
            raise ValueError(f"CPU worker audit device evidence is invalid: {path}")
        current = payload.get("selection")
        if not isinstance(current, dict) or set(current) != set(OperationSelection.__dataclass_fields__):
            raise ValueError(f"worker audit selection schema is invalid: {path}")
        index = current.get("partition_index")
        if (
            current.get("strategy") != "operation-index-modulo/v1"
            or current.get("partition_count") != partition_count
            or not isinstance(index, int)
            or isinstance(index, bool)
            or index in seen_partitions
        ):
            raise ValueError(f"worker audit partition identity is invalid or duplicated: {path}")
        expected = operation_selection(
            plan_operation_count,
            partition_index=index,
            partition_count=partition_count,
        )
        selected_value = current.get("selected_indices")
        normalized_selection = {
            **current,
            "selected_indices": tuple(selected_value) if isinstance(selected_value, list) else selected_value,
        }
        if normalized_selection != asdict(expected):
            raise ValueError(f"worker audit selected_indices or selection SHA256 is invalid: {path}")
        selected_indices = set(expected.selected_indices)
        if covered_indices.intersection(selected_indices):
            raise ValueError("worker audit selected_indices overlap")
        seen_partitions.add(index)
        covered_indices.update(selected_indices)

        operations = payload.get("operations")
        if not isinstance(operations, list) or payload.get("operation_count") != len(operations):
            raise ValueError(f"worker audit operation list is invalid: {path}")
        if any(
            not isinstance(operation, dict)
            or set(operation) != set(OperationAudit.__dataclass_fields__)
            or not isinstance(operation.get("partitions"), list)
            or any(
                not isinstance(partition, dict) or set(partition) != set(PartitionMetrics.__dataclass_fields__)
                for partition in operation.get("partitions", [])
            )
            for operation in operations
        ):
            raise ValueError(f"worker audit operation or partition schema is invalid: {path}")
        operation_indices = [operation.get("plan_operation_index") for operation in operations]
        if operation_indices != list(expected.selected_indices):
            raise ValueError(f"worker audit operations do not match selected_indices: {path}")
        if any(
            operation.get("partition_count") != len(operation["partitions"])
            or operation.get("passed") != all(partition.get("passed") is True for partition in operation["partitions"])
            for operation in operations
        ):
            raise ValueError(f"worker audit operation pass state is inconsistent: {path}")
        failed_operations = sum(operation.get("passed") is not True for operation in operations)
        failed_partitions = sum(
            partition.get("passed") is not True
            for operation in operations
            for partition in operation.get("partitions", [])
        )
        computed_pass = failed_operations == 0 and failed_partitions == 0
        expected_status = "PASS" if computed_pass else "FAIL"
        expected_decision = "PARTITION_DIAGNOSTIC_PASS" if computed_pass else _rejection_decision(target_dtype)
        if (
            payload.get("failed_operation_count") != failed_operations
            or payload.get("failed_partition_count") != failed_partitions
            or payload.get("weight_oracle_status") != expected_status
            or payload.get("decision") != expected_decision
        ):
            raise ValueError(f"worker audit decision or failure counts are inconsistent: {path}")
        if not computed_pass:
            failed_workers += 1
        workers.append(
            AggregateWorker(
                path=str(path.resolve()),
                receipt_json_sha256=file_sha256,
                partition_index=index,
                selection_sha256=expected.selection_sha256,
                device_identity=identity,
                operation_count=len(operations),
                weight_oracle_status=payload["weight_oracle_status"],
                decision=payload["decision"],
            )
        )

    if seen_partitions != set(range(partition_count)):
        raise ValueError("worker audit partition_index set is incomplete")
    if covered_indices != set(range(plan_operation_count)):
        raise ValueError("worker audit selected_indices coverage is incomplete")
    workers.sort(key=lambda worker: worker.partition_index)
    device_identities = tuple(worker.device_identity for worker in workers)
    if first_device.startswith("cuda") and len(set(device_identities)) != partition_count:
        raise ValueError("CUDA worker audit device identities must be non-empty and distinct")
    weight_status = "PASS" if not failed_workers else "FAIL"
    decision = _complete_pass_decision(target_dtype) if weight_status == "PASS" else _rejection_decision(target_dtype)
    aggregate_payload: dict[str, Any] = {
        "schema": aggregate_schema,
        "profile": first["profile"],
        "plan_sha256": first["plan_sha256"],
        "plan_json_sha256": first["plan_json_sha256"],
        "base_catalog_sha256": first["base_catalog_sha256"],
        "lora_sha256": first["lora_sha256"],
        "target_dtype": target_dtype,
        "audit_role": first["audit_role"],
        "product_gate": first["product_gate"],
        "product_gate_status": first["product_gate_status"],
        "thresholds": first["thresholds"],
        "partition_count": partition_count,
        "plan_operation_count": plan_operation_count,
        "covered_operation_count": len(covered_indices),
        "device_identities": device_identities,
        "failed_worker_count": failed_workers,
        "weight_oracle_status": weight_status,
        "activation_oracle_status": "NOT_RUN",
        "decision": decision,
        "workers": [asdict(worker) for worker in workers],
    }
    return AggregateBakeAuditReceipt(
        schema=aggregate_schema,
        profile=first["profile"],
        plan_sha256=first["plan_sha256"],
        plan_json_sha256=first["plan_json_sha256"],
        base_catalog_sha256=first["base_catalog_sha256"],
        lora_sha256=first["lora_sha256"],
        target_dtype=target_dtype,
        audit_role=first["audit_role"],
        product_gate=first["product_gate"],
        product_gate_status=first["product_gate_status"],
        thresholds=first["thresholds"],
        partition_count=partition_count,
        plan_operation_count=plan_operation_count,
        covered_operation_count=len(covered_indices),
        device_identities=device_identities,
        failed_worker_count=failed_workers,
        weight_oracle_status=weight_status,
        activation_oracle_status="NOT_RUN",
        decision=decision,
        workers=tuple(workers),
        receipt_sha256=_canonical_sha256(aggregate_payload),
    )


def audit_official_fl2va_bf16_bake(
    base_directory: Path | str,
    normalized_lora: Path | str,
    *,
    profile: str | None = None,
    scale: float = 1.0,
    thresholds: AuditThresholds = DEFAULT_AUDIT_THRESHOLDS,
    max_working_set_bytes: int = DEFAULT_MAX_WORKING_SET_BYTES,
    device: str = "cpu",
    partition_index: int = 0,
    partition_count: int = 1,
    target_dtype: str = "BF16",
) -> BakeFeasibilityReceipt:
    """Plan and audit official FL2VA weights without creating output weights."""

    plan = plan_official_fl2va_bf16_bake(
        base_directory,
        normalized_lora,
        profile=profile,
        scale=scale,
        target_dtype=target_dtype,
    )
    return audit_bf16_bake_plan(
        plan,
        thresholds=thresholds,
        max_working_set_bytes=max_working_set_bytes,
        device=device,
        partition_index=partition_index,
        partition_count=partition_count,
    )
