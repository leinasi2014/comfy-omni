"""Bounded byte producers for immutable native-export tensor actions.

The scheduling and row-layout logic is framework-free. A numerical block backend is injected for
inverse ConvRot, keeping Torch out of artifact I/O, the plan model, and this producer boundary.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Iterator, Sequence

from comfy_omni.artifacts import fileops
from comfy_omni.artifacts.sources import LocatedTensor, SafeTensorSources
from comfy_omni.contracts.models import ContractError
from comfy_omni.conversion.exporters.models import QkvLayoutPlan, TensorAction
from comfy_omni.domain.qkv import qkv_to_grouped_row_indices

ConvRotBlockBackend = Callable[..., bytes]


def _fail(detail: str) -> None:
    raise ContractError(detail, evidence={"stage": "payload-producer"})


def validate_qkv_layout(layout: QkvLayoutPlan) -> tuple[int, ...]:
    """Recompute and bind the complete runtime-QKV to grouped permutation."""

    if layout.source_layout != "runtime-qkv" or layout.target_layout != "grouped-for-official-loader":
        _fail("QKV plan names an unsupported source or target layout")
    try:
        indices = qkv_to_grouped_row_indices(
            num_query_groups=layout.num_query_groups,
            heads_per_group=layout.heads_per_group,
            head_dim=layout.head_dim,
        )
    except ValueError as exc:
        raise ContractError(str(exc), evidence={"stage": "payload-producer"}) from exc
    digest = hashlib.sha256(fileops.canonical_json(list(indices))).hexdigest()
    if len(indices) != layout.row_count or digest != layout.permutation_sha256:
        _fail("QKV permutation digest or row count disagrees with the immutable plan")
    if tuple(sorted(indices)) != tuple(range(layout.row_count)):
        _fail("QKV row indices are not a complete permutation")
    return indices


def _row_width(action: TensorAction) -> int:
    if len(action.shape) != 2 or action.shape[0] <= 0 or action.shape[1] <= 0:
        _fail(f"payload action must describe a non-empty rank-2 tensor: {action.source_name!r}")
    rows = action.shape[0]
    if action.source_bytes % rows:
        _fail(f"source tensor does not contain complete rows: {action.source_name!r}")
    return action.source_bytes // rows


def _bounded_runs(indices: Sequence[int], max_rows: int) -> Iterator[tuple[int, int]]:
    if type(max_rows) is not int or max_rows <= 0:
        _fail("payload max_rows must be a positive integer")
    if not indices:
        _fail("payload row order must not be empty")
    start = previous = indices[0]
    count = 1
    for current in indices[1:]:
        if current == previous + 1 and count < max_rows:
            count += 1
        else:
            yield start, count
            start, count = current, 1
        previous = current
    yield start, count


def qkv_raw_chunks(
    sources: SafeTensorSources,
    tensor: LocatedTensor,
    action: TensorAction,
    layout: QkvLayoutPlan,
    *,
    max_rows: int,
) -> Iterable[bytes]:
    """Yield unchanged complete rows in the exact grouped loader order."""

    indices = validate_qkv_layout(layout)
    if action.shape[0] != len(indices):
        _fail(f"QKV action row count disagrees with its permutation: {action.source_name!r}")
    row_bytes = _row_width(action)
    for start, count in _bounded_runs(indices, max_rows):
        yield from sources.iter_raw_range(tensor, start * row_bytes, count * row_bytes)


def convrot_bf16_chunks(
    sources: SafeTensorSources,
    weight: LocatedTensor,
    scale: LocatedTensor,
    action: TensorAction,
    layout: QkvLayoutPlan,
    backend: ConvRotBlockBackend,
    *,
    max_rows: int,
    reorder_qkv: bool,
) -> Iterable[bytes]:
    """Decode held I8/F32 ranges in bounded row runs and emit BF16 bytes."""

    rows, columns = action.shape
    if action.group_size is None:
        _fail(f"ConvRot action has no group-size binding: {action.source_name!r}")
    row_bytes = _row_width(action)
    if row_bytes != columns:
        _fail(f"ConvRot I8 source row width is invalid: {action.source_name!r}")
    order = validate_qkv_layout(layout) if reorder_qkv else tuple(range(rows))
    if len(order) != rows:
        _fail(f"ConvRot QKV rows disagree with the immutable layout: {action.source_name!r}")
    for start, count in _bounded_runs(order, max_rows):
        qweight = b"".join(sources.iter_raw_range(weight, start * columns, count * columns))
        rowwise_scale = b"".join(sources.iter_raw_range(scale, start * 4, count * 4))
        payload = backend(
            qweight,
            rowwise_scale,
            rows=count,
            columns=columns,
            group_size=action.group_size,
        )
        expected = count * columns * 2
        if not isinstance(payload, bytes) or len(payload) != expected:
            observed = len(payload) if isinstance(payload, bytes) else type(payload).__name__
            _fail(f"ConvRot backend produced {observed} bytes; expected {expected}")
        yield payload


__all__ = [
    "ConvRotBlockBackend",
    "convrot_bf16_chunks",
    "qkv_raw_chunks",
    "validate_qkv_layout",
]
