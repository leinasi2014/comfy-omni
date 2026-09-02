"""Bounded standard-library oracle for regular-Hadamard ConvRot math.

This is deliberately small-scale evidence code, not the model conversion
backend. It gives CPU-only CI an independent oracle for the lazy Torch path.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from comfy_omni.conversion.numerics.errors import ConvRotNumericsError

MIN_GROUP_SIZE = 4
MAX_GROUP_SIZE = 256
MAX_REFERENCE_ELEMENTS = 1_048_576
MAX_ROWS = 4096

_H4 = (
    (1, 1, 1, -1),
    (1, 1, -1, 1),
    (1, -1, 1, 1),
    (-1, 1, 1, 1),
)


def validate_group_size(size: int) -> int:
    """Return the base-four exponent of a supported regular Hadamard size."""

    if type(size) is not int or size < MIN_GROUP_SIZE or size > MAX_GROUP_SIZE:
        raise ConvRotNumericsError(
            f"regular Hadamard size must be a power of four in {MIN_GROUP_SIZE}..{MAX_GROUP_SIZE}"
        )
    exponent = 0
    current = 1
    while current < size:
        current *= 4
        exponent += 1
    if current != size:
        raise ConvRotNumericsError(
            f"regular Hadamard size must be a power of four in {MIN_GROUP_SIZE}..{MAX_GROUP_SIZE}"
        )
    return exponent


def validate_max_rows(max_rows: int) -> None:
    if type(max_rows) is not int or not 1 <= max_rows <= MAX_ROWS:
        raise ConvRotNumericsError(f"max_rows must be in 1..{MAX_ROWS}")


def row_blocks(row_count: int, max_rows: int) -> tuple[tuple[int, int], ...]:
    """Return a complete, non-overlapping bounded row partition."""

    validate_max_rows(max_rows)
    if type(row_count) is not int or row_count <= 0:
        raise ConvRotNumericsError("row_count must be a positive integer")
    return tuple((start, min(start + max_rows, row_count)) for start in range(0, row_count, max_rows))


def regular_hadamard_reference(size: int) -> tuple[tuple[float, ...], ...]:
    """Build comfy-kitchen's normalized regular Hadamard matrix."""

    exponent = validate_group_size(size)
    signs: tuple[tuple[int, ...], ...] = ((1,),)
    for _ in range(exponent):
        signs = tuple(
            tuple(left * right for left in outer_row for right in inner_row) for outer_row in signs for inner_row in _H4
        )
    normalization = math.sqrt(size)
    return tuple(tuple(value / normalization for value in row) for row in signs)


def _validated_rows(rows: Sequence[Sequence[int | float]], group_size: int) -> tuple[tuple[float, ...], ...]:
    validate_group_size(group_size)
    if not rows:
        raise ConvRotNumericsError("reference rows must not be empty")
    width = len(rows[0])
    if width <= 0 or width % group_size:
        raise ConvRotNumericsError("reference row width must be a positive multiple of group_size")
    if len(rows) * width > MAX_REFERENCE_ELEMENTS:
        raise ConvRotNumericsError("reference input exceeds its bounded element limit")
    result: list[tuple[float, ...]] = []
    for row in rows:
        if len(row) != width:
            raise ConvRotNumericsError("reference rows must have one consistent width")
        values = tuple(float(value) for value in row)
        if not all(math.isfinite(value) for value in values):
            raise ConvRotNumericsError("reference rows must contain finite values")
        result.append(values)
    return tuple(result)


def apply_regular_hadamard_reference(
    rows: Sequence[Sequence[int | float]], *, group_size: int
) -> tuple[tuple[float, ...], ...]:
    """Apply the normalized regular Hadamard independently to each group."""

    values = _validated_rows(rows, group_size)
    matrix = regular_hadamard_reference(group_size)
    transformed: list[tuple[float, ...]] = []
    for row in values:
        output: list[float] = []
        for start in range(0, len(row), group_size):
            group = row[start : start + group_size]
            output.extend(math.fsum(group[index] * target[index] for index in range(group_size)) for target in matrix)
        transformed.append(tuple(output))
    return tuple(transformed)


def inverse_convrot_reference(
    qweight: Sequence[Sequence[int]],
    rowwise_scale: Sequence[float],
    *,
    group_size: int,
) -> tuple[tuple[float, ...], ...]:
    """Dequantize and inverse-rotate a small integer fixture."""

    if len(qweight) != len(rowwise_scale):
        raise ConvRotNumericsError("reference scales must contain exactly one value per row")
    integer_rows: list[tuple[int, ...]] = []
    for row in qweight:
        if not all(type(value) is int and -128 <= value <= 127 for value in row):
            raise ConvRotNumericsError("reference qweight values must be signed INT8 integers")
        integer_rows.append(tuple(row))
    scales = tuple(float(value) for value in rowwise_scale)
    if not all(math.isfinite(value) and value > 0 for value in scales):
        raise ConvRotNumericsError("reference scales must be finite and positive")
    dequantized = tuple(tuple(value * scales[row_index] for value in row) for row_index, row in enumerate(integer_rows))
    return apply_regular_hadamard_reference(dequantized, group_size=group_size)


__all__ = [
    "MAX_GROUP_SIZE",
    "MAX_REFERENCE_ELEMENTS",
    "MAX_ROWS",
    "apply_regular_hadamard_reference",
    "inverse_convrot_reference",
    "regular_hadamard_reference",
    "row_blocks",
    "validate_group_size",
    "validate_max_rows",
]
