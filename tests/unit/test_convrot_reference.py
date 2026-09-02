from __future__ import annotations

import math

import pytest

from comfy_omni.conversion.numerics import (
    ConvRotNumericsError,
    apply_regular_hadamard_reference,
    inverse_convrot_reference,
    regular_hadamard_reference,
    row_blocks,
)
from comfy_omni.conversion.numerics.reference import validate_group_size


def test_regular_hadamard_four_matches_the_audited_base_matrix() -> None:
    assert regular_hadamard_reference(4) == (
        (0.5, 0.5, 0.5, -0.5),
        (0.5, 0.5, -0.5, 0.5),
        (0.5, -0.5, 0.5, 0.5),
        (-0.5, 0.5, 0.5, 0.5),
    )


def test_regular_hadamard_sixteen_is_symmetric_and_orthonormal() -> None:
    matrix = regular_hadamard_reference(16)

    assert all(matrix[row][column] == matrix[column][row] for row in range(16) for column in range(16))
    for left in range(16):
        for right in range(16):
            dot = math.fsum(matrix[left][index] * matrix[right][index] for index in range(16))
            assert dot == pytest.approx(1.0 if left == right else 0.0, abs=1e-12)


def test_regular_hadamard_is_its_own_inverse_for_multiple_groups() -> None:
    source = ((1.25, -2.5, 3.0, 4.5, -1.0, 0.5, 2.0, -3.0),)

    rotated = apply_regular_hadamard_reference(source, group_size=4)
    restored = apply_regular_hadamard_reference(rotated, group_size=4)

    assert restored[0] == pytest.approx(source[0], abs=1e-12)


def test_inverse_convrot_reference_dequantizes_then_inverse_rotates() -> None:
    result = inverse_convrot_reference(((1, -2, 4, 3),), (0.5,), group_size=4)

    assert result[0] == pytest.approx((0.0, -0.5, 2.5, 1.0), abs=1e-12)


@pytest.mark.parametrize("size", (4, 16, 64, 256))
def test_registered_h3_power_of_four_sizes_are_bounded(size: int) -> None:
    assert 4 ** validate_group_size(size) == size


@pytest.mark.parametrize("size", (True, 0, 8, 1024))
def test_non_contract_group_sizes_fail_closed(size: int) -> None:
    with pytest.raises(ConvRotNumericsError, match="power of four"):
        validate_group_size(size)


def test_row_blocks_are_complete_non_overlapping_and_bounded() -> None:
    assert row_blocks(9, 4) == ((0, 4), (4, 8), (8, 9))


@pytest.mark.parametrize("max_rows", (0, -1, 4097, True))
def test_row_blocks_reject_invalid_resource_limits(max_rows: int) -> None:
    with pytest.raises(ConvRotNumericsError, match="max_rows"):
        row_blocks(8, max_rows)


def test_reference_rejects_invalid_int8_or_scale_values() -> None:
    with pytest.raises(ConvRotNumericsError, match="INT8"):
        inverse_convrot_reference(((128, 0, 0, 0),), (1.0,), group_size=4)
    with pytest.raises(ConvRotNumericsError, match="finite and positive"):
        inverse_convrot_reference(((1, 0, 0, 0),), (float("nan"),), group_size=4)
