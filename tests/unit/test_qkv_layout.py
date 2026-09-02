from __future__ import annotations

import pytest

from comfy_omni.domain.qkv import grouped_to_qkv_row_indices, qkv_to_grouped_row_indices


def test_grouped_and_runtime_qkv_permutations_are_exact_inverses() -> None:
    forward = grouped_to_qkv_row_indices(num_query_groups=2, heads_per_group=1, head_dim=2)
    inverse = qkv_to_grouped_row_indices(num_query_groups=2, heads_per_group=1, head_dim=2)

    assert forward == (0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11)
    assert tuple(forward[index] for index in inverse) == tuple(range(12))
    assert tuple(inverse[index] for index in forward) == tuple(range(12))


@pytest.mark.parametrize(
    "dimensions",
    ((0, 1, 1), (1, 0, 1), (1, 1, 0), (-1, 1, 1)),
)
def test_qkv_layout_rejects_non_positive_dimensions(dimensions: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError, match="positive"):
        grouped_to_qkv_row_indices(
            num_query_groups=dimensions[0],
            heads_per_group=dimensions[1],
            head_dim=dimensions[2],
        )
