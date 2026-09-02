"""Pure row-layout rules for MiniMax H3 grouped QKV checkpoint weights.

Derived from Apache-2.0 ``h3_forge.qkv`` at commit
e9cb011d00b028c149db3978de246c54f6e34acc (blob
43f3d9e1d243b3d5aebaa6281b2c9b383970abfd).
"""

from __future__ import annotations


def grouped_to_qkv_row_indices(
    *, num_query_groups: int, heads_per_group: int, head_dim: int
) -> tuple[int, ...]:
    """Return grouped-source rows in runtime ``Q | K | V`` order."""

    if num_query_groups <= 0 or heads_per_group <= 0 or head_dim <= 0:
        raise ValueError("QKV layout dimensions must be positive")
    per_group = (heads_per_group + 2) * head_dim
    query: list[int] = []
    key: list[int] = []
    value: list[int] = []
    for group in range(num_query_groups):
        start = group * per_group
        query.extend(range(start, start + heads_per_group * head_dim))
        key.extend(range(start + heads_per_group * head_dim, start + (heads_per_group + 1) * head_dim))
        value.extend(range(start + (heads_per_group + 1) * head_dim, start + per_group))
    return tuple(query + key + value)


def qkv_to_grouped_row_indices(
    *, num_query_groups: int, heads_per_group: int, head_dim: int
) -> tuple[int, ...]:
    """Return runtime-QKV rows in grouped checkpoint order."""

    forward = grouped_to_qkv_row_indices(
        num_query_groups=num_query_groups,
        heads_per_group=heads_per_group,
        head_dim=head_dim,
    )
    inverse = [0] * len(forward)
    for runtime_row, grouped_row in enumerate(forward):
        inverse[grouped_row] = runtime_row
    return tuple(inverse)


__all__ = ["grouped_to_qkv_row_indices", "qkv_to_grouped_row_indices"]
