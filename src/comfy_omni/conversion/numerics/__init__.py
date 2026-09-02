"""Bounded numerical backends and their independent reference oracle."""

from comfy_omni.conversion.numerics.errors import ConvRotNumericsError
from comfy_omni.conversion.numerics.reference import (
    apply_regular_hadamard_reference,
    inverse_convrot_reference,
    regular_hadamard_reference,
    row_blocks,
)
from comfy_omni.conversion.numerics.torch_backend import (
    fast_inverse_convrot_rows,
    inverse_convrot_rows,
    inverse_convrot_to_bf16,
    regular_hadamard,
)

__all__ = [
    "ConvRotNumericsError",
    "apply_regular_hadamard_reference",
    "fast_inverse_convrot_rows",
    "inverse_convrot_reference",
    "inverse_convrot_rows",
    "inverse_convrot_to_bf16",
    "regular_hadamard",
    "regular_hadamard_reference",
    "row_blocks",
]
