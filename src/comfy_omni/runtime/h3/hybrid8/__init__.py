"""Hybrid8 dense DiT runtime contracts and geometry (pure).

Provenance: h3-forge@e9cb011, Apache-2.0; see docs/migration/dense-hybrid8-runtime-port-e9cb011.md.
"""

from comfy_omni.runtime.h3.hybrid8.contracts import (
    HYBRID8_BLOCK_SUFFIXES,
    HYBRID8_FINAL_LAYER_SUFFIXES,
    HYBRID8_QKV_GROUPED_LAYOUT,
    HYBRID8_SIGNATURE_NAMES,
    HYBRID8_TEMPLATE_NAME,
    HYBRID8_TOKEN_REFINER_SUFFIXES,
    Hybrid8DitForm,
    Hybrid8StructureError,
    block_indices,
    has_hybrid8_signature,
    manifest_schema_sha256,
    pinned_hybrid8_inventory,
    validate_hybrid8_census,
)
from comfy_omni.runtime.h3.hybrid8.geometry import Hybrid8Geometry, derive_hybrid8_geometry

__all__ = [
    "HYBRID8_TEMPLATE_NAME",
    "HYBRID8_SIGNATURE_NAMES",
    "HYBRID8_BLOCK_SUFFIXES",
    "HYBRID8_TOKEN_REFINER_SUFFIXES",
    "HYBRID8_FINAL_LAYER_SUFFIXES",
    "HYBRID8_QKV_GROUPED_LAYOUT",
    "Hybrid8StructureError",
    "Hybrid8DitForm",
    "pinned_hybrid8_inventory",
    "manifest_schema_sha256",
    "has_hybrid8_signature",
    "validate_hybrid8_census",
    "block_indices",
    "Hybrid8Geometry",
    "derive_hybrid8_geometry",
]
