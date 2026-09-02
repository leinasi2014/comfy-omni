"""Application orchestration for approved offline normalization.

This layer resolves installed-tool provenance and selects the approved conversion profile. It does
not implement hashing, filesystem publication, or CLI rendering.
"""

from __future__ import annotations

from pathlib import Path

from comfy_omni.artifacts.build_identity import installed_tool_identity
from comfy_omni.artifacts.normalization import NormalizationPublication
from comfy_omni.conversion.normalization import normalize_modelscope_qwen3vl_h3_text_encoder


def normalize_pinned_text_encoder(source: Path, destination: Path) -> NormalizationPublication:
    """Normalize the one pinned ModelScope text encoder with installed-wheel provenance."""

    return normalize_modelscope_qwen3vl_h3_text_encoder(
        source,
        destination,
        installed_tool_identity(),
    )


__all__ = ["normalize_pinned_text_encoder"]
