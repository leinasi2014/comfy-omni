"""The sole authorized normalization for the pinned ModelScope H3 text encoder.

The profile is intentionally exact and non-extensible: it authorizes one source digest, one
72-byte suffix, and one strict derived digest. It does not make trailing safetensors bytes valid in
general and never mutates the source.
"""

from __future__ import annotations

from pathlib import Path

from comfy_omni.artifacts.normalization import NormalizationPublication, normalize_digest_pinned_prefix
from comfy_omni.domain.normalization import ArtifactIdentity, NormalizationProfile, ToolIdentity

MODELSCOPE_QWEN3VL_H3_TEXT_ENCODER_PROFILE = NormalizationProfile(
    profile_id="modelscope-qwen3vl-h3-text-encoder-strict",
    profile_version=1,
    source=ArtifactIdentity(
        bytes=15_683_129_659,
        sha256="47babbb3e4b7e43c097351ca39cfb7f326d014ae53a584f8559dc8121abca94c",
    ),
    removed_suffix=ArtifactIdentity(
        bytes=72,
        sha256="8bbc743f1fdc67acb6b09c977485e7d8bed7ff073a12d70865e0e4b793ed8e75",
    ),
    derived=ArtifactIdentity(
        bytes=15_683_129_587,
        sha256="a166c7bbbe66a22065159e478335fee4a633c4a3e3bb34c8e8ac4cc91bf4996f",
    ),
)


def normalize_modelscope_qwen3vl_h3_text_encoder(
    source: Path,
    destination: Path,
    tool: ToolIdentity,
) -> NormalizationPublication:
    """Apply the exact pinned text-encoder profile and publish its receipt."""

    return normalize_digest_pinned_prefix(
        source,
        destination,
        MODELSCOPE_QWEN3VL_H3_TEXT_ENCODER_PROFILE,
        tool,
    )


__all__ = [
    "MODELSCOPE_QWEN3VL_H3_TEXT_ENCODER_PROFILE",
    "normalize_modelscope_qwen3vl_h3_text_encoder",
]
