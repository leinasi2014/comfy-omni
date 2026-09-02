"""Contract tests for the sole approved text-encoder normalization profile."""

from __future__ import annotations

from comfy_omni.conversion.normalization import MODELSCOPE_QWEN3VL_H3_TEXT_ENCODER_PROFILE
from comfy_omni.domain.normalization import ArtifactIdentity


def test_modelscope_text_encoder_profile_is_fully_content_bound() -> None:
    profile = MODELSCOPE_QWEN3VL_H3_TEXT_ENCODER_PROFILE

    assert profile.profile_id == "modelscope-qwen3vl-h3-text-encoder-strict"
    assert profile.profile_version == 1
    assert profile.source == ArtifactIdentity(
        15_683_129_659,
        "47babbb3e4b7e43c097351ca39cfb7f326d014ae53a584f8559dc8121abca94c",
    )
    assert profile.removed_suffix == ArtifactIdentity(
        72,
        "8bbc743f1fdc67acb6b09c977485e7d8bed7ff073a12d70865e0e4b793ed8e75",
    )
    assert profile.derived == ArtifactIdentity(
        15_683_129_587,
        "a166c7bbbe66a22065159e478335fee4a633c4a3e3bb34c8e8ac4cc91bf4996f",
    )
