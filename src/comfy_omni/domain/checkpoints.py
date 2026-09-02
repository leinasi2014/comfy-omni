"""Pure checkpoint inspection values and classification rules.

This module owns immutable inspection results and evidence-based classification. It performs no
filesystem I/O and must not import optional runtimes, Torch, API, CLI, or integration modules.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class TensorDescriptor:
    """One tensor's logical safetensors descriptor."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]


@dataclass(frozen=True)
class ArtifactInspection:
    """Serializable summary of one inspected checkpoint artifact."""

    path: str
    component: str
    quantization: tuple[str, ...]
    tensor_count: int
    metadata: Mapping[str, str]
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        """Detach and freeze caller-owned metadata."""

        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        """Return the legacy-compatible JSON-ready representation."""

        return {
            "path": self.path,
            "component": self.component,
            "quantization": self.quantization,
            "tensor_count": self.tensor_count,
            "metadata": dict(self.metadata),
            "evidence": self.evidence,
        }


def _contains_name_token(name: str, token: str) -> bool:
    return re.search(rf"(?:^|[._]){re.escape(token)}(?:[._]|$)", name) is not None


def classify_component(names: Sequence[str], metadata: Mapping[str, str]) -> tuple[str, list[str]]:
    """Classify a tensor-name set from explicit H3 evidence."""

    lowered = tuple(name.lower() for name in names)
    vae_metadata = {key for key in metadata if key in {"minimax_h3_audio_vae", "minimax_h3_video_vae"}}
    if len(vae_metadata) > 1:
        raise ValueError("contradictory H3 VAE metadata namespaces")

    identity_keys = {
        "architecture",
        "base_model",
        "base_model_name_or_path",
        "model",
        "model_type",
    }
    identity_text = "\n".join(value for key, value in metadata.items() if key.lower() in identity_keys).lower()
    adapter = any(
        _contains_name_token(name, marker)
        for name in lowered
        for marker in ("lora_a", "lora_b", "lora_up", "lora_down", "dora_scale")
    )
    metadata_h3_signature = re.search(r"(?:^|[^a-z0-9])minimax[-_ ]?h3(?:[^a-z0-9]|$)", identity_text) is not None
    transformer_signature_markers = {
        marker
        for marker in ("audio_patch_proj", "video_patch_proj", "token_refiner", "curve_model")
        if any(_contains_name_token(name, marker) for name in lowered)
    }
    transformer_marker_present = bool(transformer_signature_markers)
    coherent_transformer_signature = len(transformer_signature_markers) >= 2
    h3_signature = metadata_h3_signature or coherent_transformer_signature or bool(vae_metadata)

    candidates: dict[str, str] = {}
    if "minimax_h3_audio_vae" in vae_metadata:
        candidates["audio_vae"] = "MiniMax H3 audio VAE metadata namespace"
    if "minimax_h3_video_vae" in vae_metadata:
        candidates["video_vae"] = "MiniMax H3 video VAE metadata namespace"
    if any(_contains_name_token(name, marker) for name in lowered for marker in ("audio_vae", "audio_decoder")):
        candidates["audio_vae"] = "audio VAE tensor naming"
    if any(_contains_name_token(name, "video_vae") for name in lowered):
        candidates["video_vae"] = "video VAE tensor naming"
    if any(_contains_name_token(name, "qwen3_vl") for name in lowered) or any(
        marker in name for name in lowered for marker in ("text_model.layers.", "visual.blocks.")
    ):
        candidates["text_encoder"] = "Qwen3-VL/text encoder tensor naming"
    if (
        any(marker in name for name in lowered for marker in ("diffusion_model.blocks.", "transformer_blocks."))
        or transformer_marker_present
    ):
        candidates["transformer"] = "H3 diffusion transformer tensor naming"

    if len(candidates) > 1:
        raise ValueError(f"contradictory component evidence: {', '.join(sorted(candidates))}")
    if adapter:
        if h3_signature and set(candidates) == {"transformer"}:
            return "lora", ["adapter tensor naming", candidates["transformer"]]
        return "unknown", ["adapter tensors without a unique H3 target signature"]
    if candidates:
        component, evidence = next(iter(candidates.items()))
        if not h3_signature:
            return "unknown", [f"{component} tensors without an H3 signature"]
        return component, [evidence]
    return "unknown", []


def classify_quantization(
    tensors: Sequence[TensorDescriptor], metadata: Mapping[str, str]
) -> tuple[tuple[str, ...], list[str]]:
    """Classify storage/quantization from structured metadata, names, and dtypes."""

    names = tuple(tensor.name for tensor in tensors)
    dtypes = {tensor.dtype for tensor in tensors}
    lowered_names = tuple(name.lower() for name in names)
    quantization_keys = {
        "dtype",
        "format",
        "quant_type",
        "quantization",
        "quantization_type",
        "weight_dtype",
    }
    structured_values = tuple(value.lower() for key, value in metadata.items() if key.lower() in quantization_keys)
    structured_tokens: set[str] = set()
    for value in structured_values:
        value_tokens = set(re.findall(r"[a-z0-9]+", value))
        if value_tokens.isdisjoint({"no", "not", "none", "unquantized"}):
            structured_tokens.update(value_tokens)
    structured_joined = tuple(re.sub(r"[^a-z0-9]+", "_", value).strip("_") for value in structured_values)
    detected: list[str] = []
    evidence: list[str] = []

    def add(label: str) -> None:
        if label not in detected:
            detected.append(label)
            evidence.append(f"quantization evidence: {label}")

    if "comfy_quant" in structured_joined or any(_contains_name_token(name, "comfy_quant") for name in lowered_names):
        add("comfy_quant")
    if structured_tokens.intersection({"convrot", "hadamard"}) or any(
        _contains_name_token(name, marker) for name in lowered_names for marker in ("convrot", "hadamard")
    ):
        add("convrot")
    if structured_tokens.intersection({"nvfp4", "float4"}) or any(
        _contains_name_token(name, marker) for name in lowered_names for marker in ("nvfp4", "float4_e2m1")
    ):
        add("nvfp4")
    has_mxfp8 = (
        "mxfp8" in structured_tokens
        or any(value == "mx_fp8" for value in structured_joined)
        or any(_contains_name_token(name, marker) for name in lowered_names for marker in ("mxfp8", "mx_fp8"))
    )
    if has_mxfp8:
        add("mxfp8")
    elif dtypes.intersection({"F8_E4M3", "F8_E5M2", "F8_E4M3FNUZ", "F8_E5M2FNUZ"}) or (
        structured_tokens.intersection({"fp8", "float8"})
    ):
        add("fp8")

    if "I8" in dtypes or "int8" in structured_tokens:
        add("int8")
    if not detected:
        detected.append("unquantized-or-unspecified")
    return tuple(detected), evidence


__all__ = [
    "ArtifactInspection",
    "TensorDescriptor",
    "classify_component",
    "classify_quantization",
]
