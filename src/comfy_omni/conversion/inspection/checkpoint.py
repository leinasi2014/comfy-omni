"""Orchestrate read-only inspection of one safetensors checkpoint.

This conversion boundary composes artifact I/O with pure domain classification. It performs no
payload loading, conversion, publication, runtime registration, or host integration.
"""

from __future__ import annotations

from pathlib import Path

from comfy_omni.artifacts.safetensors import read_safetensors_header
from comfy_omni.domain.checkpoints import (
    ArtifactInspection,
    classify_component,
    classify_quantization,
)


def inspect_safetensors(path: Path | str) -> ArtifactInspection:
    """Inspect one safetensors file using header metadata only."""

    artifact = Path(path).resolve()
    if artifact.suffix.lower() != ".safetensors":
        raise ValueError(f"{artifact}: expected a .safetensors file")
    metadata, tensors = read_safetensors_header(artifact)
    component, component_evidence = classify_component(tuple(tensor.name for tensor in tensors), metadata)
    quantization, quantization_evidence = classify_quantization(tensors, metadata)
    return ArtifactInspection(
        path=str(artifact),
        component=component,
        quantization=quantization,
        tensor_count=len(tensors),
        metadata=metadata,
        evidence=tuple(component_evidence + quantization_evidence),
    )


__all__ = ["inspect_safetensors"]
