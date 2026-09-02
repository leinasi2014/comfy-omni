"""Application use cases for inspecting checkpoint paths.

The application layer owns deterministic path expansion and multi-artifact orchestration. It does
not parse safetensors, render CLI output, import runtimes, or mutate artifacts.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from comfy_omni.conversion.inspection import inspect_safetensors
from comfy_omni.domain.checkpoints import ArtifactInspection


def expand_checkpoint_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    """Expand directories recursively into sorted safetensors paths."""

    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.rglob("*.safetensors")))
        else:
            expanded.append(path)
    return tuple(expanded)


def inspect_checkpoint_paths(paths: Sequence[Path]) -> tuple[ArtifactInspection, ...]:
    """Inspect every explicitly named or recursively discovered checkpoint."""

    return tuple(inspect_safetensors(path) for path in expand_checkpoint_paths(paths))


__all__ = ["expand_checkpoint_paths", "inspect_checkpoint_paths"]
