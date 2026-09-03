# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright h3-forge contributors
#
# Provenance: wholesale migration from h3-forge@e9cb011d00b028c149db3978de246c54f6e34acc
#   source path: src/h3_forge/lora_hotswap/security.py
#   source blob: 9c55b71391899c2b5a0620cb861b13c51571c270
#   license: Apache-2.0
#   attribution: h3-forge contributors
# Migrated byte-preserving except this provenance header, import retargeting, and
# mechanical line wrapping to satisfy the repository line-length (120).
"""Explicit runtime-update gate and local adapter path policy."""

from __future__ import annotations

import os
import re
from pathlib import Path

_TRUE_VALUES = {"1", "true", "yes", "on"}
_REMOTE_REPOSITORY = re.compile(
    r"^(?![.-])[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?/"
    r"(?![.-])[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$"
)


def runtime_updates_enabled() -> bool:
    return os.getenv("VLLM_OMNI_ALLOW_RUNTIME_LORA_UPDATING", "").lower() in _TRUE_VALUES


def validate_adapter_path(path: str) -> str:
    """Validate a local path or an explicitly allowed remote repository ID."""
    candidate = Path(path).expanduser()
    if ".." in candidate.parts:
        raise ValueError("LoRA path must not contain parent-directory traversal")
    if candidate.exists():
        resolved = candidate.resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError("Runtime diffusion LoRA loading requires a PEFT adapter directory")
        configured_roots = os.getenv("VLLM_OMNI_LORA_ALLOWED_ROOTS", "")
        if not configured_roots:
            raise ValueError("Local LoRA loading requires VLLM_OMNI_LORA_ALLOWED_ROOTS")
        try:
            roots = [
                Path(item).expanduser().resolve(strict=True) for item in configured_roots.split(os.pathsep) if item
            ]
        except FileNotFoundError as exc:
            raise ValueError("A configured VLLM_OMNI_LORA_ALLOWED_ROOTS entry does not exist") from exc
        if not any(resolved == root or root in resolved.parents for root in roots):
            raise ValueError("LoRA path is outside VLLM_OMNI_LORA_ALLOWED_ROOTS")
        return str(resolved)

    if os.getenv("VLLM_OMNI_LORA_ALLOW_REMOTE", "").lower() in _TRUE_VALUES:
        if _REMOTE_REPOSITORY.fullmatch(path):
            return path
        raise ValueError("Remote LoRA path must be a repository ID in owner/name form")

    raise ValueError("LoRA path does not exist locally; remote adapters are disabled")
