"""Enforce the dependency boundary of the first migrated vertical slice."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_INTERNAL_PREFIXES = {
    "src/comfy_omni/domain/normalization.py": (),
    "src/comfy_omni/domain/checkpoints.py": (),
    "src/comfy_omni/artifacts/build_identity.py": ("comfy_omni.domain",),
    "src/comfy_omni/artifacts/normalization.py": (
        "comfy_omni.artifacts",
        "comfy_omni.domain",
    ),
    "src/comfy_omni/artifacts/safetensors.py": ("comfy_omni.domain",),
    "src/comfy_omni/conversion/normalization/text_encoder.py": (
        "comfy_omni.artifacts",
        "comfy_omni.domain",
    ),
    "src/comfy_omni/conversion/inspection/checkpoint.py": (
        "comfy_omni.artifacts",
        "comfy_omni.domain",
    ),
    "src/comfy_omni/application/inspection.py": (
        "comfy_omni.conversion",
        "comfy_omni.domain",
    ),
    "src/comfy_omni/application/normalization.py": (
        "comfy_omni.artifacts",
        "comfy_omni.conversion",
    ),
    "src/comfy_omni/cli/commands/inspect.py": ("comfy_omni.application",),
    "src/comfy_omni/cli/commands/normalize.py": ("comfy_omni.application",),
}
FORBIDDEN_OPTIONAL_IMPORTS = ("fastapi", "torch", "vllm", "vllm_omni")


def _imported_modules(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return tuple(modules)


@pytest.mark.parametrize(("relative", "allowed_prefixes"), EXPECTED_INTERNAL_PREFIXES.items())
def test_inspection_slice_has_only_declared_internal_dependencies(
    relative: str, allowed_prefixes: tuple[str, ...]
) -> None:
    modules = _imported_modules(ROOT / relative)
    internal_modules = tuple(module for module in modules if module.startswith("comfy_omni"))

    assert all(module.startswith(allowed_prefixes) for module in internal_modules)


@pytest.mark.parametrize("relative", EXPECTED_INTERNAL_PREFIXES)
def test_inspection_slice_does_not_import_optional_runtime_stacks(relative: str) -> None:
    modules = _imported_modules(ROOT / relative)

    assert not any(
        module == forbidden or module.startswith(f"{forbidden}.")
        for module in modules
        for forbidden in FORBIDDEN_OPTIONAL_IMPORTS
    )
