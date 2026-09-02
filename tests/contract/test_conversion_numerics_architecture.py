from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_INTERNAL_PREFIXES = {
    "src/comfy_omni/conversion/numerics/errors.py": ("comfy_omni.contracts",),
    "src/comfy_omni/conversion/numerics/reference.py": ("comfy_omni.conversion.numerics",),
    "src/comfy_omni/conversion/numerics/torch_backend.py": ("comfy_omni.conversion.numerics",),
}
FORBIDDEN = ("torch", "vllm", "vllm_omni", "comfy_omni.application", "comfy_omni.runtime")


def _imports(relative: str) -> tuple[str, ...]:
    path = ROOT / relative
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.append(node.module)
    return tuple(result)


@pytest.mark.parametrize(("relative", "allowed"), EXPECTED_INTERNAL_PREFIXES.items())
def test_numerics_has_only_declared_internal_dependencies(relative: str, allowed: tuple[str, ...]) -> None:
    internal = tuple(name for name in _imports(relative) if name.startswith("comfy_omni"))

    assert all(name.startswith(allowed) for name in internal)


@pytest.mark.parametrize("relative", EXPECTED_INTERNAL_PREFIXES)
def test_numerics_has_no_static_optional_or_upstream_import(relative: str) -> None:
    imports = _imports(relative)

    assert not any(name == item or name.startswith(f"{item}.") for name in imports for item in FORBIDDEN)


def test_torch_is_loaded_only_through_the_explicit_lazy_boundary() -> None:
    source = (ROOT / "src/comfy_omni/conversion/numerics/torch_backend.py").read_text(encoding="utf-8")

    assert 'importlib.import_module("torch")' in source
    assert "import torch" not in source
