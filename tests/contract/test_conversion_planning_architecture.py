from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "comfy_omni"
EXPECTED_INTERNAL_PREFIXES = {
    "src/comfy_omni/domain/qkv.py": (),
    "src/comfy_omni/contracts/conversion.py": ("comfy_omni.contracts",),
    "src/comfy_omni/conversion/exporters/models.py": (),
    "src/comfy_omni/conversion/exporters/planning.py": (
        "comfy_omni.artifacts",
        "comfy_omni.contracts",
        "comfy_omni.conversion.contract_workflows",
        "comfy_omni.conversion.exporters",
        "comfy_omni.domain",
    ),
    "src/comfy_omni/application/conversion.py": (
        "comfy_omni.application",
        "comfy_omni.contracts",
        "comfy_omni.conversion",
    ),
}
FORBIDDEN_OPTIONAL_IMPORTS = ("torch", "vllm", "vllm_omni")


def _imports(relative: str) -> tuple[str, ...]:
    path = ROOT / relative
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return tuple(modules)


@pytest.mark.parametrize(("relative", "allowed"), EXPECTED_INTERNAL_PREFIXES.items())
def test_conversion_plan_has_only_declared_internal_dependencies(relative: str, allowed: tuple[str, ...]) -> None:
    internal = tuple(name for name in _imports(relative) if name.startswith("comfy_omni"))

    assert all(name.startswith(allowed) for name in internal)


@pytest.mark.parametrize("relative", EXPECTED_INTERNAL_PREFIXES)
def test_conversion_plan_does_not_import_optional_runtime_stacks(relative: str) -> None:
    imports = _imports(relative)

    assert not any(
        name == forbidden or name.startswith(f"{forbidden}.")
        for name in imports
        for forbidden in FORBIDDEN_OPTIONAL_IMPORTS
    )


def test_conversion_layer_does_not_depend_on_upstream_surfaces() -> None:
    forbidden = (
        "comfy_omni.application",
        "comfy_omni.api",
        "comfy_omni.cli",
        "comfy_omni.integrations",
        "comfy_omni.runtime",
    )
    hits: list[tuple[str, str]] = []
    for path in (PACKAGE / "conversion").rglob("*.py"):
        relative = path.relative_to(ROOT).as_posix()
        hits.extend((relative, name) for name in _imports(relative) if name.startswith(forbidden))

    assert hits == []
