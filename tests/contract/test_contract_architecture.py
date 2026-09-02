from __future__ import annotations

import ast
from pathlib import Path

from comfy_omni.application.contracts import load_contract_catalog

PACKAGE_ROOT = Path(__file__).parents[2] / "src" / "comfy_omni"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def test_contract_core_does_not_depend_on_artifacts_or_workflows() -> None:
    forbidden = ("comfy_omni.artifacts", "comfy_omni.conversion", "comfy_omni.application", "torch", "vllm")
    for path in (PACKAGE_ROOT / "contracts").glob("*.py"):
        assert not any(name.startswith(forbidden) for name in _imports(path)), path


def test_environment_compatibility_is_confined_to_cli() -> None:
    hits = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        if "H3_FORGE_CONTRACT_DIR" in path.read_text(encoding="utf-8"):
            hits.append(path.relative_to(PACKAGE_ROOT).as_posix())
    assert hits == ["cli/commands/contract.py"]


def test_application_catalog_ignores_environment_without_explicit_path(monkeypatch) -> None:
    monkeypatch.setenv("H3_FORGE_CONTRACT_DIR", "does-not-exist")
    assert len(load_contract_catalog().records) == 3


def test_plugin_has_no_contract_store_activation_path() -> None:
    source = (PACKAGE_ROOT / "plugin.py").read_text(encoding="utf-8")
    assert "contract_workflows" not in source
    assert "snapshot_store" not in source
    assert "H3_FORGE_CONTRACT_DIR" not in source
