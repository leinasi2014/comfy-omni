"""Contract checks for the standalone native-package acceptance harness."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSEMBLY_SCRIPT = ROOT / "scripts" / "acceptance" / "native_package_assembly.py"
RUNTIME_COMPONENT_CONFIGS = ROOT / "docs" / "testing" / "e4-component-configs.v1.json"


def _load_assembly_harness():
    spec = importlib.util.spec_from_file_location("native_package_assembly", ASSEMBLY_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_package_assembly_pins_every_runtime_component_file() -> None:
    document = json.loads(RUNTIME_COMPONENT_CONFIGS.read_text(encoding="utf-8"))
    expected: dict[str, dict[str, tuple[int, str]]] = {}
    for record in document["files"]:
        expected.setdefault(record["component"], {})[record["file"]] = (record["bytes"], record["sha256"])

    harness = _load_assembly_harness()
    assert harness.RUNTIME_COMPONENT_FILES == expected
