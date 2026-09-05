"""Late source changes must fail before the commit marker, after output reread."""
from pathlib import Path

import pytest
from test_native_export_transaction import _bound_plan, _tool, _write_fixture

from comfy_omni.artifacts import fileops
from comfy_omni.contracts.models import ContractError
from comfy_omni.conversion.exporters.execution import execute_native_export


def test_native_export_rechecks_source_after_published_payload_rehash(tmp_path, monkeypatch):
    source = tmp_path / "source.safetensors"
    _write_fixture(source)
    plan = _bound_plan(source)
    output = tmp_path / "output"
    original = fileops.sha256_file_pinned
    changed = False

    def tamper_during_output_rehash(path: Path):
        nonlocal changed
        result = original(path)
        if path.parent == output and not changed:
            raw = bytearray(source.read_bytes())
            raw[-1] ^= 1
            source.write_bytes(raw)
            changed = True
        return result

    monkeypatch.setattr(fileops, "sha256_file_pinned", tamper_during_output_rehash)
    with pytest.raises(ContractError, match="source.*changed"):
        execute_native_export(plan, output, tool=_tool())
    assert changed
    assert not output.exists()
