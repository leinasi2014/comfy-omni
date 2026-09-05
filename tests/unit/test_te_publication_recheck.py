"""Late source changes must fail before the commit marker, after output reread."""

from pathlib import Path

import pytest
from test_native_export_transaction import _bound_plan, _tool, _write_fixture

from comfy_omni.artifacts import fileops
from comfy_omni.contracts.models import ContractError
from comfy_omni.conversion.exporters.execution import execute_native_export
from comfy_omni.conversion.packaging.native_export import prepare_native_export, publish_native_export, stage_document


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


@pytest.mark.parametrize("replace_parent", [False, True])
def test_final_check_cannot_redirect_manifest_into_foreign_directory(tmp_path, replace_parent):
    parent = tmp_path / "parent"
    parent.mkdir()
    output = parent / "output"
    stage = prepare_native_export(output)
    artifact = stage_document(stage, "config.json", b"{}\n", kind="config")

    def replace_directory():
        if replace_parent:
            parent.rename(tmp_path / "original-parent")
            parent.mkdir()
            output.mkdir()
        else:
            output.rename(parent / "original-output")
            output.mkdir()
        (output / "foreign.txt").write_bytes(b"untouched")

    with pytest.raises(ContractError, match="identity changed"):
        publish_native_export(stage, (artifact,), {"schema": "synthetic/v1"}, before_manifest=replace_directory)
    assert {path.name for path in output.iterdir()} == {"foreign.txt"}
    assert (output / "foreign.txt").read_bytes() == b"untouched"


@pytest.mark.parametrize("change", ["in-place", "replacement", "extra"])
def test_final_check_rejects_changed_or_added_published_file(tmp_path, change):
    output = tmp_path / "output"
    stage = prepare_native_export(output)
    artifact = stage_document(stage, "config.json", b'{"value":1}\n', kind="config")

    def tamper_after_hash():
        path = output / "config.json"
        if change == "in-place":
            path.chmod(0o600)
            path.write_bytes(b'{"value":2}\n')
        elif change == "replacement":
            path.unlink()
            path.write_bytes(b'{"value":2}\n')
        else:
            (output / "foreign.txt").write_bytes(b"untouched")

    with pytest.raises(ContractError, match="published.*changed"):
        publish_native_export(stage, (artifact,), {"schema": "synthetic/v1"}, before_manifest=tamper_after_hash)
    if change == "in-place":
        assert not output.exists()
    elif change == "replacement":
        assert {path.name for path in output.iterdir()} == {"config.json"}
        assert (output / "config.json").read_bytes() == b'{"value":2}\n'
    else:
        assert {path.name for path in output.iterdir()} == {"foreign.txt"}
        assert (output / "foreign.txt").read_bytes() == b"untouched"
