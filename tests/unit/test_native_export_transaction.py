from __future__ import annotations

import hashlib
import importlib
import json
import struct
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from comfy_omni.artifacts import fileops
from comfy_omni.artifacts import sources as source_artifacts
from comfy_omni.artifacts.safetensors import read_safetensors_header
from comfy_omni.artifacts.sources import SafeTensorSources
from comfy_omni.contracts.models import ContractError
from comfy_omni.conversion.exporters.models import (
    NativeExportPlan,
    QkvLayoutPlan,
    ResourceEnvelope,
    ShardPlan,
    SourceBinding,
    TensorAction,
)
from comfy_omni.domain.normalization import ToolIdentity


def _write_fixture(path: Path) -> None:
    tensors = (
        ("alpha", "BF16", (2,), b"\x01\x02\x03\x04"),
        ("beta", "F32", (1,), b"\x05\x06\x07\x08"),
    )
    cursor = 0
    header: dict[str, object] = {}
    payload = bytearray()
    for name, dtype, shape, raw in tensors:
        header[name] = {
            "data_offsets": [cursor, cursor + len(raw)],
            "dtype": dtype,
            "shape": list(shape),
        }
        cursor += len(raw)
        payload.extend(raw)
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _bound_plan(source: Path) -> NativeExportPlan:
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    qkv_digest = hashlib.sha256(fileops.canonical_json([0])).hexdigest()
    draft = NativeExportPlan(
        schema="comfy_omni.native_export.plan/v1",
        output_schema="h3-comfy-int8-export/v2",
        component="transformer",
        profile="dense-bf16-online-int8",
        source_contract="tiny-copy-only-v1",
        source_contract_origin="compile-time",
        source_contract_schema_sha256="a" * 64,
        source_snapshot_manifest_sha256=None,
        source_snapshot_file_sha256=None,
        template_name="tiny-copy-only",
        template_version=1,
        template_sha256="b" * 64,
        source_files=(SourceBinding(str(source), source.stat().st_size, source_digest),),
        qkv_layout=QkvLayoutPlan("runtime-qkv", "grouped-for-official-loader", 1, 1, 1, 1, qkv_digest),
        resource_envelope=ResourceEnvelope(max_rows=1, max_shard_bytes=1024, largest_target_tensor_bytes=4),
        actions=(
            TensorAction("alpha", "alpha", "BF16", "BF16", (2,), 4, 4, "copy-raw"),
            TensorAction("beta", "beta", "F32", "F32", (1,), 4, 4, "copy-raw"),
        ),
        shards=(ShardPlan("model-00001-of-00001.safetensors", ("alpha", "beta"), 8),),
        target_tensor_count=2,
        target_payload_bytes=8,
        runtime_quant_method="compressed-tensors",
        runtime_ignored_layers=(),
        payload_semantics="test-copy-only",
        content_sha256="",
    )
    digest = hashlib.sha256(fileops.canonical_json(draft.to_dict(include_content_sha256=False))).hexdigest()
    return replace(draft, content_sha256=digest)


def _tool() -> ToolIdentity:
    return ToolIdentity("comfy-omni", "0.2.0a1", "1" * 40, "2" * 64)


def _execution() -> Any:
    try:
        module = importlib.import_module("comfy_omni.conversion.exporters.execution")
    except ModuleNotFoundError as exc:
        pytest.fail(f"native export execution boundary is absent: {exc}")
    return module


def _writer() -> Any:
    try:
        module = importlib.import_module("comfy_omni.artifacts.safetensors_writer")
    except ModuleNotFoundError as exc:
        pytest.fail(f"safetensors writer boundary is absent: {exc}")
    return module


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_source_unchanged_verification_rehashes_the_held_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.safetensors"
    _write_fixture(source)
    original = source_artifacts._hash_stream
    calls = 0

    def counted(stream: Any) -> str:
        nonlocal calls
        calls += 1
        return original(stream)

    monkeypatch.setattr(source_artifacts, "_hash_stream", counted)
    with SafeTensorSources([source]) as sources:
        assert calls == 1
        sources.verify_unchanged()
        assert calls == 2


def test_safetensors_writer_refuses_short_and_long_producers(tmp_path: Path) -> None:
    writer = _writer()
    short = writer.TensorPayload("alpha", "BF16", (2,), 4, lambda: iter((b"\x00\x01",)))
    long = writer.TensorPayload("alpha", "BF16", (2,), 4, lambda: iter((b"\x00" * 5,)))

    with pytest.raises(ContractError, match="wrote 2 bytes; expected 4"):
        writer.write_safetensors_file(tmp_path / "short.safetensors", (short,))
    with pytest.raises(ContractError, match="exceeded 4 bytes"):
        writer.write_safetensors_file(tmp_path / "long.safetensors", (long,))


def test_copy_transaction_is_deterministic_strictly_verified_and_manifest_last(tmp_path: Path) -> None:
    execution = _execution()
    source = tmp_path / "source.safetensors"
    _write_fixture(source)
    plan = _bound_plan(source)

    first = execution.execute_native_export(plan, tmp_path / "first", tool=_tool())
    second = execution.execute_native_export(plan, tmp_path / "second", tool=_tool())

    assert _tree_hashes(first.output_dir) == _tree_hashes(second.output_dir)
    assert set(_tree_hashes(first.output_dir)) == {
        "config.patch.json",
        "export.plan.json",
        "manifest.json",
        "model-00001-of-00001.safetensors",
        "model.safetensors.index.json",
    }
    metadata, descriptors = read_safetensors_header(first.output_dir / "model-00001-of-00001.safetensors")
    assert metadata == {}
    assert [(item.name, item.dtype, item.shape) for item in descriptors] == [
        ("alpha", "BF16", (2,)),
        ("beta", "F32", (1,)),
    ]
    with SafeTensorSources([first.output_dir / "model-00001-of-00001.safetensors"]) as exported:
        assert exported.read_raw(exported.tensors["alpha"]) == b"\x01\x02\x03\x04"
        assert exported.read_raw(exported.tensors["beta"]) == b"\x05\x06\x07\x08"
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    claimed = manifest.pop("manifest_sha256")
    assert claimed == hashlib.sha256(fileops.canonical_json(manifest)).hexdigest()
    assert manifest["schema"] == "comfy_omni.native_export.receipt/v1"
    assert manifest["status"] == "COMMITTED"
    assert manifest["plan_content_sha256"] == plan.content_sha256
    assert manifest["tool"] == _tool().to_dict()
    assert first.manifest_sha256 == claimed


@pytest.mark.parametrize("tamper", ["plan", "source", "operation"])
def test_transaction_rejects_unbound_inputs_before_publication(tmp_path: Path, tamper: str) -> None:
    execution = _execution()
    source = tmp_path / "source.safetensors"
    _write_fixture(source)
    plan = _bound_plan(source)
    if tamper == "plan":
        plan = replace(plan, content_sha256="f" * 64)
    elif tamper == "source":
        plan = replace(plan, source_files=(replace(plan.source_files[0], sha256="e" * 64),))
        plan = replace(
            plan,
            content_sha256=hashlib.sha256(
                fileops.canonical_json(plan.to_dict(include_content_sha256=False))
            ).hexdigest(),
        )
    else:
        actions = (replace(plan.actions[0], operation="inverse-convrot-to-bf16"), plan.actions[1])
        plan = replace(plan, actions=actions)
        plan = replace(
            plan,
            content_sha256=hashlib.sha256(
                fileops.canonical_json(plan.to_dict(include_content_sha256=False))
            ).hexdigest(),
        )
    output = tmp_path / "output"

    with pytest.raises(ContractError):
        execution.execute_native_export(plan, output, tool=_tool())

    assert not (output / "manifest.json").exists()


def test_transaction_refuses_overwrite_and_preserves_existing_tree(tmp_path: Path) -> None:
    execution = _execution()
    source = tmp_path / "source.safetensors"
    _write_fixture(source)
    output = tmp_path / "output"
    output.mkdir()
    occupied = output / "manifest.json"
    occupied.write_bytes(b"owner-data")

    with pytest.raises(FileExistsError, match="overwrite"):
        execution.execute_native_export(_bound_plan(source), output, tool=_tool())

    assert occupied.read_bytes() == b"owner-data"


def test_final_source_recheck_failure_cannot_publish_a_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    execution = _execution()
    source = tmp_path / "source.safetensors"
    _write_fixture(source)
    output = tmp_path / "output"

    def refuse(_sources: SafeTensorSources) -> None:
        raise ContractError("source changed during final held-descriptor verification")

    monkeypatch.setattr(source_artifacts.SafeTensorSources, "verify_unchanged", refuse)
    with pytest.raises(ContractError, match="final held-descriptor"):
        execution.execute_native_export(_bound_plan(source), output, tool=_tool())

    assert not (output / "manifest.json").exists()


def test_publication_interruption_leaves_no_commit_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    execution = _execution()
    publication = importlib.import_module("comfy_omni.conversion.packaging.native_export")
    source = tmp_path / "source.safetensors"
    _write_fixture(source)
    output = tmp_path / "output"
    original_link = publication.os.link
    links = 0

    def interrupt(source_path: Path, target_path: Path) -> None:
        nonlocal links
        links += 1
        if links == 2:
            raise OSError("simulated publication interruption")
        original_link(source_path, target_path)

    monkeypatch.setattr(publication.os, "link", interrupt)
    with pytest.raises(ContractError, match="publication"):
        execution.execute_native_export(_bound_plan(source), output, tool=_tool())

    assert not (output / "manifest.json").exists()
