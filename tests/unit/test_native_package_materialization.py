from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from comfy_omni.artifacts import fileops
from comfy_omni.contracts.models import ContractError
from comfy_omni.conversion.packaging.models import ComponentFile, ComponentReceipt, NativePackagePlan
from comfy_omni.conversion.packaging.planning import (
    PACKAGE_COMPONENTS,
    PINNED_VLLM_OMNI_COMMIT,
    plan_native_package,
)
from comfy_omni.domain.normalization import ToolIdentity


def _fixture(tmp_path: Path) -> tuple[NativePackagePlan, dict[str, bytes]]:
    tool = ToolIdentity("comfy-omni", "0.2.0a1", "a" * 40, "b" * 64)
    receipts: list[ComponentReceipt] = []
    payloads: dict[str, bytes] = {}
    for component in PACKAGE_COMPONENTS:
        source = tmp_path / "sources" / component
        source.mkdir(parents=True)
        payload = f"{component}:payload".encode()
        relative = "nested/artifact.bin"
        target = source / relative
        target.parent.mkdir()
        target.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        receipts.append(
            ComponentReceipt(
                component=component,
                source_dir=source.as_posix(),
                receipt_schema="test.component.receipt/v1",
                receipt_sha256=hashlib.sha256(f"{component}:receipt".encode()).hexdigest(),
                tool=tool,
                files=(ComponentFile(relative, len(payload), digest),),
            )
        )
        target_path = f"Ref2VA/{component}/{relative}"
        payloads[target_path] = payload
    return plan_native_package(tuple(receipts), vllm_omni_commit=PINNED_VLLM_OMNI_COMMIT), payloads


def _expected_files(payloads: dict[str, bytes]) -> list[dict[str, object]]:
    return [
        {
            "path": path,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
        }
        for path, payload in payloads.items()
    ]


def test_materialize_package_stages_exact_planned_bytes_without_publishing(tmp_path: Path) -> None:
    from comfy_omni.conversion.packaging.materialization import (
        PACKAGE_MATERIALIZATION_SCHEMA,
        materialize_package,
    )

    plan, payloads = _fixture(tmp_path)
    output = tmp_path / "native-package"

    result = materialize_package(plan, output)

    assert result.schema == PACKAGE_MATERIALIZATION_SCHEMA
    assert result.plan_content_sha256 == plan.content_sha256
    assert result.file_count == len(payloads)
    assert result.total_bytes == sum(len(payload) for payload in payloads.values())
    assert result.files_sha256 == hashlib.sha256(fileops.canonical_json(_expected_files(payloads))).hexdigest()
    assert result.output_dir == output
    assert result.stage_dir.parent == output.parent
    assert result.stage_dir.name.startswith(f".{output.name}.stage-")
    assert not output.exists()
    assert {
        item.relative_to(result.stage_dir).as_posix(): item.read_bytes()
        for item in result.stage_dir.rglob("*")
        if item.is_file()
    } == payloads
    assert result.to_dict()["status"] == "STAGED_VERIFIED"


def test_materialize_package_refuses_existing_output_without_creating_staging(tmp_path: Path) -> None:
    from comfy_omni.conversion.packaging.materialization import materialize_package

    plan, _ = _fixture(tmp_path)
    output = tmp_path / "native-package"
    output.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        materialize_package(plan, output)

    assert list(tmp_path.glob(".native-package.stage-*")) == []


def test_materialize_package_refuses_output_inside_a_component_source(tmp_path: Path) -> None:
    from comfy_omni.conversion.packaging.materialization import materialize_package

    plan, _ = _fixture(tmp_path)
    output = tmp_path / "sources" / "transformer" / "native-package"

    with pytest.raises(ContractError, match="paths overlap") as failure:
        materialize_package(plan, output)

    assert failure.value.evidence["stage"] == "output-binding"
    assert not output.exists()


def test_materialize_package_rejects_source_drift_after_preverification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from comfy_omni.conversion.packaging import materialization

    plan, _ = _fixture(tmp_path)
    original = materialization.verify_package_sources

    def verify_then_mutate(candidate: NativePackagePlan):
        result = original(candidate)
        (tmp_path / "sources" / "transformer" / "nested" / "artifact.bin").write_bytes(b"changed")
        return result

    monkeypatch.setattr(materialization, "verify_package_sources", verify_then_mutate)
    output = tmp_path / "native-package"

    with pytest.raises(ContractError, match="differs from the package plan") as failure:
        materialization.materialize_package(plan, output)

    assert failure.value.evidence["stage"] == "file-copy"
    assert not output.exists()
    assert len(list(tmp_path.glob(".native-package.stage-*"))) == 1


def test_materialize_package_rejects_a_linked_source_before_staging(tmp_path: Path) -> None:
    from comfy_omni.conversion.packaging.materialization import materialize_package

    plan, _ = _fixture(tmp_path)
    transformer = tmp_path / "sources" / "transformer"
    (transformer / "linked.bin").symlink_to(transformer / "nested" / "artifact.bin")
    output = tmp_path / "native-package"

    with pytest.raises(ContractError, match="link"):
        materialize_package(plan, output)

    assert not output.exists()
    assert list(tmp_path.glob(".native-package.stage-*")) == []


def test_materialize_package_retains_private_stage_on_copy_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from comfy_omni.conversion.packaging import materialization

    plan, _ = _fixture(tmp_path)
    original = fileops.copy_file_pinned_exclusive
    calls = 0

    def interrupt(source: Path, destination: Path) -> tuple[str, int]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise fileops.FsopsIoError("injected copy interruption")
        return original(source, destination)

    monkeypatch.setattr(fileops, "copy_file_pinned_exclusive", interrupt)
    output = tmp_path / "native-package"

    with pytest.raises(ContractError, match="package file copy failed") as failure:
        materialization.materialize_package(plan, output)

    assert failure.value.evidence["stage"] == "file-copy"
    assert not output.exists()
    assert len(list(tmp_path.glob(".native-package.stage-*"))) == 1


def test_materialize_package_rejects_an_unexpected_staging_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from comfy_omni.conversion.packaging import materialization

    plan, _ = _fixture(tmp_path)
    original = fileops.copy_file_pinned_exclusive
    injected = False

    def copy_and_inject(source: Path, destination: Path) -> tuple[str, int]:
        nonlocal injected
        result = original(source, destination)
        if not injected:
            destination.parents[3].joinpath("unexpected.bin").write_bytes(b"unexpected")
            injected = True
        return result

    monkeypatch.setattr(fileops, "copy_file_pinned_exclusive", copy_and_inject)
    output = tmp_path / "native-package"

    with pytest.raises(ContractError, match="file census differs") as failure:
        materialization.materialize_package(plan, output)

    assert failure.value.evidence["stage"] == "staging-census"
    assert not output.exists()


def test_pinned_exclusive_copy_refuses_an_existing_target(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    target = tmp_path / "target.bin"
    source.write_bytes(b"source")
    target.write_bytes(b"existing")

    with pytest.raises(fileops.FsopsExistsError, match="refusing to overwrite"):
        fileops.copy_file_pinned_exclusive(source, target)

    assert target.read_bytes() == b"existing"


def test_pinned_exclusive_copy_detects_a_same_size_source_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    target = tmp_path / "target.bin"
    source.write_bytes(b"original")
    original_read = fileops._read_chunk
    rewritten = False

    def read_then_rewrite(descriptor: int, count: int) -> bytes:
        nonlocal rewritten
        chunk = original_read(descriptor, count)
        if chunk and not rewritten:
            source.write_bytes(b"modified")
            rewritten = True
        return chunk

    monkeypatch.setattr(fileops, "_read_chunk", read_then_rewrite)

    with pytest.raises(fileops.FsopsModifiedError, match="replaced or rewritten"):
        fileops.copy_file_pinned_exclusive(source, target)

    assert target.exists()
