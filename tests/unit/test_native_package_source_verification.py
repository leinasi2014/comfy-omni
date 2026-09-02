from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from comfy_omni.artifacts import fileops
from comfy_omni.contracts.models import ContractError
from comfy_omni.conversion.packaging.models import ComponentFile, ComponentReceipt
from comfy_omni.conversion.packaging.planning import (
    PACKAGE_COMPONENTS,
    PINNED_VLLM_OMNI_COMMIT,
    plan_native_package,
)
from comfy_omni.domain.normalization import ToolIdentity


def _fixture(tmp_path: Path):
    tool = ToolIdentity("comfy-omni", "0.2.0a1", "a" * 40, "b" * 64)
    receipts: list[ComponentReceipt] = []
    expected_files: list[dict[str, object]] = []
    for component in PACKAGE_COMPONENTS:
        source = tmp_path / component
        source.mkdir()
        payload = f"{component}:payload".encode()
        target = source / "nested" / "artifact.bin"
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
                files=(ComponentFile("nested/artifact.bin", len(payload), digest),),
            )
        )
        expected_files.append(
            {
                "component": component,
                "path": "nested/artifact.bin",
                "sha256": digest,
                "size": len(payload),
            }
        )
    return plan_native_package(tuple(receipts), vllm_omni_commit=PINNED_VLLM_OMNI_COMMIT), expected_files


def test_package_source_verification_rehashes_the_exact_planned_trees(tmp_path: Path) -> None:
    from comfy_omni.conversion.packaging.verification import (
        PACKAGE_SOURCE_VERIFICATION_SCHEMA,
        verify_package_sources,
    )

    plan, expected_files = _fixture(tmp_path)

    result = verify_package_sources(plan)

    assert result.schema == PACKAGE_SOURCE_VERIFICATION_SCHEMA
    assert result.plan_content_sha256 == plan.content_sha256
    assert result.tool == plan.tool
    assert result.component_count == 6
    assert result.file_count == 6
    assert result.total_bytes == sum(item["size"] for item in expected_files)
    assert result.files_sha256 == hashlib.sha256(fileops.canonical_json(expected_files)).hexdigest()
    assert result.to_dict()["status"] == "VERIFIED"


@pytest.mark.parametrize("mode", ["missing", "extra", "tampered", "linked"])
def test_package_source_verification_rejects_tree_or_payload_drift(tmp_path: Path, mode: str) -> None:
    from comfy_omni.conversion.packaging.verification import verify_package_sources

    plan, _ = _fixture(tmp_path)
    transformer = tmp_path / "transformer"
    artifact = transformer / "nested" / "artifact.bin"
    if mode == "missing":
        artifact.unlink()
    elif mode == "extra":
        (transformer / "extra.bin").write_bytes(b"extra")
    elif mode == "tampered":
        artifact.write_bytes(b"different")
    else:
        (transformer / "linked.bin").symlink_to(artifact)

    with pytest.raises(ContractError) as failure:
        verify_package_sources(plan)

    assert failure.value.evidence["stage"] in {"tree-census", "file-verification"}


def test_package_source_verification_rejects_plan_digest_drift(tmp_path: Path) -> None:
    from comfy_omni.conversion.packaging.verification import verify_package_sources

    plan, _ = _fixture(tmp_path)
    with pytest.raises(ContractError, match="plan fields or content SHA256 drifted") as failure:
        verify_package_sources(replace(plan, content_sha256="f" * 64))
    assert failure.value.evidence["stage"] == "plan-binding"


def test_package_source_verification_rejects_a_linked_source_directory(tmp_path: Path) -> None:
    from comfy_omni.conversion.packaging.verification import verify_package_sources

    plan, _ = _fixture(tmp_path)
    source = tmp_path / "transformer"
    linked = tmp_path / "linked-transformer"
    linked.symlink_to(source, target_is_directory=True)
    receipts = []
    for component in plan.components:
        files = tuple(
            ComponentFile(item.source_path, item.size, item.sha256)
            for item in plan.files
            if item.component == component.component
        )
        receipts.append(
            ComponentReceipt(
                component=component.component,
                source_dir=linked.as_posix() if component.component == "transformer" else component.source_dir,
                receipt_schema=component.receipt_schema,
                receipt_sha256=component.receipt_sha256,
                tool=plan.tool,
                files=files,
            )
        )
    linked_plan = plan_native_package(tuple(receipts), vllm_omni_commit=PINNED_VLLM_OMNI_COMMIT)

    with pytest.raises(ContractError, match="missing, linked, or unreadable") as failure:
        verify_package_sources(linked_plan)
    assert failure.value.evidence["stage"] == "tree-census"
