from __future__ import annotations

import hashlib
from pathlib import Path

from comfy_omni.artifacts import fileops
from comfy_omni.conversion.packaging.models import ComponentFile, ComponentReceipt
from comfy_omni.conversion.packaging.planning import (
    PACKAGE_COMPONENTS,
    PINNED_VLLM_OMNI_COMMIT,
    plan_native_package,
)
from comfy_omni.domain.normalization import ToolIdentity


def test_package_source_verification_rehashes_the_exact_planned_trees(tmp_path: Path) -> None:
    from comfy_omni.conversion.packaging.verification import (
        PACKAGE_SOURCE_VERIFICATION_SCHEMA,
        verify_package_sources,
    )

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
    plan = plan_native_package(tuple(receipts), vllm_omni_commit=PINNED_VLLM_OMNI_COMMIT)

    result = verify_package_sources(plan)

    assert result.schema == PACKAGE_SOURCE_VERIFICATION_SCHEMA
    assert result.plan_content_sha256 == plan.content_sha256
    assert result.tool == tool
    assert result.component_count == 6
    assert result.file_count == 6
    assert result.total_bytes == sum(item["size"] for item in expected_files)
    assert result.files_sha256 == hashlib.sha256(fileops.canonical_json(expected_files)).hexdigest()
    assert result.to_dict()["status"] == "VERIFIED"
