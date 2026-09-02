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


def test_materialize_package_stages_exact_planned_bytes_without_publishing(tmp_path: Path) -> None:
    from comfy_omni.conversion.packaging.materialization import (
        PACKAGE_MATERIALIZATION_SCHEMA,
        materialize_package,
    )

    tool = ToolIdentity("comfy-omni", "0.2.0a1", "a" * 40, "b" * 64)
    receipts: list[ComponentReceipt] = []
    payloads: dict[str, bytes] = {}
    expected_files: list[dict[str, object]] = []
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
        expected_files.append(
            {
                "path": target_path,
                "sha256": digest,
                "size": len(payload),
            }
        )
    plan = plan_native_package(tuple(receipts), vllm_omni_commit=PINNED_VLLM_OMNI_COMMIT)
    output = tmp_path / "native-package"

    result = materialize_package(plan, output)

    assert result.schema == PACKAGE_MATERIALIZATION_SCHEMA
    assert result.plan_content_sha256 == plan.content_sha256
    assert result.file_count == len(payloads)
    assert result.total_bytes == sum(len(payload) for payload in payloads.values())
    assert result.files_sha256 == hashlib.sha256(fileops.canonical_json(expected_files)).hexdigest()
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
