from __future__ import annotations

import hashlib
from pathlib import Path

from comfy_omni.artifacts import fileops
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


def _census(plan: NativePackagePlan) -> list[dict[str, object]]:
    return [{"path": item.target_path, "sha256": item.sha256, "size": item.size} for item in plan.files]


def test_publish_package_publishes_manifest_last_atomically(tmp_path: Path) -> None:
    from comfy_omni.conversion.packaging.publication import (
        PACKAGE_PUBLICATION_SCHEMA,
        publish_package,
    )

    from comfy_omni.conversion.packaging.materialization import materialize_package

    plan, payloads = _fixture(tmp_path)
    output = tmp_path / "native-package"

    materialized = materialize_package(plan, output)
    published = publish_package(plan, materialized)

    assert published.schema == PACKAGE_PUBLICATION_SCHEMA
    assert published.plan_content_sha256 == plan.content_sha256
    assert not materialized.stage_dir.exists()
    assert output.is_dir()
    on_disk = {item.relative_to(output).as_posix(): item.read_bytes() for item in output.rglob("*") if item.is_file()}
    assert set(on_disk) == set(payloads) | {"h3-comfy-package.json"}
    for path, payload in payloads.items():
        assert on_disk[path] == payload
    assert published.file_count == len(plan.files)
    assert published.total_bytes == sum(item.size for item in plan.files)
    assert published.output_dir == output
    assert published.to_dict()["status"] == "PUBLISHED"

    manifest = fileops.parse_json_strict((output / "h3-comfy-package.json").read_bytes())
    assert manifest["schema"] == "h3-comfy-package/v3"
    assert manifest["plan_content_sha256"] == plan.content_sha256
    assert manifest["file_count"] == len(plan.files)
    assert manifest["total_bytes"] == sum(item.size for item in plan.files)
    assert manifest["files"] == _census(plan)
    assert manifest["routing"]["serving_entrypoint"] == "Ref2VA/"
    assert manifest["routing"]["resident_dit_count"] == 1
    assert manifest["routing"]["supported_tasks"] == ["ref2va", "t2va", "fl2va"]
    digest = hashlib.sha256(
        fileops.canonical_json({k: v for k, v in manifest.items() if k != "package_manifest_sha256"})
    ).hexdigest()
    assert manifest["package_manifest_sha256"] == digest
    assert published.manifest_sha256 == digest
