from __future__ import annotations

import hashlib
from pathlib import Path

from comfy_omni.artifacts import fileops
from comfy_omni.conversion.packaging.models import ComponentFile
from comfy_omni.conversion.packaging.planning import (
    PACKAGE_COMPONENTS,
    PINNED_VLLM_OMNI_COMMIT,
    plan_native_package,
)
from comfy_omni.conversion.packaging.verification import verify_package_sources
from comfy_omni.domain.normalization import ToolIdentity

TOOL = ToolIdentity("comfy-omni", "0.2.0a1", "a" * 40, "b" * 64)


def _component_payloads(component: str) -> dict[str, bytes]:
    payloads = {"nested/artifact.bin": f"{component}:payload".encode()}
    if component == "transformer":
        payloads["top.bin"] = b"transformer:top"
    return payloads


def _expected_files(component: str) -> tuple[ComponentFile, ...]:
    return tuple(
        ComponentFile(path, len(payload), hashlib.sha256(payload).hexdigest())
        for path, payload in sorted(_component_payloads(component).items())
    )


def _build_sources(tmp_path: Path) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for component in PACKAGE_COMPONENTS:
        source = tmp_path / "sources" / component
        for relative, payload in _component_payloads(component).items():
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        sources[component] = source
    return sources


def test_parse_component_receipt_builds_plannable_census_from_real_directory(tmp_path: Path) -> None:
    from comfy_omni.conversion.packaging.receipts import (
        RECEIPT_SCHEMA,
        parse_component_receipt,
    )

    sources = _build_sources(tmp_path)
    receipts = tuple(
        parse_component_receipt(component, sources[component].resolve().as_posix(), TOOL)
        for component in PACKAGE_COMPONENTS
    )
    by_component = {receipt.component: receipt for receipt in receipts}

    for component in PACKAGE_COMPONENTS:
        receipt = by_component[component]
        assert receipt.component == component
        assert receipt.receipt_schema == RECEIPT_SCHEMA
        assert receipt.source_dir == sources[component].resolve().as_posix()
        assert receipt.tool == TOOL
        assert receipt.files == _expected_files(component)

    expected_files = _expected_files("transformer")
    expected_sha = hashlib.sha256(
        fileops.canonical_json(
            {
                "component": "transformer",
                "schema": RECEIPT_SCHEMA,
                "source_dir": sources["transformer"].resolve().as_posix(),
                "tool": TOOL.to_dict(),
                "files": [{"path": item.path, "sha256": item.sha256, "size": item.size} for item in expected_files],
            }
        )
    ).hexdigest()
    assert by_component["transformer"].receipt_sha256 == expected_sha

    receipts_in_order = tuple(by_component[component] for component in PACKAGE_COMPONENTS)
    plan = plan_native_package(receipts_in_order, vllm_omni_commit=PINNED_VLLM_OMNI_COMMIT)
    verification = verify_package_sources(plan)
    assert verification.plan_content_sha256 == plan.content_sha256
    assert verification.file_count == 7
