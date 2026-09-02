from __future__ import annotations

import hashlib

from comfy_omni.artifacts import fileops
from comfy_omni.domain.normalization import ToolIdentity


def test_package_plan_is_canonical_complete_and_bound_to_one_producer() -> None:
    from comfy_omni.conversion.packaging.models import ComponentFile, ComponentReceipt
    from comfy_omni.conversion.packaging.planning import (
        PACKAGE_COMPONENTS,
        PACKAGE_PLAN_SCHEMA,
        PINNED_VLLM_OMNI_COMMIT,
        plan_native_package,
    )

    tool = ToolIdentity("comfy-omni", "0.2.0a1", "a" * 40, "b" * 64)
    receipts = tuple(
        ComponentReceipt(
            component=component,
            source_dir=f"/verified/{component}",
            receipt_schema="test.component.receipt/v1",
            receipt_sha256=hashlib.sha256(component.encode()).hexdigest(),
            tool=tool,
            files=(
                ComponentFile(
                    "config.json",
                    len(component),
                    hashlib.sha256(f"{component}:config".encode()).hexdigest(),
                ),
            ),
        )
        for component in reversed(PACKAGE_COMPONENTS)
    )

    first = plan_native_package(receipts, vllm_omni_commit=PINNED_VLLM_OMNI_COMMIT)
    second = plan_native_package(tuple(reversed(receipts)), vllm_omni_commit=PINNED_VLLM_OMNI_COMMIT)

    assert first == second
    assert first.schema == PACKAGE_PLAN_SCHEMA
    assert tuple(item.component for item in first.components) == PACKAGE_COMPONENTS
    assert tuple(item.target_path for item in first.files) == tuple(
        f"Ref2VA/{component}/config.json" for component in PACKAGE_COMPONENTS
    )
    assert first.to_dict()["host"] == {
        "adapter": "vllm-omni",
        "commit": PINNED_VLLM_OMNI_COMMIT,
    }
    assert first.to_dict()["target"] == {
        "manifest": "h3-comfy-package.json",
        "output_schema": "h3-comfy-package/v3",
        "resident_dit_count": 1,
        "serving_entrypoint": "Ref2VA/",
        "supported_tasks": ["ref2va", "t2va", "fl2va"],
    }
    expected = hashlib.sha256(fileops.canonical_json(first.to_dict(include_content_sha256=False))).hexdigest()
    assert first.content_sha256 == expected
