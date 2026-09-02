from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from comfy_omni.artifacts import fileops
from comfy_omni.contracts.models import ContractError
from comfy_omni.domain.normalization import ToolIdentity


def _receipts():
    from comfy_omni.conversion.packaging.models import ComponentFile, ComponentReceipt
    from comfy_omni.conversion.packaging.planning import PACKAGE_COMPONENTS

    tool = ToolIdentity("comfy-omni", "0.2.0a1", "a" * 40, "b" * 64)
    return tuple(
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
        for component in PACKAGE_COMPONENTS
    )


def test_package_plan_is_canonical_complete_and_bound_to_one_producer() -> None:
    from comfy_omni.conversion.packaging.planning import (
        PACKAGE_COMPONENTS,
        PACKAGE_PLAN_SCHEMA,
        PINNED_VLLM_OMNI_COMMIT,
        plan_native_package,
    )

    receipts = tuple(reversed(_receipts()))

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


@pytest.mark.parametrize("mode", ["missing", "duplicate", "unknown"])
def test_package_plan_requires_the_exact_component_census(mode: str) -> None:
    from comfy_omni.conversion.packaging.planning import PINNED_VLLM_OMNI_COMMIT, plan_native_package

    receipts = _receipts()
    if mode == "missing":
        receipts = receipts[:-1]
    elif mode == "duplicate":
        receipts = (*receipts, receipts[0])
    else:
        receipts = (*receipts[:-1], replace(receipts[-1], component="unknown"))

    with pytest.raises(ContractError, match="component census") as failure:
        plan_native_package(receipts, vllm_omni_commit=PINNED_VLLM_OMNI_COMMIT)

    assert failure.value.evidence["stage"] == "component-census"


def test_package_plan_rejects_mixed_or_reused_component_authorities() -> None:
    from comfy_omni.conversion.packaging.planning import PINNED_VLLM_OMNI_COMMIT, plan_native_package

    receipts = _receipts()
    mixed = replace(receipts[-1], tool=replace(receipts[-1].tool, source_commit="c" * 40))
    with pytest.raises(ContractError, match="producer identities"):
        plan_native_package((*receipts[:-1], mixed), vllm_omni_commit=PINNED_VLLM_OMNI_COMMIT)

    reused = replace(receipts[-1], receipt_sha256=receipts[0].receipt_sha256)
    with pytest.raises(ContractError, match="authorities are reused"):
        plan_native_package((*receipts[:-1], reused), vllm_omni_commit=PINNED_VLLM_OMNI_COMMIT)


@pytest.mark.parametrize("path", ["", "/absolute", "../escape", "a/../escape", "a\\foreign"])
def test_package_plan_rejects_unsafe_component_file_paths(path: str) -> None:
    from comfy_omni.conversion.packaging.planning import PINNED_VLLM_OMNI_COMMIT, plan_native_package

    receipts = _receipts()
    changed_file = replace(receipts[0].files[0], path=path)
    changed = replace(receipts[0], files=(changed_file,))

    with pytest.raises(ContractError, match="safe canonical relative POSIX") as failure:
        plan_native_package((changed, *receipts[1:]), vllm_omni_commit=PINNED_VLLM_OMNI_COMMIT)

    assert failure.value.evidence["stage"] == "file-binding"


def test_package_plan_rejects_duplicate_files_bad_receipts_and_wrong_host() -> None:
    from comfy_omni.conversion.packaging.planning import PINNED_VLLM_OMNI_COMMIT, plan_native_package

    receipts = _receipts()
    duplicate = replace(receipts[0], files=(receipts[0].files[0], receipts[0].files[0]))
    with pytest.raises(ContractError, match="duplicate file"):
        plan_native_package((duplicate, *receipts[1:]), vllm_omni_commit=PINNED_VLLM_OMNI_COMMIT)

    bad_receipt = replace(receipts[0], receipt_sha256="not-a-digest")
    with pytest.raises(ContractError, match="receipt identity"):
        plan_native_package((bad_receipt, *receipts[1:]), vllm_omni_commit=PINNED_VLLM_OMNI_COMMIT)

    with pytest.raises(ContractError, match="not the pinned") as failure:
        plan_native_package(receipts, vllm_omni_commit="d" * 40)
    assert failure.value.evidence["stage"] == "host-binding"


@pytest.mark.parametrize("source_dir", ["relative", "/a/../escape", "/a\\foreign"])
def test_package_plan_rejects_noncanonical_source_directories(source_dir: str) -> None:
    from comfy_omni.conversion.packaging.planning import PINNED_VLLM_OMNI_COMMIT, plan_native_package

    receipts = _receipts()
    changed = replace(receipts[0], source_dir=source_dir)
    with pytest.raises(ContractError, match="source directory") as failure:
        plan_native_package((changed, *receipts[1:]), vllm_omni_commit=PINNED_VLLM_OMNI_COMMIT)
    assert failure.value.evidence["stage"] == "source-binding"


@pytest.mark.parametrize(("size", "digest"), [(-1, "a" * 64), (1, "bad")])
def test_package_plan_rejects_invalid_file_identities(size: int, digest: str) -> None:
    from comfy_omni.conversion.packaging.planning import PINNED_VLLM_OMNI_COMMIT, plan_native_package

    receipts = _receipts()
    changed_file = replace(receipts[0].files[0], size=size, sha256=digest)
    changed = replace(receipts[0], files=(changed_file,))
    with pytest.raises(ContractError, match="file identity") as failure:
        plan_native_package((changed, *receipts[1:]), vllm_omni_commit=PINNED_VLLM_OMNI_COMMIT)
    assert failure.value.evidence["stage"] == "file-binding"
