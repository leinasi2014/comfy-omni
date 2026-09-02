"""Pure, fail-closed planning for immutable native package assembly.

The behavior is characterized from Apache-2.0 ``h3-forge`` package assembly at
commit ``e9cb011d00b028c149db3978de246c54f6e34acc`` and blob
``e64558f1d3bb6e1ee6f714b70e783d9df907f9ce``. This module retains only the
exact-component, same-producer, pinned-host, and safe-path planning contract;
filesystem materialization and runtime/LoRA policy belong to later layers.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from pathlib import PurePosixPath

from comfy_omni.artifacts import fileops
from comfy_omni.contracts.models import ContractError
from comfy_omni.conversion.packaging.models import (
    ComponentFile,
    ComponentReceipt,
    NativePackagePlan,
    PackageComponentPlan,
    PackageFilePlan,
)
from comfy_omni.domain.normalization import ToolIdentity

PACKAGE_PLAN_SCHEMA = "comfy_omni.native_package.plan/v1"
PACKAGE_OUTPUT_SCHEMA = "h3-comfy-package/v3"
PACKAGE_MANIFEST_NAME = "h3-comfy-package.json"
PACKAGE_COMPONENTS = (
    "transformer",
    "text_encoder",
    "video_vae",
    "audio_vae",
    "tokenizer",
    "processor",
)
PACKAGE_TASKS = ("ref2va", "t2va", "fl2va")
PINNED_VLLM_OMNI_COMMIT = "17285c2f55a41bf15772676121814d59a60ace35"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RECEIPT_SCHEMA = re.compile(r"[a-z0-9_.-]+/v[1-9][0-9]*")


class PackagePlanError(ContractError):
    """A stable package-planning refusal."""


def _fail(detail: str, stage: str, **evidence: object) -> None:
    raise PackagePlanError(detail, evidence={"stage": stage, **evidence})


def _source_dir(value: str, component: str) -> None:
    if not isinstance(value, str) or "\\" in value:
        _fail("component source directory is not canonical POSIX", "source-binding", component=component)
    candidate = PurePosixPath(value)
    if candidate.anchor != "/" or str(candidate) != value or any(part in {".", ".."} for part in candidate.parts):
        _fail("component source directory must be absolute and canonical", "source-binding", component=component)


def _file(record: ComponentFile, component: str) -> PurePosixPath:
    if not isinstance(record, ComponentFile):
        _fail("component receipt contains an invalid file record", "file-binding", component=component)
    value = record.path
    if not isinstance(value, str):
        _fail("component file path must be safe canonical relative POSIX", "file-binding", component=component)
    candidate = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or candidate.is_absolute()
        or str(candidate) != value
        or any(part in {".", ".."} for part in candidate.parts)
    ):
        _fail("component file path must be safe canonical relative POSIX", "file-binding", component=component)
    if type(record.size) is not int or record.size < 0 or _SHA256.fullmatch(record.sha256) is None:
        _fail("component file identity is invalid", "file-binding", component=component, path=value)
    return candidate


def _component(receipt: ComponentReceipt) -> tuple[PackageComponentPlan, tuple[PackageFilePlan, ...]]:
    component = receipt.component
    _source_dir(receipt.source_dir, component)
    if (
        not isinstance(receipt.receipt_schema, str)
        or _RECEIPT_SCHEMA.fullmatch(receipt.receipt_schema) is None
        or not isinstance(receipt.receipt_sha256, str)
        or _SHA256.fullmatch(receipt.receipt_sha256) is None
    ):
        _fail("component receipt identity is invalid", "receipt-binding", component=component)
    if not receipt.files:
        _fail("component receipt has no committed files", "receipt-binding", component=component)
    files: list[PackageFilePlan] = []
    seen: set[str] = set()
    if any(not isinstance(item, ComponentFile) for item in receipt.files):
        _fail("component receipt contains an invalid file record", "file-binding", component=component)
    for record in sorted(receipt.files, key=lambda item: item.path):
        relative = _file(record, component).as_posix()
        if relative in seen:
            _fail(
                "component receipt contains a duplicate file path",
                "file-binding",
                component=component,
                path=relative,
            )
        seen.add(relative)
        files.append(
            PackageFilePlan(
                component=component,
                source_path=relative,
                target_path=f"Ref2VA/{component}/{relative}",
                size=record.size,
                sha256=record.sha256,
            )
        )
    return (
        PackageComponentPlan(
            component=component,
            source_dir=receipt.source_dir,
            receipt_schema=receipt.receipt_schema,
            receipt_sha256=receipt.receipt_sha256,
        ),
        tuple(files),
    )


def plan_native_package(
    receipts: tuple[ComponentReceipt, ...],
    *,
    vllm_omni_commit: str,
) -> NativePackagePlan:
    """Return one canonical package authorization without reading or writing files."""

    if vllm_omni_commit != PINNED_VLLM_OMNI_COMMIT:
        _fail("package host contract is not the pinned vLLM-Omni commit", "host-binding")
    if any(not isinstance(item, ComponentReceipt) for item in receipts):
        _fail("package receipts contain an invalid record", "component-census")
    names = tuple(item.component for item in receipts)
    duplicates = sorted(name for name in set(names) if names.count(name) > 1)
    missing = sorted(set(PACKAGE_COMPONENTS) - set(names))
    unknown = sorted(set(names) - set(PACKAGE_COMPONENTS))
    if duplicates or missing or unknown:
        _fail(
            "package component census is not exact",
            "component-census",
            duplicates=duplicates,
            missing=missing,
            unknown=unknown,
        )
    by_component = {item.component: item for item in receipts}
    ordered = tuple(by_component[name] for name in PACKAGE_COMPONENTS)
    tool = ordered[0].tool
    if not isinstance(tool, ToolIdentity) or any(item.tool != tool for item in ordered[1:]):
        _fail("package component producer identities disagree", "producer-binding")
    if tool.distribution != "comfy-omni":
        _fail("package producer is not ComfyOmni", "producer-binding")
    source_dirs = tuple(item.source_dir for item in ordered)
    receipt_digests = tuple(item.receipt_sha256 for item in ordered)
    if len(set(source_dirs)) != len(source_dirs) or len(set(receipt_digests)) != len(receipt_digests):
        _fail("package component authorities are reused", "component-census")

    components: list[PackageComponentPlan] = []
    files: list[PackageFilePlan] = []
    targets: set[str] = set()
    for receipt in ordered:
        component, planned = _component(receipt)
        components.append(component)
        for item in planned:
            if item.target_path in targets:
                _fail("package target path collision", "file-binding", path=item.target_path)
            targets.add(item.target_path)
            files.append(item)

    plan = NativePackagePlan(
        schema=PACKAGE_PLAN_SCHEMA,
        host_adapter="vllm-omni",
        host_commit=vllm_omni_commit,
        output_schema=PACKAGE_OUTPUT_SCHEMA,
        manifest_name=PACKAGE_MANIFEST_NAME,
        serving_entrypoint="Ref2VA/",
        supported_tasks=PACKAGE_TASKS,
        resident_dit_count=1,
        tool=tool,
        components=tuple(components),
        files=tuple(files),
        content_sha256="",
    )
    digest = hashlib.sha256(fileops.canonical_json(plan.to_dict(include_content_sha256=False))).hexdigest()
    return replace(plan, content_sha256=digest)


__all__ = [
    "PACKAGE_COMPONENTS",
    "PACKAGE_MANIFEST_NAME",
    "PACKAGE_OUTPUT_SCHEMA",
    "PACKAGE_PLAN_SCHEMA",
    "PACKAGE_TASKS",
    "PINNED_VLLM_OMNI_COMMIT",
    "PackagePlanError",
    "plan_native_package",
]
