"""Exact-tree and pinned-file verification for immutable native package plans.

The exact-tree, digest, and link-refusal behavior is characterized from the
Apache-2.0 ``h3-forge`` package assembler at commit
``e9cb011d00b028c149db3978de246c54f6e34acc`` and blob
``e64558f1d3bb6e1ee6f714b70e783d9df907f9ce``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import NoReturn

from comfy_omni.artifacts import fileops
from comfy_omni.contracts.models import ContractError
from comfy_omni.conversion.packaging.models import (
    ComponentFile,
    ComponentReceipt,
    NativePackagePlan,
    PackageSourceVerification,
)
from comfy_omni.conversion.packaging.planning import (
    PACKAGE_COMPONENTS,
    PackagePlanError,
    plan_native_package,
)

PACKAGE_SOURCE_VERIFICATION_SCHEMA = "comfy_omni.native_package.source_verification/v1"


class PackageSourceVerificationError(ContractError):
    """A stable package-source verification refusal."""


def _fail(detail: str, stage: str, **evidence: object) -> NoReturn:
    raise PackageSourceVerificationError(detail, evidence={"stage": stage, **evidence})


def _receipts(plan: NativePackagePlan) -> tuple[ComponentReceipt, ...]:
    by_component: dict[str, list[ComponentFile]] = {component: [] for component in PACKAGE_COMPONENTS}
    for item in plan.files:
        records = by_component.get(item.component)
        if records is None:
            _fail("package plan contains an unknown file component", "plan-binding", component=item.component)
        records.append(ComponentFile(item.source_path, item.size, item.sha256))
    components = {item.component: item for item in plan.components}
    if len(components) != len(plan.components) or set(components) != set(PACKAGE_COMPONENTS):
        _fail("package plan component census is invalid", "plan-binding")
    return tuple(
        ComponentReceipt(
            component=component,
            source_dir=components[component].source_dir,
            receipt_schema=components[component].receipt_schema,
            receipt_sha256=components[component].receipt_sha256,
            tool=plan.tool,
            files=tuple(by_component[component]),
        )
        for component in PACKAGE_COMPONENTS
    )


def _verify_plan(plan: NativePackagePlan) -> None:
    if not isinstance(plan, NativePackagePlan):
        _fail("package source verification requires a native package plan", "plan-binding")
    try:
        rebuilt = plan_native_package(_receipts(plan), vllm_omni_commit=plan.host_commit)
    except (PackagePlanError, AttributeError, KeyError, TypeError, ValueError) as exc:
        _fail("package plan cannot be reconstructed", "plan-binding", cause=str(exc))
    if rebuilt != plan:
        _fail("package plan fields or content SHA256 drifted", "plan-binding")


def _tree_files(root: Path, component: str) -> set[str]:
    result: set[str] = set()

    def visit(directory: Path, prefix: str) -> None:
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
        except OSError as exc:
            _fail("component directory cannot be enumerated", "tree-census", component=component, cause=str(exc))
        for entry in entries:
            relative = f"{prefix}/{entry.name}" if prefix else entry.name
            path = Path(entry.path)
            try:
                if fileops.is_link(path):
                    _fail("component tree contains a link", "tree-census", component=component, path=relative)
                if entry.is_dir(follow_symlinks=False):
                    visit(path, relative)
                elif entry.is_file(follow_symlinks=False):
                    result.add(relative)
                else:
                    _fail("component tree contains a special entry", "tree-census", component=component, path=relative)
            except (fileops.FsopsError, OSError) as exc:
                _fail(
                    "component entry changed during census",
                    "tree-census",
                    component=component,
                    path=relative,
                    cause=str(exc),
                )

    visit(root, "")
    return result


def _root(value: str, component: str) -> Path:
    try:
        root = fileops.reject_linked_ancestors(Path(value))
    except fileops.FsopsError as exc:
        _fail(
            "component source path is missing, linked, or unreadable",
            "tree-census",
            component=component,
            cause=str(exc),
        )
    if not root.is_dir():
        _fail("component source path is not a directory", "tree-census", component=component)
    return root


def verify_package_sources(plan: NativePackagePlan) -> PackageSourceVerification:
    """Re-hash the exact source trees named by ``plan`` without writing output."""

    _verify_plan(plan)
    component_plans = {item.component: item for item in plan.components}
    planned_files = {component: [] for component in PACKAGE_COMPONENTS}
    for item in plan.files:
        planned_files[item.component].append(item)
    census: list[dict[str, object]] = []
    total_bytes = 0
    for component in PACKAGE_COMPONENTS:
        root = _root(component_plans[component].source_dir, component)
        expected = {item.source_path for item in planned_files[component]}
        before = _tree_files(root, component)
        if before != expected:
            _fail(
                "component tree file census differs from the package plan",
                "tree-census",
                component=component,
                missing=sorted(expected - before),
                unexpected=sorted(before - expected),
            )
        for item in planned_files[component]:
            path = root / item.source_path
            try:
                digest, size = fileops.sha256_file_pinned(path)
            except fileops.FsopsError as exc:
                _fail(
                    "component file changed or could not be hashed",
                    "file-verification",
                    component=component,
                    path=item.source_path,
                    cause=str(exc),
                )
            if (digest, size) != (item.sha256, item.size):
                _fail(
                    "component file size or SHA256 differs from the package plan",
                    "file-verification",
                    component=component,
                    path=item.source_path,
                )
            census.append(
                {
                    "component": component,
                    "path": item.source_path,
                    "sha256": digest,
                    "size": size,
                }
            )
            total_bytes += size
        if _tree_files(root, component) != before:
            _fail("component tree changed during verification", "tree-census", component=component)
    files_sha256 = hashlib.sha256(fileops.canonical_json(census)).hexdigest()
    return PackageSourceVerification(
        schema=PACKAGE_SOURCE_VERIFICATION_SCHEMA,
        plan_content_sha256=plan.content_sha256,
        tool=plan.tool,
        component_count=len(plan.components),
        file_count=len(plan.files),
        total_bytes=total_bytes,
        files_sha256=files_sha256,
    )


__all__ = [
    "PACKAGE_SOURCE_VERIFICATION_SCHEMA",
    "PackageSourceVerificationError",
    "verify_package_sources",
]
