"""Component receipts: exact-tree authority for immutable native package plans.

The exact-tree census, link/special refusal, deterministic ordering, and
pinned-hold hashing are characterized from the Apache-2.0 ``h3-forge``
package_assembler.py blob ``e64558f1d3bb6e1ee6f714b70e783d9df907f9ce``
(exact tree census, link/special refusal, deterministic order) and
``fsops.py`` blob ``ae40e46eef808f979ee085e806f2380e50b6c01d`` (pinned
held-descriptor hashing, canonical JSON) at commit
``e9cb011d00b028c149db3978de246c54f6e34acc``. The ``ComponentReceipt`` value
is the ComfyOmni slice-1 planning contract.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import NoReturn

from comfy_omni.artifacts import fileops
from comfy_omni.contracts.models import ContractError
from comfy_omni.conversion.packaging.models import ComponentFile, ComponentReceipt
from comfy_omni.conversion.packaging.planning import PACKAGE_COMPONENTS
from comfy_omni.domain.normalization import ToolIdentity

RECEIPT_SCHEMA = "comfy_omni.component_receipt/v1"


class ComponentReceiptError(ContractError):
    """A stable component-receipt refusal."""


def _fail(detail: str, stage: str, **evidence: object) -> NoReturn:
    raise ComponentReceiptError(detail, evidence={"stage": stage, **evidence})


def _tree_files(root: Path, component: str) -> set[str]:
    result: set[str] = set()

    def visit(directory: Path, prefix: str) -> None:
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
        except OSError as exc:
            _fail("component directory cannot be enumerated", "census", component=component, cause=str(exc))
        for entry in entries:
            relative = f"{prefix}/{entry.name}" if prefix else entry.name
            path = Path(entry.path)
            try:
                if fileops.is_link(path):
                    _fail("component tree contains a link", "census", component=component, path=relative)
                if entry.is_dir(follow_symlinks=False):
                    visit(path, relative)
                elif entry.is_file(follow_symlinks=False):
                    result.add(relative)
                else:
                    _fail("component tree contains a special entry", "census", component=component, path=relative)
            except (fileops.FsopsError, OSError) as exc:
                _fail(
                    "component entry changed during census",
                    "census",
                    component=component,
                    path=relative,
                    cause=str(exc),
                )

    visit(root, "")
    return result


def _verify_unchanged(root: Path, component: str, before: set[str], files: list[ComponentFile]) -> None:
    after = _tree_files(root, component)
    if after != before:
        _fail(
            "component tree changed during receipt",
            "census-recheck",
            component=component,
            missing=sorted(before - after),
            unexpected=sorted(after - before),
        )
    for record in files:
        path = root / record.path
        try:
            digest, size = fileops.sha256_file_pinned(path)
        except fileops.FsopsError as exc:
            _fail(
                "component file changed during receipt",
                "census-recheck",
                component=component,
                path=record.path,
                cause=str(exc),
            )
        if (digest, size) != (record.sha256, record.size):
            _fail("component file changed during receipt", "census-recheck", component=component, path=record.path)


def parse_component_receipt(component: str, source_dir: Path | str, tool: ToolIdentity) -> ComponentReceipt:
    """Build one content-bound component receipt from ``source_dir`` without writing."""

    if not isinstance(tool, ToolIdentity) or tool.distribution != "comfy-omni":
        _fail("component receipt tool is not ComfyOmni", "tool-binding")
    if component not in PACKAGE_COMPONENTS:
        _fail("component is not a package component", "component-binding", component=component)
    try:
        root = fileops.reject_linked_ancestors(Path(source_dir)).resolve(strict=True)
    except (fileops.FsopsError, OSError) as exc:
        _fail(
            "component source is missing, linked, or unreadable",
            "source-binding",
            component=component,
            cause=str(exc),
        )
    if not root.is_dir():
        _fail("component source is not a directory", "source-binding", component=component)
    before = _tree_files(root, component)
    if not before:
        _fail("component tree has no files", "census", component=component)
    files: list[ComponentFile] = []
    for relative in sorted(before):
        path = root / relative
        try:
            digest, size = fileops.sha256_file_pinned(path)
        except fileops.FsopsError as exc:
            _fail(
                "component file changed or could not be hashed",
                "file-hashing",
                component=component,
                path=relative,
                cause=str(exc),
            )
        files.append(ComponentFile(relative, size, digest))
    files.sort(key=lambda item: item.path)
    _verify_unchanged(root, component, before, files)
    source = root.as_posix()
    content = {
        "component": component,
        "schema": RECEIPT_SCHEMA,
        "source_dir": source,
        "tool": tool.to_dict(),
        "files": [{"path": item.path, "sha256": item.sha256, "size": item.size} for item in files],
    }
    return ComponentReceipt(
        component=component,
        source_dir=source,
        receipt_schema=RECEIPT_SCHEMA,
        receipt_sha256=hashlib.sha256(fileops.canonical_json(content)).hexdigest(),
        tool=tool,
        files=tuple(files),
    )


__all__ = [
    "ComponentReceiptError",
    "RECEIPT_SCHEMA",
    "parse_component_receipt",
]
