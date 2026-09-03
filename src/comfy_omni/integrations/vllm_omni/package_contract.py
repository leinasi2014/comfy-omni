"""Fail-closed verification of a published vLLM-Omni Ref2VA runtime package.

The local-package resolution and the fail-closed package verification chain are
characterized from the Apache-2.0 ``h3-forge`` ``h3/runtime_pipeline.py`` blob
``fa94f86da746ff9a11105584081464c1162d07b6`` at commit
``e9cb011d00b028c149db3978de246c54f6e34acc`` (the ``_converted_partition_path``
local-package resolution and the fail-closed package verification chain) plus the
already-migrated packaging verification semantics in
:mod:`comfy_omni.conversion.packaging.verification` and
:mod:`comfy_omni.conversion.packaging.publication`. The validator is
host-free by design: it never imports ``vllm_omni``, ``torch``, ``vllm``, or
``fastapi``.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

from comfy_omni.artifacts import fileops
from comfy_omni.contracts.models import ContractError
from comfy_omni.conversion.packaging.planning import PACKAGE_COMPONENTS

MANIFEST_NAME = "h3-comfy-package.json"
MODEL_INDEX_NAME = "model_index.json"
OUTPUT_SCHEMA = "h3-comfy-package/v3"


class RuntimePackageContractError(ContractError):
    """A stable runtime package verification refusal."""


def _fail(detail: str, stage: str, **evidence: object) -> NoReturn:
    raise RuntimePackageContractError(detail, evidence={"stage": stage, **evidence})


def _tree_files(root: Path) -> set[str]:
    """Census the regular files below ``root`` as sorted POSIX-relative paths.

    Symlinks and special entries are refused fail-closed; enumeration errors are
    refused as ``tree-census`` exactly like publication's ``_tree_files``.
    """

    result: set[str] = set()

    def visit(directory: Path, prefix: str) -> None:
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
        except OSError as exc:
            _fail("runtime package directory cannot be enumerated", "tree-census", cause=str(exc))
        for entry in entries:
            relative = f"{prefix}/{entry.name}" if prefix else entry.name
            path = Path(entry.path)
            try:
                if fileops.is_link(path):
                    _fail("runtime package tree contains a link", "tree-census", path=relative)
                if entry.is_dir(follow_symlinks=False):
                    visit(path, relative)
                elif entry.is_file(follow_symlinks=False):
                    result.add(relative)
                else:
                    _fail(
                        "runtime package tree contains a special entry",
                        "tree-census",
                        path=relative,
                    )
            except (fileops.FsopsError, OSError) as exc:
                _fail(
                    "runtime package entry changed during census",
                    "tree-census",
                    path=relative,
                    cause=str(exc),
                )

    visit(root, "")
    return result


def _check_census_records(files: list[object]) -> None:
    """Bind every manifest census record to a canonical path/digest/size object."""

    for record in files:
        if not isinstance(record, dict):
            _fail("runtime package manifest contains an invalid file record", "manifest")
        path = record.get("path")
        digest = record.get("sha256")
        size = record.get("size")
        if not isinstance(path, str) or not path or "\\" in path:
            _fail("runtime package manifest file path is invalid", "manifest", path=path)
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            _fail("runtime package manifest file digest is invalid", "manifest", path=path)
        if type(size) is not int or size < 0:
            _fail("runtime package manifest file size is invalid", "manifest", path=path)


@dataclass(frozen=True)
class RuntimePackageContract:
    """The verified, immutable identity of one validated runtime package."""

    package_root: Path
    plan_content_sha256: str
    manifest_sha256: str
    model_index_sha256: str
    schema: str
    class_name: str
    partition: str
    supported_tasks: tuple[str, ...]
    components: tuple[tuple[str, int, int], ...]
    file_count: int
    total_bytes: int
    files_sha256: str
    component_paths: dict[str, Path]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready binding record for this verified package."""

        return {
            "status": "RUNTIME_VERIFIED",
            "package_root": self.package_root.as_posix(),
            "plan_content_sha256": self.plan_content_sha256,
            "manifest_sha256": self.manifest_sha256,
            "model_index_sha256": self.model_index_sha256,
            "schema": self.schema,
            "class_name": self.class_name,
            "partition": self.partition,
            "supported_tasks": list(self.supported_tasks),
            "components": [
                {
                    "component": component,
                    "file_count": file_count,
                    "total_bytes": total_bytes,
                }
                for component, file_count, total_bytes in self.components
            ],
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "files_sha256": self.files_sha256,
            "component_paths": {component: path.as_posix() for component, path in self.component_paths.items()},
        }


def validate_runtime_package(
    package_root: Path | str,
    *,
    expected_class_name: str = "MiniMaxH3Pipeline",
) -> RuntimePackageContract:
    """Verify a published runtime package and return its bindable contract.

    Refusals raise :class:`RuntimePackageContractError` with structured
    ``evidence["stage"]`` in this fail-closed order: ``package-binding``,
    ``model-index``, ``manifest``, ``routing``, ``tree-census``,
    ``file-verification``, and ``components``.
    """

    try:
        root = fileops.reject_linked_ancestors(Path(package_root)).resolve(strict=True)
    except (fileops.FsopsError, OSError) as exc:
        _fail("runtime package path is missing, linked, or unreadable", "package-binding", cause=str(exc))
    if not root.is_dir():
        _fail("runtime package root is not a directory", "package-binding")

    index_path = root / MODEL_INDEX_NAME
    try:
        index_bytes = index_path.read_bytes()
    except OSError as exc:
        _fail("runtime package model index is missing or unreadable", "model-index", cause=str(exc))
    try:
        parsed_index = fileops.parse_json_strict(index_bytes)
    except fileops.FsopsError as exc:
        _fail("runtime package model index is not strict JSON", "model-index", cause=str(exc))
    if index_bytes != fileops.canonical_json(parsed_index):
        _fail("runtime package model index is not canonical", "model-index")
    if not isinstance(parsed_index, dict):
        _fail("runtime package model index is not an object", "model-index")
    class_name = parsed_index.get("_class_name")
    if class_name != expected_class_name:
        _fail(
            "runtime package class name differs from the expected host pipeline",
            "model-index",
            class_name=class_name,
        )
    minimax = parsed_index.get("_minimax_h3")
    if not isinstance(minimax, dict) or minimax.get("partition") != "ref2va":
        _fail("runtime package model index partition is not the Ref2VA partition", "model-index")
    index_tasks = minimax.get("tasks")
    if (
        not isinstance(index_tasks, list)
        or not index_tasks
        or any(not isinstance(task, str) or not task for task in index_tasks)
    ):
        _fail("runtime package model index task list is empty or invalid", "model-index")

    manifest_path = root / MANIFEST_NAME
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        _fail("runtime package manifest is missing or unreadable", "manifest", cause=str(exc))
    try:
        manifest = fileops.parse_json_strict(manifest_bytes)
    except fileops.FsopsError as exc:
        _fail("runtime package manifest is not strict JSON", "manifest", cause=str(exc))
    if not isinstance(manifest, dict):
        _fail("runtime package manifest is not an object", "manifest")
    if manifest.get("schema") != OUTPUT_SCHEMA:
        _fail(
            "runtime package schema differs from the output contract",
            "manifest",
            schema=manifest.get("schema"),
        )
    routing = manifest.get("routing")
    if (
        not isinstance(routing, dict)
        or routing.get("serving_entrypoint") != "Ref2VA/"
        or routing.get("resident_dit_count") != 1
        or not isinstance(routing.get("supported_tasks"), list)
        or not routing["supported_tasks"]
        or any(not isinstance(task, str) or not task for task in routing["supported_tasks"])
    ):
        _fail("runtime package manifest routing block is invalid", "manifest")
    routing_tasks = routing["supported_tasks"]
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        _fail("runtime package manifest file census is invalid", "manifest")
    _check_census_records(files)
    self_digest = hashlib.sha256(
        fileops.canonical_json({key: value for key, value in manifest.items() if key != "package_manifest_sha256"})
    ).hexdigest()
    if manifest.get("package_manifest_sha256") != self_digest:
        _fail("runtime package manifest self-digest does not bind", "manifest")
    if manifest.get("model_index_sha256") != hashlib.sha256(index_bytes).hexdigest():
        _fail("runtime package manifest does not bind the model index", "manifest")
    recorded_total = sum(record["size"] for record in files)
    if manifest.get("file_count") != len(files) or manifest.get("total_bytes") != recorded_total:
        _fail("runtime package manifest file counts do not match its census", "manifest")

    if sorted(index_tasks) != sorted(routing_tasks):
        _fail(
            "runtime package index tasks differ from the manifest routing tasks",
            "routing",
            index_tasks=index_tasks,
            routing_tasks=routing_tasks,
        )

    expected = {record["path"] for record in files} | {MANIFEST_NAME, MODEL_INDEX_NAME}
    observed = _tree_files(root)
    if observed != expected:
        _fail(
            "runtime package tree file census differs from the manifest",
            "tree-census",
            missing=sorted(expected - observed),
            unexpected=sorted(observed - expected),
        )

    component_totals: dict[str, tuple[int, int]] = {}
    seen_components: set[str] = set()
    for record in files:
        parts = PurePosixPath(record["path"]).parts
        if len(parts) < 3 or parts[0] != "Ref2VA":
            _fail(
                "runtime package file path is outside the Ref2VA component tree",
                "components",
                path=record["path"],
            )
        component = parts[1]
        if component not in PACKAGE_COMPONENTS:
            _fail(
                "runtime package file names an unknown component",
                "components",
                component=component,
                path=record["path"],
            )
        try:
            digest, size = fileops.sha256_file_pinned(root.joinpath(*parts))
        except fileops.FsopsError as exc:
            _fail(
                "runtime package file changed or could not be hashed",
                "file-verification",
                path=record["path"],
                cause=str(exc),
            )
        if (digest, size) != (record["sha256"], record["size"]):
            _fail(
                "runtime package file size or SHA256 differs from the manifest",
                "file-verification",
                path=record["path"],
            )
        seen_components.add(component)
        count, total = component_totals.get(component, (0, 0))
        component_totals[component] = (count + 1, total + size)
    if seen_components != set(PACKAGE_COMPONENTS):
        _fail(
            "runtime package component census is invalid",
            "components",
            missing=sorted(set(PACKAGE_COMPONENTS) - seen_components),
            unexpected=sorted(seen_components - set(PACKAGE_COMPONENTS)),
        )

    components = tuple((component, *component_totals[component]) for component in PACKAGE_COMPONENTS)
    return RuntimePackageContract(
        package_root=root,
        plan_content_sha256=manifest["plan_content_sha256"],
        manifest_sha256=self_digest,
        model_index_sha256=hashlib.sha256(index_bytes).hexdigest(),
        schema=manifest["schema"],
        class_name=class_name,
        partition="ref2va",
        supported_tasks=tuple(routing_tasks),
        components=components,
        file_count=len(files),
        total_bytes=recorded_total,
        files_sha256=hashlib.sha256(fileops.canonical_json(files)).hexdigest(),
        component_paths={component: (root / "Ref2VA" / component).resolve() for component in PACKAGE_COMPONENTS},
    )


__all__ = [
    "MANIFEST_NAME",
    "MODEL_INDEX_NAME",
    "OUTPUT_SCHEMA",
    "RuntimePackageContract",
    "RuntimePackageContractError",
    "validate_runtime_package",
]
