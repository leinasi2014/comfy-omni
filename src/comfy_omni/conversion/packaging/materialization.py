"""Private, exact-file staging for immutable native package plans.

The exclusive bounded-copy, link-refusal, readback, and manifest-less failure
behavior is characterized from the Apache-2.0 ``h3-forge`` package assembler
at commit ``e9cb011d00b028c149db3978de246c54f6e34acc`` and blob
``e64558f1d3bb6e1ee6f714b70e783d9df907f9ce``.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import NoReturn

from comfy_omni.artifacts import fileops
from comfy_omni.artifacts.immutable_links import reuse_file_pinned_exclusive
from comfy_omni.contracts.models import ContractError
from comfy_omni.conversion.packaging.models import NativePackagePlan, PackageMaterialization
from comfy_omni.conversion.packaging.verification import verify_package_sources

PACKAGE_MATERIALIZATION_SCHEMA = "comfy_omni.native_package.materialization/v1"


class PackageMaterializationError(ContractError):
    """A stable package-staging refusal."""


def _fail(detail: str, stage: str, **evidence: object) -> NoReturn:
    raise PackageMaterializationError(detail, evidence={"stage": stage, **evidence})


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _prepare_output(plan: NativePackagePlan, output_dir: Path) -> tuple[Path, Path, tuple[int, int]]:
    try:
        output = fileops.reject_linked_ancestors(Path(output_dir), allow_missing_final=True)
        parent = fileops.reject_linked_ancestors(output.parent).resolve(strict=True)
    except (fileops.FsopsError, OSError) as exc:
        _fail("package output path is missing, linked, or unreadable", "output-binding", cause=str(exc))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite package path: {output}")
    if not parent.is_dir():
        _fail("package output parent is not a directory", "output-binding")
    for component in plan.components:
        try:
            source = fileops.reject_linked_ancestors(Path(component.source_dir)).resolve(strict=True)
        except (fileops.FsopsError, OSError) as exc:
            _fail(
                "package component source is missing, linked, or unreadable",
                "output-binding",
                component=component.component,
                cause=str(exc),
            )
        if _inside(output, source) or _inside(source, output):
            _fail(
                "package output and component source paths overlap",
                "output-binding",
                component=component.component,
            )
    try:
        status = parent.stat()
    except OSError as exc:
        _fail("package output parent cannot be inspected", "output-binding", cause=str(exc))
    return output, parent, (status.st_dev, status.st_ino)


def _tree_files(root: Path) -> set[str]:
    result: set[str] = set()

    def visit(directory: Path, prefix: str) -> None:
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
        except OSError as exc:
            _fail("package staging directory cannot be enumerated", "staging-census", cause=str(exc))
        for entry in entries:
            relative = f"{prefix}/{entry.name}" if prefix else entry.name
            path = Path(entry.path)
            try:
                if fileops.is_link(path):
                    _fail("package staging contains a link", "staging-census", path=relative)
                if entry.is_dir(follow_symlinks=False):
                    visit(path, relative)
                elif entry.is_file(follow_symlinks=False):
                    result.add(relative)
                else:
                    _fail("package staging contains a special entry", "staging-census", path=relative)
            except (fileops.FsopsError, OSError) as exc:
                _fail(
                    "package staging entry changed during census",
                    "staging-census",
                    path=relative,
                    cause=str(exc),
                )

    visit(root, "")
    return result


def _unchanged_directory(path: Path, identity: tuple[int, int], *, label: str) -> None:
    try:
        status = path.lstat()
        is_directory = path.is_dir()
        linked = fileops.is_link(path)
    except (fileops.FsopsError, OSError) as exc:
        _fail(f"{label} directory cannot be re-inspected", "staging", cause=str(exc))
    if not is_directory or linked or (status.st_dev, status.st_ino) != identity:
        _fail(f"{label} directory identity changed", "staging")


def materialize_package(
    plan: NativePackagePlan,
    output_dir: Path | str,
    *,
    reuse_immutable: bool = False,
    max_copy_bytes: int | None = None,
) -> PackageMaterialization:
    """Materialize private staging, optionally sharing immutable source inodes.

    Default copying and its serialized result remain unchanged. Reuse requires
    read-only source permissions and metadata write access to a shared mount.
    Cross-mount/filesystem fallback copies must fit an optional explicit budget.
    """

    if type(reuse_immutable) is not bool or (
        max_copy_bytes is not None and (type(max_copy_bytes) is not int or max_copy_bytes < 0)
    ):
        _fail("invalid package storage policy", "storage-policy")

    output, parent, parent_identity = _prepare_output(plan, Path(output_dir))
    source_verification = verify_package_sources(plan)
    _unchanged_directory(parent, parent_identity, label="package parent")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite package path: {output}")
    try:
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=parent))
        stage_status = stage.lstat()
    except OSError as exc:
        _fail("private package staging directory could not be created", "staging", cause=str(exc))
    stage_identity = (stage_status.st_dev, stage_status.st_ino)
    component_roots = {item.component: Path(item.source_dir) for item in plan.components}
    census: list[dict[str, object]] = []
    total_bytes = 0
    shared_bytes = 0
    copied_bytes = 0
    try:
        for item in plan.files:
            source = component_roots[item.component].joinpath(*PurePosixPath(item.source_path).parts)
            target = stage.joinpath(*PurePosixPath(item.target_path).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            remaining_copy_bytes = None if max_copy_bytes is None else max_copy_bytes - copied_bytes
            if reuse_immutable:
                digest, size, shared = reuse_file_pinned_exclusive(source, target, max_copy_bytes=remaining_copy_bytes)
            else:
                if remaining_copy_bytes is None:
                    digest, size = fileops.copy_file_pinned_exclusive(source, target)
                else:
                    digest, size = fileops.copy_file_pinned_exclusive(source, target, max_bytes=remaining_copy_bytes)
                shared = False
            if (digest, size) != (item.sha256, item.size):
                _fail(
                    "materialized file differs from the package plan",
                    "file-copy",
                    path=item.target_path,
                    staging_dir=stage.as_posix(),
                )
            census.append({"path": item.target_path, "sha256": digest, "size": size})
            total_bytes += size
            shared_bytes += size if shared else 0
            copied_bytes += 0 if shared else size
        expected = {item.target_path for item in plan.files}
        observed = _tree_files(stage)
        if observed != expected:
            _fail(
                "package staging file census differs from the package plan",
                "staging-census",
                missing=sorted(expected - observed),
                unexpected=sorted(observed - expected),
                staging_dir=stage.as_posix(),
            )
        _unchanged_directory(stage, stage_identity, label="package staging")
        _unchanged_directory(parent, parent_identity, label="package parent")
        if output.exists() or output.is_symlink():
            _fail("package output appeared during staging", "output-binding", staging_dir=stage.as_posix())
    except (fileops.FsopsError, OSError) as exc:
        _fail(
            "package file copy failed",
            "file-copy",
            cause=str(exc),
            staging_dir=stage.as_posix(),
        )
    return PackageMaterialization(
        schema=PACKAGE_MATERIALIZATION_SCHEMA,
        plan_content_sha256=plan.content_sha256,
        source_files_sha256=source_verification.files_sha256,
        stage_dir=stage,
        output_dir=output,
        stage_identity=stage_identity,
        file_count=len(census),
        total_bytes=total_bytes,
        files_sha256=hashlib.sha256(fileops.canonical_json(census)).hexdigest(),
        reuse_immutable=reuse_immutable,
        shared_bytes=shared_bytes,
    )


__all__ = [
    "PACKAGE_MATERIALIZATION_SCHEMA",
    "PackageMaterializationError",
    "materialize_package",
]
