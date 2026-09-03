"""Manifest-last atomic publication for immutable native package staging.

The manifest-last exclusive write and self-digest semantics are characterized
from the Apache-2.0 ``h3-forge`` package_assembler.py blob
``e64558f1d3bb6e1ee6f714b70e783d9df907f9ce`` at commit
``e9cb011d00b028c149db3978de246c54f6e34acc`` and the ``fsops.py`` blob
``ae40e46eef808f979ee085e806f2380e50b6c01d``. The same-parent atomic rename is
a deliberate strengthening over legacy in-place assembly, which assembled
directly into the destination and left only the manifest as a last-write
commit point.

The host-discovery model index written before the manifest restores the legacy
v3 root ``model_index.json`` behavior (the official
``MiniMaxAI/MiniMax-H3@42ed227e`` Ref2VA index shape with the legacy hybrid
task list), characterized from the ``h3-forge`` ``package_assembler.py`` blob
``e64558f1d3bb6e1ee6f714b70e783d9df907f9ce`` at commit
``e9cb011d00b028c149db3978de246c54f6e34acc`` (the
``_deep_merge(source_index, {"_minimax_h3": {"tasks": HYBRID_TASKS}})`` plus
``_write_exclusive(output_root / "model_index.json", ...)`` semantics).
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import NoReturn, cast

from comfy_omni.artifacts import fileops
from comfy_omni.contracts.models import ContractError
from comfy_omni.conversion.packaging.models import NativePackagePlan, PackageMaterialization, PackagePublication

PACKAGE_PUBLICATION_SCHEMA = "comfy_omni.native_package.publication/v1"

_MODEL_INDEX_NAME = "model_index.json"
"""Host-discovery index emitted into every native package staging tree."""

_MODEL_INDEX_TEMPLATE: Mapping[str, object] = MappingProxyType(
    {
        "_class_name": "MiniMaxH3Pipeline",
        "_diffusers_version": "0.32.2",
        "_minimax_h3": MappingProxyType(
            {
                "partition": "ref2va",
                "sigma_shift_scales": MappingProxyType({"audio": 3.0, "video": 12.0}),
                "schema_version": 1,
                "task_aliases": MappingProxyType({}),
                "tasks": (),
            }
        ),
        "audio_vae": ("diffusers", "MiniMaxH3AudioVAE"),
        "processor": ("transformers", "Qwen3VLProcessor"),
        "scheduler": None,
        "text_encoder": ("transformers", "MiniMaxH3Qwen3VLHFEncoder"),
        "tokenizer": ("transformers", "Qwen2TokenizerFast"),
        "transformer": ("diffusers", "MiniMaxH3DiTModel"),
        "video_vae": ("diffusers", "MiniMaxH3VideoVAE"),
    }
)
"""Frozen host-discovery model-index template; ``tasks`` is injected per plan."""


def _materialize_model_index(tasks: tuple[str, ...]) -> dict[str, object]:
    """Return a plain JSON-ready copy of the template with ``tasks`` injected."""

    def unwrap(value: object) -> object:
        if isinstance(value, MappingProxyType):
            return {key: unwrap(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [unwrap(item) for item in value]
        return value

    index = cast(dict[str, object], unwrap(_MODEL_INDEX_TEMPLATE))
    minimax = cast(dict[str, object], index["_minimax_h3"])
    minimax["tasks"] = list(tasks)
    return index


class PackagePublicationError(ContractError):
    """A stable package-publication refusal."""


def _fail(detail: str, stage: str, **evidence: object) -> NoReturn:
    raise PackagePublicationError(detail, evidence={"stage": stage, **evidence})


def _unchanged_directory(path: Path, identity: tuple[int, int], *, label: str) -> None:
    try:
        status = path.lstat()
        is_directory = path.is_dir()
        linked = fileops.is_link(path)
    except (fileops.FsopsError, OSError) as exc:
        _fail(f"{label} directory cannot be re-inspected", "staging", cause=str(exc))
    if not is_directory or linked or (status.st_dev, status.st_ino) != identity:
        _fail(f"{label} directory identity changed", "staging")


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


def publish_package(plan: NativePackagePlan, materialization: PackageMaterialization) -> PackagePublication:
    """Publish verified native package staging as one same-parent atomic rename."""

    if materialization.plan_content_sha256 != plan.content_sha256:
        _fail("publication plan digest does not match the materialized handle", "plan-binding")

    stage = Path(materialization.stage_dir)
    output = Path(materialization.output_dir)
    _unchanged_directory(stage, materialization.stage_identity, label="package staging")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite package path: {output}")
    try:
        parent = fileops.reject_linked_ancestors(output.parent).resolve(strict=True)
    except (fileops.FsopsError, OSError) as exc:
        _fail("package output parent is missing, linked, or unreadable", "output-binding", cause=str(exc))
    if not parent.is_dir():
        _fail("package output parent is not a directory", "output-binding")
    if parent != stage.parent:
        _fail("package staging and output parents differ", "output-binding")

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
    census: list[dict[str, object]] = []
    for item in plan.files:
        staged = stage.joinpath(*PurePosixPath(item.target_path).parts)
        try:
            digest, size = fileops.sha256_file_pinned(staged)
        except fileops.FsopsError as exc:
            _fail(
                "package staged file could not be hashed",
                "file-verification",
                path=item.target_path,
                cause=str(exc),
            )
        if (digest, size) != (item.sha256, item.size):
            _fail(
                "package staged file size or SHA256 differs from the package plan",
                "file-verification",
                path=item.target_path,
            )
        census.append({"path": item.target_path, "sha256": digest, "size": size})
    if hashlib.sha256(fileops.canonical_json(census)).hexdigest() != materialization.files_sha256:
        _fail("package staged tree digest differs from the materialized handle", "staging-census")

    model_index_bytes = fileops.canonical_json(_materialize_model_index(plan.supported_tasks))
    model_index_sha256 = hashlib.sha256(model_index_bytes).hexdigest()
    try:
        fileops.write_exclusive(stage / _MODEL_INDEX_NAME, model_index_bytes)
    except (fileops.FsopsError, OSError) as exc:
        _fail(
            "package host-discovery model index could not be written to staging",
            "model-index",
            cause=str(exc),
        )

    manifest = {
        "schema": plan.output_schema,
        "plan_content_sha256": plan.content_sha256,
        "tool": plan.tool.to_dict(),
        "host": {"adapter": plan.host_adapter, "commit": plan.host_commit},
        "components": [component.to_dict() for component in plan.components],
        "source_files_sha256": materialization.source_files_sha256,
        "staged_files_sha256": materialization.files_sha256,
        "model_index_sha256": model_index_sha256,
        "files": [{"path": item.target_path, "sha256": item.sha256, "size": item.size} for item in plan.files],
        "file_count": len(plan.files),
        "total_bytes": sum(item.size for item in plan.files),
        "routing": {
            "manifest": plan.manifest_name,
            "serving_entrypoint": plan.serving_entrypoint,
            "resident_dit_count": plan.resident_dit_count,
            "supported_tasks": list(plan.supported_tasks),
        },
    }
    manifest_sha256 = hashlib.sha256(fileops.canonical_json(manifest)).hexdigest()
    manifest["package_manifest_sha256"] = manifest_sha256
    try:
        fileops.write_exclusive(stage / plan.manifest_name, fileops.canonical_json(manifest))
    except (fileops.FsopsError, OSError) as exc:
        _fail("package manifest could not be written to staging", "manifest", cause=str(exc))

    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite package path: {output}")
    try:
        os.rename(stage, output)
    except OSError as exc:
        _fail("package staging could not be renamed to its output", "publication", cause=str(exc))
    fileops.fsync_dir(output.parent)

    return PackagePublication(
        schema=PACKAGE_PUBLICATION_SCHEMA,
        plan_content_sha256=plan.content_sha256,
        manifest_sha256=manifest_sha256,
        file_count=len(plan.files),
        total_bytes=sum(item.size for item in plan.files),
        output_dir=output,
    )


__all__ = [
    "PACKAGE_PUBLICATION_SCHEMA",
    "PackagePublicationError",
    "publish_package",
]
