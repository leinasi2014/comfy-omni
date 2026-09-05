"""Fresh-directory, manifest-last publication for native checkpoint exports.

Publication invariants are characterized from Apache-2.0 ``h3-forge`` sources
``native_export.py@475cee5523be64e5b24a95e16c5de3f371cbdf67`` and
``fsops.py@ae40e46eef808f979ee085e806f2380e50b6c01d``.
"""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from comfy_omni.artifacts import fileops
from comfy_omni.contracts.models import ContractError

MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class StagedArtifact:
    name: str
    size: int
    sha256: str
    kind: str
    tensor_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "kind": self.kind,
            "name": self.name,
            "sha256": self.sha256,
            "size": self.size,
        }
        if self.tensor_count is not None:
            value["tensor_count"] = self.tensor_count
        return value


@dataclass(frozen=True)
class NativeExportStage:
    path: Path
    parent: Path
    parent_identity: tuple[int, int]
    output_dir: Path


@dataclass(frozen=True)
class NativeExportPublication:
    output_dir: Path
    manifest_path: Path
    manifest_sha256: str


def _regular_file(path: Path) -> None:
    status = path.lstat()
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise ContractError(
            f"native export staging contains a non-regular file: {path.name}",
            evidence={"stage": "publication"},
        )


def _safe_name(name: str) -> None:
    if (
        not name
        or "\0" in name
        or "/" in name
        or "\\" in name
        or Path(name).name != name
        or name in {".", "..", MANIFEST_NAME}
    ):
        raise ContractError(f"unsafe staged artifact name: {name!r}", evidence={"stage": "publication"})


def prepare_native_export(output_dir: Path) -> NativeExportStage:
    """Resolve a fresh output location and create a private sibling staging directory."""

    requested = fileops.reject_linked_ancestors(Path(output_dir), allow_missing_final=True)
    parent = fileops.reject_linked_ancestors(requested.parent).resolve(strict=True)
    resolved = parent / requested.name
    if resolved.exists() or resolved.is_symlink():
        raise FileExistsError(f"refusing to overwrite output path: {resolved}")
    status = parent.stat()
    identity = (status.st_dev, status.st_ino)
    stage = Path(tempfile.mkdtemp(prefix=f".{resolved.name}.stage-", dir=parent))
    current = parent.stat()
    if (current.st_dev, current.st_ino) != identity:
        raise ContractError("output parent identity changed while staging", evidence={"stage": "publication"})
    return NativeExportStage(stage, parent, identity, resolved)


def stage_document(stage: NativeExportStage, name: str, payload: bytes, *, kind: str) -> StagedArtifact:
    """Write and independently rehash one canonical sidecar in private staging."""

    _safe_name(name)
    path = stage.path / name
    fileops.write_exclusive(path, payload)
    digest, size = fileops.sha256_file_pinned(path)
    if digest != hashlib.sha256(payload).hexdigest() or size != len(payload):
        raise ContractError(f"staged document verification failed: {name}", evidence={"stage": "staging"})
    return StagedArtifact(name, size, digest, kind)


def publish_native_export(
    stage: NativeExportStage,
    artifacts: tuple[StagedArtifact, ...],
    unsigned_manifest: dict[str, Any],
    *,
    before_manifest: Callable[[], None] | None = None,
) -> NativeExportPublication:
    """Publish verified artifacts without overwrite, then write the receipt last."""

    names = tuple(item.name for item in artifacts)
    if len(names) != len(set(names)):
        raise ContractError("duplicate staged artifact names", evidence={"stage": "publication"})
    for name in names:
        _safe_name(name)
    parent_status = stage.parent.stat()
    if (parent_status.st_dev, parent_status.st_ino) != stage.parent_identity:
        raise ContractError("output parent identity changed before publication", evidence={"stage": "publication"})
    if stage.output_dir.exists() or stage.output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite output path: {stage.output_dir}")
    try:
        stage.output_dir.mkdir(mode=0o755)
        claimed = stage.output_dir.lstat()
        output_identity = (claimed.st_dev, claimed.st_ino)
        for artifact in sorted(artifacts, key=lambda item: item.name):
            staged_path = stage.path / artifact.name
            _regular_file(staged_path)
            os.link(staged_path, stage.output_dir / artifact.name)
        current = stage.output_dir.lstat()
        if (current.st_dev, current.st_ino) != output_identity:
            raise ContractError("output directory identity changed during publication")
        for artifact in artifacts:
            digest, size = fileops.sha256_file_pinned(stage.output_dir / artifact.name)
            if (digest, size) != (artifact.sha256, artifact.size):
                raise ContractError(f"published artifact verification failed: {artifact.name}")
        if before_manifest is not None:
            try:
                before_manifest()
            except BaseException:
                # Remove only this call's exclusive directory and its own staged links.
                # A foreign replacement must never be removed during failure cleanup.
                current = stage.output_dir.lstat()
                if (current.st_dev, current.st_ino) == output_identity:
                    for artifact in artifacts:
                        target = stage.output_dir / artifact.name
                        linked, staged = target.lstat(), (stage.path / artifact.name).lstat()
                        if (linked.st_dev, linked.st_ino) == (staged.st_dev, staged.st_ino):
                            target.unlink()
                    if not any(stage.output_dir.iterdir()):
                        stage.output_dir.rmdir()
                    fileops.fsync_dir(stage.parent)
                raise
        manifest_sha256 = hashlib.sha256(fileops.canonical_json(unsigned_manifest)).hexdigest()
        manifest = {**unsigned_manifest, "manifest_sha256": manifest_sha256}
        manifest_path = stage.output_dir / MANIFEST_NAME
        payload = fileops.canonical_json(manifest)
        fileops.write_exclusive(manifest_path, payload)
        observed, _ = fileops.read_file_pinned(manifest_path)
        if observed != payload or fileops.parse_json_strict(observed) != manifest:
            raise ContractError("published manifest verification failed")
        fileops.fsync_dir(stage.output_dir)
    except FileExistsError:
        raise
    except ContractError:
        raise
    except OSError as exc:
        raise ContractError(
            f"native export publication failed: {exc}",
            evidence={"stage": "publication"},
        ) from exc
    return NativeExportPublication(stage.output_dir, manifest_path, manifest_sha256)


__all__ = [
    "MANIFEST_NAME",
    "NativeExportPublication",
    "NativeExportStage",
    "StagedArtifact",
    "prepare_native_export",
    "publish_native_export",
    "stage_document",
]
