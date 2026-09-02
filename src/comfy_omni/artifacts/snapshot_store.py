"""Immutable publication and explicit loading of contract snapshot stores."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from comfy_omni.artifacts import fileops
from comfy_omni.artifacts.snapshot_schema import (
    ContractSnapshot,
    load_snapshot,
    snapshot_manifest_sha256,
    snapshot_record,
)
from comfy_omni.contracts.models import ContractCatalog, ContractError, ContractRecord


def _discard_owned(target: Path, captured: tuple[int, int, int, int, int], payload: bytes) -> None:
    try:
        current = target.stat()
        same_inode = (current.st_dev, current.st_ino) == captured[:2]
    except OSError:
        return
    if not same_inode:
        return
    if fileops.fd_identity(current) != captured:
        try:
            if target.read_bytes() == payload:
                return
        except OSError:
            return
    try:
        os.chmod(target, 0o644)
        target.unlink()
    except OSError:
        return


def _verify_publish(target: Path, payload: bytes, digest: str, captured: tuple[int, int, int, int, int]) -> None:
    try:
        published, identity = fileops.read_file_pinned(target)
        document = fileops.parse_json_strict(published)
    except fileops.FsopsError as exc:
        _discard_owned(target, captured, payload)
        raise ContractError(
            f"published snapshot could not be re-read: {target}",
            evidence={"stage": "snapshot-verify", "reason": "unreadable"},
        ) from exc
    if snapshot_manifest_sha256(document) != digest or published != payload or identity != captured:
        _discard_owned(target, captured, payload)
        raise ContractError(
            f"published snapshot failed its digest/byte/identity gate: {target}",
            evidence={"stage": "snapshot-verify", "reason": "publication-mismatch"},
        )


def write_snapshot(directory: Path | str, document: Mapping[str, Any]) -> ContractSnapshot:
    """Publish a canonical content-addressed snapshot with O_EXCL and post-write verification."""

    root = Path(directory)
    digest = snapshot_manifest_sha256(document)
    final_document = {**document, "manifest_sha256": digest}
    payload = fileops.canonical_json(final_document)
    target = root / f"{digest}.json"
    try:
        root.mkdir(parents=True, exist_ok=True)
        root = fileops.reject_linked_ancestors(root)
        target = root / f"{digest}.json"
        captured = fileops.write_exclusive(target, payload)
        fileops.fsync_dir(root)
        _verify_publish(target, payload, digest, captured)
    except fileops.FsopsExistsError as exc:
        raise ContractError(
            f"refusing to overwrite existing snapshot: {target}",
            evidence={"stage": "snapshot-write", "path": str(target)},
        ) from exc
    except fileops.FsopsError as exc:
        raise ContractError(
            f"snapshot could not be published: {target}: {exc}",
            evidence={"stage": "snapshot-write", "path": str(target)},
        ) from exc
    return ContractSnapshot(target, digest, MappingProxyType(final_document), payload)


def load_store(directory: Path | str) -> dict[str, ContractRecord]:
    """Load a store into a fresh mapping; no global registry is mutated."""

    root = Path(directory)
    try:
        if fileops.is_link(root) or not root.is_dir():
            raise ContractError(f"contract store must be a non-linked directory: {root}")
        entries = sorted(root.iterdir(), key=lambda item: item.name)
    except (fileops.FsopsError, OSError) as exc:
        raise ContractError(str(exc), evidence={"stage": "store-load", "directory": str(root)}) from exc
    records: dict[str, ContractRecord] = {}
    owners: dict[str, str] = {}
    for path in entries:
        if path.suffix != ".json":
            continue
        record = snapshot_record(load_snapshot(path))
        if record.name in records:
            raise ContractError(
                f"external contract {record.name!r} is pinned twice",
                evidence={"stage": "store-load", "paths": [owners[record.name], str(path)]},
            )
        records[record.name] = record
        owners[record.name] = str(path)
    return records


def catalog_with_store(base: ContractCatalog, directory: Path | str) -> ContractCatalog:
    """Return a new catalog combining an explicit base with a validated external store."""

    return base.extend(load_store(directory))


__all__ = ["catalog_with_store", "load_store", "write_snapshot"]
