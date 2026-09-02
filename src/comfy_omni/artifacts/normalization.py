"""Bounded, no-overwrite publication for digest-pinned normalization."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol

from comfy_omni.artifacts.safetensors import read_safetensors_header
from comfy_omni.domain.normalization import (
    ArtifactIdentity,
    NormalizationError,
    NormalizationProfile,
    NormalizationReceipt,
    ToolIdentity,
)

COPY_CHUNK_BYTES = 8 * 1024 * 1024


class _Digest(Protocol):
    def update(self, value: bytes) -> None: ...


@dataclass(frozen=True)
class NormalizationPublication:
    """Published artifact, commit-marker receipt, and portable receipt value."""

    artifact_path: Path
    receipt_path: Path
    receipt: NormalizationReceipt


@dataclass(frozen=True)
class _ResolvedPaths:
    source: Path
    parent: Path
    destination: Path
    receipt: Path


@dataclass(frozen=True)
class _VerifiedArtifact:
    source: ArtifactIdentity
    removed_suffix: ArtifactIdentity
    derived: ArtifactIdentity
    tensor_count: int


def _canonical_receipt(receipt: NormalizationReceipt) -> bytes:
    return (json.dumps(receipt.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _receipt_path(destination: Path) -> Path:
    return destination.with_name(f"{destination.name}.normalization.json")


def _temporary_path(destination: Path, kind: str) -> Path:
    return destination.parent / f".{destination.name}.{kind}.{secrets.token_hex(16)}.tmp"


def _open_exclusive(path: Path) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    return os.open(path, flags, 0o600)


def _open_source(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    return os.open(path, flags)


def _is_link_or_reparse(path: Path) -> bool:
    try:
        attributes = os.lstat(path)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(attributes.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(getattr(attributes, "st_file_attributes", 0) & reparse_flag)


def _reject_link_components(path: Path, reason_code: str) -> None:
    absolute = path.absolute()
    for candidate in (*reversed(absolute.parents), absolute):
        if _is_link_or_reparse(candidate):
            raise NormalizationError(
                reason_code,
                f"symbolic-link or reparse-point path component is forbidden: {candidate}",
            )


def _write_all(stream: BinaryIO, value: bytes) -> None:
    view = memoryview(value)
    while view:
        written = stream.write(view)
        if written is None or written <= 0:
            raise OSError("normalization staging write made no progress")
        view = view[written:]


def _copy_exact_prefix(
    source: BinaryIO,
    target: BinaryIO,
    prefix_bytes: int,
    source_digest: _Digest,
    derived_digest: _Digest,
) -> None:
    remaining = prefix_bytes
    while remaining:
        chunk = source.read(min(COPY_CHUNK_BYTES, remaining))
        if not chunk:
            raise NormalizationError(
                "normalization-source-truncated",
                "source ended before the authorized derived prefix",
            )
        source_digest.update(chunk)
        derived_digest.update(chunk)
        _write_all(target, chunk)
        remaining -= len(chunk)


def _read_exact_suffix(
    source: BinaryIO,
    suffix_bytes: int,
    source_digest: _Digest,
    suffix_digest: _Digest,
) -> None:
    remaining = suffix_bytes
    while remaining:
        chunk = source.read(min(COPY_CHUNK_BYTES, remaining))
        if not chunk:
            raise NormalizationError(
                "normalization-source-truncated",
                "source ended before the authorized suffix",
            )
        source_digest.update(chunk)
        suffix_digest.update(chunk)
        remaining -= len(chunk)
    if source.read(1):
        raise NormalizationError(
            "normalization-source-extended",
            "source contains bytes beyond the authorized suffix",
        )


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unlink_if_owned(staging: Path, published: Path) -> None:
    try:
        if staging.exists() and published.exists() and os.path.samefile(staging, published):
            published.unlink()
    except FileNotFoundError:
        pass


def _resolve_paths(source: Path, destination: Path) -> _ResolvedPaths:
    _reject_link_components(source, "normalization-source-link-forbidden")
    _reject_link_components(destination.parent, "normalization-destination-link-forbidden")
    resolved_source = source.resolve(strict=True)
    resolved_parent = destination.parent.resolve(strict=True)
    resolved_destination = resolved_parent / destination.name
    resolved_receipt = _receipt_path(resolved_destination)
    if resolved_destination.suffix.lower() != ".safetensors":
        raise NormalizationError(
            "normalization-invalid-destination",
            "destination must end with .safetensors",
        )
    if resolved_source == resolved_destination:
        raise NormalizationError(
            "normalization-source-destination-collision",
            "source and destination must be different paths",
        )
    if resolved_destination.exists():
        raise NormalizationError(
            "normalization-destination-exists",
            f"destination already exists: {resolved_destination}",
        )
    if resolved_receipt.exists():
        raise NormalizationError(
            "normalization-receipt-exists",
            f"receipt already exists: {resolved_receipt}",
        )
    return _ResolvedPaths(resolved_source, resolved_parent, resolved_destination, resolved_receipt)


def _stream_to_staging(
    source_descriptor: int,
    artifact_descriptor: int,
    profile: NormalizationProfile,
) -> tuple[str, str, str]:
    source_digest = hashlib.sha256()
    derived_digest = hashlib.sha256()
    suffix_digest = hashlib.sha256()
    with os.fdopen(source_descriptor, "rb", closefd=False) as source_stream:
        with os.fdopen(artifact_descriptor, "wb", closefd=False) as artifact_stream:
            _copy_exact_prefix(
                source_stream,
                artifact_stream,
                profile.derived.bytes,
                source_digest,
                derived_digest,
            )
            _read_exact_suffix(
                source_stream,
                profile.removed_suffix.bytes,
                source_digest,
                suffix_digest,
            )
            artifact_stream.flush()
            os.fsync(artifact_descriptor)
    return source_digest.hexdigest(), suffix_digest.hexdigest(), derived_digest.hexdigest()


def _stage_profiled_bytes(
    source: Path,
    staging: Path,
    profile: NormalizationProfile,
) -> tuple[os.stat_result, os.stat_result, ArtifactIdentity, ArtifactIdentity, ArtifactIdentity]:
    source_descriptor = _open_source(source)
    try:
        before = os.fstat(source_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise NormalizationError("normalization-source-not-regular", "source must be a regular file")
        if before.st_size != profile.source.bytes:
            raise NormalizationError(
                "normalization-source-size-mismatch",
                f"source has {before.st_size} bytes; expected {profile.source.bytes}",
            )
        artifact_descriptor = _open_exclusive(staging)
        try:
            source_sha256, suffix_sha256, derived_sha256 = _stream_to_staging(
                source_descriptor,
                artifact_descriptor,
                profile,
            )
        finally:
            os.close(artifact_descriptor)
        staging.chmod(0o644)
        after = os.fstat(source_descriptor)
    finally:
        os.close(source_descriptor)
    return (
        before,
        after,
        ArtifactIdentity(before.st_size, source_sha256),
        ArtifactIdentity(profile.removed_suffix.bytes, suffix_sha256),
        ArtifactIdentity(profile.derived.bytes, derived_sha256),
    )


def _require_identity(observed: ArtifactIdentity, expected: ArtifactIdentity, kind: str) -> None:
    if observed != expected:
        raise NormalizationError(
            f"normalization-{kind}-digest-mismatch",
            f"{kind} SHA256 is {observed.sha256}; expected {expected.sha256}",
        )


def _verify_staged_artifact(source: Path, staging: Path, profile: NormalizationProfile) -> _VerifiedArtifact:
    before, after, observed_source, observed_suffix, observed_derived = _stage_profiled_bytes(
        source,
        staging,
        profile,
    )
    if _file_identity(before) != _file_identity(after):
        raise NormalizationError(
            "normalization-source-changed",
            "source identity changed during normalization",
        )
    _require_identity(observed_source, profile.source, "source")
    _require_identity(observed_suffix, profile.removed_suffix, "suffix")
    _require_identity(observed_derived, profile.derived, "derived")
    try:
        _, tensors = read_safetensors_header(staging)
    except (OSError, ValueError) as exc:
        raise NormalizationError(
            "normalization-strict-reread-failed",
            f"derived artifact failed strict safetensors reread: {exc}",
        ) from exc
    return _VerifiedArtifact(observed_source, observed_suffix, observed_derived, len(tensors))


def _stage_receipt(path: Path, receipt: NormalizationReceipt) -> None:
    receipt_descriptor = _open_exclusive(path)
    try:
        with os.fdopen(receipt_descriptor, "wb", closefd=False) as receipt_stream:
            _write_all(receipt_stream, _canonical_receipt(receipt))
            receipt_stream.flush()
            os.fsync(receipt_descriptor)
    finally:
        os.close(receipt_descriptor)
    path.chmod(0o644)


def _publish_staging(
    artifact_staging: Path,
    receipt_staging: Path,
    paths: _ResolvedPaths,
) -> None:
    try:
        os.link(artifact_staging, paths.destination)
        os.link(receipt_staging, paths.receipt)
        _fsync_directory(paths.parent)
    except OSError as exc:
        _unlink_if_owned(receipt_staging, paths.receipt)
        _unlink_if_owned(artifact_staging, paths.destination)
        _fsync_directory(paths.parent)
        raise NormalizationError(
            "normalization-publication-failed",
            f"no-overwrite publication failed: {exc}",
        ) from exc


def normalize_digest_pinned_prefix(
    source: Path,
    destination: Path,
    profile: NormalizationProfile,
    tool: ToolIdentity,
) -> NormalizationPublication:
    """Remove one exact suffix and atomically publish artifact plus receipt marker."""

    paths = _resolve_paths(source, destination)
    artifact_staging = _temporary_path(paths.destination, "artifact")
    receipt_staging = _temporary_path(paths.destination, "receipt")
    try:
        verified = _verify_staged_artifact(paths.source, artifact_staging, profile)
        receipt = NormalizationReceipt(
            profile_id=profile.profile_id,
            profile_version=profile.profile_version,
            source=verified.source,
            removed_suffix=verified.removed_suffix,
            derived=verified.derived,
            strict_reread_tensor_count=verified.tensor_count,
            tool=tool,
        )
        _stage_receipt(receipt_staging, receipt)
        _publish_staging(artifact_staging, receipt_staging, paths)
        return NormalizationPublication(paths.destination, paths.receipt, receipt)
    finally:
        for staging in (artifact_staging, receipt_staging):
            try:
                staging.unlink()
            except FileNotFoundError:
                pass


__all__ = ["COPY_CHUNK_BYTES", "NormalizationPublication", "normalize_digest_pinned_prefix"]
