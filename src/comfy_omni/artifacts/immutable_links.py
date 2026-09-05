"""Explicit immutable storage sharing; payload descriptors are always read-only.

This is a new ComfyOmni implementation following the existing pinned-copy protocol.
Hardlink creation changes link count/ctime, so it is a metadata write operation.
It requires a shared writable mount containing distinct source/output subtrees.
Ordinary conversion and runtime callers continue to use read-only model mounts.
"""

from __future__ import annotations

import errno
import hashlib
import os
import stat
from contextlib import contextmanager
from pathlib import Path

from comfy_omni.artifacts import fileops


def _require(condition, message):
    if not condition:
        raise fileops.FsopsModifiedError(message)


def _identity(info):
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_ctime_ns,
        info.st_nlink,
    )


@contextmanager
def _directory(path):
    fileops.reject_linked_ancestors(path)
    before = path.lstat()
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        _require(
            (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino), "storage directory changed during open"
        )
        yield descriptor
        after = path.lstat()
        _require((after.st_dev, after.st_ino) == (opened.st_dev, opened.st_ino), "storage directory path changed")
    finally:
        os.close(descriptor)


def _entry(directory, name):
    info = os.stat(name, dir_fd=directory, follow_symlinks=False)
    _require(stat.S_ISREG(info.st_mode), "immutable reuse requires a regular file")
    return _identity(info)


def _digest(descriptor):
    os.lseek(descriptor, 0, os.SEEK_SET)
    result = hashlib.sha256()
    while chunk := os.read(descriptor, fileops.HASH_CHUNK_BYTES):
        result.update(chunk)
    return result.hexdigest()


def _stable(source_fd, directory, name, expected):
    _require(_identity(os.fstat(source_fd)) == expected, "immutable source descriptor changed")
    _require(_entry(directory, name) == expected, "immutable source path changed")


def _link(source_fd, source_directory, source_name, target_directory, target_name, before):
    digest = _digest(source_fd)
    _stable(source_fd, source_directory, source_name, before)
    try:
        os.link(
            source_name, target_name, src_dir_fd=source_directory, dst_dir_fd=target_directory, follow_symlinks=False
        )
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        _stable(source_fd, source_directory, source_name, before)
        return None
    linked = _identity(os.fstat(source_fd))
    _require(linked[:7] == before[:7] and linked[8] == before[8] + 1, "immutable source changed during link")
    _require(_entry(target_directory, target_name) == linked, "new storage link points to another inode")
    _stable(source_fd, source_directory, source_name, linked)
    _require(_digest(source_fd) == digest, "immutable payload changed during link")
    _stable(source_fd, source_directory, source_name, linked)
    _require(_entry(target_directory, target_name) == linked, "new storage link changed during verification")
    os.fsync(target_directory)
    return digest, before[2], True


def _copy(source_fd, source_directory, source_name, target_directory, target_name, before, max_copy_bytes):
    if max_copy_bytes is not None and (
        type(max_copy_bytes) is not int or max_copy_bytes < 0 or before[2] > max_copy_bytes
    ):
        raise fileops.FsopsIoError("exclusive copy exceeds its allocation budget")
    _stable(source_fd, source_directory, source_name, before)
    target_fd = os.open(target_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444, dir_fd=target_directory)
    digest = hashlib.sha256()
    total = 0
    try:
        os.lseek(source_fd, 0, os.SEEK_SET)
        while chunk := os.read(source_fd, fileops.HASH_CHUNK_BYTES):
            total += len(chunk)
            _require(total <= before[2], "immutable source grew while copying")
            digest.update(chunk)
            remaining = memoryview(chunk)
            while remaining:
                written = os.write(target_fd, remaining)
                if written <= 0:
                    raise OSError("short write while copying immutable file")
                remaining = remaining[written:]
        _require(total == before[2], "immutable source shrank while copying")
        os.fsync(target_fd)
        target_identity = _identity(os.fstat(target_fd))
        _require(_entry(target_directory, target_name) == target_identity, "copy target changed during write")
        _stable(source_fd, source_directory, source_name, before)
    finally:
        os.close(target_fd)
    read_fd = os.open(target_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=target_directory)
    try:
        _stable(read_fd, target_directory, target_name, target_identity)
        _require(_digest(read_fd) == digest.hexdigest(), "copied payload differs from immutable source")
        _stable(read_fd, target_directory, target_name, target_identity)
        _stable(source_fd, source_directory, source_name, before)
        os.fsync(target_directory)
    finally:
        os.close(read_fd)
    return digest.hexdigest(), total, False


def reuse_file_pinned_exclusive(
    source: Path, destination: Path, *, max_copy_bytes: int | None = None
) -> tuple[str, int, bool]:
    """Share an immutable inode, or copy across mounts/filesystems.

    Return digest, size and whether storage was shared. Source write permissions
    are forbidden even when copying is necessary. No chmod, replacement or
    payload write is performed on a source. An existing destination refuses.
    Failed exclusive targets remain inside the caller's unpublished staging.
    """
    if os.name != "posix" or not hasattr(os, "O_NOFOLLOW"):
        raise fileops.FsopsIoError("immutable reuse requires POSIX directory descriptors; use default copying")
    if max_copy_bytes is not None and (type(max_copy_bytes) is not int or max_copy_bytes < 0):
        raise fileops.FsopsIoError("invalid immutable copy allocation budget")
    source = fileops.reject_linked_ancestors(source).absolute()
    destination = destination.absolute()
    try:
        with _directory(source.parent) as source_directory, _directory(destination.parent) as target_directory:
            before = _entry(source_directory, source.name)
            _require(before[4] & 0o222 == 0, "immutable reuse refuses writable sources")
            source_fd = os.open(source.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=source_directory)
            try:
                _stable(source_fd, source_directory, source.name, before)
                if before[0] == os.fstat(target_directory).st_dev:
                    result = _link(source_fd, source_directory, source.name, target_directory, destination.name, before)
                    if result is not None:
                        return result
                return _copy(
                    source_fd, source_directory, source.name, target_directory, destination.name, before, max_copy_bytes
                )
            finally:
                os.close(source_fd)
    except FileExistsError as exc:
        raise fileops.FsopsExistsError(f"refusing to overwrite existing file: {destination}") from exc
    except OSError as exc:
        raise fileops.FsopsIoError(f"immutable storage materialization failed: {exc}") from exc
