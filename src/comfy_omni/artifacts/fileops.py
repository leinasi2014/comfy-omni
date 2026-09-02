"""Low-level immutable artifact filesystem primitives for ComfyOmni.

Derived from Apache-2.0 h3-forge fsops.py blob ae40e46eef808f979ee085e806f2380e50b6c01d
at commit e9cb011d00b028c149db3978de246c54f6e34acc.

This is the convergence target for the six duplicated fs/hash helper families
(``native_export``, ``package_assembler``, ``vae_export``,
``converter/package_v6``, ``converter/key_rewrite``, ``oracle/fingerprint``).
It is a **zero-dependency leaf**: it imports nothing from ``comfy_omni`` and only
stdlib modules, so every domain layer may depend on it without cycles.

P1a did NOT wire any existing caller to this module; the behavioral differences
between the legacy families are locked by ``tests/test_fsops_characterization.py``
first, and P1b migrates consumers one per commit (each consumer keeps its
private helper names as thin boundary adapters that convert ``FsopsError`` back
to the legacy domain error types -- the locked divergences are the contract and
are deliberately preserved).

Every raised error carries the offending filesystem location as ``.path``
(``Path | None``) so domain adapters can rebuild their legacy message texts
without parsing exception strings.

Typed error mapping (deliberately the fail-closed union of the legacy
behaviors -- see the characterization matrix for the divergences):

=====================  =========================  ====================================
primitive              legacy behavior            fsops behavior
=====================  =========================  ====================================
``is_link`` missing    pkg_v6 ``False`` / others  ``FsopsMissingPathError``
                       raw ``FileNotFoundError``
``is_link`` lstat err   pkg_v6 ``False`` / others  ``FsopsIoError`` (never swallowed)
                       raw ``OSError``
``reject_linked_...``  raw ``FileNotFoundError``  ``FsopsMissingPathError``
missing                or wrapped domain error
``write_exclusive``    raw ``FileExistsError``    ``FsopsExistsError``
collision
=====================  =========================  ====================================
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

__all__ = [
    "HASH_CHUNK_BYTES",
    "FsopsError",
    "FsopsExistsError",
    "FsopsIoError",
    "FsopsJsonError",
    "FsopsLinkError",
    "FsopsMissingPathError",
    "FsopsModifiedError",
    "canonical_json",
    "fd_identity",
    "fsync_dir",
    "is_link",
    "parse_json_strict",
    "path_inside",
    "read_file_pinned",
    "read_json",
    "reject_linked_ancestors",
    "sha256_file",
    "sha256_file_pinned",
    "write_exclusive",
]

HASH_CHUNK_BYTES = 8 * 1024 * 1024
"""Read size for streaming hashes (matches the export-path copy chunk)."""

_READ_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
"""Read-only open flags for pinned reads.

``O_NOFOLLOW`` is the POSIX leaf-link guard; **Windows has no such bit**
(recorded here) and ``os.open`` follows junctions/reparse points there, so
callers that must refuse linked leaves keep their own pre-open
:func:`is_link` walk -- the descriptor identity checks below are the rest
of the net.
"""

_REPARSE_POINT = 0x400
"""``FILE_ATTRIBUTE_REPARSE_POINT`` (covers symlinks, junctions, other reparse)."""

_WRITE_MODE = 0o444
"""Mode for exclusively created files: read-only by contract, never rewritten."""


class FsopsError(RuntimeError):
    """Base type for every fsops contract failure.

    Instances carry the offending filesystem location as ``.path``
    (``Path | None``) so callers can report or re-wrap with full context.
    """

    path: Path | None = None


class FsopsIoError(FsopsError):
    """A raw ``OSError`` escaped a primitive and was wrapped with context."""


class FsopsMissingPathError(FsopsIoError):
    """The path (or one of its components) does not exist."""


class FsopsExistsError(FsopsIoError):
    """An exclusive (``O_EXCL``) create collided with an existing file."""


class FsopsLinkError(FsopsError):
    """A symlink/junction/reparse component appeared where links are forbidden."""


class FsopsModifiedError(FsopsError):
    """The path or its open descriptor moved while being read (TOCTOU rejected).

    Raised by the open-first/verify-later pinned-read primitives
    (:func:`read_file_pinned`, :func:`sha256_file_pinned`) when the path named
    a different file across the open, the descriptor identity changed during
    the single read pass, or the path no longer names the read file after it.
    """


class FsopsJsonError(FsopsError):
    """A JSON document could not be read or parsed.

    Chained (``__cause__``) from the underlying ``UnicodeError`` /
    ``json.JSONDecodeError`` / ``ValueError`` so domain adapters can rebuild
    their legacy message texts verbatim.
    """


def canonical_json(value: Any) -> bytes:
    """Serialize ``value`` to the repo-wide canonical JSON byte form.

    Sorted keys, compact separators, ``ensure_ascii=False`` (raw UTF-8), and a
    trailing newline.  All four legacy ``_canonical_json`` copies produce
    exactly these bytes; the digests embedded in manifests depend on it.
    """
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def read_json(path: Path) -> Any:
    """Read one JSON document from ``path`` (lenient parse semantics).

    Plain ``json.loads`` semantics -- duplicate keys silently collapse (last
    wins) and ``NaN``/``Infinity`` are accepted -- matching the legacy
    ``package_assembler._read_json`` / ``package_v6._read_json`` contract (D5);
    use :func:`parse_json_strict` where those defects must fail instead.
    Raises ``FsopsIoError`` (chained from the ``OSError``/``UnicodeError``)
    when the file cannot be read or decoded and ``FsopsJsonError`` (chained
    from the ``json.JSONDecodeError``) on invalid JSON.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        error = FsopsIoError(f"cannot read JSON file: {path}: {exc}")
        error.path = path
        raise error from exc
    try:
        return json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        error = FsopsJsonError(f"invalid JSON in {path}: {exc}")
        error.path = path
        raise error from exc


def parse_json_strict(raw: str | bytes) -> Any:
    """Parse one JSON document rejecting duplicate keys and non-standard constants.

    The strict D5 variant (the ``vae_export._parse_unique_json`` contract):
    ``{"a": 1, "a": 2}`` fails with ``duplicate key 'a'`` and ``NaN`` /
    ``Infinity`` fail with ``non-standard JSON constant 'NaN'`` instead of
    silently collapsing.  Raises ``FsopsJsonError`` chained from the underlying
    ``ValueError``/``JSONDecodeError``/``UnicodeError``.
    """

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value!r}")

    try:
        return json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise FsopsJsonError(f"invalid strict JSON: {exc}") from exc


def sha256_file(path: Path) -> str:
    """Stream-hash one regular file; never loads it whole.

    Raises ``FsopsMissingPathError`` when the file is absent and ``FsopsIoError``
    for any other OS-level failure (directory passed in, permissions, I/O).
    """
    digest = hashlib.sha256()
    try:
        with path.open("rb", buffering=0) as stream:
            while chunk := stream.read(HASH_CHUNK_BYTES):
                digest.update(chunk)
    except FileNotFoundError as exc:
        error = FsopsMissingPathError(f"cannot hash missing file: {path}")
        error.path = path
        raise error from exc
    except OSError as exc:
        error = FsopsIoError(f"cannot hash file: {path}: {exc}")
        error.path = path
        raise error from exc
    return digest.hexdigest()


def fd_identity(status: os.stat_result) -> tuple[int, int, int, int, int]:
    """The five-field descriptor identity ``(dev, ino, size, mtime_ns, ctime_ns)``.

    Comparing only ``(dev, ino, size)`` cannot distinguish an honest read
    from a same-size in-place rewrite -- the same file object, the same
    length, different bytes.  ``mtime_ns`` moves on every in-place rewrite
    (user-resettable, but the leg that catches the probe); ``ctime_ns`` is
    the kernel-maintained metadata-change time on POSIX and the creation
    time on current Windows interpreters, where it anchors identity but is
    not bumped by rewrites.  Any field moving between a pre-read snapshot
    and a post-read re-check means the file changed under the descriptor.
    Timestamp resolution is filesystem-dependent and two writes inside one
    update tick can share a timestamp, so the timestamp legs are best-effort
    against sub-tick rewrites -- the ``dev/ino/size`` legs and the callers'
    digest comparisons carry the rest.
    """

    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns)


def _path_identity(status: os.stat_result) -> tuple[int, int, int]:
    return (status.st_dev, status.st_ino, status.st_size)


def _read_chunk(descriptor: int, count: int) -> bytes:
    """Read one chunk from a pinned descriptor (seam for TOCTOU tests)."""

    return os.read(descriptor, count)


def _modified_error(path: Path, message: str) -> FsopsModifiedError:
    error = FsopsModifiedError(message)
    error.path = path
    return error


def _open_pinned(path: Path) -> tuple[int, os.stat_result]:
    """lstat + open read-only + fstat with the path->fd identity check.

    Returns ``(descriptor, opened_stat)``; the descriptor is open and the
    caller owes the close.  Raises :class:`FsopsModifiedError` when the path
    named a different ``(dev, ino, size)`` file across the open (an open fd
    cannot be repointed by later path games; the swappable path can), and
    the typed IO errors otherwise.
    """

    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        error = FsopsMissingPathError(f"cannot inspect missing file: {path}")
        error.path = path
        raise error from exc
    except OSError as exc:
        error = FsopsIoError(f"cannot inspect file: {path}: {exc}")
        error.path = path
        raise error from exc
    try:
        descriptor = os.open(path, _READ_OPEN_FLAGS)
    except FileNotFoundError as exc:
        error = FsopsMissingPathError(f"cannot open missing file: {path}")
        error.path = path
        raise error from exc
    except OSError as exc:
        error = FsopsIoError(f"cannot open file: {path}: {exc}")
        error.path = path
        raise error from exc
    try:
        opened = os.fstat(descriptor)
    except OSError as exc:
        os.close(descriptor)
        error = FsopsIoError(f"cannot fstat open file: {path}: {exc}")
        error.path = path
        raise error from exc
    if not stat.S_ISREG(opened.st_mode):
        os.close(descriptor)
        error = FsopsIoError(f"not a regular file: {path}")
        error.path = path
        raise error
    if _path_identity(before) != _path_identity(opened):
        os.close(descriptor)
        raise _modified_error(path, f"path named a different file across the open (TOCTOU rejected): {path}")
    return descriptor, opened


def _iter_pinned_chunks(descriptor: int, path: Path, size: int):
    """Yield the descriptor's bytes in one sequential pass, guarding the length."""

    total = 0
    while chunk := _read_chunk(descriptor, HASH_CHUNK_BYTES):
        total += len(chunk)
        if total > size:
            raise _modified_error(path, f"file grew while being read (TOCTOU rejected): {path}")
        yield chunk
    if total != size:
        raise _modified_error(path, f"file shrank or was rewritten while being read (TOCTOU rejected): {path}")


def _verify_pinned_unchanged(path: Path, descriptor: int, identity: tuple[int, int, int, int, int]) -> None:
    """Post-read legs: descriptor identity stability plus path re-binding.

    The descriptor-only quintuple re-check catches a same-size in-place
    rewrite (mtime_ns moves); the fresh path lstat must still name the read
    file, so a path swapped for another file after the read is refused too.
    """

    try:
        after = os.fstat(descriptor)
    except OSError as exc:
        raise _modified_error(path, f"descriptor failed its post-read identity check: {path}") from exc
    if fd_identity(after) != identity:
        raise _modified_error(path, f"file was replaced or rewritten while being read (TOCTOU rejected): {path}")
    try:
        final = path.lstat()
    except FileNotFoundError as exc:
        raise _modified_error(path, f"path disappeared while being read: {path}") from exc
    except OSError as exc:
        raise _modified_error(path, f"path could not be re-inspected after the read: {path}") from exc
    if _path_identity(final) != _path_identity(after):
        raise _modified_error(path, f"path named a different file after the read (TOCTOU rejected): {path}")


def read_file_pinned(path: Path) -> tuple[bytes, tuple[int, int, int, int, int]]:
    """Read one whole file through a held descriptor: open first, verify later.

    The pinned-read protocol (the census/:class:`SafeTensorSources` shape):
    the path is lstat-ed, opened read-only (``O_NOFOLLOW`` where the platform
    has it), and the descriptor fstat-pinned with the path->fd identity
    check; the payload is then read in one sequential pass and, after the
    read, the descriptor identity is re-checked (full five-tuple) together
    with a fresh path lstat that must still name the same ``(dev, ino,
    size)``.  Returns ``(payload, identity)``.  Raises
    :class:`FsopsModifiedError` when any leg of the loop sees the file or
    its path move; :class:`FsopsMissingPathError`/:class:`FsopsIoError` for
    missing/un-openable targets.  Whole-file in memory -- use
    :func:`sha256_file_pinned` for huge shards.
    """

    descriptor, opened = _open_pinned(path)
    try:
        identity = fd_identity(opened)
        payload = b"".join(_iter_pinned_chunks(descriptor, path, opened.st_size))
        _verify_pinned_unchanged(path, descriptor, identity)
    except FsopsError:
        raise
    except OSError as exc:
        error = FsopsIoError(f"cannot read file through a pinned descriptor: {path}: {exc}")
        error.path = path
        raise error from exc
    finally:
        os.close(descriptor)
    return payload, identity


def sha256_file_pinned(path: Path) -> tuple[str, int]:
    """Stream-hash one file through a held descriptor: open first, verify later.

    The same pinned-read loop as :func:`read_file_pinned`, but the single
    sequential pass feeds the SHA-256 directly (the file is never loaded
    whole, so huge shards stay constant-memory) and the returned size is the
    fstat size of the very descriptor whose bytes produced the digest.
    Returns ``(sha256, size)``; raises :class:`FsopsModifiedError` when the
    path named a different file across the open, the descriptor identity
    moved during or after the hash (five-field quintuple -- a same-size
    in-place rewrite moves ``mtime_ns``), or a post-hash path lstat no
    longer names the hashed file; :class:`FsopsMissingPathError` /
    :class:`FsopsIoError` for missing/un-openable targets.
    """

    descriptor, opened = _open_pinned(path)
    try:
        identity = fd_identity(opened)
        digest = hashlib.sha256()
        for chunk in _iter_pinned_chunks(descriptor, path, opened.st_size):
            digest.update(chunk)
        _verify_pinned_unchanged(path, descriptor, identity)
    except FsopsError:
        raise
    except OSError as exc:
        error = FsopsIoError(f"cannot hash file through a pinned descriptor: {path}: {exc}")
        error.path = path
        raise error from exc
    finally:
        os.close(descriptor)
    return digest.hexdigest(), opened.st_size


def is_link(path: Path) -> bool:
    """True for symlinks, Windows junctions, and any other reparse point.

    Fail-closed: a missing path raises ``FsopsMissingPathError`` and an
    un-inspectable path raises ``FsopsIoError`` (``package_v6._is_link``
    instead swallows ``OSError`` and answers ``False`` -- that divergence is
    intentional there and must not be reintroduced here).
    """
    try:
        if path.is_symlink():
            return True
        junction = getattr(path, "is_junction", None)
        if junction is not None and junction():
            return True
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except FileNotFoundError as exc:
        error = FsopsMissingPathError(f"path does not exist: {path}")
        error.path = path
        raise error from exc
    except OSError as exc:
        error = FsopsIoError(f"path could not be inspected: {path}: {exc}")
        error.path = path
        raise error from exc
    return bool(attributes & _REPARSE_POINT)


def reject_linked_ancestors(path: Path, *, allow_missing_final: bool = False) -> Path:
    """Forbid linked ancestors; return the absolute (unresolved) path.

    Walks every component from the filesystem root down to ``path`` and raises
    ``FsopsLinkError`` if any is a link/reparse point.  A missing component
    raises ``FsopsMissingPathError`` unless it is the final one and
    ``allow_missing_final`` is true (the pre-create validation shape).
    Permission/inspection failures always raise ``FsopsIoError`` -- unlike
    ``package_assembler``/``vae_export``, which let the raw ``OSError`` leak.
    The returned path is ``os.path.abspath``-normalized but **not** resolved:
    lexical ``..`` segments are collapsed, symlinks are not followed.
    """
    absolute = Path(os.path.abspath(path))
    components = [*reversed(absolute.parents), absolute]
    for index, component in enumerate(components):
        final = index == len(components) - 1
        try:
            linked = is_link(component)
        except FsopsMissingPathError:
            if not (allow_missing_final and final):
                raise
            continue
        if linked:
            error = FsopsLinkError(f"linked path component is forbidden: {component}")
            error.path = component
            raise error
    return absolute


def path_inside(path: Path, root: Path) -> bool:
    """Lexical containment check (``path.relative_to(root)`` succeeds).

    Purely lexical, exactly like ``package_assembler._inside``: no symlink
    resolution and **no ``..`` rejection** -- ``root/a/../x`` counts as inside
    ``root/a``.  Callers needing post-resolution containment must resolve both
    sides first (the ``package_v6._assert_resolved_inside`` shape).
    """
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def write_exclusive(path: Path, payload: bytes) -> tuple[int, int, int, int, int]:
    """Create ``path`` with ``payload`` or fail: one file, written once.

    ``O_EXCL`` semantics: an existing path raises ``FsopsExistsError`` (the
    create-or-refuse race-free primitive used for manifests and pins).  The
    file is created read-only (``0o444``), written whole, flushed, and
    fsync-ed before returning.  Other OS failures raise ``FsopsIoError``.
    Returns the write descriptor's final ``(dev, ino, size, mtime_ns, ctime_ns)``
    identity (fstat-ed after the fsync, before the close) so a caller can
    prove that a later re-open of the path still names the file it wrote.
    """
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _WRITE_MODE)
    except FileExistsError as exc:
        error = FsopsExistsError(f"refusing to overwrite existing file: {path}")
        error.path = path
        raise error from exc
    except OSError as exc:
        error = FsopsIoError(f"exclusive create failed: {path}: {exc}")
        error.path = path
        raise error from exc
    try:
        with os.fdopen(descriptor, "wb", buffering=0) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            final = os.fstat(stream.fileno())
    except OSError as exc:
        error = FsopsIoError(f"failed writing exclusive file: {path}: {exc}")
        error.path = path
        raise error from exc
    return fd_identity(final)


def fsync_dir(directory: Path) -> None:
    """Best-effort durability barrier for a directory entry.

    Opens the directory read-only and fsyncs it; silently does nothing when the
    platform/filesystem does not support directory fsync or the directory
    cannot be opened.  This mirrors the two legacy best-effort copies
    (``key_rewrite._fsync_directory`` / ``package_v6._fsync_directory``) --
    unlike ``vae_export._fsync_tree``, which propagates ``OSError``.
    """
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
