"""Resolve installed-wheel provenance for content-bound operation receipts."""

from __future__ import annotations

import importlib
import json
import re
from importlib.metadata import PackageNotFoundError, distribution
from typing import Any
from urllib.parse import urlparse

from comfy_omni.domain.normalization import NormalizationError, ToolIdentity

DISTRIBUTION_NAME = "comfy-omni"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _wheel_sha256(direct_url: dict[str, Any]) -> str:
    archive_url = direct_url.get("url")
    if not isinstance(archive_url, str) or not urlparse(archive_url).path.lower().endswith(".whl"):
        raise NormalizationError(
            "normalization-unbound-wheel",
            "installed distribution does not identify a wheel archive",
        )
    archive_info = direct_url.get("archive_info")
    if not isinstance(archive_info, dict):
        raise NormalizationError(
            "normalization-unbound-wheel",
            "installed distribution has no direct wheel archive identity",
        )
    hashes = archive_info.get("hashes")
    digest = hashes.get("sha256") if isinstance(hashes, dict) else None
    if not isinstance(digest, str):
        legacy_hash = archive_info.get("hash")
        if isinstance(legacy_hash, str) and legacy_hash.startswith("sha256="):
            digest = legacy_hash.removeprefix("sha256=")
    if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest.lower()) is None:
        raise NormalizationError(
            "normalization-unbound-wheel",
            "installed distribution has no valid wheel SHA256",
        )
    return digest.lower()


def installed_tool_identity() -> ToolIdentity:
    """Return proven source and archive identities for the installed wheel."""

    try:
        build = importlib.import_module("comfy_omni._build_identity")
    except ModuleNotFoundError as exc:
        raise NormalizationError(
            "normalization-unbound-build",
            "installed package has no generated source identity",
        ) from exc
    source_commit = getattr(build, "SOURCE_COMMIT", None)
    source_dirty = getattr(build, "SOURCE_DIRTY", None)
    if source_dirty is not False:
        raise NormalizationError(
            "normalization-dirty-build",
            "normalization requires a clean source build",
        )
    try:
        installed = distribution(DISTRIBUTION_NAME)
    except PackageNotFoundError as exc:
        raise NormalizationError(
            "normalization-unbound-build",
            "ComfyOmni distribution metadata is unavailable",
        ) from exc
    raw_direct_url = installed.read_text("direct_url.json")
    if raw_direct_url is None:
        raise NormalizationError(
            "normalization-unbound-wheel",
            "installed distribution has no PEP 610 wheel identity",
        )
    try:
        direct_url = json.loads(raw_direct_url)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise NormalizationError(
            "normalization-unbound-wheel",
            "installed distribution has invalid PEP 610 metadata",
        ) from exc
    if not isinstance(direct_url, dict):
        raise NormalizationError(
            "normalization-unbound-wheel",
            "installed distribution has invalid PEP 610 metadata",
        )
    try:
        return ToolIdentity(
            distribution=DISTRIBUTION_NAME,
            version=installed.version,
            source_commit=source_commit,
            wheel_sha256=_wheel_sha256(direct_url),
        )
    except NormalizationError:
        raise
    except (TypeError, ValueError) as exc:
        raise NormalizationError(
            "normalization-unbound-build",
            "installed package source identity is invalid",
        ) from exc


__all__ = ["installed_tool_identity"]
