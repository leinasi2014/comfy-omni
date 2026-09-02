"""Pure values for explicit, digest-pinned artifact normalization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


class NormalizationError(ValueError):
    """A fail-closed normalization error with a stable reason code."""

    def __init__(self, reason_code: str, detail: str) -> None:
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(f"{reason_code}: {detail}")


def _require_sha256(value: str, field: str) -> None:
    if SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA256 digest")


@dataclass(frozen=True)
class ArtifactIdentity:
    """A complete immutable file identity."""

    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if self.bytes < 0:
            raise ValueError("artifact byte count must be non-negative")
        _require_sha256(self.sha256, "artifact sha256")

    def to_dict(self) -> dict[str, Any]:
        return {"bytes": self.bytes, "sha256": self.sha256}


@dataclass(frozen=True)
class NormalizationProfile:
    """One exact source-to-derived transformation authorization."""

    profile_id: str
    profile_version: int
    source: ArtifactIdentity
    removed_suffix: ArtifactIdentity
    derived: ArtifactIdentity

    def __post_init__(self) -> None:
        if not self.profile_id or self.profile_version < 1:
            raise ValueError("normalization profile identity is invalid")
        if self.source.bytes != self.derived.bytes + self.removed_suffix.bytes:
            raise ValueError("normalization profile byte counts are incoherent")


@dataclass(frozen=True)
class ToolIdentity:
    """Installed tool provenance required for a publishable receipt."""

    distribution: str
    version: str
    source_commit: str
    wheel_sha256: str

    def __post_init__(self) -> None:
        if not self.distribution or not self.version:
            raise ValueError("tool distribution identity is incomplete")
        if COMMIT_PATTERN.fullmatch(self.source_commit) is None:
            raise ValueError("tool source commit must be a lowercase 40-character Git SHA")
        _require_sha256(self.wheel_sha256, "tool wheel sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "distribution": self.distribution,
            "source_commit": self.source_commit,
            "version": self.version,
            "wheel_sha256": self.wheel_sha256,
        }


@dataclass(frozen=True)
class NormalizationReceipt:
    """Portable proof of one exact normalization operation."""

    profile_id: str
    profile_version: int
    source: ArtifactIdentity
    removed_suffix: ArtifactIdentity
    derived: ArtifactIdentity
    strict_reread_tensor_count: int
    tool: ToolIdentity

    def __post_init__(self) -> None:
        if not self.profile_id or self.profile_version < 1:
            raise ValueError("normalization receipt profile identity is invalid")
        if self.strict_reread_tensor_count < 0:
            raise ValueError("strict reread tensor count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "derived": self.derived.to_dict(),
            "profile": {"id": self.profile_id, "version": self.profile_version},
            "removed_suffix": self.removed_suffix.to_dict(),
            "schema_id": "comfy-omni.normalization-receipt/v1",
            "source": self.source.to_dict(),
            "strict_reread": {"passed": True, "tensor_count": self.strict_reread_tensor_count},
            "tool": self.tool.to_dict(),
        }


__all__ = [
    "ArtifactIdentity",
    "NormalizationError",
    "NormalizationProfile",
    "NormalizationReceipt",
    "ToolIdentity",
]
