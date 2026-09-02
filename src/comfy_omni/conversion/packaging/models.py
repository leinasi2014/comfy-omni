"""Immutable values for native package composition plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from comfy_omni.domain.normalization import ToolIdentity


@dataclass(frozen=True)
class ComponentFile:
    """One content-bound file exposed by a verified component receipt."""

    path: str
    size: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}


@dataclass(frozen=True)
class ComponentReceipt:
    """Portable component authority normalized before package planning."""

    component: str
    source_dir: str
    receipt_schema: str
    receipt_sha256: str
    tool: ToolIdentity
    files: tuple[ComponentFile, ...]


@dataclass(frozen=True)
class PackageComponentPlan:
    """One component receipt bound into a package plan."""

    component: str
    source_dir: str
    receipt_schema: str
    receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "receipt": {"schema": self.receipt_schema, "sha256": self.receipt_sha256},
            "source_dir": self.source_dir,
        }


@dataclass(frozen=True)
class PackageFilePlan:
    """One exact source file and its final package-relative destination."""

    component: str
    source_path: str
    target_path: str
    size: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "sha256": self.sha256,
            "size": self.size,
            "source_path": self.source_path,
            "target_path": self.target_path,
        }


@dataclass(frozen=True)
class NativePackagePlan:
    """Canonical authorization for one immutable native package."""

    schema: str
    host_adapter: str
    host_commit: str
    output_schema: str
    manifest_name: str
    serving_entrypoint: str
    supported_tasks: tuple[str, ...]
    resident_dit_count: int
    tool: ToolIdentity
    components: tuple[PackageComponentPlan, ...]
    files: tuple[PackageFilePlan, ...]
    content_sha256: str

    def to_dict(self, *, include_content_sha256: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "components": [item.to_dict() for item in self.components],
            "files": [item.to_dict() for item in self.files],
            "host": {"adapter": self.host_adapter, "commit": self.host_commit},
            "schema": self.schema,
            "status": "AUTHORIZED_PLAN",
            "target": {
                "manifest": self.manifest_name,
                "output_schema": self.output_schema,
                "resident_dit_count": self.resident_dit_count,
                "serving_entrypoint": self.serving_entrypoint,
                "supported_tasks": list(self.supported_tasks),
            },
            "tool": self.tool.to_dict(),
        }
        if include_content_sha256:
            value["content_sha256"] = self.content_sha256
        return value


@dataclass(frozen=True)
class PackageSourceVerification:
    """Portable result of re-reading every source named by a package plan."""

    schema: str
    plan_content_sha256: str
    tool: ToolIdentity
    component_count: int
    file_count: int
    total_bytes: int
    files_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "components": self.component_count,
            "file_count": self.file_count,
            "files_sha256": self.files_sha256,
            "plan_content_sha256": self.plan_content_sha256,
            "schema": self.schema,
            "status": "VERIFIED",
            "tool": self.tool.to_dict(),
            "total_bytes": self.total_bytes,
        }


__all__ = [
    "ComponentFile",
    "ComponentReceipt",
    "NativePackagePlan",
    "PackageComponentPlan",
    "PackageFilePlan",
    "PackageSourceVerification",
]
