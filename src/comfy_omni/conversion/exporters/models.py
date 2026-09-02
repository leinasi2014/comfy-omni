"""Frozen values describing an authorized native checkpoint export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SourceBinding:
    path: str
    size: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "size": self.size, "sha256": self.sha256}


@dataclass(frozen=True)
class QkvLayoutPlan:
    source_layout: str
    target_layout: str
    num_query_groups: int
    heads_per_group: int
    head_dim: int
    row_count: int
    permutation_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_layout": self.source_layout,
            "target_layout": self.target_layout,
            "num_query_groups": self.num_query_groups,
            "heads_per_group": self.heads_per_group,
            "head_dim": self.head_dim,
            "row_count": self.row_count,
            "permutation_sha256": self.permutation_sha256,
        }


@dataclass(frozen=True)
class TensorAction:
    source_name: str
    target_name: str | None
    source_dtype: str
    target_dtype: str | None
    shape: tuple[int, ...]
    source_bytes: int
    target_bytes: int
    operation: str
    group_prefix: str | None = None
    group_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "target_name": self.target_name,
            "source_dtype": self.source_dtype,
            "target_dtype": self.target_dtype,
            "shape": list(self.shape),
            "source_bytes": self.source_bytes,
            "target_bytes": self.target_bytes,
            "operation": self.operation,
            "group_prefix": self.group_prefix,
            "group_size": self.group_size,
        }


@dataclass(frozen=True)
class ShardPlan:
    name: str
    tensor_names: tuple[str, ...]
    payload_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "tensor_names": list(self.tensor_names), "payload_bytes": self.payload_bytes}


@dataclass(frozen=True)
class ResourceEnvelope:
    max_rows: int
    max_shard_bytes: int
    largest_target_tensor_bytes: int

    def to_dict(self) -> dict[str, int]:
        return {
            "max_rows": self.max_rows,
            "max_shard_bytes": self.max_shard_bytes,
            "largest_target_tensor_bytes": self.largest_target_tensor_bytes,
        }


@dataclass(frozen=True)
class NativeExportPlan:
    schema: str
    output_schema: str
    component: str
    profile: str
    source_contract: str
    source_contract_origin: str
    source_contract_schema_sha256: str
    source_snapshot_manifest_sha256: str | None
    source_snapshot_file_sha256: str | None
    template_name: str
    template_version: int
    template_sha256: str
    source_files: tuple[SourceBinding, ...]
    qkv_layout: QkvLayoutPlan
    resource_envelope: ResourceEnvelope
    actions: tuple[TensorAction, ...]
    shards: tuple[ShardPlan, ...]
    target_tensor_count: int
    target_payload_bytes: int
    runtime_quant_method: str
    runtime_ignored_layers: tuple[str, ...]
    payload_semantics: str
    content_sha256: str

    def to_dict(self, *, include_content_sha256: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "status": "AUTHORIZED_PLAN",
            "output_schema": self.output_schema,
            "component": self.component,
            "profile": self.profile,
            "source_contract": {
                "name": self.source_contract,
                "origin": self.source_contract_origin,
                "schema_sha256": self.source_contract_schema_sha256,
                "snapshot_manifest_sha256": self.source_snapshot_manifest_sha256,
                "snapshot_file_sha256": self.source_snapshot_file_sha256,
            },
            "architecture_template": {
                "name": self.template_name,
                "version": self.template_version,
                "sha256": self.template_sha256,
            },
            "source_files": [item.to_dict() for item in self.source_files],
            "qkv_layout": self.qkv_layout.to_dict(),
            "resource_envelope": self.resource_envelope.to_dict(),
            "actions": [item.to_dict() for item in self.actions],
            "shards": [item.to_dict() for item in self.shards],
            "target": {
                "tensor_count": self.target_tensor_count,
                "payload_bytes": self.target_payload_bytes,
            },
            "runtime_quantization": {
                "required": True,
                "method": self.runtime_quant_method,
                "ignored_layers": list(self.runtime_ignored_layers),
                "checkpoint_int8_serialized": False,
            },
            "semantics": {
                "description": self.payload_semantics,
                "payload_preserving": False,
                "lossless_claim": False,
                "direct_convrot_loading": False,
            },
        }
        if include_content_sha256:
            payload["content_sha256"] = self.content_sha256
        return payload


__all__ = [
    "NativeExportPlan",
    "QkvLayoutPlan",
    "ResourceEnvelope",
    "ShardPlan",
    "SourceBinding",
    "TensorAction",
]
