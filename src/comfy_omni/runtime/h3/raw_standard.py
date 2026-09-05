"""Direct, in-memory loading of the fixed standard H3 ConvRot transformer.

This binding authenticates the original Ref2VA source once and then yields
only its 532 logical host weights.  It never creates a converted checkpoint:
the inherited reader preserves native F16/F32 copies and decodes ConvRot to
BF16 in bounded CPU memory. The host may hold the previous and current tensor
during iteration and performs explicit execution-dtype adaptation. It initializes its two documented auxiliary slots
separately; they are not source weights and are deliberately absent here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from comfy_omni.artifacts import fileops
from comfy_omni.artifacts.sources import SafeTensorSources
from comfy_omni.contracts.models import ContractError
from comfy_omni.contracts.registry import COMPILE_TIME_CATALOG, SOURCE_PROFILE_DASIWA_REF2VA_HYBRID
from comfy_omni.contracts.templates import ARCHITECTURE_TEMPLATES
from comfy_omni.conversion.contract_workflows.census import FileRecord, census_tensors, schema_sha256
from comfy_omni.conversion.exporters.planning import build_native_export_plan
from comfy_omni.domain.checkpoints import TensorDescriptor
from comfy_omni.runtime.h3.raw_beta4 import (
    RawBeta4Binding,
    _actions_by_target,
    _identity,
    _validate_plan,
)

RAW_STANDARD_SOURCE_NAME = "minimax_h3_ref2va_pruned_zs05_int8_convrot.safetensors"
RAW_STANDARD_SOURCE_BYTES = 20_970_379_680
RAW_STANDARD_SOURCE_SHA256 = "71b8085ac4221ee036708c230a007d617dccca1b0028b95bb4ee106cb2a385c5"
RAW_STANDARD_SOURCE_SCHEMA_SHA256 = "cc7976f678e6d4a567e718aca56c1db4aa91adfa27108db84066cce3213edf9d"


def _fail(detail: str, *, stage: str) -> None:
    raise ContractError(detail, evidence={"stage": stage})


@dataclass(frozen=True)
class RawStandardIdentity:
    """Trusted identity of the one supported standard H3 original source."""

    name: str
    size: int
    sha256: str
    source_schema_sha256: str
    tensor_count: int | None = None
    target_tensor_count: int | None = None
    target_schema_sha256: str | None = None


PRIMARY_RAW_STANDARD_IDENTITY = RawStandardIdentity(
    RAW_STANDARD_SOURCE_NAME,
    RAW_STANDARD_SOURCE_BYTES,
    RAW_STANDARD_SOURCE_SHA256,
    RAW_STANDARD_SOURCE_SCHEMA_SHA256,
    932,
    532,
)


@dataclass(frozen=True)
class RawStandardBinding(RawBeta4Binding):
    """Authenticated standard-source binding with its real 532-weight census.

    The shared base owns the bounded source reader, ConvRot inverse transform,
    QKV row reorder, and lightweight identity/header recheck.  This subclass
    changes only the authenticated source authority and the plan builder.
    """

    trusted_identity: RawStandardIdentity

    @classmethod
    def establish(
        cls,
        source_path: Path | str,
        *,
        identity: RawStandardIdentity = PRIMARY_RAW_STANDARD_IDENTITY,
    ) -> RawStandardBinding:
        path = fileops.reject_linked_ancestors(Path(source_path)).resolve(strict=True)
        record = COMPILE_TIME_CATALOG.resolve("transformer", SOURCE_PROFILE_DASIWA_REF2VA_HYBRID)
        template = ARCHITECTURE_TEMPLATES[record.template_name]
        with SafeTensorSources((path,)) as sources:
            if sources.sizes != [identity.size] or sources.hashes != [identity.sha256]:
                _fail("raw standard source size or SHA256 differs from trusted identity", stage="raw-authentication")
            descriptors = tuple(
                sorted((item.descriptor for item in sources.tensors.values()), key=lambda item: item.name)
            )
            if schema_sha256(descriptors) != identity.source_schema_sha256:
                _fail("raw standard source descriptor schema differs from trusted identity", stage="raw-authentication")
            if identity.tensor_count is not None and len(descriptors) != identity.tensor_count:
                _fail("raw standard source tensor count differs from trusted identity", stage="raw-authentication")
            report = census_tensors(
                descriptors,
                {
                    name: sources.read_raw(item)
                    for name, item in sources.tensors.items()
                    if name.endswith(".comfy_quant")
                },
                files=(FileRecord(str(path), sources.sizes[0], sources.hashes[0]),),
            )
            plan = build_native_export_plan(report, record, template)
            _validate_plan(plan, identity, path, descriptors)
            source_file_identity = sources._sources[0].identity
        if _identity(path) != source_file_identity:
            _fail("raw standard source changed after authentication", stage="raw-authentication")
        targets = _actions_by_target(plan)
        target_descriptors = tuple(
            TensorDescriptor(
                action.target_name or "", action.target_dtype or "", action.shape, (0, action.target_bytes)
            )
            for action in targets
        )
        return cls(path, identity, source_file_identity, descriptors, target_descriptors, plan)


__all__ = [
    "PRIMARY_RAW_STANDARD_IDENTITY",
    "RAW_STANDARD_SOURCE_BYTES",
    "RAW_STANDARD_SOURCE_NAME",
    "RAW_STANDARD_SOURCE_SCHEMA_SHA256",
    "RAW_STANDARD_SOURCE_SHA256",
    "RawStandardBinding",
    "RawStandardIdentity",
]
