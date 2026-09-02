"""Application use cases for offline native checkpoint conversion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from comfy_omni.application.contracts import load_contract_catalog
from comfy_omni.contracts.conversion import PROFILE_DENSE_BF16_ONLINE_INT8
from comfy_omni.contracts.models import ArchitectureTemplate, ContractCatalog, ContractError
from comfy_omni.contracts.templates import ARCHITECTURE_TEMPLATES
from comfy_omni.conversion.contract_workflows.census import CensusEngine
from comfy_omni.conversion.exporters.models import NativeExportPlan
from comfy_omni.conversion.exporters.execution import execute_native_export
from comfy_omni.conversion.exporters.planning import (
    DEFAULT_MAX_ROWS,
    DEFAULT_MAX_SHARD_BYTES,
    build_native_export_plan,
)
from comfy_omni.conversion.packaging.native_export import NativeExportPublication
from comfy_omni.domain.normalization import ToolIdentity


def plan_native_export(
    sources: Sequence[Path | str],
    *,
    component: str = "transformer",
    source_profile: str | None = None,
    profile_name: str = PROFILE_DENSE_BF16_ONLINE_INT8,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
    catalog: ContractCatalog | None = None,
    templates: Mapping[str, ArchitectureTemplate] = ARCHITECTURE_TEMPLATES,
) -> NativeExportPlan:
    """Scan source headers and return an authorized, payload-free export plan."""

    active_catalog = catalog if catalog is not None else load_contract_catalog()
    record = active_catalog.resolve(component, source_profile)
    template = templates.get(record.template_name)
    if template is None:
        raise ContractError(
            f"source contract references an unavailable template: {record.template_name!r}",
            evidence={"stage": "architecture-template", "template": record.template_name},
        )
    report = CensusEngine().scan_paths(sources)
    return build_native_export_plan(
        report,
        record,
        template,
        profile_name=profile_name,
        max_rows=max_rows,
        max_shard_bytes=max_shard_bytes,
    )


def convert_native_export(
    sources: Sequence[Path | str],
    output_dir: Path | str,
    *,
    tool: ToolIdentity,
    component: str = "transformer",
    source_profile: str | None = None,
    profile_name: str = PROFILE_DENSE_BF16_ONLINE_INT8,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
    catalog: ContractCatalog | None = None,
    templates: Mapping[str, ArchitectureTemplate] = ARCHITECTURE_TEMPLATES,
    source_contract_snapshot: Path | None = None,
) -> NativeExportPublication:
    """Plan and execute one explicit source through the immutable export transaction."""

    plan = plan_native_export(
        sources,
        component=component,
        source_profile=source_profile,
        profile_name=profile_name,
        max_rows=max_rows,
        max_shard_bytes=max_shard_bytes,
        catalog=catalog,
        templates=templates,
    )
    return execute_native_export(
        plan,
        Path(output_dir),
        tool=tool,
        source_contract_snapshot=source_contract_snapshot,
    )


__all__ = ["convert_native_export", "plan_native_export"]
