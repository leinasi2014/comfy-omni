"""Explicit fixed-asset beta4 planning; the legacy catalogs remain unchanged."""

from comfy_omni.contracts.beta4 import BETA4_SOURCE_RECORD, BETA4_SOURCE_TEMPLATE
from comfy_omni.contracts.conversion import PROFILE_BETA4_DENSE_BF16
from comfy_omni.conversion.contract_workflows.census import CensusReport
from comfy_omni.conversion.exporters.models import NativeExportPlan
from comfy_omni.conversion.exporters.planning import (
    DEFAULT_MAX_ROWS,
    DEFAULT_MAX_SHARD_BYTES,
    build_native_export_plan,
)


def build_beta4_dense_plan(
    report: CensusReport,
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
) -> NativeExportPlan:
    """Authorize exactly A934 -> dense534 without an online INT8 requirement."""
    return build_native_export_plan(
        report,
        BETA4_SOURCE_RECORD,
        BETA4_SOURCE_TEMPLATE,
        profile_name=PROFILE_BETA4_DENSE_BF16,
        max_rows=max_rows,
        max_shard_bytes=max_shard_bytes,
    )
