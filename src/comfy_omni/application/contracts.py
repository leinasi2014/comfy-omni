"""Application use cases for explicit immutable native-source contract workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from comfy_omni.artifacts.snapshot_store import catalog_with_store
from comfy_omni.contracts.models import ContractCatalog
from comfy_omni.contracts.registry import COMPILE_TIME_CATALOG
from comfy_omni.conversion.contract_workflows import (
    CensusEngine,
    ContractDraft,
    PinResult,
    ScanReport,
    build_draft,
    build_scan_report,
    pin_draft,
)


def load_contract_catalog(contract_dir: Path | None = None) -> ContractCatalog:
    """Return compile-time records or a new catalog extended from one explicit store."""

    if contract_dir is None:
        return COMPILE_TIME_CATALOG
    return catalog_with_store(COMPILE_TIME_CATALOG, contract_dir)


def scan_contract_source(paths: Sequence[Path]) -> ScanReport:
    """Observe and exactly match one logical checkpoint source."""

    census = CensusEngine().scan_paths(paths)
    return build_scan_report(census)


def draft_contract_source(
    paths: Sequence[Path],
    *,
    generator_identity: Mapping[str, str | None],
    catalog: ContractCatalog = COMPILE_TIME_CATALOG,
) -> ContractDraft:
    """Create an immutable pending-review draft from one exact L3 match."""

    report = scan_contract_source(paths)
    return build_draft(report.census, report.match, generator_identity=generator_identity, catalog=catalog)


def pin_contract_draft(
    draft_path: Path,
    *,
    name: str,
    reviewer: str,
    evidence_path: Path,
    contract_dir: Path,
    enforce_observed_schema: bool = False,
    catalog: ContractCatalog = COMPILE_TIME_CATALOG,
) -> PinResult:
    """Review and publish one draft without mutating any active catalog."""

    return pin_draft(
        draft_path,
        name=name,
        reviewer=reviewer,
        evidence_path=evidence_path,
        contract_dir=contract_dir,
        enforce_observed_schema=enforce_observed_schema,
        catalog=catalog,
    )


def catalog_document(catalog: ContractCatalog) -> list[dict[str, Any]]:
    """Return a stable JSON-ready catalog overview."""

    result: list[dict[str, Any]] = []
    for name, record in sorted(catalog.records.items()):
        contract = record.contract
        result.append(
            {
                "name": name,
                "component": contract.component,
                "tensor_count": contract.tensor_count,
                "convrot_group_count": contract.convrot_group_count,
                "schema_sha256": contract.schema_sha256,
                "template_name": record.template_name,
                "storage_kind": record.storage_kind,
                "origin": "external-snapshot" if record.snapshot_manifest_sha256 else "compile-time",
                "snapshot_manifest_sha256": record.snapshot_manifest_sha256,
            }
        )
    return result


__all__ = [
    "catalog_document",
    "draft_contract_source",
    "load_contract_catalog",
    "pin_contract_draft",
    "scan_contract_source",
]
