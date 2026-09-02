"""Strict legacy-compatible native-source contract snapshot schema.

Derived from Apache-2.0 h3-forge snapshots.py blob
b0a438df386278a1b0e7dcf783ca182748ec77ea at commit e9cb011d00b028c149db3978de246c54f6e34acc.
Global activation and environment lookup are deliberately absent.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from comfy_omni.artifacts import fileops
from comfy_omni.contracts.models import (
    STORAGE_BF16_PLAIN,
    STORAGE_INT8_CONVROT,
    ContractError,
    ContractRecord,
    NativeSourceContract,
)
from comfy_omni.contracts.templates import ARCHITECTURE_TEMPLATES, template_digest

SNAPSHOT_SCHEMA = "h3_forge.contract.snapshot/v1"
CONTRACT_BLOCK_SCHEMA = "h3_forge.contract/v1"
PIN_BLOCK_SCHEMA = "h3_forge.contract.pin/v1"
DECISION_INHERITED_ENFORCED_PIN = "inherited-enforced-pin"
DECISION_OBSERVED_FROZEN = "observed-frozen"
DECISION_CENSUS_ONLY = "census-only"

_HEX64 = re.compile(r"[0-9a-f]{64}")
_COMPONENTS = {"transformer", "text_encoder"}
_STORAGE_KINDS = {STORAGE_INT8_CONVROT, STORAGE_BF16_PLAIN}
_DECISIONS = {DECISION_INHERITED_ENFORCED_PIN, DECISION_OBSERVED_FROZEN, DECISION_CENSUS_ONLY}
_SNAPSHOT_KEYS = {"schema", "status", "pending_review", "contract", "pin", "manifest_sha256"}
_CONTRACT_KEYS = {
    "schema",
    "name",
    "component",
    "tensor_count",
    "convrot_group_count",
    "schema_sha256",
    "include_transformer_adaln",
    "transformer_adaln_group_size",
    "template_name",
    "template_version",
    "storage_kind",
}
_PIN_KEYS = {
    "schema",
    "reviewed_by",
    "evidence_sha256",
    "generated_by",
    "draft_sha256",
    "source_files",
    "census_sha256",
    "template",
    "enforced_schema_decision",
}
_PIN_TEMPLATE_KEYS = {"name", "version", "digest"}
_SOURCE_FILE_KEYS = {"path", "size", "sha256"}


@dataclass(frozen=True)
class ContractSnapshot:
    path: Path
    manifest_sha256: str
    document: Mapping[str, Any]
    payload: bytes = field(repr=False)

    @property
    def contract_block(self) -> Mapping[str, Any]:
        return self.document["contract"]


def _error(message: str, field: str | None = None) -> ContractError:
    evidence = {"stage": "snapshot-schema"}
    if field is not None:
        evidence["field"] = field
    return ContractError(message, evidence=evidence)


def _keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise _error(f"{label} must contain exactly {sorted(expected)}, got {observed}", label)
    return value


def _hex64(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise _error(f"{label} must be a lowercase SHA-256 digest", label)
    return value


def _integer(value: Any, label: str, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise _error(f"{label} must be an integer >= {minimum}", label)
    return value


def snapshot_manifest_sha256(document: Mapping[str, Any]) -> str:
    unsigned = dict(document)
    unsigned.pop("manifest_sha256", None)
    return hashlib.sha256(fileops.canonical_json(unsigned)).hexdigest()


def contract_block(
    contract: NativeSourceContract, *, template_name: str, template_version: int, storage_kind: str
) -> dict[str, Any]:
    return {
        "schema": CONTRACT_BLOCK_SCHEMA,
        "name": contract.name,
        "component": contract.component,
        "tensor_count": contract.tensor_count,
        "convrot_group_count": contract.convrot_group_count,
        "schema_sha256": contract.schema_sha256,
        "include_transformer_adaln": contract.include_transformer_adaln,
        "transformer_adaln_group_size": contract.transformer_adaln_group_size,
        "template_name": template_name,
        "template_version": template_version,
        "storage_kind": storage_kind,
    }


def _validate_storage(block: Mapping[str, Any], group_count: int, has_inventory: bool) -> None:
    storage = block["storage_kind"]
    if storage not in _STORAGE_KINDS:
        raise _error(f"storage_kind must be one of {sorted(_STORAGE_KINDS)}", "storage_kind")
    if storage == STORAGE_INT8_CONVROT and (block["convrot_group_count"] < 1 or group_count < 1):
        raise _error("int8-convrot requires a positive ConvRot template", "storage_kind")
    if storage != STORAGE_BF16_PLAIN:
        return
    if block["convrot_group_count"] != 0 or group_count != 0:
        raise _error("bf16-plain requires a zero-group dense template", "storage_kind")
    if block["include_transformer_adaln"] or block["transformer_adaln_group_size"] is not None:
        raise _error("bf16-plain carries no AdaLN quantization flags", "include_transformer_adaln")
    if not has_inventory:
        raise _error("bf16-plain requires a complete pinned inventory", "template_name")


def _validate_contract(value: Any) -> dict[str, Any]:
    block = _keys(value, _CONTRACT_KEYS, "snapshot.contract")
    if block["schema"] != CONTRACT_BLOCK_SCHEMA:
        raise _error(f"contract schema must be {CONTRACT_BLOCK_SCHEMA!r}", "contract.schema")
    if not isinstance(block["name"], str) or not block["name"]:
        raise _error("contract name must be non-empty", "name")
    if block["component"] not in _COMPONENTS:
        raise _error(f"contract component must be one of {sorted(_COMPONENTS)}", "component")
    _integer(block["tensor_count"], "tensor_count")
    _integer(block["convrot_group_count"], "convrot_group_count", 0)
    if block["schema_sha256"] is not None:
        _hex64(block["schema_sha256"], "schema_sha256")
    if not isinstance(block["include_transformer_adaln"], bool):
        raise _error("include_transformer_adaln must be boolean", "include_transformer_adaln")
    if block["transformer_adaln_group_size"] is not None:
        _integer(block["transformer_adaln_group_size"], "transformer_adaln_group_size")
    template = ARCHITECTURE_TEMPLATES.get(block["template_name"])
    if template is None or block["template_version"] != template.template_version:
        raise _error("snapshot references an unknown or drifted template", "template_name")
    if block["convrot_group_count"] != len(template.convrot_table()):
        raise _error("snapshot group count disagrees with its template", "convrot_group_count")
    _validate_storage(block, len(template.convrot_table()), bool(template.non_quantized_inventory))
    return block


def _validate_source_files(value: Any) -> None:
    if not isinstance(value, list) or not value:
        raise _error("pin.source_files must be a non-empty array", "source_files")
    for item in value:
        record = _keys(item, _SOURCE_FILE_KEYS, "pin.source_files[]")
        if not isinstance(record["path"], str) or not record["path"]:
            raise _error("source path must be non-empty", "source_files[].path")
        _integer(record["size"], "source_files[].size", 0)
        _hex64(record["sha256"], "source_files[].sha256")


def _validate_pin(value: Any) -> dict[str, Any]:
    pin = _keys(value, _PIN_KEYS, "snapshot.pin")
    if pin["schema"] != PIN_BLOCK_SCHEMA:
        raise _error(f"pin schema must be {PIN_BLOCK_SCHEMA!r}", "pin.schema")
    reviewer = pin["reviewed_by"]
    if not isinstance(reviewer, str) or not reviewer:
        raise _error("pin reviewer must be non-empty", "reviewed_by")
    for field_name in ("evidence_sha256", "draft_sha256", "census_sha256"):
        _hex64(pin[field_name], field_name)
    generated = pin["generated_by"]
    if not isinstance(generated, dict) or not all(
        isinstance(key, str) and (value is None or isinstance(value, str)) for key, value in generated.items()
    ):
        raise _error("generated_by must map strings to string-or-null", "generated_by")
    if reviewer in {item for item in generated.values() if item is not None}:
        raise _error("reviewer must differ from every generator identity", "reviewed_by")
    template = _keys(pin["template"], _PIN_TEMPLATE_KEYS, "snapshot.pin.template")
    if not isinstance(template["name"], str):
        raise _error("pin template name must be a string", "pin.template.name")
    _integer(template["version"], "pin.template.version")
    _hex64(template["digest"], "pin.template.digest")
    _validate_source_files(pin["source_files"])
    if pin["enforced_schema_decision"] not in _DECISIONS:
        raise _error(f"invalid enforced schema decision {pin['enforced_schema_decision']!r}")
    return pin


def _validate_cross_binding(contract: Mapping[str, Any], pin: Mapping[str, Any]) -> None:
    template = ARCHITECTURE_TEMPLATES[contract["template_name"]]
    reference = pin["template"]
    if (reference["name"], reference["version"]) != (template.template_name, template.template_version):
        raise _error("pin template reference disagrees with the contract block", "pin.template")
    if reference["digest"] != template_digest(template):
        raise _error("pin template digest drifted from current audited tables", "pin.template.digest")
    decision = pin["enforced_schema_decision"]
    if decision == DECISION_CENSUS_ONLY and contract["schema_sha256"] is not None:
        raise _error("census-only decision cannot carry an enforced schema", "schema_sha256")
    if decision != DECISION_CENSUS_ONLY and contract["schema_sha256"] is None:
        raise _error("enforced decision requires an enforced schema", "schema_sha256")


def load_snapshot(path: Path | str, *, require_digest_name: bool = True) -> ContractSnapshot:
    """Load one canonical, content-addressed, fully bound snapshot."""

    target = Path(path)
    try:
        if fileops.is_link(target) or not target.is_file():
            raise _error(f"snapshot must be a non-linked regular file: {target}")
        payload, _ = fileops.read_file_pinned(target)
        document = fileops.parse_json_strict(payload)
    except (fileops.FsopsError, OSError) as exc:
        raise ContractError(str(exc), evidence={"stage": "snapshot-load", "path": str(target)}) from exc
    block = _keys(document, _SNAPSHOT_KEYS, "snapshot")
    if block["schema"] != SNAPSHOT_SCHEMA or block["status"] != "PINNED" or block["pending_review"] is not False:
        raise _error("snapshot state/schema is not PINNED", "status")
    contract = _validate_contract(block["contract"])
    pin = _validate_pin(block["pin"])
    _validate_cross_binding(contract, pin)
    claimed = _hex64(block["manifest_sha256"], "manifest_sha256")
    if snapshot_manifest_sha256(block) != claimed:
        raise _error("snapshot manifest digest is not self-consistent", "manifest_sha256")
    if require_digest_name and target.name != f"{claimed}.json":
        raise _error("snapshot filename must equal its manifest digest", "manifest_sha256")
    if fileops.canonical_json(block) != payload:
        raise _error("snapshot is not in canonical byte form")
    return ContractSnapshot(target, claimed, MappingProxyType(block), payload)


def snapshot_record(snapshot: ContractSnapshot) -> ContractRecord:
    block = snapshot.contract_block
    contract = NativeSourceContract(
        block["name"],
        block["component"],
        block["tensor_count"],
        block["convrot_group_count"],
        block["schema_sha256"],
        block["include_transformer_adaln"],
        block["transformer_adaln_group_size"],
    )
    return ContractRecord(
        contract,
        block["template_name"],
        block["storage_kind"],
        snapshot.manifest_sha256,
        snapshot.payload,
    )


__all__ = [
    "CONTRACT_BLOCK_SCHEMA",
    "DECISION_CENSUS_ONLY",
    "DECISION_INHERITED_ENFORCED_PIN",
    "DECISION_OBSERVED_FROZEN",
    "PIN_BLOCK_SCHEMA",
    "SNAPSHOT_SCHEMA",
    "ContractSnapshot",
    "contract_block",
    "load_snapshot",
    "snapshot_manifest_sha256",
    "snapshot_record",
]
