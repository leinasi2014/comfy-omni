"""Fail-closed draft-to-snapshot review, derived from audited Apache-2.0 h3-forge pin.py."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from comfy_omni.artifacts import fileops
from comfy_omni.artifacts.snapshot_schema import (
    DECISION_CENSUS_ONLY,
    DECISION_INHERITED_ENFORCED_PIN,
    DECISION_OBSERVED_FROZEN,
    PIN_BLOCK_SCHEMA,
    SNAPSHOT_SCHEMA,
    ContractSnapshot,
    contract_block,
)
from comfy_omni.artifacts.snapshot_store import write_snapshot
from comfy_omni.contracts.models import (
    STORAGE_BF16_PLAIN,
    STORAGE_INT8_CONVROT,
    ContractCatalog,
    ContractError,
    NativeSourceContract,
)
from comfy_omni.contracts.registry import COMPILE_TIME_CATALOG
from comfy_omni.contracts.templates import ARCHITECTURE_TEMPLATES, template_digest
from comfy_omni.conversion.contract_workflows.drafting import DRAFT_SCHEMA

_HEX64 = re.compile(r"[0-9a-f]{64}")

_DRAFT_KEYS = frozenset(
    {
        "schema",
        "status",
        "pending_review",
        "candidate_contract",
        "enforced_schema_sha256",
        "promotion",
        "nearest_known_contract",
        "diff_vs_known",
        "red_flags",
        "template_evidence",
        "l1_ambiguity_resolved_by_h3_evidence",
        "three_level_match",
        "census",
        "provenance",
    }
)
_CANDIDATE_KEYS = frozenset(
    {
        "name",
        "component",
        "tensor_count",
        "convrot_group_count",
        "observed_schema_sha256",
        "include_transformer_adaln",
        "transformer_adaln_group_size",
        "storage_kind",
        "template_name",
        "template_version",
    }
)
_PROVENANCE_KEYS = frozenset({"source_files", "census_sha256", "template", "generated_by"})
_PROVENANCE_TEMPLATE_KEYS = frozenset({"name", "version", "digest"})
_SOURCE_FILE_KEYS = frozenset({"path", "size", "sha256"})


@dataclass(frozen=True)
class PinResult:
    """One successful pin: the snapshot plus the decision chain."""

    snapshot: ContractSnapshot
    contract: NativeSourceContract
    enforced_schema_decision: str
    enforced_schema_sha256: str | None
    evidence_sha256: str
    reviewer: str
    draft_sha256: str


@dataclass(frozen=True)
class _PreparedPin:
    document: Mapping[str, Any]
    candidate: Mapping[str, Any]
    draft_sha256: str
    census_sha256: str
    source_files: list[dict[str, Any]]
    generated_by: Mapping[str, str | None]
    evidence_sha256: str
    decision: str
    enforced: str | None


def _keys(value: Any, expected: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or frozenset(value) != expected:
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ContractError(
            f"{label} must be an object with exactly the keys {sorted(expected)}, got {observed}",
            evidence={"stage": "pin-draft", "block": label},
        )
    return value


def _hex64(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ContractError(
            f"{label} must be a 64-character lowercase hex digest",
            evidence={"stage": "pin-draft", "field": label},
        )
    return value


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ContractError(f"{label} must be a positive integer", evidence={"stage": "pin-draft", "field": label})
    return value


def _non_negative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ContractError(f"{label} must be a non-negative integer", evidence={"stage": "pin-draft", "field": label})
    return value


def load_draft_document(path: Path | str) -> tuple[dict[str, Any], str]:
    """Load one draft file: canonical strict JSON in the DRAFTED state.

    Returns ``(document, draft_sha256)`` where the digest is over the exact
    file bytes (the writer emits canonical bytes, which is re-asserted).
    """

    target = Path(path)
    try:
        if fileops.is_link(target) or not target.is_file():
            raise ContractError(
                f"draft must be a regular non-linked file: {target}",
                evidence={"stage": "pin-draft", "path": str(target)},
            )
        payload = target.read_bytes()
        document = fileops.parse_json_strict(payload)
    except (fileops.FsopsError, OSError) as error:
        raise ContractError(
            f"draft could not be read: {target}: {error}",
            evidence={"stage": "pin-draft", "path": str(target)},
        ) from error
    if not isinstance(document, dict):
        raise ContractError("draft must be a JSON object", evidence={"stage": "pin-draft", "path": str(target)})
    _keys(document, _DRAFT_KEYS, "draft")
    if document["schema"] != DRAFT_SCHEMA:
        raise ContractError(
            f"draft schema must be {DRAFT_SCHEMA!r}", evidence={"stage": "pin-draft", "path": str(target)}
        )
    if document["status"] != "DRAFTED":
        raise ContractError(
            f"only a DRAFTED draft can be pinned (status is {document['status']!r})",
            evidence={"stage": "pin-draft", "path": str(target)},
        )
    if document["pending_review"] is not True:
        raise ContractError(
            "draft must be in the pending_review state",
            evidence={"stage": "pin-draft", "path": str(target)},
        )
    if document["enforced_schema_sha256"] is not None:
        raise ContractError(
            "draft carries a non-None enforced_schema_sha256; the generator never writes "
            "enforced values (promotion is the human pin's decision)",
            evidence={"stage": "pin-draft", "path": str(target)},
        )
    if fileops.canonical_json(document) != payload:
        raise ContractError(
            "draft is not in canonical byte form (rewritten or tampered?)",
            evidence={"stage": "pin-draft", "path": str(target)},
        )
    return document, hashlib.sha256(payload).hexdigest()


def _validate_candidate(document: Mapping[str, Any]) -> dict[str, Any]:
    candidate = _keys(document["candidate_contract"], _CANDIDATE_KEYS, "draft.candidate_contract")
    if candidate["name"] is not None:
        raise ContractError(
            "draft candidate name must be None (the name is assigned by the human pin)",
            evidence={"stage": "pin-draft", "field": "candidate_contract.name"},
        )
    if candidate["component"] not in {"transformer", "text_encoder"}:
        raise ContractError(
            f"draft candidate component must be transformer/text_encoder, got {candidate['component']!r}",
            evidence={"stage": "pin-draft", "field": "candidate_contract.component"},
        )
    _positive_int(candidate["tensor_count"], "candidate_contract.tensor_count")
    # P4d: a bf16-plain (dense) draft pins zero ConvRot groups, so the arity
    # gate is non-negative here; the storage-kind coherence block below is
    # what decides what zero groups may mean (dense template) or forbids it
    # (an int8-convrot draft without a positive group count is a disguise).
    _non_negative_int(candidate["convrot_group_count"], "candidate_contract.convrot_group_count")
    _hex64(candidate["observed_schema_sha256"], "candidate_contract.observed_schema_sha256")
    if not isinstance(candidate["include_transformer_adaln"], bool):
        raise ContractError(
            "candidate_contract.include_transformer_adaln must be a boolean",
            evidence={"stage": "pin-draft", "field": "candidate_contract.include_transformer_adaln"},
        )
    if candidate["transformer_adaln_group_size"] is not None:
        _positive_int(candidate["transformer_adaln_group_size"], "candidate_contract.transformer_adaln_group_size")
    if not isinstance(candidate["template_name"], str) or candidate["template_name"] not in ARCHITECTURE_TEMPLATES:
        raise ContractError(
            f"draft references unknown template {candidate['template_name']!r}",
            evidence={"stage": "pin-draft", "field": "candidate_contract.template_name"},
        )
    template = ARCHITECTURE_TEMPLATES[candidate["template_name"]]
    table_size = len(template.convrot_table())
    if candidate["convrot_group_count"] != table_size:
        raise ContractError(
            f"draft pins {candidate['convrot_group_count']} ConvRot groups but template "
            f"{candidate['template_name']!r} pins {table_size}",
            evidence={"stage": "pin-draft", "field": "candidate_contract.convrot_group_count"},
        )
    if candidate["storage_kind"] == STORAGE_INT8_CONVROT:
        # Fail-closed (P4d): an int8-convrot draft must carry a positive group
        # count against a ConvRot template -- never a zero-group dense template.
        if candidate["convrot_group_count"] < 1 or table_size < 1:
            raise ContractError(
                "int8-convrot drafts pin at least one ConvRot group against a ConvRot template "
                "(a zero-group dense template is not an int8-convrot checkpoint)",
                evidence={"stage": "pin-draft", "field": "candidate_contract.storage_kind"},
            )
    elif candidate["storage_kind"] == STORAGE_BF16_PLAIN:
        # Fail-closed (P4d): a dense draft pins zero ConvRot groups against a
        # dense template (empty ConvRot table) and never carries AdaLN
        # quantization flags -- mirroring the snapshot loader exactly, so the
        # pin product is loadable instead of a write-only dead file.
        if candidate["convrot_group_count"] != 0 or table_size != 0:
            raise ContractError(
                "bf16-plain drafts pin zero ConvRot groups against a dense template "
                "(zero-group convrot disguise is forbidden)",
                evidence={"stage": "pin-draft", "field": "candidate_contract.storage_kind"},
            )
        if candidate["include_transformer_adaln"] or candidate["transformer_adaln_group_size"] is not None:
            raise ContractError(
                "bf16-plain drafts carry no AdaLN ConvRot quantization flags",
                evidence={"stage": "pin-draft", "field": "candidate_contract.include_transformer_adaln"},
            )
        if not template.non_quantized_inventory:
            raise ContractError(
                "bf16-plain drafts require a template with a complete pinned tensor manifest",
                evidence={"stage": "pin-draft", "field": "candidate_contract.template_name"},
            )
    else:
        raise ContractError(
            f"draft candidate storage_kind must be one of "
            f"{sorted((STORAGE_BF16_PLAIN, STORAGE_INT8_CONVROT))}, got {candidate['storage_kind']!r}",
            evidence={"stage": "pin-draft", "field": "candidate_contract.storage_kind"},
        )
    _positive_int(candidate["template_version"], "candidate_contract.template_version")
    return candidate


def _verify_template_binding(document: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    """The recorded template digest must equal the current template tables."""

    provenance = document["provenance"]
    recorded = _keys(provenance["template"], _PROVENANCE_TEMPLATE_KEYS, "draft.provenance.template")
    template = ARCHITECTURE_TEMPLATES[candidate["template_name"]]
    if recorded["name"] != candidate["template_name"] or recorded["version"] != candidate["template_version"]:
        raise ContractError(
            "draft provenance template reference disagrees with the candidate contract",
            evidence={
                "stage": "pin-template",
                "provenance_template": recorded,
                "candidate_template": [candidate["template_name"], candidate["template_version"]],
            },
        )
    current_digest = template_digest(template)
    if recorded["digest"] != current_digest:
        raise ContractError(
            "the template drifted between draft and pin (recorded digest != current digest); "
            "the draft is stale and must be regenerated",
            evidence={
                "stage": "pin-template",
                "template": candidate["template_name"],
                "recorded_digest": recorded["digest"],
                "current_digest": current_digest,
            },
        )


def _verify_census_digest(document: Mapping[str, Any]) -> str:
    provenance = document["provenance"]
    recorded = _hex64(provenance["census_sha256"], "provenance.census_sha256")
    recomputed = hashlib.sha256(fileops.canonical_json(document["census"])).hexdigest()
    if recorded != recomputed:
        raise ContractError(
            "draft census block does not match its recorded census digest (tampered?)",
            evidence={"stage": "pin-census", "recorded": recorded, "recomputed": recomputed},
        )
    return recorded


def _verify_source_digests(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Re-hash every source file recorded in the draft; stale means refuse.

    Each source is re-hashed through the census/:class:`SafeTensorSources`
    protocol (:func:`comfy_omni.fileops.sha256_file_pinned`): the leaf is checked
    un-linked, then the file is opened (``O_NOFOLLOW`` best effort), the
    descriptor fstat-pinned with the path->fd identity check, hashed in one
    sequential pass, its five-field identity re-checked on the descriptor
    after the hash, and the path re-stat'ed to still name the hashed file --
    so the digest compared below is provably the digest of the file the path
    named, taken while nothing swapped or rewrote it (a same-size rewrite or
    a swapped path refuses as a TOCTOU, not as a silent mismatch).
    """

    provenance = document["provenance"]
    files = provenance["source_files"]
    if not isinstance(files, list) or not files:
        raise ContractError(
            "draft provenance source_files must be a non-empty array",
            evidence={"stage": "pin-source", "field": "provenance.source_files"},
        )
    verified: list[dict[str, Any]] = []
    for entry in files:
        record = _keys(entry, _SOURCE_FILE_KEYS, "draft.provenance.source_files[]")
        if not isinstance(record["path"], str) or not record["path"]:
            raise ContractError(
                "recorded source path must be a non-empty string",
                evidence={"stage": "pin-source", "field": "source_files[].path"},
            )
        _hex64(record["sha256"], "source_files[].sha256")
        if not isinstance(record["size"], int) or isinstance(record["size"], bool) or record["size"] < 0:
            raise ContractError(
                "recorded source size must be a non-negative integer",
                evidence={"stage": "pin-source", "field": "source_files[].size"},
            )
        source = Path(record["path"])
        try:
            if fileops.is_link(source) or not source.is_file():
                raise ContractError(
                    f"recorded source file is missing (stale draft): {source}",
                    evidence={"stage": "pin-source", "path": str(source), "recorded_sha256": record["sha256"]},
                )
            observed_sha, observed_size = fileops.sha256_file_pinned(source)
        except fileops.FsopsModifiedError as error:
            raise ContractError(
                f"source file changed while being re-hashed (TOCTOU rejected): {source}",
                evidence={
                    "stage": "pin-source",
                    "path": str(source),
                    "reason": "modified-during-read",
                    "recorded": {"size": record["size"], "sha256": record["sha256"]},
                },
            ) from error
        except (fileops.FsopsError, OSError) as error:
            raise ContractError(
                f"recorded source file could not be re-hashed: {source}: {error}",
                evidence={"stage": "pin-source", "path": str(source)},
            ) from error
        if observed_size != record["size"] or observed_sha != record["sha256"]:
            raise ContractError(
                f"source file changed after the draft (stale draft, refusing to pin): {source}",
                evidence={
                    "stage": "pin-source",
                    "path": str(source),
                    "recorded": {"size": record["size"], "sha256": record["sha256"]},
                    "observed": {"size": observed_size, "sha256": observed_sha},
                },
            )
        verified.append({"path": record["path"], "size": record["size"], "sha256": record["sha256"]})
    return verified


def _verify_reviewer_separation(reviewer: str, generated_by: Mapping[str, Any]) -> None:
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ContractError(
            "reviewer must be a non-empty name", evidence={"stage": "pin-reviewer", "reviewer": reviewer}
        )
    for key, value in sorted(generated_by.items()):
        if value is not None and value == reviewer:
            raise ContractError(
                f"reviewer {reviewer!r} equals the recorded generator identity {key}={value!r}; "
                "the generator of a draft cannot be its reviewer (generated_by != reviewed_by)",
                evidence={"stage": "pin-reviewer", "reviewer": reviewer, "generator_field": key},
            )


def _evidence_digest(path: Path | str) -> str:
    """The SHA-256 of the evidence file bytes (the path is never recorded)."""

    target = Path(path)
    try:
        if fileops.is_link(target) or not target.is_file():
            raise ContractError(
                f"evidence report must be a regular non-linked file: {target}",
                evidence={"stage": "pin-evidence", "path": str(target)},
            )
        digest, _ = fileops.sha256_file_pinned(target)
        return digest
    except (fileops.FsopsError, OSError) as error:
        raise ContractError(
            f"evidence report could not be hashed: {target}: {error}",
            evidence={"stage": "pin-evidence", "path": str(target)},
        ) from error


def _promotion_decision(
    document: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    enforce_observed_schema: bool,
    catalog: ContractCatalog,
) -> tuple[str, str | None]:
    nearest = document["nearest_known_contract"]
    if nearest is None:
        # P4d dense route: the compile-time registry pins no bf16-plain
        # contract, so a dense draft has no nearest *known* contract to
        # inherit an enforced pin from.  Only the dense storage kind may
        # arrive here -- an int8-convrot draft without a known contract is
        # a draft the generator refuses to produce and the pin refuses too.
        if candidate["storage_kind"] != STORAGE_BF16_PLAIN:
            raise ContractError(
                "draft names no nearest known contract; only bf16-plain (dense) drafts may pin "
                "without one (int8-convrot drafts always name their nearest known contract)",
                evidence={"stage": "pin-promotion", "known": nearest},
            )
        observed_schema: str = candidate["observed_schema_sha256"]
        if enforce_observed_schema:
            return DECISION_OBSERVED_FROZEN, observed_schema
        return DECISION_CENSUS_ONLY, None
    known_record = catalog.records.get(nearest)
    known = None if known_record is None else known_record.contract
    if known is None:
        raise ContractError(
            f"draft names unknown nearest known contract {document['nearest_known_contract']!r}",
            evidence={"stage": "pin-promotion", "known": document["nearest_known_contract"]},
        )
    structural = {
        "component": known.component,
        "tensor_count": known.tensor_count,
        "convrot_group_count": known.convrot_group_count,
        "include_transformer_adaln": known.include_transformer_adaln,
        "transformer_adaln_group_size": known.transformer_adaln_group_size,
    }
    observed = {
        "component": candidate["component"],
        "tensor_count": candidate["tensor_count"],
        "convrot_group_count": candidate["convrot_group_count"],
        "include_transformer_adaln": candidate["include_transformer_adaln"],
        "transformer_adaln_group_size": candidate["transformer_adaln_group_size"],
    }
    if structural != observed:
        raise ContractError(
            "draft candidate drifted from the nearest known contract (stale or tampered draft)",
            evidence={"stage": "pin-promotion", "known": structural, "candidate": observed},
        )
    observed_schema = candidate["observed_schema_sha256"]
    if known.schema_sha256 is not None:
        if observed_schema != known.schema_sha256:
            raise ContractError(
                "the known contract enforces a schema hash the draft no longer reproduces",
                evidence={
                    "stage": "pin-promotion",
                    "enforced_schema_sha256": known.schema_sha256,
                    "observed_schema_sha256": observed_schema,
                },
            )
        return DECISION_INHERITED_ENFORCED_PIN, observed_schema
    if enforce_observed_schema:
        return DECISION_OBSERVED_FROZEN, observed_schema
    return DECISION_CENSUS_ONLY, None


def _prepare_pin(
    draft_path: Path | str,
    *,
    reviewer: str,
    evidence_path: Path | str,
    enforce_observed_schema: bool,
    catalog: ContractCatalog,
) -> _PreparedPin:
    document, draft_sha256 = load_draft_document(draft_path)
    candidate = _validate_candidate(document)
    _keys(document["provenance"], _PROVENANCE_KEYS, "draft.provenance")
    _verify_template_binding(document, candidate)
    census_sha256 = _verify_census_digest(document)
    source_files = _verify_source_digests(document)
    generated_by = document["provenance"]["generated_by"]
    if not isinstance(generated_by, dict) or not all(
        isinstance(key, str) and (value is None or isinstance(value, str)) for key, value in generated_by.items()
    ):
        raise ContractError(
            "draft provenance generated_by must be an object of string keys to string-or-null values",
            evidence={"stage": "pin-draft", "field": "provenance.generated_by"},
        )
    _verify_reviewer_separation(reviewer, generated_by)
    evidence_sha256 = _evidence_digest(evidence_path)
    decision, enforced = _promotion_decision(
        document,
        candidate,
        enforce_observed_schema=enforce_observed_schema,
        catalog=catalog,
    )
    return _PreparedPin(
        document,
        candidate,
        draft_sha256,
        census_sha256,
        source_files,
        generated_by,
        evidence_sha256,
        decision,
        enforced,
    )


def _snapshot_document(contract: NativeSourceContract, prepared: _PreparedPin, reviewer: str) -> dict[str, Any]:
    candidate = prepared.candidate
    template = ARCHITECTURE_TEMPLATES[candidate["template_name"]]
    return {
        "schema": SNAPSHOT_SCHEMA,
        "status": "PINNED",
        "pending_review": False,
        "contract": contract_block(
            contract,
            template_name=template.template_name,
            template_version=template.template_version,
            storage_kind=candidate["storage_kind"],
        ),
        "pin": {
            "schema": PIN_BLOCK_SCHEMA,
            "reviewed_by": reviewer,
            "evidence_sha256": prepared.evidence_sha256,
            "generated_by": dict(prepared.generated_by),
            "draft_sha256": prepared.draft_sha256,
            "source_files": prepared.source_files,
            "census_sha256": prepared.census_sha256,
            "template": {
                "name": candidate["template_name"],
                "version": candidate["template_version"],
                "digest": template_digest(template),
            },
            "enforced_schema_decision": prepared.decision,
        },
    }


def pin_draft(
    draft_path: Path | str,
    *,
    name: str,
    reviewer: str,
    evidence_path: Path | str,
    contract_dir: Path | str,
    enforce_observed_schema: bool = False,
    catalog: ContractCatalog = COMPILE_TIME_CATALOG,
) -> PinResult:
    """Pin one DRAFTED draft into an immutable content-addressed snapshot."""

    prepared = _prepare_pin(
        draft_path,
        reviewer=reviewer,
        evidence_path=evidence_path,
        enforce_observed_schema=enforce_observed_schema,
        catalog=catalog,
    )
    if not isinstance(name, str) or not name.strip():
        raise ContractError("contract name must be a non-empty string", evidence={"stage": "pin-name", "name": name})
    if name in catalog.records:
        raise ContractError(
            f"contract name {name!r} shadows a compile-time profile name",
            evidence={"stage": "pin-name", "name": name},
        )
    candidate = prepared.candidate
    contract = NativeSourceContract(
        name=name,
        component=candidate["component"],
        tensor_count=candidate["tensor_count"],
        convrot_group_count=candidate["convrot_group_count"],
        schema_sha256=prepared.enforced,
        include_transformer_adaln=candidate["include_transformer_adaln"],
        transformer_adaln_group_size=candidate["transformer_adaln_group_size"],
    )
    snapshot = write_snapshot(contract_dir, _snapshot_document(contract, prepared, reviewer))
    return PinResult(
        snapshot=snapshot,
        contract=contract,
        enforced_schema_decision=prepared.decision,
        enforced_schema_sha256=prepared.enforced,
        evidence_sha256=prepared.evidence_sha256,
        reviewer=reviewer,
        draft_sha256=prepared.draft_sha256,
    )


__all__ = [
    "PinResult",
    "load_draft_document",
    "pin_draft",
]
