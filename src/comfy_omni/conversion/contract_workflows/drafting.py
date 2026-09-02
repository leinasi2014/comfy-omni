"""Contract draft generator (P4b, design v2 §2/§3/§8).

Turns one census + three-level match into a
:class:`~comfy_omni.h3.contracts.NativeSourceContract`-shaped **draft** with

* the observed candidate contract fields (``observed_schema_sha256``
  included),
* the field-by-field diff against the nearest *known* pinned contract
  (same template), and
* the L3 template-match evidence.

Observed vs enforced separation (design review #6, hardened by QA P4b-qa2):
the generator records ``observed_schema_sha256`` and *never* writes an
enforced value -- whether the observed hash is frozen as
``enforced_schema_sha256`` or the pinned None-enforced (census-only) policy
is kept is a human promotion decision (P4c).  The *known* contract's
enforced pin is a hard gate, not advice: when the nearest known contract
pins a non-None ``schema_sha256`` and the observed hash deviates, the draft
is refused fail-closed with both hashes and the census summary as evidence
-- an enforced pin authorizes exactly one schema, so a deviation means the
source is not the pinned source.  The red-flag class survives only for a
future enforced=None template form without inventory coverage (no current
template has that form); every structural census deviation (component /
tensor count / group count / AdaLN flags) fails closed with the census diff
summary as evidence.

Registration/pinning is out of scope by construction: the generator
produces a draft document only (the four-step milestone stops at DRAFTED).

Dense bf16-plain exception (P4d, QA P4d-qa3): the one shape that may draft
without a *known* pinned contract is a dense ``bf16-plain`` census whose
unique L3 pass is against a zero-group dense template carrying a complete
pinned tensor manifest -- the manifest is the item-by-item authority for
every tensor of the source (name, dtype and shape all matched at L3), so
there is nothing a known contract could add.  The draft honestly records
``nearest_known_contract=None`` (serialized as JSON ``null``).  Every other
unpinned architecture -- int8-convrot above all -- keeps failing closed at
the known-contract gate: drafting those stays human work (registration is
P4c).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from comfy_omni.artifacts import fileops
from comfy_omni.contracts.models import STORAGE_BF16_PLAIN, ContractCatalog, NativeSourceContract
from comfy_omni.contracts.registry import COMPILE_TIME_CATALOG
from comfy_omni.contracts.templates import ARCHITECTURE_TEMPLATES, template_digest
from comfy_omni.conversion.contract_workflows.census import CensusReport, ContractScanError
from comfy_omni.conversion.contract_workflows.matching import MatchReport, TemplateMatchResult

#: JSON schema identifier of the draft document.
DRAFT_SCHEMA = "h3_forge.contract_auto.draft/v1"

#: The promotion note every draft carries (observed/enforced separation).
PROMOTION_NOTE = (
    "enforced_schema_sha256 is a human promotion decision (P4c pin): the generator "
    "records the observed hash only; freezing it or keeping the None-enforced "
    "census-only policy stays with the reviewer"
)

#: Default generator identity: empty until the caller records one (the CLI
#: records the executing wheel identity plus the operator name).
EMPTY_GENERATOR_IDENTITY: MappingProxyType[str, str | None] = MappingProxyType({})


@dataclass(frozen=True)
class FieldDiff:
    """One field of the draft-vs-known-contract diff.

    ``red`` marks a field that needs a human promotion decision without
    being a structural failure.  QA P4b-qa2 reclassified the schema hash:
    a deviation from a non-None enforced pin is a hard REJECT in
    :func:`build_draft`, never a draft-time red flag, so the red class
    survives only for a future enforced=None template form without
    inventory coverage (no current template has that form).
    """

    field: str
    known: Any
    observed: Any
    equal: bool
    red: bool = False
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "field": self.field,
            "known": self.known,
            "observed": self.observed,
            "equal": self.equal,
            "red": self.red,
        }
        if self.note is not None:
            payload["note"] = self.note
        return payload


@dataclass(frozen=True)
class ContractDraft:
    """The DRAFTED artifact: candidate contract + diff + evidence.

    P4c binding chain (design v2, section 2, review #7): the serialized
    draft records the full provenance the pin later re-verifies -- the
    scanned source file digest census, the census digest, the matched
    template's digest + version, and the generator identity (executing
    wheel/commit plus the operator who ran the generator).  A draft is
    ``pending_review`` until a human pins it.
    """

    template_name: str
    template_version: int
    component: str
    storage_kind: str
    tensor_count: int
    convrot_group_count: int
    observed_schema_sha256: str
    include_transformer_adaln: bool
    transformer_adaln_group_size: int | None
    #: ``None`` exactly on the P4d dense bf16-plain route (no compile-time
    #: bf16-plain contract exists; the template manifest is the authority) --
    #: serialized honestly as JSON ``null``.
    nearest_known_contract: str | None
    diff: tuple[FieldDiff, ...]
    census: CensusReport
    match: MatchReport
    ambiguity_resolved_by_h3_evidence: bool
    generator_identity: Mapping[str, str | None] = field(default=EMPTY_GENERATOR_IDENTITY, repr=False)

    @property
    def red_flags(self) -> tuple[FieldDiff, ...]:
        return tuple(item for item in self.diff if item.red)

    def provenance_dict(self) -> dict[str, Any]:
        """The binding-chain block the pin re-verifies (review #7)."""

        return {
            "source_files": [
                {"path": item.path, "size": item.size, "sha256": item.sha256} for item in self.census.files
            ],
            "census_sha256": hashlib.sha256(fileops.canonical_json(self.census.to_dict())).hexdigest(),
            "template": {
                "name": self.template_name,
                "version": self.template_version,
                "digest": template_digest(ARCHITECTURE_TEMPLATES[self.template_name]),
            },
            "generated_by": dict(self.generator_identity),
        }

    def candidate_dict(self) -> dict[str, Any]:
        return {
            "name": None,  # assigned by the human pin (P4c), never generated
            "component": self.component,
            "tensor_count": self.tensor_count,
            "convrot_group_count": self.convrot_group_count,
            "observed_schema_sha256": self.observed_schema_sha256,
            "include_transformer_adaln": self.include_transformer_adaln,
            "transformer_adaln_group_size": self.transformer_adaln_group_size,
            "storage_kind": self.storage_kind,
            "template_name": self.template_name,
            "template_version": self.template_version,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DRAFT_SCHEMA,
            "status": "DRAFTED",
            "pending_review": True,
            "candidate_contract": self.candidate_dict(),
            "enforced_schema_sha256": None,
            "promotion": {
                "decision": "human",
                "pin": "out of scope for the generator (P4c)",
                "observed_vs_enforced": PROMOTION_NOTE,
            },
            "nearest_known_contract": self.nearest_known_contract,
            "diff_vs_known": [item.to_dict() for item in self.diff],
            "red_flags": [item.field for item in self.red_flags],
            "template_evidence": self.template_result.to_dict(),
            "l1_ambiguity_resolved_by_h3_evidence": self.ambiguity_resolved_by_h3_evidence,
            "three_level_match": self.match.to_dict(),
            "census": self.census.to_dict(),
            "provenance": self.provenance_dict(),
        }

    def to_json(self) -> str:
        return fileops.canonical_json(self.to_dict()).decode("utf-8")

    @property
    def template_result(self) -> TemplateMatchResult:
        for result in self.match.template_results:
            if result.template_name == self.template_name:
                return result
        raise ContractScanError(
            f"internal: matched template {self.template_name!r} missing from the match report",
            evidence={"stage": "draft"},
        )

    def write_to(self, path: Path | str) -> None:
        """Write the draft JSON exclusively (no overwrite, ever)."""

        target = Path(path)
        payload = fileops.canonical_json(self.to_dict())
        try:
            fileops.reject_linked_ancestors(target, allow_missing_final=True)
            fileops.write_exclusive(target, payload)
        except fileops.FsopsExistsError as error:
            cause = error.__cause__
            assert isinstance(cause, FileExistsError)
            raise ContractScanError(
                f"refusing to overwrite an existing draft file: {target}",
                evidence={"stage": "draft-write", "path": str(target)},
            ) from cause
        except fileops.FsopsIoError as error:
            cause = error.__cause__
            assert isinstance(cause, OSError)
            raise ContractScanError(
                f"draft file could not be written: {target}: {cause}",
                evidence={"stage": "draft-write", "path": str(target)},
            ) from cause
        except fileops.FsopsError as error:
            raise ContractScanError(
                f"draft path is not safe for immutable publication: {target}: {error}",
                evidence={"stage": "draft-write", "path": str(target)},
            ) from error


def _observed_adaln_group_size(census: CensusReport, template: Any) -> int | None:
    """Observe the group size of the curve-quantized AdaLN weights.

    Mechanical derivation from the template tables: the quantized weight
    names that are also curve-AdaLN census members (exactly the 50
    ``blocks.N.adaln_proj.linear.weight`` groups of the adaln64 template)
    must share one observed group size (64 for the pinned source).
    """

    if not template.curve_adaln_tensors:
        return None
    quantized_curve_weights = template.quantized_weight_names() & template.curve_adaln_tensors
    sizes = {group.group_size for group in census.groups if f"{group.prefix}.weight" in quantized_curve_weights}
    if len(sizes) != 1:
        raise ContractScanError(
            "the AdaLN curve-quantized ConvRot groups do not share one observed group size",
            evidence={"stage": "draft", "observed_group_sizes": sorted(sizes)},
        )
    return sizes.pop()


def _field_diffs(
    census: CensusReport,
    *,
    component: str,
    include_transformer_adaln: bool,
    transformer_adaln_group_size: int | None,
    known: Any,
) -> tuple[FieldDiff, ...]:
    schema_equal = known.schema_sha256 is None or known.schema_sha256 == census.observed_schema_sha256
    schema_note = (
        "enforced hash is None on the known contract (census-only policy); the observed "
        "hash is recorded for the human promotion decision"
        if known.schema_sha256 is None
        else "observed schema hash equals the enforced pin of the known contract"
    )
    return (
        FieldDiff(field="component", known=known.component, observed=component, equal=known.component == component),
        FieldDiff(
            field="tensor_count",
            known=known.tensor_count,
            observed=census.tensor_count,
            equal=known.tensor_count == census.tensor_count,
        ),
        FieldDiff(
            field="convrot_group_count",
            known=known.convrot_group_count,
            observed=census.convrot_group_count,
            equal=known.convrot_group_count == census.convrot_group_count,
        ),
        FieldDiff(
            field="include_transformer_adaln",
            known=known.include_transformer_adaln,
            observed=include_transformer_adaln,
            equal=known.include_transformer_adaln == include_transformer_adaln,
        ),
        FieldDiff(
            field="transformer_adaln_group_size",
            known=known.transformer_adaln_group_size,
            observed=transformer_adaln_group_size,
            equal=known.transformer_adaln_group_size == transformer_adaln_group_size,
        ),
        FieldDiff(
            field="schema_sha256",
            known=known.schema_sha256,
            observed=census.observed_schema_sha256,
            equal=schema_equal,
            # QA P4b-qa2: with a non-None enforced pin a deviation is a hard
            # REJECT (see build_draft), never a draft-time red flag; the red
            # class stays reserved for a future enforced=None template form
            # without inventory coverage (no current template has that form).
            red=False,
            note=schema_note,
        ),
    )


def _dense_unpinned_exception(census: CensusReport, template: Any) -> bool:
    """Whether the P4d dense route may draft with no known pinned contract.

    Strictly limited to ``bf16-plain`` (QA P4d-qa3): the census observed a
    marker-free dense checkpoint, the (uniquely) matched template is a
    *dense* template -- an empty ConvRot table, never a zero-group convrot
    disguise -- and that template pins a complete tensor manifest, which L3
    has just validated item by item (name, dtype and shape of every tensor
    of the source).  Anything else (int8-convrot above all) keeps requiring
    exactly one known pinned contract.
    """

    return (
        census.storage_kind == STORAGE_BF16_PLAIN
        and census.convrot_group_count == 0
        and not template.convrot_table()
        and bool(template.non_quantized_inventory)
    )


def _winning_template(census: CensusReport, match: MatchReport) -> tuple[TemplateMatchResult, bool]:
    if match.routing.ambiguous_unresolvable:
        raise ContractScanError(
            "L1 routing is ambiguous across signature families and no sufficient H3-specific "
            "tensor combination can disambiguate; draft refused (design v2 section 3)",
            evidence={
                "stage": "l1-routing",
                "routing": match.routing.to_dict(),
                "census_summary": census.census_summary(),
            },
        )
    passed = [result for result in match.template_results if result.passed]
    if not passed:
        raise ContractScanError(
            "no architecture template matched the census exactly; draft refused (fail-closed)",
            evidence={
                "stage": "l3-template-match",
                "census_summary": census.census_summary(),
                "template_results": [result.to_dict() for result in match.template_results],
            },
        )
    if len(passed) > 1:
        raise ContractScanError(
            f"{len(passed)} architecture templates matched exactly; the draft refuses to pick between them",
            evidence={
                "stage": "l3-template-match",
                "matched_templates": [result.template_name for result in passed],
                "census_summary": census.census_summary(),
            },
        )
    resolved = match.routing.status == "ambiguous" and match.routing.h3_evidence_resolves
    return passed[0], resolved


def _known_contract(
    census: CensusReport,
    winner: TemplateMatchResult,
    catalog: ContractCatalog,
) -> NativeSourceContract | None:
    template = ARCHITECTURE_TEMPLATES[winner.template_name]
    known_views = [record for record in catalog.records.values() if record.template_name == winner.template_name]
    if not known_views and _dense_unpinned_exception(census, template):
        return None
    if len(known_views) != 1:
        raise ContractScanError(
            f"no unique pinned contract exists for template {winner.template_name!r}; drafting "
            "an unpinned architecture is human work (registration is P4c)",
            evidence={
                "stage": "known-contract-resolution",
                "template": winner.template_name,
                "known_contracts": sorted(record.name for record in catalog.records.values()),
                "census_summary": census.census_summary(),
            },
        )
    return known_views[0].contract


def _validate_known_contract(
    census: CensusReport,
    winner: TemplateMatchResult,
    known: NativeSourceContract,
) -> tuple[tuple[FieldDiff, ...], bool, int | None]:
    template = ARCHITECTURE_TEMPLATES[winner.template_name]
    include_adaln = bool(template.curve_adaln_tensors)
    adaln_group_size = _observed_adaln_group_size(census, template)
    diff = _field_diffs(
        census,
        component=winner.component,
        include_transformer_adaln=include_adaln,
        transformer_adaln_group_size=adaln_group_size,
        known=known,
    )
    structural = [item for item in diff if item.field != "schema_sha256" and not item.red and not item.equal]
    if structural:
        census_diff = {item.field: {"known": item.known, "observed": item.observed} for item in structural}
        raise ContractScanError(
            f"nearest known contract {known.name!r} census diff is non-empty "
            f"({', '.join(item.field for item in structural)}); draft refused (fail-closed)",
            evidence={
                "stage": "known-contract-diff",
                "known_contract": known.name,
                "census_diff": census_diff,
                "census_summary": census.census_summary(),
            },
        )
    schema_diff = next(item for item in diff if item.field == "schema_sha256")
    if schema_diff.known is not None and not schema_diff.equal:
        raise ContractScanError(
            f"observed schema sha256 deviates from the enforced pin of known contract "
            f"{known.name!r}: an enforced pin authorizes exactly one schema, so the source "
            "is not the pinned source; draft refused (fail-closed)",
            evidence={
                "stage": "schema-hash-enforcement",
                "known_contract": known.name,
                "enforced_schema_sha256": schema_diff.known,
                "observed_schema_sha256": schema_diff.observed,
                "census_summary": census.census_summary(),
            },
        )
    return diff, include_adaln, adaln_group_size


def _make_draft(
    census: CensusReport,
    match: MatchReport,
    winner: TemplateMatchResult,
    *,
    known: NativeSourceContract | None,
    diff: tuple[FieldDiff, ...],
    ambiguity_resolved: bool,
    generator_identity: Mapping[str, str | None] | None,
) -> ContractDraft:
    template = ARCHITECTURE_TEMPLATES[winner.template_name]
    return ContractDraft(
        template_name=winner.template_name,
        template_version=winner.template_version,
        component=winner.component,
        storage_kind=census.storage_kind,
        tensor_count=census.tensor_count,
        convrot_group_count=census.convrot_group_count,
        observed_schema_sha256=census.observed_schema_sha256,
        include_transformer_adaln=bool(template.curve_adaln_tensors),
        transformer_adaln_group_size=_observed_adaln_group_size(census, template),
        nearest_known_contract=None if known is None else known.name,
        diff=diff,
        census=census,
        match=match,
        ambiguity_resolved_by_h3_evidence=ambiguity_resolved,
        generator_identity=MappingProxyType(dict(generator_identity or {})),
    )


def build_draft(
    census: CensusReport,
    match: MatchReport,
    *,
    generator_identity: Mapping[str, str | None] | None = None,
    catalog: ContractCatalog = COMPILE_TIME_CATALOG,
) -> ContractDraft:
    """Generate a pending-review draft only after exact, fail-closed matching."""

    winner, ambiguity_resolved = _winning_template(census, match)
    known = _known_contract(census, winner, catalog)
    diff: tuple[FieldDiff, ...] = ()
    if known is not None:
        diff, _, _ = _validate_known_contract(census, winner, known)
    return _make_draft(
        census,
        match,
        winner,
        known=known,
        diff=diff,
        ambiguity_resolved=ambiguity_resolved,
        generator_identity=generator_identity,
    )
