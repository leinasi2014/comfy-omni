"""Three-level contract matching with exact templates as the sole authority.

Derived from Apache-2.0 h3-forge matcher.py blob
521bbd63051a2a617ddb2385751f9458bc024625 at commit e9cb011d00b028c149db3978de246c54f6e34acc.
The broad legacy Oracle dependency is retired; L1 now reports H3-only weak evidence and never
authorizes a draft. Persisted report schema and L2/L3 meanings remain compatible.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from comfy_omni.contracts.models import ArchitectureTemplate
from comfy_omni.contracts.templates import ARCHITECTURE_TEMPLATES
from comfy_omni.conversion.contract_workflows.census import CensusReport

SCAN_SCHEMA = "h3_forge.contract_auto.scan/v1"
ROUTING_WEAK_PASS = "weak-pass"
ROUTING_NO_MATCH = "no-match"
ROUTING_AMBIGUOUS = "ambiguous"
MIN_H3_DISAMBIGUATION_EVIDENCE_GROUPS = 2
_H3_SPECIFIC_TOKENS = (
    "audio_patch_proj",
    "video_patch_proj",
    "token_refiner",
    "curve_model",
    "adaln_t_table",
)


def _contains_name_token(name: str, token: str) -> bool:
    return re.search(rf"(?:^|[._]){re.escape(token)}(?:[._]|$)", name) is not None


def h3_specific_evidence(names: tuple[str, ...]) -> tuple[str, ...]:
    """Return grouped H3-only tensor-name evidence for weak routing."""

    evidence: list[str] = []
    for token in _H3_SPECIFIC_TOKENS:
        hits = sorted(name for name in names if _contains_name_token(name, token))
        if hits:
            evidence.append(f"{token}: {len(hits)} tensors (e.g. {hits[0]})")
    return tuple(evidence)


@dataclass(frozen=True)
class Level1Routing:
    status: str
    fired_families: tuple[str, ...]
    candidates: tuple[dict[str, Any], ...]
    family_evidence: tuple[dict[str, Any], ...]
    h3_specific_evidence: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def h3_evidence_resolves(self) -> bool:
        return len(self.h3_specific_evidence) >= MIN_H3_DISAMBIGUATION_EVIDENCE_GROUPS

    @property
    def ambiguous_unresolvable(self) -> bool:
        return self.status == ROUTING_AMBIGUOUS and not self.h3_evidence_resolves

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "role": "weak-routing-only (cannot authorize; L3 is the sole authority)",
            "fired_families": list(self.fired_families),
            "candidates": [dict(item) for item in self.candidates],
            "family_evidence": [dict(item) for item in self.family_evidence],
            "h3_specific_evidence": list(self.h3_specific_evidence),
            "h3_evidence_resolves": self.h3_evidence_resolves,
            "notes": list(self.notes),
            "ambiguous_unresolvable": self.ambiguous_unresolvable,
        }


def route_level1(report: CensusReport) -> Level1Routing:
    """Produce weak H3 routing without importing the retired broad Oracle."""

    names = tuple(descriptor.name for descriptor in report.descriptors)
    evidence = h3_specific_evidence(names)
    if evidence:
        return Level1Routing(
            ROUTING_WEAK_PASS,
            ("minimax-h3",),
            (),
            ({"family": "minimax-h3", "coverage": 1.0, "pipelines": []},),
            evidence,
            ("H3-only tensor groups fired; weak routing only",),
        )
    return Level1Routing(
        ROUTING_NO_MATCH,
        (),
        (),
        (),
        (),
        ("no H3-only weak-routing token fired; absence never blocks an exact L3 pass",),
    )


@dataclass(frozen=True)
class Level2Classification:
    component: str
    evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "role": "advisory (component authority of a draft is the L3-matched template)",
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class TemplateMatchResult:
    template_name: str
    template_version: int
    component: str
    screened_in: bool
    passed: bool
    checks: tuple[str, ...]
    census_diff: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_name": self.template_name,
            "template_version": self.template_version,
            "component": self.component,
            "screened_in": self.screened_in,
            "passed": self.passed,
            "checks": list(self.checks),
            "census_diff": self.census_diff,
        }


def _table_diff(report: CensusReport, template: ArchitectureTemplate) -> tuple[dict[str, Any], list[str]]:
    expected = template.convrot_table()
    observed = {group.prefix: (group.weight.shape, group.group_size) for group in report.groups}
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    common = set(expected) & set(observed)
    wrong_shapes = sorted(
        (name, list(expected[name][0]), list(observed[name][0]))
        for name in common
        if expected[name][0] != observed[name][0]
    )
    wrong_sizes = sorted(
        (name, expected[name][1], observed[name][1]) for name in common if expected[name][1] != observed[name][1]
    )
    diff: dict[str, Any] = {}
    checks: list[str] = []
    for key, values, message in (
        ("missing_group_prefixes", missing, "missing ConvRot prefixes"),
        ("extra_group_prefixes", extra, "extra ConvRot prefixes"),
        ("wrong_weight_shapes", wrong_shapes, "wrong weight shapes"),
        ("wrong_group_sizes", wrong_sizes, "wrong group sizes"),
    ):
        if values:
            diff[key] = [list(value) if isinstance(value, tuple) else value for value in values[:8]]
            checks.append(f"{message}: {values[:4]}")
    return diff, checks


def _inventory_diff(report: CensusReport, template: ArchitectureTemplate) -> dict[str, Any] | None:
    expected = dict(template.non_quantized_inventory)
    if not expected:
        return None
    triplets = {name for group in report.groups for name in (group.weight.name, group.scale.name, group.marker.name)}
    observed = {
        descriptor.name: (descriptor.dtype, descriptor.shape)
        for descriptor in report.descriptors
        if descriptor.name not in triplets
    }
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    wrong = sorted(
        (name, [expected[name][0], list(expected[name][1])], [observed[name][0], list(observed[name][1])])
        for name in set(expected) & set(observed)
        if expected[name] != observed[name]
    )
    if not missing and not extra and not wrong:
        return None
    return {
        "expected_count": len(expected),
        "observed_count": len(observed),
        "missing": missing[:8],
        "extra": extra[:8],
        "wrong_dtype_or_shape": [list(item) for item in wrong[:8]],
    }


def validate_level3(report: CensusReport, template: ArchitectureTemplate) -> TemplateMatchResult:
    """Validate group tables and any complete non-quantized inventory item by item."""

    diff, checks = _table_diff(report, template)
    expected_count = len(template.convrot_table())
    if report.convrot_group_count != expected_count:
        diff["convrot_group_count"] = {"expected": expected_count, "observed": report.convrot_group_count}
        checks.append(f"convrot group count: observed {report.convrot_group_count}, template pins {expected_count}")
    expected_census = {
        f"{rows}x{columns}": count for (rows, columns), count in sorted(template.scale_shape_census.items())
    }
    observed_census = dict(sorted(report.scale_shape_census.items()))
    if observed_census != expected_census:
        diff["scale_shape_census"] = {"expected": expected_census, "observed": observed_census}
        checks.append("scale-shape census mismatch")
    if inventory := _inventory_diff(report, template):
        diff["non_quantized_inventory"] = inventory
        checks.append("non-quantized tensor inventory mismatch")
    return TemplateMatchResult(
        template.template_name,
        template.template_version,
        template.component,
        True,
        not diff,
        tuple(checks),
        diff,
    )


@dataclass(frozen=True)
class MatchReport:
    routing: Level1Routing
    component: Level2Classification
    template_results: tuple[TemplateMatchResult, ...]
    matched_template: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "l1_routing": self.routing.to_dict(),
            "l2_component": self.component.to_dict(),
            "l3_templates": [result.to_dict() for result in self.template_results],
            "matched_template": self.matched_template,
        }


@dataclass(frozen=True)
class ScanReport:
    census: CensusReport
    match: MatchReport

    @property
    def matched_template(self) -> str | None:
        return self.match.matched_template

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCAN_SCHEMA,
            "status": "DISCOVERED",
            "census": self.census.to_dict(),
            "match": self.match.to_dict(),
        }


def _screened_result(report: CensusReport, template: ArchitectureTemplate) -> TemplateMatchResult:
    expected = len(template.convrot_table())
    if expected == report.convrot_group_count:
        return validate_level3(report, template)
    return TemplateMatchResult(
        template.template_name,
        template.template_version,
        template.component,
        False,
        False,
        (f"screened out: convrot groups {report.convrot_group_count} != template table size {expected}",),
        {},
    )


def build_match_report(
    census: CensusReport, templates: Mapping[str, ArchitectureTemplate] = ARCHITECTURE_TEMPLATES
) -> MatchReport:
    routing = route_level1(census)
    component = Level2Classification(census.component_hint, census.component_evidence)
    results = tuple(_screened_result(census, template) for template in templates.values())
    passed = [result for result in results if result.passed]
    matched = passed[0].template_name if len(passed) == 1 else None
    return MatchReport(routing, component, results, matched)


def build_scan_report(census: CensusReport) -> ScanReport:
    return ScanReport(census, build_match_report(census))


__all__ = [
    "Level1Routing",
    "MatchReport",
    "ScanReport",
    "TemplateMatchResult",
    "build_match_report",
    "build_scan_report",
    "validate_level3",
]
