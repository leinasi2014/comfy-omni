"""CLI adapter for immutable native-source contract workflows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from comfy_omni.application.contracts import (
    catalog_document,
    draft_contract_source,
    load_contract_catalog,
    pin_contract_draft,
    scan_contract_source,
)
from comfy_omni.artifacts.build_identity import installed_tool_identity
from comfy_omni.contracts.models import ContractError

LEGACY_CONTRACT_DIR_ENV = "H3_FORGE_CONTRACT_DIR"


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Declare scan/draft/pin/list commands without embedding workflow policy."""

    commands = parser.add_subparsers(dest="contract_command", required=True)
    scan = commands.add_parser("scan", help="read-only census and exact three-level match")
    scan.add_argument("paths", nargs="+", type=Path)
    scan.add_argument("--json", action="store_true")
    draft = commands.add_parser("draft", help="write one immutable pending-review draft")
    draft.add_argument("paths", nargs="+", type=Path)
    draft.add_argument("-o", "--output", required=True, type=Path)
    draft.add_argument("--generated-by", required=True)
    draft.add_argument("--json", action="store_true")
    pin = commands.add_parser("pin", help="review and publish one immutable snapshot")
    pin.add_argument("draft", type=Path)
    pin.add_argument("--name", required=True)
    pin.add_argument("--reviewer", required=True)
    pin.add_argument("--evidence", required=True, type=Path)
    pin.add_argument("--contract-dir", type=Path)
    pin.add_argument("--enforce-observed-schema", action="store_true")
    pin.add_argument("--json", action="store_true")
    listing = commands.add_parser("list", help="list compile-time and explicitly loaded snapshots")
    listing.add_argument("--contract-dir", type=Path)
    listing.add_argument("--json", action="store_true")


def _contract_dir(args: argparse.Namespace, *, required: bool) -> Path | None:
    explicit = getattr(args, "contract_dir", None)
    if explicit is not None:
        return explicit
    legacy = os.environ.get(LEGACY_CONTRACT_DIR_ENV, "").strip()
    if legacy:
        return Path(legacy)
    if required:
        raise ContractError(
            f"contract pin requires --contract-dir or ${LEGACY_CONTRACT_DIR_ENV}",
            evidence={"stage": "pin-cli"},
        )
    return None


def _generator_identity(operator: str) -> dict[str, str]:
    identity = installed_tool_identity().to_dict()
    return {**identity, "operator": operator}


def _render(payload: Any, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif isinstance(payload, list):
        for item in payload:
            print(f"{item['name']}: {item['component']} {item['storage_kind']} ({item['origin']})")
    else:
        print(payload)


def _run_scan(args: argparse.Namespace) -> int:
    report = scan_contract_source(args.paths)
    if args.json:
        _render(report.to_dict(), as_json=True)
    else:
        matched = report.matched_template or "NONE"
        print(
            f"contract scan: tensors={report.census.tensor_count} groups={report.census.convrot_group_count} "
            f"storage={report.census.storage_kind} template={matched}"
        )
    return 0 if report.matched_template is not None else 3


def _run_draft(args: argparse.Namespace) -> int:
    draft = draft_contract_source(args.paths, generator_identity=_generator_identity(args.generated_by))
    draft.write_to(args.output)
    if args.json:
        _render(draft.to_dict(), as_json=True)
    else:
        print(f"contract draft: {args.output} template={draft.template_name} status=DRAFTED")
    return 0


def _run_pin(args: argparse.Namespace) -> int:
    contract_dir = _contract_dir(args, required=True)
    assert contract_dir is not None
    result = pin_contract_draft(
        args.draft,
        name=args.name,
        reviewer=args.reviewer,
        evidence_path=args.evidence,
        contract_dir=contract_dir,
        enforce_observed_schema=args.enforce_observed_schema,
    )
    payload = {
        "name": result.contract.name,
        "status": "PINNED",
        "snapshot": str(result.snapshot.path),
        "manifest_sha256": result.snapshot.manifest_sha256,
        "reviewer": result.reviewer,
        "enforced_schema_decision": result.enforced_schema_decision,
    }
    rendered = payload if args.json else f"contract pin: {payload['snapshot']} ({payload['name']})"
    _render(rendered, as_json=args.json)
    return 0


def _run_list(args: argparse.Namespace) -> int:
    payload = catalog_document(load_contract_catalog(_contract_dir(args, required=False)))
    _render(payload, as_json=args.json)
    return 0


def run(args: argparse.Namespace) -> int:
    """Dispatch one contract subcommand and render fail-closed evidence."""

    handlers = {"scan": _run_scan, "draft": _run_draft, "pin": _run_pin, "list": _run_list}
    try:
        return handlers[args.contract_command](args)
    except (ContractError, OSError, ValueError) as exc:
        evidence = getattr(exc, "evidence", {})
        if getattr(args, "json", False):
            print(json.dumps({"error": str(exc), "evidence": evidence}, sort_keys=True), file=sys.stderr)
        else:
            print(f"comfy-omni contract: error: {exc}", file=sys.stderr)
        return 2


__all__ = ["configure_parser", "run"]
