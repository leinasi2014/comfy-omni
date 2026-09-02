#!/usr/bin/env python3
"""Plan and execute the pinned complete Ref2VA conversion in acceptance Docker."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from collections import Counter
from pathlib import Path
from typing import Any

from comfy_omni.application.conversion import convert_native_export, plan_native_export
from comfy_omni.artifacts import fileops
from comfy_omni.artifacts.build_identity import installed_tool_identity
from comfy_omni.contracts.registry import SOURCE_PROFILE_DASIWA_REF2VA_HYBRID
from comfy_omni.conversion.exporters.models import NativeExportPlan
from comfy_omni.conversion.exporters.planning import PLAN_SCHEMA

SOURCE_NAME = "minimax_h3_ref2va_pruned_zs05_int8_convrot.safetensors"
SOURCE_SIZE = 20_970_379_680
SOURCE_SHA256 = "71b8085ac4221ee036708c230a007d617dccca1b0028b95bb4ee106cb2a385c5"
SOURCE_SCHEMA_SHA256 = "cc7976f678e6d4a567e718aca56c1db4aa91adfa27108db84066cce3213edf9d"
EXPECTED_OPERATIONS = {
    "copy-raw": 330,
    "copy-runtime-qkv-to-grouped": 2,
    "inverse-convrot-to-bf16": 150,
    "inverse-convrot-to-bf16-runtime-qkv-to-grouped": 50,
    "omit-comfy-quant-marker": 200,
    "omit-source-rowwise-scale": 200,
}
TARGET_TENSOR_COUNT = 532
TARGET_PAYLOAD_BYTES = 40_225_668_192
SHARD_COUNT = 10
MAX_SHARD_BYTES = 4 * 1024**3


def _fail(detail: str) -> None:
    raise RuntimeError(f"Ref2VA acceptance contract failed: {detail}")


def _tool(commit: str, wheel_sha256: str):
    tool = installed_tool_identity()
    if tool.source_commit != commit or tool.wheel_sha256 != wheel_sha256:
        _fail("installed wheel identity disagrees with command authority")
    return tool


def _validate_plan(plan: NativeExportPlan, source: Path, max_rows: int) -> dict[str, int]:
    if source.name != SOURCE_NAME:
        _fail("unexpected source filename")
    if plan.schema != PLAN_SCHEMA:
        _fail("unexpected plan schema")
    if plan.output_schema != "h3-comfy-int8-export/v2" or plan.profile != "dense-bf16-online-int8":
        _fail("unexpected output schema or profile")
    if (
        plan.source_contract != SOURCE_PROFILE_DASIWA_REF2VA_HYBRID
        or plan.source_contract_origin != "compile-time"
        or plan.source_contract_schema_sha256 != SOURCE_SCHEMA_SHA256
        or plan.source_snapshot_manifest_sha256 is not None
        or plan.source_snapshot_file_sha256 is not None
    ):
        _fail("source contract authority drifted")
    if plan.template_name != "h3-transformer-50l-convrot" or plan.template_version != 1:
        _fail("architecture template drifted")
    if len(plan.source_files) != 1:
        _fail("plan must bind exactly one source")
    binding = plan.source_files[0]
    if (binding.path, binding.size, binding.sha256) != (str(source), SOURCE_SIZE, SOURCE_SHA256):
        _fail("source path, size, or digest drifted")
    operations = Counter(action.operation for action in plan.actions)
    if len(plan.actions) != 932 or dict(operations) != EXPECTED_OPERATIONS:
        _fail(f"operation census drifted: {dict(operations)}")
    if plan.target_tensor_count != TARGET_TENSOR_COUNT or plan.target_payload_bytes != TARGET_PAYLOAD_BYTES:
        _fail("target census drifted")
    if len(plan.shards) != SHARD_COUNT or sum(shard.payload_bytes for shard in plan.shards) != TARGET_PAYLOAD_BYTES:
        _fail("shard census drifted")
    if plan.resource_envelope.max_rows != max_rows or plan.resource_envelope.max_shard_bytes != MAX_SHARD_BYTES:
        _fail("resource envelope drifted")
    if plan.qkv_layout.to_dict() != {
        "source_layout": "runtime-qkv",
        "target_layout": "grouped-for-official-loader",
        "num_query_groups": 56,
        "heads_per_group": 1,
        "head_dim": 128,
        "row_count": 21_504,
        "permutation_sha256": "610267ea3e93bafc9bde40cd210aa3e3187a3b6add4750ffd6faac1b85fd1c47",
    }:
        _fail("QKV layout drifted")
    observed = hashlib.sha256(fileops.canonical_json(plan.to_dict(include_content_sha256=False))).hexdigest()
    if observed != plan.content_sha256:
        _fail("plan self-digest mismatch")
    return dict(operations)


def _max_rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fileops.write_exclusive(path, fileops.canonical_json(value))


def _plan(args: argparse.Namespace) -> None:
    started = time.monotonic()
    tool = _tool(args.commit, args.wheel_sha256)
    plan = plan_native_export(
        (args.source,),
        source_profile=SOURCE_PROFILE_DASIWA_REF2VA_HYBRID,
        max_rows=args.max_rows,
        max_shard_bytes=MAX_SHARD_BYTES,
    )
    operations = _validate_plan(plan, args.source, args.max_rows)
    plan_payload = fileops.canonical_json(plan.to_dict())
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    fileops.write_exclusive(args.plan, plan_payload)
    result = {
        "action_count": len(plan.actions),
        "candidate_commit": tool.source_commit,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "max_rss_bytes": _max_rss_bytes(),
        "operation_counts": operations,
        "plan_content_sha256": plan.content_sha256,
        "plan_file_sha256": hashlib.sha256(plan_payload).hexdigest(),
        "shard_count": len(plan.shards),
        "source_sha256": SOURCE_SHA256,
        "status": "AUTHORIZED",
        "target_payload_bytes": plan.target_payload_bytes,
        "target_tensor_count": plan.target_tensor_count,
        "wheel_sha256": tool.wheel_sha256,
    }
    _write(args.result, result)
    print(json.dumps(result, sort_keys=True))


def _run(args: argparse.Namespace) -> None:
    started = time.monotonic()
    tool = _tool(args.commit, args.wheel_sha256)
    preflight = fileops.parse_json_strict(args.preflight_result.read_bytes())
    if not isinstance(preflight, dict) or preflight.get("status") != "AUTHORIZED":
        _fail("preflight result is not authorized")
    if preflight.get("candidate_commit") != tool.source_commit or preflight.get("wheel_sha256") != tool.wheel_sha256:
        _fail("preflight tool identity drifted")
    publication = convert_native_export(
        (args.source,),
        args.output,
        tool=tool,
        source_profile=SOURCE_PROFILE_DASIWA_REF2VA_HYBRID,
        max_rows=args.max_rows,
        max_shard_bytes=MAX_SHARD_BYTES,
    )
    output_plan = publication.output_dir / "export.plan.json"
    output_plan_payload = output_plan.read_bytes()
    if output_plan_payload != args.preflight_plan.read_bytes():
        _fail("executed plan bytes disagree with preflight")
    plan_document = fileops.parse_json_strict(output_plan_payload)
    manifest = fileops.parse_json_strict(publication.manifest_path.read_bytes())
    if not isinstance(plan_document, dict) or not isinstance(manifest, dict):
        _fail("published plan or manifest is not an object")
    if plan_document.get("content_sha256") != preflight.get("plan_content_sha256"):
        _fail("published plan identity disagrees with preflight")
    import torch

    result = {
        "candidate_commit": tool.source_commit,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "manifest_file_sha256": hashlib.sha256(publication.manifest_path.read_bytes()).hexdigest(),
        "manifest_sha256": publication.manifest_sha256,
        "max_rss_bytes": _max_rss_bytes(),
        "plan_content_sha256": plan_document["content_sha256"],
        "plan_file_sha256": hashlib.sha256(output_plan_payload).hexdigest(),
        "source_sha256": SOURCE_SHA256,
        "status": "EXECUTED",
        "target_payload_bytes": manifest["target"]["payload_bytes"],
        "target_tensor_count": manifest["target"]["tensor_count"],
        "torch_version": torch.__version__,
        "wheel_sha256": tool.wheel_sha256,
    }
    _write(args.result, result)
    print(json.dumps(result, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("source", type=Path)
    plan.add_argument("plan", type=Path)
    plan.add_argument("result", type=Path)
    run = subparsers.add_parser("run")
    run.add_argument("source", type=Path)
    run.add_argument("output", type=Path)
    run.add_argument("preflight_plan", type=Path)
    run.add_argument("preflight_result", type=Path)
    run.add_argument("result", type=Path)
    for command in (plan, run):
        command.add_argument("--commit", required=True)
        command.add_argument("--wheel-sha256", required=True)
        command.add_argument("--max-rows", type=int, default=4096)
    args = parser.parse_args()
    if args.command == "plan":
        _plan(args)
    else:
        _run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
