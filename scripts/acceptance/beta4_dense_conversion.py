#!/usr/bin/env python3
"""Installed-wheel, CPU-only fixed beta4 conversion with explicit resource gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import time
from collections import Counter
from pathlib import Path

from comfy_omni.artifacts import fileops
from comfy_omni.artifacts.build_identity import installed_tool_identity
from comfy_omni.contracts.beta4 import BETA4_SOURCE_BYTES, BETA4_SOURCE_SHA256, BETA4_TARGET_PAYLOAD_BYTES
from comfy_omni.conversion.contract_workflows.census import CensusEngine
from comfy_omni.conversion.exporters.beta4 import build_beta4_dense_plan
from comfy_omni.conversion.exporters.execution import execute_native_export

OUTPUT_BUDGET = 45 * 1024**3
FREE_RESERVE = 12 * 1024**3
MEMORY_LIMIT = 4 * 1024**3
DOCUMENT_RESERVE = 1024**2
MANIFEST_RESERVE = 64 * 1024


def _memory_limit() -> int:
    for path in (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        if path.is_file():
            value = path.read_text().strip()
            if value.isdecimal() and 0 < int(value) <= MEMORY_LIMIT:
                return int(value)
    raise ValueError("acceptance requires an enforced Docker memory limit of at most 4 GiB")


def _estimate_output(plan) -> dict[str, int]:
    actions = {action.target_name: action for action in plan.actions if action.target_name}
    shard_bytes = 0
    for shard in plan.shards:
        cursor = 0
        header = {}
        for name in shard.tensor_names:
            action = actions[name]
            header[name] = {
                "dtype": action.target_dtype,
                "shape": list(action.shape),
                "data_offsets": [cursor, cursor + action.target_bytes],
            }
            cursor += action.target_bytes
        encoded = json.dumps(header, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        shard_bytes += 8 + len(encoded) + (-len(encoded) % 8) + cursor
    # Exact payload/shard geometry, plus a checked upper bound on small
    # plan/index/config/manifest documents; publication uses hard links, so
    # staging and output do not duplicate tensor allocation.
    upper = shard_bytes + len(fileops.canonical_json(plan.to_dict())) + DOCUMENT_RESERVE
    if plan.target_payload_bytes != BETA4_TARGET_PAYLOAD_BYTES or upper > OUTPUT_BUDGET:
        raise ValueError("beta4 output estimate exceeds the fixed allocation")
    return {
        "payload_bytes": plan.target_payload_bytes,
        "shard_file_bytes": shard_bytes,
        "maximum_output_bytes": upper,
        "output_budget_bytes": OUTPUT_BUDGET,
        "minimum_free_bytes": FREE_RESERVE,
    }


def _disk_gate(parent: Path, additional: int) -> int:
    free = shutil.disk_usage(parent).free
    if free < FREE_RESERVE + additional:
        raise ValueError("output filesystem cannot retain the 12 GiB reserve")
    return free


def _publication_gate(stage: Path, estimate: dict[str, int]) -> None:
    total = sum(path.stat().st_size for path in stage.iterdir())
    if total + MANIFEST_RESERVE > estimate["maximum_output_bytes"]:
        raise ValueError("staged output exceeds its preflight allocation")
    _disk_gate(stage.parent, MANIFEST_RESERVE)


def _read_json(path: Path):
    payload, _ = fileops.read_file_pinned(path)
    return payload, fileops.parse_json_strict(payload)


def _run(args) -> dict:
    started = time.monotonic()
    tool = installed_tool_identity()
    if tool.source_commit != args.commit or tool.wheel_sha256 != args.wheel_sha256:
        raise ValueError("installed wheel identity disagrees with the candidate")
    if os.environ.get("COMFY_OMNI_CONVROT_DEVICE", "cpu") != "cpu":
        raise ValueError("beta4 E3 acceptance requires CPU conversion")
    memory_limit = _memory_limit()
    source = fileops.reject_linked_ancestors(args.source).resolve(strict=True)
    output = fileops.reject_linked_ancestors(args.output, allow_missing_final=True)
    parent = output.parent.resolve(strict=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError("acceptance refuses an existing output")
    if source.stat().st_size != BETA4_SOURCE_BYTES:
        raise ValueError("unexpected fixed beta4 source size")
    if source.stat().st_dev == parent.stat().st_dev:
        raise ValueError("source and output must use separate filesystems for this acceptance")
    for result in (args.result, args.plan if args.action == "plan" else None):
        if result is not None:
            result = fileops.reject_linked_ancestors(result, allow_missing_final=True)
            if result.exists() or result.is_relative_to(output):
                raise FileExistsError("receipt must be fresh and outside the model output")
    free_before = _disk_gate(parent, BETA4_TARGET_PAYLOAD_BYTES + 2 * DOCUMENT_RESERVE)
    report = CensusEngine().scan(source)
    plan = build_beta4_dense_plan(report, max_rows=args.max_rows)
    estimate = _estimate_output(plan)
    _disk_gate(parent, estimate["maximum_output_bytes"])
    plan_payload = fileops.canonical_json(plan.to_dict())
    result = {
        "candidate_commit": tool.source_commit,
        "wheel_sha256": tool.wheel_sha256,
        "source_sha256": BETA4_SOURCE_SHA256,
        "plan_content_sha256": plan.content_sha256,
        "plan_file_sha256": hashlib.sha256(plan_payload).hexdigest(),
        "operation_counts": dict(Counter(action.operation for action in plan.actions)),
        "target_tensor_count": plan.target_tensor_count,
        "target_payload_bytes": plan.target_payload_bytes,
        "shard_count": len(plan.shards),
        "resource_preflight": {**estimate, "free_bytes_before": free_before, "memory_limit_bytes": memory_limit},
    }
    if args.action == "plan":
        fileops.write_exclusive(args.plan, plan_payload)
        result["status"] = "AUTHORIZED"
    else:
        expected_payload, _ = _read_json(args.preflight_plan)
        _, previous = _read_json(args.preflight_result)
        if (
            expected_payload != plan_payload
            or any(
                previous.get(key) != result[key]
                for key in (
                    "candidate_commit",
                    "wheel_sha256",
                    "source_sha256",
                    "plan_content_sha256",
                    "plan_file_sha256",
                )
            )
            or previous.get("status") != "AUTHORIZED"
        ):
            raise ValueError("executed candidate/plan differs from the preflight")
        publication = execute_native_export(
            plan,
            output,
            tool=tool,
            before_publication=lambda stage: _publication_gate(stage, estimate),
        )
        manifest_payload, _ = _read_json(publication.manifest_path)
        actual_bytes = sum(path.stat().st_size for path in output.iterdir())
        if actual_bytes > estimate["maximum_output_bytes"]:
            raise ValueError("published size exceeds the preflight bound")
        result.update(
            status="EXECUTED",
            manifest_sha256=publication.manifest_sha256,
            manifest_file_sha256=hashlib.sha256(manifest_payload).hexdigest(),
            output_bytes=actual_bytes,
            free_bytes_after=_disk_gate(parent, 0),
        )
    result.update(
        elapsed_seconds=time.monotonic() - started,
        max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )
    fileops.write_exclusive(args.result, fileops.canonical_json(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--wheel-sha256", required=True)
    parser.add_argument("--max-rows", type=int, default=128)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--preflight-plan", type=Path)
    parser.add_argument("--preflight-result", type=Path)
    args = parser.parse_args()
    if args.action == "plan" and args.plan is None:
        parser.error("plan requires --plan")
    if args.action == "run" and (args.preflight_plan is None or args.preflight_result is None):
        parser.error("run requires --preflight-plan and --preflight-result")
    print(json.dumps(_run(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
