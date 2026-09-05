#!/usr/bin/env python3
"""Plan/run the fixed TE using an installed wheel inside the bounded CPU container."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import shutil
import time
from pathlib import Path

from comfy_omni.artifacts import fileops
from comfy_omni.artifacts.build_identity import installed_tool_identity
from comfy_omni.contracts import te_nvfp4 as contract
from comfy_omni.conversion.exporters.te_nvfp4 import execute_te_dense_export
from comfy_omni.conversion.exporters.te_nvfp4_plan import plan_te_dense_export


def container_limits(source: Path, config: Path) -> dict:
    memory = None
    for path in (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        if path.is_file():
            value = path.read_text().strip()
            if value.isdecimal() and 0 < int(value) <= 4 * 1024**3:
                memory = int(value)
                break
    if memory is None or not Path("/.dockerenv").exists():
        raise ValueError("requires Docker memory cap <=4 GiB")
    if {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise ValueError("requires network none")
    if list(Path("/dev").glob("nvidia*")) or os.environ.get("NVIDIA_VISIBLE_DEVICES") not in {"void", "none", ""}:
        raise ValueError("requires no GPU devices")
    if any(not os.statvfs(p).f_flag & os.ST_RDONLY for p in (source, config)):
        raise ValueError("source and config mounts must be read-only")
    return {"memory_limit_bytes": memory, "network": "none", "gpu_devices": [], "sources_read_only": True}


def estimate(plan) -> dict:
    cursor, header = 0, {}
    for tensor in plan.tensors:
        header[tensor.target_name] = {"dtype": "BF16", "shape": list(tensor.shape), "data_offsets": [cursor, cursor + tensor.byte_length]}
        cursor += tensor.byte_length
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    file_size = 8 + len(raw) + (-len(raw) % 8) + cursor
    upper = file_size + len(fileops.canonical_json(plan.to_dict())) + contract.CONFIG_BYTES + 1024**2
    if cursor != contract.TARGET_PAYLOAD_BYTES or upper > contract.MAX_OUTPUT_BYTES:
        raise ValueError("fixed TE output allocation exceeded")
    return {"payload_bytes": cursor, "safetensors_file_bytes": file_size, "maximum_output_bytes": upper, "output_budget_bytes": contract.MAX_OUTPUT_BYTES, "free_reserve_bytes": contract.RESERVE_BYTES}


def run(args):
    start = time.monotonic()
    tool = installed_tool_identity()
    if tool.source_commit != args.commit or tool.wheel_sha256 != args.wheel_sha256:
        raise ValueError("installed wheel is not the candidate")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", args.image_id) is None:
        raise ValueError("image ID must be an immutable digest")
    source = fileops.reject_linked_ancestors(args.source)
    config = fileops.reject_linked_ancestors(args.config)
    output = fileops.reject_linked_ancestors(args.output, allow_missing_final=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError("TE output must be fresh")
    evidence = (args.result, args.plan) if args.action == "plan" else (args.result,)
    for item in evidence:
        path = fileops.reject_linked_ancestors(item, allow_missing_final=True)
        if path.exists() or path.is_relative_to(output) or path in {source, config}:
            raise FileExistsError("evidence must be fresh outside output and sources")
    if args.action == "plan" and args.result.absolute() == args.plan.absolute():
        raise ValueError("plan and result must be distinct")
    limits = container_limits(source, config)
    free = shutil.disk_usage(output.parent).free
    if free < contract.MIN_FREE_BYTES:
        raise ValueError("TE output requires60 GiB free before execution")
    plan = plan_te_dense_export(source, config)
    budget = estimate(plan)
    if free < budget["maximum_output_bytes"] + contract.RESERVE_BYTES:
        raise ValueError("TE output cannot retain12 GiB reserve")
    plan_raw = fileops.canonical_json(plan.to_dict())
    result = {"candidate_commit": tool.source_commit, "wheel_sha256": tool.wheel_sha256, "image_id": args.image_id, "source_sha256": plan.source_sha256, "config_sha256": plan.config_sha256, "consumer": plan.consumer, "plan_content_sha256": plan.content_sha256, "plan_file_sha256": hashlib.sha256(plan_raw).hexdigest(), "target_tensor_count": len(plan.tensors), "resource_preflight": {**limits, **budget, "free_bytes_before": free}}
    if args.action == "plan":
        fileops.write_exclusive(args.plan, plan_raw)
        result["status"] = "AUTHORIZED"
    else:
        previous_plan, _ = fileops.read_file_pinned(args.preflight_plan)
        previous_raw, _ = fileops.read_file_pinned(args.preflight_result)
        previous = fileops.parse_json_strict(previous_raw)
        keys = ("candidate_commit", "wheel_sha256", "image_id", "source_sha256", "config_sha256", "consumer", "plan_content_sha256", "plan_file_sha256")
        if previous_plan != plan_raw or previous.get("status") != "AUTHORIZED" or any(previous.get(key) != result[key] for key in keys):
            raise ValueError("execution does not match the authorized preflight")
        publication = execute_te_dense_export(plan, output, tool=tool)
        manifest_raw, _ = fileops.read_file_pinned(publication.manifest_path)
        actual_bytes = sum(p.stat().st_size for p in output.iterdir())
        after = shutil.disk_usage(output.parent).free
        if actual_bytes > budget["maximum_output_bytes"] or after < contract.RESERVE_BYTES:
            raise ValueError("published output violated allocation or free reserve")
        result.update(status="EXECUTED", manifest_sha256=publication.manifest_sha256, manifest_file_sha256=hashlib.sha256(manifest_raw).hexdigest(), output_bytes=actual_bytes, free_bytes_after=after)
    result.update(elapsed_seconds=time.monotonic() - start, max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    fileops.write_exclusive(args.result, fileops.canonical_json(result))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "run"))
    for name in ("source", "config", "output", "result"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("commit", "wheel-sha256", "image-id"):
        parser.add_argument("--" + name, required=True)
    for name in ("plan", "preflight-plan", "preflight-result"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.action == "plan" and args.plan is None:
        parser.error("plan requires --plan")
    if args.action == "run" and (args.preflight_plan is None or args.preflight_result is None):
        parser.error("run requires preflight plan and result")
    try:
        print(json.dumps(run(args), sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
