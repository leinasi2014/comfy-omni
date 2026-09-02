#!/usr/bin/env python3
"""Real-Torch equivalence and throughput proof for the fast ConvRot backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import struct
import time
from pathlib import Path

from comfy_omni.artifacts import fileops
from comfy_omni.artifacts.build_identity import installed_tool_identity
from comfy_omni.conversion.numerics.serialization import torch_convrot_bf16_block
from comfy_omni.conversion.numerics.torch_backend import fast_inverse_convrot_rows, inverse_convrot_rows


def _fail(detail: str) -> None:
    raise RuntimeError(f"fast ConvRot acceptance failed: {detail}")


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fileops.write_exclusive(path, fileops.canonical_json(value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--wheel-sha256", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    args = parser.parse_args()

    tool = installed_tool_identity()
    if (tool.source_commit, tool.wheel_sha256) != (args.commit, args.wheel_sha256):
        _fail("installed wheel identity drifted")

    import torch

    torch.manual_seed(20260903)
    cases: list[dict[str, object]] = []
    for group_size in (4, 16, 64, 256):
        weight = torch.randint(-128, 128, (5, group_size * 2), dtype=torch.int8)
        scale = torch.rand((5, 1), dtype=torch.float32).add_(0.01)
        dense = inverse_convrot_rows(weight, scale, group_size=group_size)
        fast = fast_inverse_convrot_rows(weight, scale, group_size=group_size)
        maximum_error = float(torch.max(torch.abs(dense - fast)).item())
        if not bool(torch.allclose(dense, fast, rtol=1e-5, atol=1e-5)):
            _fail(f"fast transform disagrees with dense oracle for group size {group_size}: {maximum_error}")
        cases.append({"group_size": group_size, "maximum_absolute_error": maximum_error})

    rows = 4096
    columns = 7168
    pattern = bytes(range(256))
    byte_count = rows * columns
    qweight = (pattern * ((byte_count + len(pattern) - 1) // len(pattern)))[:byte_count]
    scales = struct.pack("<f", 0.0078125) * rows
    started = time.perf_counter()
    payload = torch_convrot_bf16_block(
        qweight,
        scales,
        rows=rows,
        columns=columns,
        group_size=256,
    )
    elapsed = time.perf_counter() - started
    if len(payload) != rows * columns * 2:
        _fail("representative block emitted the wrong byte count")
    if elapsed > args.max_seconds:
        _fail(f"representative block exceeded {args.max_seconds} seconds: {elapsed}")

    result: dict[str, object] = {
        "candidate_commit": tool.source_commit,
        "dense_oracle_cases": cases,
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "representative_block": {
            "columns": columns,
            "elapsed_seconds": round(elapsed, 6),
            "group_size": 256,
            "payload_bytes": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "rows": rows,
        },
        "status": "VERIFIED",
        "torch_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "wheel_sha256": tool.wheel_sha256,
    }
    _write(args.result, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
