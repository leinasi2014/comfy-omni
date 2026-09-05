#!/usr/bin/env python3
"""Inspect one fixed LoRA against the fixed raw primary from an installed wheel.

Run twice in serialized, offline CPU Docker acceptance, with readonly inputs.
No default server paths or model payloads are embedded in this harness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
from pathlib import Path

from comfy_omni.artifacts import fileops
from comfy_omni.artifacts.build_identity import installed_tool_identity
from comfy_omni.conversion.oracle.checkpoint_contract import CheckpointInputError
from comfy_omni.conversion.oracle.checkpoint_preflight import preflight_checkpoint_candidate

PRIMARY_PIN = (20_967_637_320, "54d56b15c65923b54c9ca16b494dae641bfe9455cfcb1c19c49b1008e270bbc1")
CANDIDATE_PINS = {
    "spatial-physics-lora": (155_109_672, "7d14f3701560068e7004159c8b2a7278bd2dbfc9e5e3b60d0bc9aef6c049919d"),
    "realism-people-lora": (131_229_656, "acc529601d2da117fb81179e76c56e488a3beab1171659d305f04fa3655b787e"),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--candidate-id", required=True, choices=sorted(CANDIDATE_PINS))
    parser.add_argument("--scale", required=True, type=float)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-wheel-sha256", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--result-out", required=True, type=Path)
    args = parser.parse_args(argv)
    started = time.monotonic()
    try:
        output = fileops.reject_linked_ancestors(args.result_out, allow_missing_final=True)
        if output.exists():
            raise ValueError("receipt output already exists")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", args.image_digest) is None:
            raise ValueError("container image identity must be a digest")
        tool = installed_tool_identity()
        if (tool.distribution, tool.source_commit, tool.wheel_sha256) != (
            "comfy-omni",
            args.expected_commit,
            args.expected_wheel_sha256,
        ):
            raise ValueError("installed tool identity differs from the acceptance candidate")
        candidate_size, candidate_sha = CANDIDATE_PINS[args.candidate_id]
        verdict = preflight_checkpoint_candidate(
            args.candidate_id,
            args.base,
            args.adapter,
            base_sha256=PRIMARY_PIN[1],
            base_bytes=PRIMARY_PIN[0],
            pinned_sha256=candidate_sha,
            pinned_bytes=candidate_size,
            scale=args.scale,
        ).to_dict()
        result = {
            "schema": "comfy-omni.acceptance.checkpoint-lora/v1",
            "status": "INSPECTED_UNSUPPORTED",
            "tool": tool.to_dict(),
            "environment": {
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "declared_container_image_digest": args.image_digest,
            },
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "verdict": verdict,
        }
        result["receipt_sha256"] = hashlib.sha256(fileops.canonical_json(result)).hexdigest()
        fileops.write_exclusive(output, fileops.canonical_json(result))
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "receipt_sha256": result["receipt_sha256"],
                    "reason_code": verdict["reason_code"],
                },
                sort_keys=True,
            )
        )
        return 0
    except CheckpointInputError as exc:
        print(json.dumps(exc.to_dict(), sort_keys=True), file=sys.stderr)
        return 2
    except (ValueError, OSError, fileops.FsopsError) as exc:
        print(json.dumps({"status": "ACCEPTANCE_FAILED", "reason_code": type(exc).__name__}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
