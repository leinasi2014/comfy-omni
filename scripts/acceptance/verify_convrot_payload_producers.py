#!/usr/bin/env python3
"""Independent stdlib verifier for the srv-00 bounded producer acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

QKV_PREFIX = "blocks.0.attn.qkv_proj"
MLP_PREFIX = "blocks.0.mlp"
DENSE_QKV = "token_refiner.blocks.0.attn.qkv_proj.weight"


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _json(payload: bytes) -> dict[str, Any]:
    value = json.loads(payload, object_pairs_hook=_unique, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safetensors(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    if len(payload) < 9:
        raise ValueError("safetensors file is truncated")
    header_length = struct.unpack("<Q", payload[:8])[0]
    header = _json(payload[8 : 8 + header_length].rstrip(b" "))
    tensor_payload = payload[8 + header_length :]
    cursor = 0
    for name, record in sorted(header.items(), key=lambda item: item[1]["data_offsets"][0]):
        start, end = record["data_offsets"]
        if start != cursor or end <= start or end > len(tensor_payload):
            raise ValueError(f"invalid safetensors range for {name}")
        cursor = end
    if cursor != len(tensor_payload):
        raise ValueError("safetensors payload is not exactly indexed")
    return header, tensor_payload


def _bf16(value: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return struct.pack("<H", (rounded >> 16) & 0xFFFF)


def _expected_row(value: float) -> bytes:
    return _bf16(value) + b"\x00\x00" * 3


def _tensor_bytes(header: dict[str, Any], payload: bytes, name: str) -> bytes:
    start, end = header[name]["data_offsets"]
    return payload[start:end]


def _verify(args: argparse.Namespace) -> dict[str, Any]:
    result = _json(args.result.read_bytes())
    if result["candidate_commit"] != args.expected_commit or result["wheel_sha256"] != args.expected_wheel_sha256:
        raise ValueError("result identity does not match the accepted candidate")
    if _sha256(args.source) != result["source_sha256"]:
        raise ValueError("source digest mismatch")
    expected_files = {
        "config.patch.json",
        "export.plan.json",
        "manifest.json",
        "model-00001-of-00001.safetensors",
        "model.safetensors.index.json",
    }
    observed_files = {path.name for path in args.output.iterdir() if path.is_file()}
    if observed_files != expected_files:
        raise ValueError(f"unexpected output files: {sorted(observed_files ^ expected_files)}")
    for name, record in result["output_files"].items():
        path = args.output / name
        if _sha256(path) != record["sha256"] or path.stat().st_size != record["size"]:
            raise ValueError(f"result file identity mismatch: {name}")

    manifest = _json((args.output / "manifest.json").read_bytes())
    claimed_manifest = manifest.pop("manifest_sha256")
    if hashlib.sha256(_canonical(manifest)).hexdigest() != claimed_manifest:
        raise ValueError("manifest self digest mismatch")
    if claimed_manifest != result["manifest_sha256"] or manifest["status"] != "COMMITTED":
        raise ValueError("manifest receipt identity/status mismatch")
    if manifest["tool"] != {
        "distribution": "comfy-omni",
        "source_commit": args.expected_commit,
        "version": "0.2.0a1",
        "wheel_sha256": args.expected_wheel_sha256,
    }:
        raise ValueError("manifest tool identity mismatch")
    for record in manifest["files"]:
        path = args.output / record["name"]
        if _sha256(path) != record["sha256"] or path.stat().st_size != record["size"]:
            raise ValueError(f"manifest file identity mismatch: {record['name']}")

    plan = _json((args.output / "export.plan.json").read_bytes())
    claimed_plan = plan.pop("content_sha256")
    if plan["schema"] != "comfy_omni.native_export.plan/v2":
        raise ValueError("unexpected plan schema")
    if hashlib.sha256(_canonical(plan)).hexdigest() != claimed_plan or claimed_plan != result["plan_content_sha256"]:
        raise ValueError("plan content digest mismatch")
    operations = {item["source_name"]: item for item in plan["actions"]}
    if operations[f"{QKV_PREFIX}.weight"]["operation"] != "inverse-convrot-to-bf16-runtime-qkv-to-grouped":
        raise ValueError("combined ConvRot/QKV operation is absent")
    if operations[f"{MLP_PREFIX}.weight"]["operation"] != "inverse-convrot-to-bf16":
        raise ValueError("plain ConvRot operation is absent")
    if operations[DENSE_QKV]["operation"] != "copy-runtime-qkv-to-grouped":
        raise ValueError("dense QKV operation is absent")
    grouped = [item for item in operations.values() if item["group_prefix"] in {QKV_PREFIX, MLP_PREFIX}]
    if {item["group_size"] for item in grouped} != {4}:
        raise ValueError("ConvRot group-size binding drifted")

    header, payload = _safetensors(args.output / "model-00001-of-00001.safetensors")
    expected = {
        f"{MLP_PREFIX}.weight": b"".join(_expected_row(value) for value in (4, 8, 12)),
        f"{QKV_PREFIX}.weight": b"".join(_expected_row(value) for value in (4, 12, 20, 8, 16, 24)),
        DENSE_QKV: b"".join(bytes((row, 0, row, 0)) for row in (0, 2, 4, 1, 3, 5)),
    }
    expected_descriptors = {
        f"{MLP_PREFIX}.weight": ("BF16", [3, 4]),
        f"{QKV_PREFIX}.weight": ("BF16", [6, 4]),
        DENSE_QKV: ("BF16", [6, 2]),
    }
    if set(header) != set(expected):
        raise ValueError("output tensor census mismatch")
    for name, raw in expected.items():
        if _tensor_bytes(header, payload, name) != raw:
            raise ValueError(f"output tensor bytes mismatch: {name}")
        if (header[name]["dtype"], header[name]["shape"]) != expected_descriptors[name]:
            raise ValueError(f"output descriptor mismatch: {name}")
    return {
        "candidate_commit": args.expected_commit,
        "manifest_sha256": claimed_manifest,
        "plan_content_sha256": claimed_plan,
        "shard_sha256": _sha256(args.output / "model-00001-of-00001.safetensors"),
        "status": "VERIFIED",
        "tensor_count": len(expected),
        "wheel_sha256": args.expected_wheel_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-wheel-sha256", required=True)
    parser.add_argument("--verification", type=Path, required=True)
    args = parser.parse_args()
    verification = _verify(args)
    args.verification.parent.mkdir(parents=True, exist_ok=True)
    args.verification.write_bytes(_canonical(verification))
    print(json.dumps(verification, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
