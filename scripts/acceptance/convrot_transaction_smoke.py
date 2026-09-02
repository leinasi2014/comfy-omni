"""Designated-server smoke for the copy-only native-export transaction."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import replace
from pathlib import Path

from comfy_omni.artifacts import fileops
from comfy_omni.artifacts.build_identity import installed_tool_identity
from comfy_omni.conversion.exporters.execution import execute_native_export
from comfy_omni.conversion.exporters.models import (
    NativeExportPlan,
    QkvLayoutPlan,
    ResourceEnvelope,
    ShardPlan,
    SourceBinding,
    TensorAction,
)

FIXTURE_PAYLOADS = {
    "alpha": b"\x01\x02\x03\x04",
    "beta": b"\x05\x06\x07\x08",
}


def _fixture_bytes() -> bytes:
    tensors = (
        ("alpha", "BF16", (2,), FIXTURE_PAYLOADS["alpha"]),
        ("beta", "F32", (1,), FIXTURE_PAYLOADS["beta"]),
    )
    cursor = 0
    header: dict[str, object] = {}
    payload = bytearray()
    for name, dtype, shape, raw in tensors:
        header[name] = {
            "data_offsets": [cursor, cursor + len(raw)],
            "dtype": dtype,
            "shape": list(shape),
        }
        cursor += len(raw)
        payload.extend(raw)
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded + payload


def _plan(source: Path) -> NativeExportPlan:
    source_sha256, source_size = fileops.sha256_file_pinned(source)
    qkv_digest = hashlib.sha256(fileops.canonical_json([0])).hexdigest()
    draft = NativeExportPlan(
        schema="comfy_omni.native_export.plan/v1",
        output_schema="h3-comfy-int8-export/v2",
        component="transformer",
        profile="dense-bf16-online-int8",
        source_contract="acceptance-copy-only-v1",
        source_contract_origin="compile-time",
        source_contract_schema_sha256="a" * 64,
        source_snapshot_manifest_sha256=None,
        source_snapshot_file_sha256=None,
        template_name="acceptance-copy-only",
        template_version=1,
        template_sha256="b" * 64,
        source_files=(SourceBinding(str(source), source_size, source_sha256),),
        qkv_layout=QkvLayoutPlan("runtime-qkv", "grouped-for-official-loader", 1, 1, 1, 1, qkv_digest),
        resource_envelope=ResourceEnvelope(1, 1024, 4),
        actions=(
            TensorAction("alpha", "alpha", "BF16", "BF16", (2,), 4, 4, "copy-raw"),
            TensorAction("beta", "beta", "F32", "F32", (1,), 4, 4, "copy-raw"),
        ),
        shards=(ShardPlan("model-00001-of-00001.safetensors", ("alpha", "beta"), 8),),
        target_tensor_count=2,
        target_payload_bytes=8,
        runtime_quant_method="compressed-tensors",
        runtime_ignored_layers=(),
        payload_semantics="acceptance-copy-only",
        content_sha256="",
    )
    content_sha256 = hashlib.sha256(fileops.canonical_json(draft.to_dict(include_content_sha256=False))).hexdigest()
    return replace(draft, content_sha256=content_sha256)


def _prepare(source: Path) -> int:
    source.parent.mkdir(parents=True, exist_ok=False)
    fileops.write_exclusive(source, _fixture_bytes())
    digest, size = fileops.sha256_file_pinned(source)
    print(json.dumps({"source": str(source), "sha256": digest, "size": size}, sort_keys=True))
    return 0


def _run(source: Path, evidence: Path) -> int:
    evidence.mkdir(parents=True, exist_ok=False)
    plan = _plan(source)
    tool = installed_tool_identity()
    publication = execute_native_export(plan, evidence / "output", tool=tool)
    tree: dict[str, dict[str, int | str]] = {}
    for path in sorted(publication.output_dir.iterdir(), key=lambda value: value.name):
        digest, size = fileops.sha256_file_pinned(path)
        tree[path.name] = {"sha256": digest, "size": size}
    result = {
        "manifest_sha256": publication.manifest_sha256,
        "output_files": tree,
        "plan_content_sha256": plan.content_sha256,
        "schema": "comfy_omni.acceptance.native-export-transaction/v1",
        "source": plan.source_files[0].to_dict(),
        "status": "PASSED",
        "tool": tool.to_dict(),
    }
    fileops.write_exclusive(evidence / "result.json", fileops.canonical_json(result))
    print(json.dumps(result, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare = subcommands.add_parser("prepare")
    prepare.add_argument("source", type=Path)
    run = subcommands.add_parser("run")
    run.add_argument("source", type=Path)
    run.add_argument("evidence", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        return _prepare(arguments.source)
    return _run(arguments.source, arguments.evidence)


if __name__ == "__main__":
    raise SystemExit(main())
