#!/usr/bin/env python3
"""Create and execute the bounded producer fixture inside acceptance containers."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import replace
from pathlib import Path

from comfy_omni import __version__
from comfy_omni.artifacts import fileops
from comfy_omni.conversion.exporters.execution import execute_native_export
from comfy_omni.conversion.exporters.models import (
    NativeExportPlan,
    QkvLayoutPlan,
    ResourceEnvelope,
    ShardPlan,
    SourceBinding,
    TensorAction,
)
from comfy_omni.conversion.exporters.planning import (
    OP_COPY_QKV_TO_GROUPED,
    OP_INVERSE_CONVROT_BF16,
    OP_INVERSE_CONVROT_BF16_QKV_TO_GROUPED,
    OP_OMIT_MARKER,
    OP_OMIT_SCALE,
    PLAN_SCHEMA,
)
from comfy_omni.domain.normalization import ToolIdentity
from comfy_omni.domain.qkv import qkv_to_grouped_row_indices

QKV_PREFIX = "blocks.0.attn.qkv_proj"
MLP_PREFIX = "blocks.0.mlp"
DENSE_QKV = "token_refiner.blocks.0.attn.qkv_proj.weight"
MARKER = b'{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":4}'


def _write_safetensors(path: Path) -> None:
    pattern = bytes((2, 2, 2, 254))
    tensors = (
        (f"{QKV_PREFIX}.comfy_quant", "U8", (len(MARKER),), MARKER),
        (f"{QKV_PREFIX}.weight", "I8", (6, 4), pattern * 6),
        (f"{QKV_PREFIX}.weight_scale", "F32", (6, 1), struct.pack("<6f", 1, 2, 3, 4, 5, 6)),
        (f"{MLP_PREFIX}.comfy_quant", "U8", (len(MARKER),), MARKER),
        (f"{MLP_PREFIX}.weight", "I8", (3, 4), pattern * 3),
        (f"{MLP_PREFIX}.weight_scale", "F32", (3, 1), struct.pack("<3f", 1, 2, 3)),
        (DENSE_QKV, "BF16", (6, 2), b"".join(bytes((row, 0, row, 0)) for row in range(6))),
    )
    header: dict[str, object] = {}
    payload = bytearray()
    for name, dtype, shape, raw in sorted(tensors):
        start = len(payload)
        payload.extend(raw)
        header[name] = {"data_offsets": [start, len(payload)], "dtype": dtype, "shape": list(shape)}
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    path.parent.mkdir(parents=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _action_triplet(prefix: str, rows: int, *, qkv: bool) -> tuple[TensorAction, ...]:
    operation = OP_INVERSE_CONVROT_BF16_QKV_TO_GROUPED if qkv else OP_INVERSE_CONVROT_BF16
    return (
        TensorAction(
            f"{prefix}.comfy_quant",
            None,
            "U8",
            None,
            (len(MARKER),),
            len(MARKER),
            0,
            OP_OMIT_MARKER,
            prefix,
            4,
        ),
        TensorAction(
            f"{prefix}.weight",
            f"{prefix}.weight",
            "I8",
            "BF16",
            (rows, 4),
            rows * 4,
            rows * 8,
            operation,
            prefix,
            4,
        ),
        TensorAction(
            f"{prefix}.weight_scale",
            None,
            "F32",
            None,
            (rows, 1),
            rows * 4,
            0,
            OP_OMIT_SCALE,
            prefix,
            4,
        ),
    )


def _plan(source: Path) -> NativeExportPlan:
    permutation = qkv_to_grouped_row_indices(num_query_groups=2, heads_per_group=1, head_dim=1)
    permutation_sha256 = hashlib.sha256(fileops.canonical_json(list(permutation))).hexdigest()
    dense = TensorAction(
        DENSE_QKV,
        DENSE_QKV,
        "BF16",
        "BF16",
        (6, 2),
        24,
        24,
        OP_COPY_QKV_TO_GROUPED,
        DENSE_QKV.removesuffix(".weight"),
    )
    unsorted_actions = (
        *_action_triplet(QKV_PREFIX, 6, qkv=True),
        *_action_triplet(MLP_PREFIX, 3, qkv=False),
        dense,
    )
    actions = tuple(sorted(unsorted_actions, key=lambda item: item.source_name))
    targets = tuple(sorted(item.target_name for item in actions if item.target_name is not None))
    source_payload = source.read_bytes()
    draft = NativeExportPlan(
        schema=PLAN_SCHEMA,
        output_schema="h3-comfy-int8-export/v2",
        component="transformer",
        profile="dense-bf16-online-int8",
        source_contract="srv00-bounded-producer-v1",
        source_contract_origin="compile-time",
        source_contract_schema_sha256="a" * 64,
        source_snapshot_manifest_sha256=None,
        source_snapshot_file_sha256=None,
        template_name="srv00-bounded-producer",
        template_version=1,
        template_sha256="b" * 64,
        source_files=(SourceBinding(str(source), len(source_payload), hashlib.sha256(source_payload).hexdigest()),),
        qkv_layout=QkvLayoutPlan(
            "runtime-qkv",
            "grouped-for-official-loader",
            2,
            1,
            1,
            6,
            permutation_sha256,
        ),
        resource_envelope=ResourceEnvelope(max_rows=2, max_shard_bytes=4096, largest_target_tensor_bytes=48),
        actions=actions,
        shards=(ShardPlan("model-00001-of-00001.safetensors", targets, 96),),
        target_tensor_count=3,
        target_payload_bytes=96,
        runtime_quant_method="compressed-tensors",
        runtime_ignored_layers=(),
        payload_semantics="bounded-srv00-actual-torch",
        content_sha256="",
    )
    digest = hashlib.sha256(fileops.canonical_json(draft.to_dict(include_content_sha256=False))).hexdigest()
    return replace(draft, content_sha256=digest)


def _run(source: Path, output: Path, result: Path, commit: str, wheel_sha256: str) -> None:
    plan = _plan(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    publication = execute_native_export(
        plan,
        output,
        tool=ToolIdentity("comfy-omni", __version__, commit, wheel_sha256),
    )
    import torch

    files = {
        path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": path.stat().st_size}
        for path in sorted(publication.output_dir.iterdir())
        if path.is_file()
    }
    payload = {
        "candidate_commit": commit,
        "manifest_sha256": publication.manifest_sha256,
        "output_files": files,
        "plan_content_sha256": plan.content_sha256,
        "qkv_permutation_sha256": plan.qkv_layout.permutation_sha256,
        "source_sha256": plan.source_files[0].sha256,
        "status": "EXECUTED",
        "torch_version": torch.__version__,
        "wheel_sha256": wheel_sha256,
    }
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_bytes(fileops.canonical_json(payload))


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("source", type=Path)
    run = subparsers.add_parser("run")
    run.add_argument("source", type=Path)
    run.add_argument("output", type=Path)
    run.add_argument("result", type=Path)
    run.add_argument("--commit", required=True)
    run.add_argument("--wheel-sha256", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        _write_safetensors(args.source)
    else:
        _run(args.source, args.output, args.result, args.commit, args.wheel_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
