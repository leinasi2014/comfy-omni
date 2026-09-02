#!/usr/bin/env python3
"""Independent stdlib verification of a complete pinned Ref2VA conversion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import struct
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any, BinaryIO

SOURCE_SIZE = 20_970_379_680
SOURCE_SHA256 = "71b8085ac4221ee036708c230a007d617dccca1b0028b95bb4ee106cb2a385c5"
SOURCE_SCHEMA_SHA256 = "cc7976f678e6d4a567e718aca56c1db4aa91adfa27108db84066cce3213edf9d"
TARGET_PAYLOAD_BYTES = 40_225_668_192
EXPECTED_OPERATIONS = {
    "copy-raw": 330,
    "copy-runtime-qkv-to-grouped": 2,
    "inverse-convrot-to-bf16": 150,
    "inverse-convrot-to-bf16-runtime-qkv-to-grouped": 50,
    "omit-comfy-quant-marker": 200,
    "omit-source-rowwise-scale": 200,
}
DTYPE_BITS = {
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "U16": 16,
    "I16": 16,
    "F16": 16,
    "BF16": 16,
    "U32": 32,
    "I32": 32,
    "F32": 32,
    "U64": 64,
    "I64": 64,
    "F64": 64,
    "C64": 64,
    "F4": 4,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
    "F8_E4M3": 8,
    "F8_E5M2": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E5M2FNUZ": 8,
    "F8_E8M0": 8,
}
HASH_CHUNK = 8 * 1024 * 1024


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise ValueError(detail)


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        _require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _json(payload: bytes) -> dict[str, Any]:
    value = json.loads(payload, object_pairs_hook=_unique, parse_constant=_reject_constant)
    _require(isinstance(value, dict), "expected a JSON object")
    return value


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode()


def _identity(path: Path) -> tuple[str, int]:
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode), f"not a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        _require((opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino), f"identity changed: {path}")
        while chunk := os.read(descriptor, HASH_CHUNK):
            digest.update(chunk)
        after = os.fstat(descriptor)
        _require(
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            == (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns),
            f"file changed while hashing: {path}",
        )
    finally:
        os.close(descriptor)
    final = path.lstat()
    _require((final.st_dev, final.st_ino) == (before.st_dev, before.st_ino), f"path replaced: {path}")
    return digest.hexdigest(), before.st_size


def _span_bytes(dtype: str, shape: list[int]) -> int:
    _require(dtype in DTYPE_BITS and isinstance(shape, list), "invalid tensor dtype or shape")
    elements = 1
    for dimension in shape:
        _require(type(dimension) is int and dimension >= 0, "invalid tensor dimension")
        elements *= dimension
    bits = elements * DTYPE_BITS[dtype]
    _require(bits % 8 == 0, "tensor payload is not byte aligned")
    return bits // 8


def _safetensors(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        _require(len(raw_length) == 8, f"truncated safetensors length: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        _require(0 < header_length <= 64 * 1024 * 1024, f"unsafe safetensors header: {path}")
        raw_header = stream.read(header_length)
        _require(len(raw_header) == header_length, f"truncated safetensors header: {path}")
    header = _json(raw_header)
    metadata = header.pop("__metadata__", {})
    _require(isinstance(metadata, dict), f"invalid safetensors metadata: {path}")
    payload_bytes = size - 8 - header_length
    cursor = 0
    records: dict[str, dict[str, Any]] = {}
    for name, record in sorted(header.items(), key=lambda item: item[1]["data_offsets"][0]):
        _require(isinstance(record, dict), f"invalid tensor record: {name}")
        dtype, shape, offsets = record.get("dtype"), record.get("shape"), record.get("data_offsets")
        _require(
            isinstance(dtype, str)
            and isinstance(shape, list)
            and isinstance(offsets, list)
            and len(offsets) == 2,
            f"incomplete tensor record: {name}",
        )
        start, end = offsets
        _require(type(start) is int and type(end) is int and start == cursor and end >= start, f"bad span: {name}")
        _require(end - start == _span_bytes(dtype, shape), f"dtype/shape/span mismatch: {name}")
        cursor = end
        records[name] = {"dtype": dtype, "shape": shape, "start": start, "end": end}
    _require(cursor == payload_bytes, f"safetensors payload is not exactly indexed: {path}")
    return records, 8 + header_length


def _qkv_indices(layout: dict[str, Any]) -> tuple[int, ...]:
    groups = layout["num_query_groups"]
    heads = layout["heads_per_group"]
    head_dim = layout["head_dim"]
    per_group = (heads + 2) * head_dim
    runtime: list[int] = []
    for section in (0, 1, 2):
        for group in range(groups):
            start = group * per_group
            if section == 0:
                runtime.extend(range(start, start + heads * head_dim))
            elif section == 1:
                runtime.extend(range(start + heads * head_dim, start + (heads + 1) * head_dim))
            else:
                runtime.extend(range(start + (heads + 1) * head_dim, start + per_group))
    inverse = [0] * len(runtime)
    for runtime_row, grouped_row in enumerate(runtime):
        inverse[grouped_row] = runtime_row
    return tuple(inverse)


def _read_at(stream: BinaryIO, offset: int, length: int) -> bytes:
    stream.seek(offset)
    payload = stream.read(length)
    _require(len(payload) == length, "short tensor range read")
    return payload


def _hash_range(path: Path, offset: int, length: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = length
        while remaining:
            chunk = stream.read(min(remaining, HASH_CHUNK))
            _require(bool(chunk), f"short range read: {path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _contiguous_runs(indices: Iterable[int]) -> Iterable[tuple[int, int]]:
    iterator = iter(indices)
    start = previous = next(iterator)
    count = 1
    for current in iterator:
        if current == previous + 1:
            count += 1
        else:
            yield start, count
            start, count = current, 1
        previous = current
    yield start, count


def _hash_reordered_rows(path: Path, offset: int, row_bytes: int, indices: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for start, count in _contiguous_runs(indices):
            digest.update(_read_at(stream, offset + start * row_bytes, count * row_bytes))
    return digest.hexdigest()


def _bf16_values(payload: bytes) -> tuple[float, ...]:
    _require(len(payload) % 2 == 0, "misaligned BF16 payload")
    return tuple(
        struct.unpack("<f", struct.pack("<I", value << 16))[0]
        for (value,) in struct.iter_unpack("<H", payload)
    )


def _regular_hadamard(values: tuple[float, ...], group_size: int) -> tuple[float, ...]:
    _require(group_size in {4, 16, 64, 256}, "unsupported sample group size")
    _require(len(values) % group_size == 0, "sample row width is not group aligned")
    output = list(values)
    for group_start in range(0, len(output), group_size):
        stride = 1
        while stride < group_size:
            width = stride * 4
            for block in range(group_start, group_start + group_size, width):
                for offset in range(stride):
                    indexes = tuple(block + offset + lane * stride for lane in range(4))
                    a, b, c, d = (output[index] for index in indexes)
                    transformed = (
                        (a + b + c - d) / 2,
                        (a + b - c + d) / 2,
                        (a - b + c + d) / 2,
                        (-a + b + c + d) / 2,
                    )
                    for index, value in zip(indexes, transformed, strict=True):
                        output[index] = value
            stride = width
    return tuple(output)


def _output_locations(
    output: Path, plan: dict[str, Any], manifest: dict[str, Any]
) -> tuple[dict[str, tuple[Path, dict[str, Any], int]], dict[str, dict[str, Any]]]:
    locations: dict[str, tuple[Path, dict[str, Any], int]] = {}
    shard_headers: dict[str, dict[str, Any]] = {}
    manifest_files = {item["name"]: item for item in manifest["files"]}
    for shard in plan["shards"]:
        path = output / shard["name"]
        records, payload_offset = _safetensors(path)
        _require(set(records) == set(shard["tensor_names"]), f"shard tensor census mismatch: {path.name}")
        _require(
            manifest_files[path.name]["tensor_count"] == len(records),
            f"manifest tensor count mismatch: {path.name}",
        )
        shard_headers[path.name] = records
        for name, record in records.items():
            _require(name not in locations, f"duplicate output tensor: {name}")
            locations[name] = (path, record, payload_offset)
    return locations, shard_headers


def _verify_semantics(
    source: Path,
    source_records: dict[str, dict[str, Any]],
    source_offset: int,
    locations: dict[str, tuple[Path, dict[str, Any], int]],
    plan: dict[str, Any],
    indices: tuple[int, ...],
) -> dict[str, Any]:
    actions = {item["source_name"]: item for item in plan["actions"]}
    copied = reordered = sampled_rows = sampled_elements = 0
    max_absolute_error = 0.0
    max_relative_error = 0.0
    with source.open("rb") as source_stream:
        for source_name, action in actions.items():
            source_record = source_records[source_name]
            operation = action["operation"]
            if action["target_name"] is None:
                continue
            output_path, output_record, output_offset = locations[action["target_name"]]
            output_start = output_offset + output_record["start"]
            source_start = source_offset + source_record["start"]
            if operation == "copy-raw":
                _require(
                    _hash_range(source, source_start, action["source_bytes"])
                    == _hash_range(output_path, output_start, action["target_bytes"]),
                    f"raw-copy semantic mismatch: {source_name}",
                )
                copied += 1
                continue
            if operation == "copy-runtime-qkv-to-grouped":
                row_bytes = action["source_bytes"] // action["shape"][0]
                expected = _hash_reordered_rows(source, source_start, row_bytes, indices)
                actual = _hash_range(output_path, output_start, action["target_bytes"])
                _require(expected == actual, f"QKV-copy semantic mismatch: {source_name}")
                reordered += 1
                continue
            if operation not in {
                "inverse-convrot-to-bf16",
                "inverse-convrot-to-bf16-runtime-qkv-to-grouped",
            }:
                raise ValueError(f"unverified materializing operation: {operation}")
            rows, columns = action["shape"]
            group_size = action["group_size"]
            scale_name = f"{action['group_prefix']}.weight_scale"
            scale_record = source_records[scale_name]
            scale_offset = source_offset + scale_record["start"]
            target_rows = tuple(sorted({0, rows // 2, rows - 1}))
            with output_path.open("rb") as output_stream:
                for target_row in target_rows:
                    source_row = indices[target_row] if operation.endswith("runtime-qkv-to-grouped") else target_row
                    qweight = _read_at(source_stream, source_start + source_row * columns, columns)
                    scale = struct.unpack("<f", _read_at(source_stream, scale_offset + source_row * 4, 4))[0]
                    _require(math.isfinite(scale) and scale > 0, f"invalid sampled scale: {source_name}")
                    signed = struct.unpack(f"<{columns}b", qweight)
                    expected = _regular_hadamard(tuple(value * scale for value in signed), group_size)
                    actual = _bf16_values(
                        _read_at(output_stream, output_start + target_row * columns * 2, columns * 2)
                    )
                    for expected_value, actual_value in zip(expected, actual, strict=True):
                        absolute = abs(actual_value - expected_value)
                        relative = absolute / max(abs(expected_value), 1e-12)
                        max_absolute_error = max(max_absolute_error, absolute)
                        max_relative_error = max(max_relative_error, relative)
                        _require(
                            math.isfinite(actual_value)
                            and math.isclose(actual_value, expected_value, rel_tol=0.02, abs_tol=0.05),
                            f"ConvRot sample mismatch: {source_name} row {target_row}",
                        )
                    sampled_rows += 1
                    sampled_elements += columns
    return {
        "convrot_sample_elements": sampled_elements,
        "convrot_sample_rows": sampled_rows,
        "copy_raw_tensors": copied,
        "qkv_reordered_tensors": reordered,
        "sample_max_absolute_error": max_absolute_error,
        "sample_max_relative_error": max_relative_error,
    }


def _verify(args: argparse.Namespace) -> dict[str, Any]:
    preflight = _json(args.preflight_result.read_bytes())
    result = _json(args.result.read_bytes())
    for document in (preflight, result):
        _require(document["candidate_commit"] == args.expected_commit, "candidate identity mismatch")
        _require(document["wheel_sha256"] == args.expected_wheel_sha256, "wheel identity mismatch")
        _require(document["source_sha256"] == SOURCE_SHA256, "source identity mismatch")
    _require(preflight["status"] == "AUTHORIZED" and result["status"] == "EXECUTED", "stage status mismatch")

    source_sha256, source_size = _identity(args.source)
    _require((source_sha256, source_size) == (SOURCE_SHA256, SOURCE_SIZE), "source file identity mismatch")
    output_names = {path.name for path in args.output.iterdir() if path.is_file()}
    shards = {f"model-{index:05d}-of-00010.safetensors" for index in range(1, 11)}
    expected_names = shards | {"config.patch.json", "export.plan.json", "manifest.json", "model.safetensors.index.json"}
    _require(output_names == expected_names, f"output file census mismatch: {sorted(output_names ^ expected_names)}")

    manifest_payload = (args.output / "manifest.json").read_bytes()
    _require(hashlib.sha256(manifest_payload).hexdigest() == result["manifest_file_sha256"], "manifest file mismatch")
    manifest = _json(manifest_payload)
    manifest_sha256 = manifest.pop("manifest_sha256")
    _require(hashlib.sha256(_canonical(manifest)).hexdigest() == manifest_sha256, "manifest self-digest mismatch")
    _require(manifest_sha256 == result["manifest_sha256"], "manifest result binding mismatch")
    expected_tool = {
        "distribution": "comfy-omni",
        "source_commit": args.expected_commit,
        "version": "0.2.0a1",
        "wheel_sha256": args.expected_wheel_sha256,
    }
    _require(manifest["tool"] == expected_tool and manifest["status"] == "COMMITTED", "manifest tool/status drift")
    _require(manifest["target"] == {"payload_bytes": TARGET_PAYLOAD_BYTES, "tensor_count": 532}, "target drift")

    plan_payload = (args.output / "export.plan.json").read_bytes()
    _require(plan_payload == args.preflight_plan.read_bytes(), "executed plan differs from preflight")
    _require(hashlib.sha256(plan_payload).hexdigest() == result["plan_file_sha256"], "plan file mismatch")
    plan = _json(plan_payload)
    plan_sha256 = plan.pop("content_sha256")
    _require(hashlib.sha256(_canonical(plan)).hexdigest() == plan_sha256, "plan self-digest mismatch")
    _require(plan_sha256 == result["plan_content_sha256"] == preflight["plan_content_sha256"], "plan binding drift")
    _require(plan["schema"] == "comfy_omni.native_export.plan/v2", "plan schema drift")
    _require(
        plan["source_contract"]
        == {
            "name": "minimax-h3-dasiwa-ref2va-hybrid-int8-convrot-v1",
            "origin": "compile-time",
            "schema_sha256": SOURCE_SCHEMA_SHA256,
            "snapshot_file_sha256": None,
            "snapshot_manifest_sha256": None,
        },
        "source contract drift",
    )
    expected_source = [{"path": str(args.source), "sha256": SOURCE_SHA256, "size": SOURCE_SIZE}]
    _require(plan["source_files"] == expected_source == manifest["source_files"], "source binding drift")
    operations = Counter(item["operation"] for item in plan["actions"])
    _require(len(plan["actions"]) == 932 and dict(operations) == EXPECTED_OPERATIONS, "operation census drift")
    _require(len(plan["shards"]) == 10, "shard count drift")
    _require(sum(item["payload_bytes"] for item in plan["shards"]) == TARGET_PAYLOAD_BYTES, "shard bytes drift")
    _require(
        plan["target"] == {"payload_bytes": TARGET_PAYLOAD_BYTES, "tensor_count": 532}, "plan target drift"
    )
    indices = _qkv_indices(plan["qkv_layout"])
    _require(len(indices) == plan["qkv_layout"]["row_count"] == 21_504, "QKV row-count drift")
    _require(tuple(sorted(indices)) == tuple(range(len(indices))), "QKV mapping is not a permutation")
    permutation_sha256 = hashlib.sha256(_canonical(list(indices))).hexdigest()
    _require(permutation_sha256 == plan["qkv_layout"]["permutation_sha256"], "QKV digest drift")

    manifest_files = {item["name"]: item for item in manifest["files"]}
    _require(set(manifest_files) == expected_names - {"manifest.json"}, "manifest file census drift")
    file_identities: dict[str, dict[str, Any]] = {}
    for name, record in manifest_files.items():
        digest, size = _identity(args.output / name)
        _require((digest, size) == (record["sha256"], record["size"]), f"manifest file mismatch: {name}")
        file_identities[name] = {"sha256": digest, "size": size}

    config = _json((args.output / "config.patch.json").read_bytes())
    _require(config["_comfy_omni"]["plan_content_sha256"] == plan_sha256, "config plan binding drift")
    _require(
        config["quantization_config"]
        == {
            "ignored_layers": plan["runtime_quantization"]["ignored_layers"],
            "quant_method": plan["runtime_quantization"]["method"],
        },
        "runtime configuration drift",
    )
    index = _json((args.output / "model.safetensors.index.json").read_bytes())
    expected_weight_map = {
        name: shard["name"] for shard in plan["shards"] for name in shard["tensor_names"]
    }
    _require(
        index == {"metadata": {"total_size": TARGET_PAYLOAD_BYTES}, "weight_map": expected_weight_map},
        "index drift",
    )

    source_records, source_offset = _safetensors(args.source)
    actions = {item["source_name"]: item for item in plan["actions"]}
    _require(set(source_records) == set(actions), "source/action census mismatch")
    for name, action in actions.items():
        record = source_records[name]
        _require(
            (record["dtype"], record["shape"], record["end"] - record["start"])
            == (action["source_dtype"], action["shape"], action["source_bytes"]),
            f"source action descriptor mismatch: {name}",
        )
    locations, _ = _output_locations(args.output, plan, manifest)
    targets = {item["target_name"]: item for item in plan["actions"] if item["target_name"] is not None}
    _require(set(locations) == set(targets) and len(locations) == 532, "output/action census mismatch")
    for name, action in targets.items():
        _, record, _ = locations[name]
        _require(
            (record["dtype"], record["shape"], record["end"] - record["start"])
            == (action["target_dtype"], action["shape"], action["target_bytes"]),
            f"target action descriptor mismatch: {name}",
        )
    semantics = _verify_semantics(args.source, source_records, source_offset, locations, plan, indices)
    return {
        "candidate_commit": args.expected_commit,
        "file_count": len(expected_names),
        "manifest_sha256": manifest_sha256,
        "operation_counts": dict(operations),
        "output_files": file_identities,
        "plan_content_sha256": plan_sha256,
        "source_sha256": source_sha256,
        "status": "VERIFIED",
        "target_payload_bytes": TARGET_PAYLOAD_BYTES,
        "target_tensor_count": len(locations),
        "wheel_sha256": args.expected_wheel_sha256,
        **semantics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-plan", type=Path, required=True)
    parser.add_argument("--preflight-result", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-wheel-sha256", required=True)
    parser.add_argument("--verification", type=Path, required=True)
    args = parser.parse_args()
    verification = _verify(args)
    args.verification.parent.mkdir(parents=True, exist_ok=True)
    with args.verification.open("xb") as stream:
        stream.write(_canonical(verification))
    print(json.dumps(verification, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
