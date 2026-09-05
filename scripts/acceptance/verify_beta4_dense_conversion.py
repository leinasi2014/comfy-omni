#!/usr/bin/env python3
"""Independent stdlib verification of pinned beta4 A934 -> dense BF16 534.

Derived from ComfyOmni@0925862033b0a9fdf48935ce538f364bbc317e2d,
scripts/acceptance/verify_ref2va_full_conversion.py blob
1e9056553b969366426b3c7dc6ad30b61ff43fc9 (Apache-2.0).
The scalar normalized regular-Hadamard oracle is retained. Beta4 authority
and held-file transactions are new. Numerical checks sample three rows per
ConvRot matrix with explicit tolerance; they do not prove all numeric bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import struct
import sys
import time
from collections import Counter
from contextlib import ExitStack, contextmanager
from pathlib import Path

SOURCE_SIZE = 20_967_637_320
SOURCE_SHA256 = "54d56b15c65923b54c9ca16b494dae641bfe9455cfcb1c19c49b1008e270bbc1"
SOURCE_SCHEMA_SHA256 = "ae2456bc6ac904929a4b773f703f8a1baa99b6356b5a389994faf64a1a2d80f2"
TARGET_SCHEMA_SHA256 = "3684a0d21eebe12c27cbf2b54d0b8cef74bd9d2119d94a14a89d5f77ffd0ec4b"
TARGET_PAYLOAD_BYTES = 40_222_925_872
SOURCE_CONTRACT = "minimax-h3-10eros-beta4-int8-convrot-v1"
TARGET_CONTRACT = "minimax-h3-10eros-beta4-dense-bf16-v1"
SOURCE_TEMPLATE = "h3-transformer-50l-beta4-convrot"
SOURCE_TEMPLATE_SHA256 = "ec6eca0257af5230d1d05864069b765ed7943ad1052ef90e5c175bcab3ae4a89"
PROFILE = "beta4-dense-bf16"
OUTPUT_SCHEMA = "h3-comfy-int8-export/v2"
SOURCE_TENSOR_COUNT, TARGET_TENSOR_COUNT, GROUP_COUNT, SHARD_COUNT = 934, 534, 200, 10
SOURCE_DTYPES = {"BF16": 334, "I8": 200, "F32": 200, "U8": 200}
GROUP_SIZE = 256
QKV_DIMENSIONS = (56, 1, 128)
EXPECTED_OPERATIONS = {
    "copy-raw": 332,
    "copy-runtime-qkv-to-grouped": 2,
    "inverse-convrot-to-bf16": 150,
    "inverse-convrot-to-bf16-runtime-qkv-to-grouped": 50,
    "omit-comfy-quant-marker": 200,
    "omit-source-rowwise-scale": 200,
}
HASH_CHUNK = 8 * 1024 * 1024
MAX_DOCUMENT_BYTES = 64 * 1024 * 1024
MAX_SHARD_BYTES = 4 * 1024**3
DTYPE_BYTES = {"BF16": 2, "I8": 1, "F32": 4, "U8": 1}
REL_TOL, ABS_TOL = 0.02, 0.05


def _require(condition, detail):
    if not condition:
        raise ValueError(detail)


def _canonical(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _json(payload):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def integer(raw):
        _require(len(raw.removeprefix("-")) <= 20, "oversized JSON integer")
        return int(raw)

    def floating(raw):
        value = float(raw)
        _require(math.isfinite(value), "non-finite JSON number")
        return value

    def constant(_):
        raise ValueError("non-standard JSON constant")

    value = json.loads(
        payload, object_pairs_hook=unique, parse_int=integer, parse_float=floating, parse_constant=constant
    )
    _require(isinstance(value, dict), "expected JSON object")
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            _require(not any(0xD800 <= ord(c) <= 0xDFFF for c in item), "invalid Unicode string")
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return value


def _safe_path(path, *, missing=False):
    path = Path(os.path.abspath(path))
    for item in (*reversed(path.parents), path):
        try:
            info = item.lstat()
        except FileNotFoundError:
            _require(missing and item == path, "missing path or ancestor")
            continue
        _require(not stat.S_ISLNK(info.st_mode) and not getattr(info, "st_file_attributes", 0) & 1024, "linked path")
    return path


def _fd_identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


class HeldFile:
    """One descriptor survives hashing, inspection and final content recheck."""

    def __init__(self, path, *, limit):
        self.path = _safe_path(path)
        before = self.path.lstat()
        _require(stat.S_ISREG(before.st_mode), "input is not a regular file")
        _require(0 <= before.st_size <= limit, "input exceeds bounded file size")
        fd = os.open(self.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        self.stream = os.fdopen(fd, "rb", buffering=0)
        try:
            self.identity = _fd_identity(os.fstat(fd))
            self.size = self.identity[2]
            _require(self.identity == _fd_identity(before), "input changed during opening")
            self.sha256 = self.hash_range(0, self.size)
            self._check_identity()
        except BaseException:
            self.stream.close()
            raise

    def _check_identity(self):
        _require(_fd_identity(os.fstat(self.stream.fileno())) == self.identity, "held input changed")
        _require(_fd_identity(_safe_path(self.path).lstat()) == self.identity, "input path changed")

    def read(self, offset, length):
        _require(
            type(offset) is int
            and type(length) is int
            and 0 <= offset <= self.size
            and 0 <= length <= self.size - offset,
            "range outside input",
        )
        _require(length <= MAX_DOCUMENT_BYTES, "unbounded range read")
        self.stream.seek(offset)
        raw = self.stream.read(length)
        _require(len(raw) == length, "short input range")
        return raw

    def hash_range(self, offset, length):
        _require(0 <= offset <= self.size and 0 <= length <= self.size - offset, "hash range outside input")
        digest = hashlib.sha256()
        while length:
            count = min(length, HASH_CHUNK)
            digest.update(self.read(offset, count))
            offset, length = offset + count, length - count
        return digest.hexdigest()

    def verify(self):
        self._check_identity()
        _require(self.hash_range(0, self.size) == self.sha256, "held input contents changed")
        self._check_identity()

    def document(self):
        _require(self.size <= MAX_DOCUMENT_BYTES, "oversized JSON document")
        return _json(self.read(0, self.size))


@contextmanager
def _held(path, limit):
    held = HeldFile(path, limit=limit)
    try:
        yield held
    finally:
        try:
            held.verify()
        finally:
            held.stream.close()


def _tree(root):
    _require(stat.S_ISDIR(_safe_path(root).lstat().st_mode), "output is not a directory")
    names = []
    for item in root.iterdir():
        info = item.lstat()
        _require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode), "output has non-regular entry")
        _safe_path(item)
        names.append(item.name)
    return tuple(sorted(names))


def _safetensors(held):
    _require(held.size >= 8, "truncated safetensors prefix")
    header_length = struct.unpack("<Q", held.read(0, 8))[0]
    _require(0 < header_length <= MAX_DOCUMENT_BYTES and header_length <= held.size - 8, "unsafe safetensors header")
    header = _json(held.read(8, header_length))
    metadata = header.pop("__metadata__", {})
    _require(
        isinstance(metadata, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()),
        "invalid metadata",
    )
    _require(0 < len(header) <= 2000, "invalid tensor census size")
    records, payload_offset = {}, 8 + header_length
    for name, record in header.items():
        _require(
            isinstance(record, dict) and set(record) == {"dtype", "shape", "data_offsets"}, "invalid tensor descriptor"
        )
        dtype, shape, offsets = record["dtype"], record["shape"], record["data_offsets"]
        _require(isinstance(dtype, str) and dtype in DTYPE_BYTES, "unsupported tensor dtype")
        _require(
            isinstance(shape, list) and len(shape) <= 8 and all(type(v) is int and 0 <= v < 2**63 for v in shape),
            "invalid tensor shape",
        )
        _require(
            isinstance(offsets, list) and len(offsets) == 2 and all(type(v) is int for v in offsets),
            "invalid tensor offsets",
        )
        start, end = offsets
        _require(0 <= start <= end <= held.size - payload_offset, "tensor span outside payload")
        _require(end - start == math.prod(shape) * DTYPE_BYTES[dtype], "tensor span disagrees with shape")
        records[name] = {"dtype": dtype, "shape": shape, "start": start, "end": end}
    cursor = 0
    for record in sorted(records.values(), key=lambda r: (r["start"], r["end"])):
        _require(record["start"] == cursor, "safetensors payload gap or overlap")
        cursor = record["end"]
    _require(cursor == held.size - payload_offset, "safetensors trailing payload")
    return records, payload_offset


def _schema(records):
    return _digest([{"name": name, "dtype": r["dtype"], "shape": r["shape"]} for name, r in sorted(records.items())])


def _qkv_indices(layout):
    groups, heads, dim = QKV_DIMENSIONS
    rows = groups * (heads + 2) * dim
    runtime = []
    for section in range(3):
        for group in range(groups):
            start = group * (heads + 2) * dim + (0, heads * dim, (heads + 1) * dim)[section]
            runtime.extend(range(start, start + (heads if section == 0 else 1) * dim))
    inverse = [0] * rows
    for runtime_row, grouped_row in enumerate(runtime):
        inverse[grouped_row] = runtime_row
    expected = {
        "source_layout": "runtime-qkv",
        "target_layout": "grouped-for-official-loader",
        "num_query_groups": groups,
        "heads_per_group": heads,
        "head_dim": dim,
        "row_count": rows,
        "permutation_sha256": _digest(inverse),
    }
    _require(layout == expected, "QKV declaration drift")
    return tuple(inverse)


def _regular_hadamard(values, group_size):
    _require(group_size in {4, 16, 64, 256} and len(values) % group_size == 0, "unsupported inverse group")
    output = list(values)
    for group in range(0, len(output), group_size):
        stride = 1
        while stride < group_size:
            width = stride * 4
            for block in range(group, group + group_size, width):
                for offset in range(stride):
                    indexes = tuple(block + offset + lane * stride for lane in range(4))
                    a, b, c, d = (output[index] for index in indexes)
                    values = ((a + b + c - d) / 2, (a + b - c + d) / 2, (a - b + c + d) / 2, (-a + b + c + d) / 2)
                    for index, value in zip(indexes, values, strict=True):
                        output[index] = value
            stride = width
    return output


def _expected_actions(source, records, offset, indices):
    roles = {}
    groups = sorted(name.removesuffix(".comfy_quant") for name in records if name.endswith(".comfy_quant"))
    _require(len(groups) == GROUP_COUNT, "ConvRot group count drift")
    for prefix in groups:
        names = [prefix + suffix for suffix in (".weight", ".weight_scale", ".comfy_quant")]
        _require(all(name in records for name in names), "incomplete ConvRot triplet")
        weight, scale, marker = (records[name] for name in names)
        _require(
            weight["dtype"] == "I8"
            and len(weight["shape"]) == 2
            and min(weight["shape"]) > 0
            and weight["shape"][1] % GROUP_SIZE == 0,
            "invalid ConvRot weight",
        )
        _require(scale["dtype"] == "F32" and scale["shape"] == [weight["shape"][0], 1], "invalid ConvRot scale")
        length = marker["end"] - marker["start"]
        _require(
            marker["dtype"] == "U8" and marker["shape"] == [length] and 0 < length <= 4096, "invalid ConvRot marker"
        )
        declaration = _json(source.read(offset + marker["start"], length))
        _require(
            set(declaration) == {"format", "convrot", "convrot_groupsize"}
            and declaration["format"] == "int8_tensorwise"
            and declaration["convrot"] is True
            and type(declaration["convrot_groupsize"]) is int
            and declaration["convrot_groupsize"] == GROUP_SIZE,
            "ConvRot declaration drift",
        )
        for name, role in zip(names, ("weight", "scale", "marker"), strict=True):
            _require(name not in roles, "duplicate ConvRot role")
            roles[name] = (prefix, role)
    actions = []
    for name, record in sorted(records.items()):
        prefix, role = roles.get(name, (None, "copy"))
        dtype, shape = record["dtype"], record["shape"]
        size = record["end"] - record["start"]
        qkv = name.endswith(".attn.qkv_proj.weight")
        if qkv:
            _require(len(shape) == 2 and shape[0] == len(indices), "QKV source geometry drift")
        if role == "copy":
            _require(dtype == "BF16", "unclaimed quantization tensor")
            op = "copy-runtime-qkv-to-grouped" if qkv else "copy-raw"
            target, target_dtype, target_bytes = name, dtype, size
            if qkv:
                prefix = name.removesuffix(".weight")
        elif role == "weight":
            op = "inverse-convrot-to-bf16-runtime-qkv-to-grouped" if qkv else "inverse-convrot-to-bf16"
            target, target_dtype, target_bytes = name, "BF16", math.prod(shape) * 2
        else:
            op = "omit-comfy-quant-marker" if role == "marker" else "omit-source-rowwise-scale"
            target, target_dtype, target_bytes = None, None, 0
        actions.append(
            {
                "source_name": name,
                "target_name": target,
                "source_dtype": dtype,
                "target_dtype": target_dtype,
                "shape": shape,
                "source_bytes": size,
                "target_bytes": target_bytes,
                "operation": op,
                "group_prefix": prefix,
                "group_size": GROUP_SIZE if role != "copy" else None,
            }
        )
    _require(dict(Counter(a["operation"] for a in actions)) == EXPECTED_OPERATIONS, "operation census drift")
    return actions


def _semantics(source, records, source_offset, locations, actions, indices):
    copied = reordered = groups = sample_rows = sample_elements = 0
    max_absolute = max_relative = 0.0
    numeric_seconds, total_elements = 0.0, 0
    for action in actions:
        if action["target_name"] is None:
            continue
        name = action["source_name"]
        target, record, target_offset = locations[name]
        source_start, target_start = source_offset + records[name]["start"], target_offset + record["start"]
        operation = action["operation"]
        if operation == "copy-raw":
            _require(
                source.hash_range(source_start, action["source_bytes"])
                == target.hash_range(target_start, action["target_bytes"]),
                "passthrough bytes differ",
            )
            copied += 1
        elif operation == "copy-runtime-qkv-to-grouped":
            row_bytes = action["source_bytes"] // len(indices)
            digest = hashlib.sha256()
            for row in indices:
                digest.update(source.read(source_start + row * row_bytes, row_bytes))
            _require(
                digest.hexdigest() == target.hash_range(target_start, action["target_bytes"]), "QKV copy bytes differ"
            )
            reordered += 1
        else:
            numeric_started = time.perf_counter()
            groups += 1
            rows, columns = action["shape"]
            total_elements += rows * columns
            scale = records[action["group_prefix"] + ".weight_scale"]
            for row in sorted({0, rows // 2, rows - 1}):
                source_row = indices[row] if operation.endswith("runtime-qkv-to-grouped") else row
                factor = struct.unpack("<f", source.read(source_offset + scale["start"] + source_row * 4, 4))[0]
                _require(math.isfinite(factor) and factor > 0, "invalid sampled scale")
                quantized = struct.unpack(f"<{columns}b", source.read(source_start + source_row * columns, columns))
                expected = _regular_hadamard(tuple(value * factor for value in quantized), GROUP_SIZE)
                raw = target.read(target_start + row * columns * 2, columns * 2)
                actual = (
                    struct.unpack("<f", struct.pack("<I", value << 16))[0] for (value,) in struct.iter_unpack("<H", raw)
                )
                for wanted, observed in zip(expected, actual, strict=True):
                    _require(
                        math.isfinite(wanted)
                        and math.isfinite(observed)
                        and math.isclose(observed, wanted, rel_tol=REL_TOL, abs_tol=ABS_TOL),
                        "ConvRot numerical sample differs",
                    )
                    absolute = abs(observed - wanted)
                    max_absolute = max(max_absolute, absolute)
                    max_relative = max(max_relative, absolute / max(abs(wanted), 1e-12))
                sample_rows += 1
                sample_elements += columns
            numeric_seconds += time.perf_counter() - numeric_started
    _require(groups == GROUP_COUNT, "numerical group coverage incomplete")
    return {
        "copy_raw_tensors": copied,
        "raw_copy_all_bytes": True,
        "qkv_reordered_tensors": reordered,
        "qkv_copy_all_bytes": True,
        "convrot_groups_checked": groups,
        "convrot_sample_rows": sample_rows,
        "convrot_sample_elements": sample_elements,
        "convrot_total_elements": total_elements,
        "numeric_elapsed_seconds": numeric_seconds,
        "estimated_full_numeric_seconds": numeric_seconds * total_elements / sample_elements,
        "numeric_estimate_is_rough": True,
        "numeric_estimate_method": (
            "sample numeric elapsed * total ConvRot elements / checked sample elements; "
            "excludes whole-file hashing and copy verification, assumes linear cost"
        ),
        "numerical_coverage": "deterministic-row-samples",
        "sample_row_policy": "sorted unique {0, rows//2, rows-1}, all columns, each ConvRot matrix",
        "oracle": "independent scalar normalized regular-Hadamard; BF16 result tolerance",
        "relative_tolerance": REL_TOL,
        "absolute_tolerance": ABS_TOL,
        "sample_max_absolute_error": max_absolute,
        "sample_max_relative_error": max_relative,
        "all_numeric_bitwise": False,
    }


def _verify(args):
    _require(
        re.fullmatch(r"[0-9a-f]{40}", args.expected_commit) is not None
        and re.fullmatch(r"[0-9a-f]{64}", args.expected_wheel_sha256) is not None,
        "invalid candidate identity",
    )
    root = _safe_path(args.output)
    initial_tree, directory_identity = _tree(root), _fd_identity(root.stat())
    with ExitStack() as stack:

        def hold(path, limit=MAX_DOCUMENT_BYTES):
            return stack.enter_context(_held(path, limit))

        source = hold(args.source, SOURCE_SIZE)
        _require((source.sha256, source.size) == (SOURCE_SHA256, SOURCE_SIZE), "source identity mismatch")
        source_records, source_offset = _safetensors(source)
        _require(
            len(source_records) == SOURCE_TENSOR_COUNT
            and _schema(source_records) == SOURCE_SCHEMA_SHA256
            and dict(Counter(r["dtype"] for r in source_records.values())) == SOURCE_DTYPES,
            "source descriptor authority mismatch",
        )
        preflight_file, result_file, preflight_plan_file = (
            hold(args.preflight_result),
            hold(args.result),
            hold(args.preflight_plan),
        )
        preflight, result = preflight_file.document(), result_file.document()
        for document, status in ((preflight, "AUTHORIZED"), (result, "EXECUTED")):
            _require(
                document["status"] == status
                and document["candidate_commit"] == args.expected_commit
                and document["wheel_sha256"] == args.expected_wheel_sha256
                and document["source_sha256"] == SOURCE_SHA256,
                "producer stage identity mismatch",
            )
        shards = {f"model-{index:05d}-of-{SHARD_COUNT:05d}.safetensors" for index in range(1, SHARD_COUNT + 1)}
        expected_tree = shards | {
            "manifest.json",
            "export.plan.json",
            "config.patch.json",
            "model.safetensors.index.json",
        }
        _require(set(initial_tree) == expected_tree, "output tree census mismatch")
        files = {
            name: hold(root / name, MAX_SHARD_BYTES + MAX_DOCUMENT_BYTES if name in shards else MAX_DOCUMENT_BYTES)
            for name in initial_tree
        }
        manifest, plan = files["manifest.json"].document(), files["export.plan.json"].document()
        manifest_sha, plan_sha = manifest.pop("manifest_sha256"), plan.pop("content_sha256")
        _require(
            _digest(manifest) == manifest_sha == result["manifest_sha256"]
            and files["manifest.json"].sha256 == result["manifest_file_sha256"],
            "manifest identity mismatch",
        )
        _require(
            _digest(plan) == plan_sha == result["plan_content_sha256"] == preflight["plan_content_sha256"],
            "plan digest mismatch",
        )
        _require(
            files["export.plan.json"].sha256
            == preflight_plan_file.sha256
            == result["plan_file_sha256"]
            == preflight["plan_file_sha256"],
            "preflight plan differs from executed plan",
        )
        _require(
            plan["schema"] == "comfy_omni.native_export.plan/v2"
            and plan["status"] == "AUTHORIZED_PLAN"
            and plan["profile"] == PROFILE
            and plan["output_schema"] == OUTPUT_SCHEMA
            and plan["component"] == "transformer",
            "plan profile drift",
        )
        _require(
            plan["source_contract"]
            == {
                "name": SOURCE_CONTRACT,
                "origin": "compile-time",
                "schema_sha256": SOURCE_SCHEMA_SHA256,
                "snapshot_manifest_sha256": None,
                "snapshot_file_sha256": None,
            },
            "source contract drift",
        )
        _require(
            plan["architecture_template"] == {"name": SOURCE_TEMPLATE, "version": 1, "sha256": SOURCE_TEMPLATE_SHA256},
            "template authority drift",
        )
        expected_source = [{"path": str(source.path), "size": SOURCE_SIZE, "sha256": SOURCE_SHA256}]
        _require(plan["source_files"] == manifest["source_files"] == expected_source, "source file declaration drift")
        target_census = {
            "tensor_count": TARGET_TENSOR_COUNT,
            "payload_bytes": TARGET_PAYLOAD_BYTES,
            "contract": TARGET_CONTRACT,
            "schema_sha256": TARGET_SCHEMA_SHA256,
        }
        _require(plan["target"] == manifest["target"] == target_census, "target contract drift")
        _require(
            plan["runtime_quantization"]
            == manifest["runtime_quantization"]
            == {"required": False, "method": None, "ignored_layers": [], "checkpoint_int8_serialized": False},
            "false runtime quantization claim",
        )
        expected_tool = {
            "distribution": "comfy-omni",
            "version": "0.2.0a1",
            "source_commit": args.expected_commit,
            "wheel_sha256": args.expected_wheel_sha256,
        }
        _require(
            manifest["schema"] == "comfy_omni.native_export.receipt/v1"
            and manifest["status"] == "COMMITTED"
            and manifest["component"] == "transformer"
            and manifest["profile"] == PROFILE
            and manifest["output_schema"] == OUTPUT_SCHEMA
            and manifest["tool"] == expected_tool
            and manifest["plan_content_sha256"] == plan_sha,
            "manifest policy drift",
        )
        indices = _qkv_indices(plan["qkv_layout"])
        _require(manifest["qkv_layout"] == plan["qkv_layout"], "manifest QKV drift")
        actions = _expected_actions(source, source_records, source_offset, indices)
        _require(plan["actions"] == actions, "plan actions differ from independent source interpretation")
        targets = {a["target_name"]: a for a in actions if a["target_name"] is not None}
        _require(
            len(targets) == TARGET_TENSOR_COUNT
            and sum(a["target_bytes"] for a in targets.values()) == TARGET_PAYLOAD_BYTES,
            "target action census drift",
        )
        envelope = plan["resource_envelope"]
        _require(
            type(envelope["max_rows"]) is int
            and 1 <= envelope["max_rows"] <= 4096
            and envelope["max_shard_bytes"] == MAX_SHARD_BYTES
            and envelope["largest_target_tensor_bytes"] == max(a["target_bytes"] for a in targets.values()),
            "resource declaration drift",
        )
        manifest_records = {item["name"]: item for item in manifest["files"]}
        _require(
            len(manifest_records) == len(manifest["files"])
            and set(manifest_records) == expected_tree - {"manifest.json"},
            "manifest file census drift",
        )
        for name, entry in manifest_records.items():
            _require(
                (files[name].sha256, files[name].size) == (entry["sha256"], entry["size"]),
                "manifest file bytes mismatch",
            )
        _require(
            files["config.patch.json"].document()
            == {
                "_comfy_omni": {"output_schema": OUTPUT_SCHEMA, "plan_content_sha256": plan_sha, "profile": PROFILE},
                "quantization_config": None,
            },
            "runtime config drift",
        )
        locations, weight_map = {}, {}
        _require(len(plan["shards"]) == SHARD_COUNT, "shard count drift")
        for index, shard in enumerate(plan["shards"], start=1):
            name = f"model-{index:05d}-of-{SHARD_COUNT:05d}.safetensors"
            _require(shard["name"] == name, "unsafe or reordered shard name")
            records, offset = _safetensors(files[name])
            _require(
                len(shard["tensor_names"]) == len(records) and set(shard["tensor_names"]) == set(records),
                "shard descriptor coverage drift",
            )
            _require(manifest_records[name]["tensor_count"] == len(records), "manifest tensor census drift")
            _require(
                shard["payload_bytes"] == sum(r["end"] - r["start"] for r in records.values()) <= MAX_SHARD_BYTES,
                "shard payload budget drift",
            )
            for tensor, record in records.items():
                _require(tensor not in locations and tensor in targets, "duplicate or unexpected target tensor")
                action = targets[tensor]
                _require(
                    record["dtype"] == "BF16"
                    and record["shape"] == action["shape"]
                    and record["end"] - record["start"] == action["target_bytes"],
                    "target descriptor mismatch",
                )
                locations[tensor], weight_map[tensor] = (files[name], record, offset), name
        _require(
            set(locations) == set(targets)
            and _schema({n: r for n, (_, r, _) in locations.items()}) == TARGET_SCHEMA_SHA256,
            "independent target schema mismatch",
        )
        _require(
            files["model.safetensors.index.json"].document()
            == {"metadata": {"total_size": TARGET_PAYLOAD_BYTES}, "weight_map": weight_map},
            "safetensors index drift",
        )
        semantics = _semantics(source, source_records, source_offset, locations, actions, indices)
        verification = {
            "schema": "comfy-omni.acceptance.beta4-dense-verification/v1",
            "status": "VERIFIED",
            "scope": "beta4-dense-bf16-offline",
            "candidate_commit": args.expected_commit,
            "wheel_sha256": args.expected_wheel_sha256,
            "source_sha256": source.sha256,
            "source_bytes": source.size,
            "source_schema_sha256": SOURCE_SCHEMA_SHA256,
            "target_schema_sha256": TARGET_SCHEMA_SHA256,
            "target_tensor_count": TARGET_TENSOR_COUNT,
            "target_payload_bytes": TARGET_PAYLOAD_BYTES,
            "manifest_sha256": manifest_sha,
            "plan_content_sha256": plan_sha,
            "operation_counts": EXPECTED_OPERATIONS,
            "output_files": {name: {"sha256": held.sha256, "size": held.size} for name, held in files.items()},
            **semantics,
        }
    _require(
        _tree(root) == initial_tree and _fd_identity(root.stat()) == directory_identity,
        "output tree changed during verification",
    )
    verification["verification_sha256"] = _digest(verification)
    return verification


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("source", "output", "preflight-plan", "preflight-result", "result", "verification"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-wheel-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        destination = _safe_path(args.verification, missing=True)
        _require(not destination.exists(), "verification receipt already exists")
        _require(not destination.is_relative_to(_safe_path(args.output)), "receipt must be outside immutable output")
        verification = _verify(args)
        _safe_path(destination, missing=True)
        created = None
        try:
            with destination.open("x+b") as stream:
                created = _fd_identity(os.fstat(stream.fileno()))[:2]
                payload = _canonical(verification)
                _require(stream.write(payload) == len(payload), "short verification receipt write")
                stream.flush()
                os.fsync(stream.fileno())
                stream.seek(0)
                _require(stream.read() == payload, "verification receipt readback differs")
                _require(
                    _fd_identity(_safe_path(destination).lstat()) == _fd_identity(os.fstat(stream.fileno())),
                    "verification receipt path changed",
                )
        except BaseException:
            # Remove only the exclusively created inode, never a raced replacement.
            if created is not None:
                try:
                    if _fd_identity(_safe_path(destination).lstat())[:2] == created:
                        destination.unlink()
                except OSError:
                    pass
            raise
        print(
            json.dumps(
                {"status": "VERIFIED", "verification_sha256": verification["verification_sha256"]}, sort_keys=True
            )
        )
        return 0
    except (ValueError, OSError, KeyError, TypeError, RecursionError, struct.error) as exc:
        print(json.dumps({"status": "VERIFICATION_FAILED", "reason_code": type(exc).__name__}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
