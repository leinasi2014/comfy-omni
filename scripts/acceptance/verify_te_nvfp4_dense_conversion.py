#!/usr/bin/env python3
"""Independent stdlib validation of the exact fixed TE dense component.

Generic held-file/strict-JSON primitives are reused from the accepted ComfyOmni
beta4 verifier at a8a3f783da9058af187893b2afa13e5c678171a6, blob
aec3263cf7344688adb46fe17659de014e8b8056 (Apache-2.0). No producer, contract module,
Torch or upstream numerical implementation is imported. Numerical sampling is
three full rows per NVFP4 matrix, not all-element equivalence.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import struct
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

from te_nvfp4_oracle import int8_values, nvfp4_row
from verify_beta4_dense_conversion import _canonical, _digest, _held, _json, _require, _safe_path, _schema, _tree

SOURCE_SIZE = 15_683_129_587
SOURCE_SHA = "a166c7bbbe66a22065159e478335fee4a633c4a3e3bb34c8e8ac4cc91bf4996f"
CONFIG_SIZE = 1474
CONFIG_SHA = "d2dd0c60d01b9e195d9447c52da61c7302d28828524914c044d9c6e1b81d0427"
SOURCE_SCHEMA = "807a68e6a06b2bd7f2736aea15b5ef111be8929495d98b9a9b517afd042c3c29"
TARGET_SCHEMA = "81262d6f94f41d39c4e1ae0ab0190a8b209f81f62eda3226a89419a11cee8011"
HOST_SCHEMA = "1ba50e5ba2d6f408dbedfde50a574fc7c172d045a9e17fed29b81bed1a0a0d70"
TARGET_BYTES = 51_506_191_840
PROFILE = "qwen3vl-h3-nvfp4-native-bf16-v1"
CONSUMER = "comfy-kitchen/b678fdf63378409676aa5596721445d33794d0ea/eager-bf16"
CHUNK = 8 * 1024**2
HEADER_LIMIT = 4 * 1024**2


def header(file):
    length, = struct.unpack("<Q", file.read(0, 8))
    _require(0 < length <= HEADER_LIMIT and length <= file.size - 8, "unsafe TE header")
    document = _json(file.read(8, length))
    metadata = document.pop("__metadata__", {})
    _require(isinstance(metadata, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()), "invalid TE metadata")
    _require(0 < len(document) <= 1954, "invalid TE descriptor count")
    records = {}
    for name, value in document.items():
        _require(isinstance(value, dict) and set(value) == {"dtype", "shape", "data_offsets"}, "invalid TE descriptor")
        dtype, shape, offsets = value["dtype"], value["shape"], value["data_offsets"]
        _require(isinstance(dtype, str) and dtype in {"U8", "I8", "F32", "BF16", "F8_E4M3"}, "invalid TE dtype")
        _require(isinstance(shape, list) and len(shape) <= 8 and all(type(v) is int and 0 <= v < 2**63 for v in shape), "invalid TE shape")
        _require(isinstance(offsets, list) and len(offsets) == 2 and all(type(v) is int for v in offsets), "invalid TE offsets")
        start, end = offsets
        width = {"U8": 1, "I8": 1, "F32": 4, "BF16": 2, "F8_E4M3": 1}[dtype]
        _require(0 <= start <= end <= file.size - 8 - length and end - start == math.prod(shape) * width, "TE byte span mismatch")
        records[name] = {"dtype": dtype, "shape": shape, "start": start, "end": end}
    cursor = 0
    for value in sorted(records.values(), key=lambda item: (item["start"], item["end"])):
        _require(value["start"] == cursor, "TE payload gap or overlap")
        cursor = value["end"]
    _require(cursor == file.size - 8 - length, "TE trailing payload")
    return records, 8 + length


def mapped(name):
    if name == "model.embed_tokens.weight" or name.startswith("model.layers."):
        return "model.language_model." + name[6:]
    _require(name.startswith("visual."), "unmapped source key")
    return "model." + name


def actions_from_source(file, records, offset):
    result, consumed = [], set()
    for name, record in sorted(records.items()):
        if name.endswith((".comfy_quant", ".weight_scale", ".weight_scale_2")):
            continue
        shape = list(record["shape"])
        if record["dtype"] == "BF16":
            op, members = "copy-bf16", {name}
        else:
            _require(name.endswith(".weight"), "non-weight quantized tensor")
            prefix = name[:-7]
            marker_name = prefix + ".comfy_quant"
            marker = records[marker_name]
            _require(marker["end"] - marker["start"] <= 4096, "oversized marker")
            declaration = _json(file.read(offset + marker["start"], marker["end"] - marker["start"]))
            members = {name, marker_name, prefix + ".weight_scale"}
            if name == "model.embed_tokens.weight":
                op = "int8-f32-to-bf16"
                _require(declaration == {"format": "int8_tensorwise"} and record["dtype"] == "I8", "wrong embedding declaration")
            else:
                op = "nvfp4-blocked-to-bf16"
                _require(declaration == {"format": "nvfp4"} and record["dtype"] == "U8" and len(shape) == 2, "wrong NVFP4 declaration")
                members.add(prefix + ".weight_scale_2")
                shape[1] *= 2
            _require(members <= records.keys(), "incomplete quantization group")
        _require(not consumed & members, "duplicate source accounting")
        consumed.update(members)
        result.append({"source_name": name, "target_name": mapped(name), "operation": op, "shape": shape, "byte_length": math.prod(shape) * 2})
    _require(consumed == records.keys(), "unconsumed source descriptor")
    _require(Counter(x["operation"] for x in result) == {"copy-bf16": 551, "nvfp4-blocked-to-bf16": 350, "int8-f32-to-bf16": 1}, "TE operation census drift")
    return sorted(result, key=lambda item: item["target_name"])


def host_projection(records):
    slots, groups = {}, {}
    for name, record in records.items():
        if name.startswith("model.visual."):
            slot, shard = "vision." + name[13:], None
        else:
            _require(name.startswith("model.language_model."), "host would ignore checkpoint name")
            slot, shard = "text_model." + name[21:], None
            for source, target, value in (("q_proj.weight", "qkv_proj.weight", "q"), ("k_proj.weight", "qkv_proj.weight", "k"), ("v_proj.weight", "qkv_proj.weight", "v"), ("gate_proj.weight", "gate_up_proj.weight", 0), ("up_proj.weight", "gate_up_proj.weight", 1)):
                if slot.endswith("." + source):
                    slot, shard = slot[:-len(source)] + target, value
                    break
        shape = list(record["shape"])
        if shard is None:
            _require(slot not in slots, "duplicate host parameter")
            slots[slot] = {"dtype": "BF16", "shape": shape}
        else:
            seen = groups.setdefault(slot, set())
            _require(shard not in seen, "duplicate host fused shard")
            seen.add(shard)
            if slot in slots:
                _require(slots[slot]["shape"][1:] == shape[1:], "incompatible fused dimensions")
                slots[slot]["shape"][0] += shape[0]
            else:
                slots[slot] = {"dtype": "BF16", "shape": shape}
    _require(len(slots) == 752 and len(groups) == 100, "host slot census drift")
    _require(all(shards == ({"q", "k", "v"} if key.endswith("qkv_proj.weight") else {0, 1}) for key, shards in groups.items()), "incomplete fused host parameter")
    _require(_schema(slots) == HOST_SCHEMA, "host shape projection drift")
    return {"logical_parameter_count": 752, "fused_qkv_groups": 50, "fused_gate_up_groups": 50, "schema_sha256": HOST_SCHEMA, "actual_host_loaded": False}


def numerical(source, src, src_offset, output, dst, dst_offset, actions):
    comparisons = []
    copied, copied_bytes, sampled_values = 0, 0, 0
    for action in actions:
        name, target = action["source_name"], action["target_name"]
        a, b = src[name], dst[target]
        source_start, target_start = src_offset + a["start"], dst_offset + b["start"]
        if action["operation"] == "copy-bf16":
            for start in range(0, action["byte_length"], CHUNK):
                count = min(CHUNK, action["byte_length"] - start)
                _require(source.read(source_start + start, count) == output.read(target_start + start, count), "BF16 passthrough byte mismatch")
            copied += 1
            copied_bytes += action["byte_length"]
            continue
        rows, cols = action["shape"]
        prefix = name[:-7]
        if action["operation"] == "int8-f32-to-bf16":
            selected = sorted({0, rows - 1, *(rows * i // 10 for i in range(1, 10))})
            scalar = source.read(src_offset + src[prefix + ".weight_scale"]["start"], 4)
        else:
            selected = [0, rows // 2, rows - 1]
            scalar = source.read(src_offset + src[prefix + ".weight_scale_2"]["start"], 4)
        for row in selected:
            if action["operation"] == "int8-f32-to-bf16":
                expected = int8_values(source.read(source_start + row * cols, cols), scalar)
            else:
                band = row // 128 * 128
                scales = source.read(src_offset + src[prefix + ".weight_scale"]["start"] + band * cols // 16, 128 * cols // 16)
                expected = nvfp4_row(source.read(source_start + row * cols // 2, cols // 2), scales, scalar, row_in_band=row % 128)
            actual = output.read(target_start + row * cols * 2, cols * 2)
            _require(actual == expected, f"exact BF16 numerical mismatch in {target}, row {row}")
            sampled_values += cols
        comparisons.append({"target_name": target, "rows": selected, "columns_per_row": cols})
    return {"all_plain_tensors_byte_equal": copied, "plain_bytes_compared": copied_bytes, "sampled_numeric_values": sampled_values, "sampled_matrices": comparisons, "all_numeric_elements_verified": False}


def verify(args):
    start = time.monotonic()
    root = _safe_path(args.output)
    expected_names = {"model.safetensors", "config.json", "export.plan.json", "manifest.json"}
    _require(set(_tree(root)) == expected_names, "unexpected component file census")
    with ExitStack() as stack:
        source = stack.enter_context(_held(args.source, SOURCE_SIZE))
        config = stack.enter_context(_held(args.config, CONFIG_SIZE))
        files = {name: stack.enter_context(_held(root / name, 50 * 1024**3 if name == "model.safetensors" else HEADER_LIMIT)) for name in expected_names}
        _require((source.size, source.sha256) == (SOURCE_SIZE, SOURCE_SHA), "fixed source identity mismatch")
        _require((config.size, config.sha256) == (CONFIG_SIZE, CONFIG_SHA), "fixed config identity mismatch")
        _require(files["config.json"].sha256 == config.sha256 and files["config.json"].size == config.size, "published config differs")
        src, src_offset = header(source)
        dst, dst_offset = header(files["model.safetensors"])
        _require(len(src) == 1954 and _schema(src) == SOURCE_SCHEMA, "complete source schema drift")
        _require(len(dst) == 902 and _schema(dst) == TARGET_SCHEMA, "complete native target schema drift")
        _require(all(r["dtype"] == "BF16" for r in dst.values()) and sum(r["end"] - r["start"] for r in dst.values()) == TARGET_BYTES, "dense output representation drift")
        actions = actions_from_source(source, src, src_offset)
        plan, manifest = files["export.plan.json"].document(), files["manifest.json"].document()
        plan_sha = plan.pop("content_sha256")
        manifest_sha = manifest.pop("manifest_sha256")
        _require(plan_sha == _digest(plan) and manifest_sha == _digest(manifest), "document content digest mismatch")
        expected_plan = {"source_path": str(source.path), "config_path": str(config.path), "source_sha256": SOURCE_SHA, "source_bytes": SOURCE_SIZE, "config_sha256": CONFIG_SHA, "config_bytes": CONFIG_SIZE, "source_schema_sha256": SOURCE_SCHEMA, "target_schema_sha256": TARGET_SCHEMA, "target_payload_bytes": TARGET_BYTES, "tensors": actions, "profile": PROFILE, "consumer": CONSUMER, "max_rows": 128, "schema": "comfy_omni.te_dense.plan/v1"}
        _require(plan == expected_plan, "plan differs from independent source interpretation")
        _require(manifest["schema"] == "comfy_omni.te_dense.export/v1" and manifest["component"] == "text_encoder" and manifest["profile"] == PROFILE and manifest["consumer"] == CONSUMER, "manifest consumer drift")
        _require(manifest["historical_writer_identity_proven"] is False and manifest["plan_content_sha256"] == plan_sha, "false writer or plan binding")
        _require(manifest["source"] == {"size": SOURCE_SIZE, "sha256": SOURCE_SHA, "schema_sha256": SOURCE_SCHEMA} and manifest["config"] == {"size": CONFIG_SIZE, "sha256": CONFIG_SHA}, "manifest input binding drift")
        _require(manifest["target"] == {"schema_sha256": TARGET_SCHEMA, "tensor_count": 902, "payload_bytes": TARGET_BYTES}, "manifest target binding drift")
        _require(manifest["source_tensor_count"] == 1954 and manifest["consumed_auxiliary_count"] == 1052, "manifest accounting drift")
        _require(manifest["tool"] == {"distribution": "comfy-omni", "version": args.expected_version, "source_commit": args.expected_commit, "wheel_sha256": args.expected_wheel_sha256}, "tool identity mismatch")
        entries = {item["name"]: item for item in manifest["files"]}
        _require(len(entries) == len(manifest["files"]) == 3 and set(entries) == expected_names - {"manifest.json"}, "manifest file coverage drift")
        for name, entry in entries.items():
            _require((entry["size"], entry["sha256"]) == (files[name].size, files[name].sha256), "manifest file identity mismatch")
        _require(set(manifest["tensor_sha256"]) == dst.keys(), "tensor digest coverage drift")
        output = files["model.safetensors"]
        for name, record in dst.items():
            _require(output.hash_range(dst_offset + record["start"], record["end"] - record["start"]) == manifest["tensor_sha256"][name], "target tensor digest mismatch")
        semantics = numerical(source, src, src_offset, output, dst, dst_offset, actions)
        projection = host_projection(dst)
        result = {"schema": "comfy-omni.acceptance.te-dense-verification/v1", "status": "VERIFIED", "candidate_commit": args.expected_commit, "wheel_sha256": args.expected_wheel_sha256, "consumer": CONSUMER, "source_sha256": source.sha256, "config_sha256": config.sha256, "plan_content_sha256": plan_sha, "manifest_sha256": manifest_sha, "output_files": {n: {"size": f.size, "sha256": f.sha256} for n, f in files.items()}, "target_schema_sha256": TARGET_SCHEMA, "host_projection": projection, **semantics}
    _require(set(_tree(root)) == expected_names, "component tree changed during verification")
    result["elapsed_seconds"] = time.monotonic() - start
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "config", "output", "result"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-wheel-sha256", required=True)
    parser.add_argument("--expected-version", default="0.2.0a1")
    args = parser.parse_args()
    result_path = _safe_path(args.result, missing=True)
    _require(not result_path.exists() and not result_path.is_relative_to(_safe_path(args.output)), "result must be fresh outside the component")
    result = verify(args)
    with result_path.open("xb") as stream:
        stream.write(_canonical(result))
        stream.flush()
        os.fsync(stream.fileno())
    print(_canonical(result).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
