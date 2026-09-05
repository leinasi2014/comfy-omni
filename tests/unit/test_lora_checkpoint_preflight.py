"""Checkpoint-only LoRA evidence, independent of a native runtime package."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path


def _write(path: Path, records, metadata=None):
    header = {"__metadata__": metadata or {"base_model": "MiniMax-H3"}}
    payload = bytearray()
    for name, dtype, shape, raw in records:
        start = len(payload)
        payload.extend(raw)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(payload)]}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size


def _pair(tmp_path):
    base, adapter = tmp_path / "base.safetensors", tmp_path / "adapter.safetensors"
    marker = b'{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}'
    base_pin = _write(base, [
        ("blocks.0.attn.out_proj.weight", "I8", [4, 256], bytes(4 * 256)),
        ("blocks.0.attn.out_proj.weight_scale", "F32", [4, 1], bytes(16)),
        ("blocks.0.attn.out_proj.comfy_quant", "U8", [len(marker)], marker),
    ])
    adapter_pin = _write(adapter, [
        ("diffusion_model.blocks.0.attn.out_proj.lora_A.weight", "BF16", [2, 256], bytes(2 * 256 * 2)),
        ("diffusion_model.blocks.0.attn.out_proj.lora_B.weight", "BF16", [4, 2], bytes(4 * 2 * 2)),
    ])
    return base, adapter, base_pin, adapter_pin


def test_checkpoint_pair_retains_both_identities_without_native_package(tmp_path):
    from comfy_omni.conversion.oracle.preflight import preflight_candidate

    base, adapter, base_pin, adapter_pin = _pair(tmp_path)
    receipt = preflight_candidate(
        "candidate", base, adapter, pinned_sha256=adapter_pin[0], pinned_bytes=adapter_pin[1]
    ).to_dict()
    assert receipt["status"] == "UNSUPPORTED"
    assert receipt["evidence"].get("scope") == "checkpoint-only"
    for role, pin in (("base", base_pin), ("adapter", adapter_pin)):
        assert receipt["evidence"][role]["actual_sha256"] == pin[0]
        assert receipt["evidence"][role]["expected_sha256"] == pin[0]
        assert receipt["evidence"][role]["actual_bytes"] == pin[1]
    assert receipt["evidence"]["promotion_capable"] is False
