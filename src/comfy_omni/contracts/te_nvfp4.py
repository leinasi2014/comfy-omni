"""Closed descriptor authority for the fixed 50-layer Qwen3VL H3 encoder.

Header-derived inventories and explicit consumer semantics are documented in
docs/migration/te-nvfp4-dense.md. No model bytes or runtime dependencies are loaded.
"""
from __future__ import annotations

import hashlib
import json
from math import prod
from types import MappingProxyType

SOURCE_BYTES = 15_683_129_587
SOURCE_SHA256 = "a166c7bbbe66a22065159e478335fee4a633c4a3e3bb34c8e8ac4cc91bf4996f"
CONFIG_BYTES = 1474
CONFIG_SHA256 = "d2dd0c60d01b9e195d9447c52da61c7302d28828524914c044d9c6e1b81d0427"
SOURCE_SCHEMA_SHA256 = "807a68e6a06b2bd7f2736aea15b5ef111be8929495d98b9a9b517afd042c3c29"
TARGET_SCHEMA_SHA256 = "81262d6f94f41d39c4e1ae0ab0190a8b209f81f62eda3226a89419a11cee8011"
TARGET_PAYLOAD_BYTES = 51_506_191_840
PROFILE = "qwen3vl-h3-nvfp4-native-bf16-v1"
CONSUMER = "comfy-kitchen/b678fdf63378409676aa5596721445d33794d0ea/eager-bf16"
MAX_ROWS = 128
MAX_CHUNK_BYTES = 8 * 1024**2
MAX_OUTPUT_BYTES = 50 * 1024**3
MIN_FREE_BYTES = 60 * 1024**3
RESERVE_BYTES = 12 * 1024**3


def schema_sha256(inventory):
    entries = [{"name": name, "dtype": dtype, "shape": list(shape)} for name, (dtype, shape) in sorted(inventory.items())]
    return hashlib.sha256((json.dumps(entries, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()


def native_name(name: str) -> str:
    if name == "model.embed_tokens.weight" or name.startswith("model.layers."):
        return "model.language_model." + name.removeprefix("model.")
    if name.startswith("visual."):
        return "model." + name
    raise ValueError("name is outside the closed text encoder prefixes")


_weights = {}
_plain = {}
_text_shapes = {
    "self_attn.q_proj": (8192, 5120),
    "self_attn.k_proj": (1024, 5120),
    "self_attn.v_proj": (1024, 5120),
    "self_attn.o_proj": (5120, 8192),
    "mlp.gate_proj": (25600, 5120),
    "mlp.up_proj": (25600, 5120),
    "mlp.down_proj": (5120, 25600),
}
for _layer in range(50):
    for _suffix, _shape in _text_shapes.items():
        _weights[f"model.layers.{_layer}.{_suffix}"] = _shape
    for _suffix, _width in (("input_layernorm", 5120), ("post_attention_layernorm", 5120), ("self_attn.q_norm", 128), ("self_attn.k_norm", 128)):
        _plain[f"model.layers.{_layer}.{_suffix}.weight"] = ("BF16", (_width,))
for _layer in range(27):
    for _suffix, _shape in {
        "attn.qkv": (3456, 1152), "attn.proj": (1152, 1152),
        "mlp.linear_fc1": (4304, 1152), "mlp.linear_fc2": (1152, 4304),
        "norm1": (1152,), "norm2": (1152,),
    }.items():
        _plain[f"visual.blocks.{_layer}.{_suffix}.weight"] = ("BF16", _shape)
        _plain[f"visual.blocks.{_layer}.{_suffix}.bias"] = ("BF16", (_shape[0],))
for _base in ("visual.merger", *(f"visual.deepstack_merger_list.{i}" for i in range(3))):
    for _suffix, _shape in {
        "norm": (1152,) if _base == "visual.merger" else (4608,),
        "linear_fc1": (4608, 4608), "linear_fc2": (5120, 4608),
    }.items():
        _plain[f"{_base}.{_suffix}.weight"] = ("BF16", _shape)
        _plain[f"{_base}.{_suffix}.bias"] = ("BF16", (_shape[0],))
_plain.update({
    "visual.patch_embed.proj.weight": ("BF16", (1152, 3, 2, 16, 16)),
    "visual.patch_embed.proj.bias": ("BF16", (1152,)),
    "visual.pos_embed.weight": ("BF16", (2304, 1152)),
})
_source = dict(_plain)
_dense = dict(_plain)
for _name, (_rows, _cols) in _weights.items():
    _source.update({
        _name + ".weight": ("U8", (_rows, _cols // 2)),
        _name + ".weight_scale": ("F8_E4M3", (_rows, _cols // 16)),
        _name + ".weight_scale_2": ("F32", (1,)),
        _name + ".comfy_quant": ("U8", (19,)),
    })
    _dense[_name + ".weight"] = ("BF16", (_rows, _cols))
_source.update({
    "model.embed_tokens.weight": ("I8", (151936, 5120)),
    "model.embed_tokens.weight_scale": ("F32", ()),
    "model.embed_tokens.comfy_quant": ("U8", (29,)),
})
_dense["model.embed_tokens.weight"] = ("BF16", (151936, 5120))
_target = {native_name(name): descriptor for name, descriptor in _dense.items()}
if (
    len(_source) != 1954 or len(_target) != 902 or len(_plain) != 551
    or schema_sha256(_source) != SOURCE_SCHEMA_SHA256
    or schema_sha256(_target) != TARGET_SCHEMA_SHA256
    or sum(prod(shape) * 2 for _, shape in _target.values()) != TARGET_PAYLOAD_BYTES
):
    raise RuntimeError("fixed TE descriptor inventories disagree with independent pins")
SOURCE_INVENTORY = MappingProxyType(_source)
TARGET_INVENTORY = MappingProxyType(_target)
NVFP4_SHAPES = MappingProxyType(_weights)
