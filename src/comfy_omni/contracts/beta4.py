"""Exact beta4 A source and dense target; no runtime support is implied.

The already distributed beta3 inventory derives from Apache-2.0 h3-forge
e9cb011d00b028c149db3978de246c54f6e34acc, templates blob
443a5cc9ca58891c3852079c8589fbe2f5af6484. The beta4 header comparison and
new authorization are recorded in docs/migration/beta4-dense-conversion.md.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import MappingProxyType

from comfy_omni.contracts.models import STORAGE_INT8_CONVROT, ContractRecord, NativeSourceContract
from comfy_omni.contracts.templates import ARCHITECTURE_TEMPLATES

BETA4_SOURCE_SHA256 = "54d56b15c65923b54c9ca16b494dae641bfe9455cfcb1c19c49b1008e270bbc1"
BETA4_SOURCE_BYTES = 20_967_637_320
BETA4_SOURCE_SCHEMA_SHA256 = "ae2456bc6ac904929a4b773f703f8a1baa99b6356b5a389994faf64a1a2d80f2"
BETA4_TARGET_SCHEMA_SHA256 = "3684a0d21eebe12c27cbf2b54d0b8cef74bd9d2119d94a14a89d5f77ffd0ec4b"
BETA4_TARGET_PAYLOAD_BYTES = 40_222_925_872
BETA4_SOURCE_NAME = "minimax-h3-10eros-beta4-int8-convrot-v1"
BETA4_TARGET_NAME = "minimax-h3-10eros-beta4-dense-bf16-v1"


def _schema(inventory):
    entries = [
        {"name": name, "dtype": dtype, "shape": list(shape)} for name, (dtype, shape) in sorted(inventory.items())
    ]
    payload = (json.dumps(entries, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


_beta3 = ARCHITECTURE_TEMPLATES["h3-transformer-50l-hybrid8-bf16-plain"]
_convrot = ARCHITECTURE_TEMPLATES["h3-transformer-50l-convrot"]
_target = dict(_beta3.non_quantized_inventory)
_target.pop("silu_t_emb_grid")
_source = dict(_target)
_non_quantized = dict(_target)
for _prefix, (_shape, _group_size) in _convrot.convrot_table().items():
    _source[f"{_prefix}.weight"] = ("I8", _shape)
    _source[f"{_prefix}.weight_scale"] = ("F32", (_shape[0], 1))
    _source[f"{_prefix}.comfy_quant"] = ("U8", (72,))
    _non_quantized.pop(f"{_prefix}.weight")
if _schema(_source) != BETA4_SOURCE_SCHEMA_SHA256 or _schema(_target) != BETA4_TARGET_SCHEMA_SHA256:
    raise RuntimeError("beta4 descriptor inventories disagree with their independent pins")
BETA4_SOURCE_INVENTORY = MappingProxyType(_source)
BETA4_TARGET_INVENTORY = MappingProxyType(_target)
BETA4_SOURCE_TEMPLATE = replace(
    _convrot,
    template_name="h3-transformer-50l-beta4-convrot",
    non_quantized_inventory=MappingProxyType(_non_quantized),
)
BETA4_TARGET_TEMPLATE = replace(
    _beta3,
    template_name="h3-transformer-50l-beta4-dense-bf16",
    non_quantized_inventory=BETA4_TARGET_INVENTORY,
)
BETA4_SOURCE_RECORD = ContractRecord(
    NativeSourceContract(BETA4_SOURCE_NAME, "transformer", 934, 200, BETA4_SOURCE_SCHEMA_SHA256),
    BETA4_SOURCE_TEMPLATE.template_name,
    STORAGE_INT8_CONVROT,
)
