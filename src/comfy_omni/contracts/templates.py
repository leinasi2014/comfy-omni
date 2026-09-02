"""Load and verify the exact architecture templates migrated from h3-forge.

The generated resource is pinned to h3-forge commit e9cb011d00b028c149db3978de246c54f6e34acc
and templates.py blob 443a5cc9ca58891c3852079c8589fbe2f5af6484 (Apache-2.0).
"""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from types import MappingProxyType
from typing import Any

from comfy_omni.contracts.models import ArchitectureTemplate

RESOURCE_NAME = "templates.v1.json"
RESOURCE_SCHEMA = "comfy_omni.contract_templates/v1"
RESOURCE_SHA256 = "294d8cf5b790d7de42b91c385a72030dcbd318eddd44e6bd72d6f5b886c6125d"
SOURCE_COMMIT = "e9cb011d00b028c149db3978de246c54f6e34acc"
SOURCE_BLOB = "443a5cc9ca58891c3852079c8589fbe2f5af6484"

EXPECTED_TEMPLATE_DIGESTS = {
    "h3-te-pruned24-convrot": "ca3196815cec871f606ac14f8cd50e995674008b64c835867dd06423a3889a8e",
    "h3-transformer-50l-convrot": "1bc9c6241c0b1e7c6b95f494b09b4bc8aee8ae07e59804522ebcc1ab657361d1",
    "h3-transformer-50l-convrot-adaln64": "41430e728c2ef641f2b1f5ee3db796fd9ec37a2c237f9bf9f7bc1c11f5833a25",
    "h3-transformer-50l-hybrid8-bf16-plain": "9e6124e9d90c121198637ae42f6384cea4e72983a721fbc208304eac89ca94fb",
}


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _parse_strict(payload: bytes) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"duplicate contract template resource key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise RuntimeError(f"non-standard contract template constant {value!r}")

    document = json.loads(payload, object_pairs_hook=unique, parse_constant=reject_constant)
    if not isinstance(document, dict) or _canonical_json(document) != payload:
        raise RuntimeError("contract template resource is not a canonical JSON object")
    return document


def _template_from_document(raw: dict[str, Any]) -> ArchitectureTemplate:
    suffixes = {name: (tuple(value["shape"]), value["group_size"]) for name, value in raw["convrot_suffixes"].items()}
    scale_census = {
        tuple(int(part) for part in key.split("x", 1)): count for key, count in raw["scale_shape_census"].items()
    }
    inventory = {
        name: (value["dtype"], tuple(value["shape"])) for name, value in raw["non_quantized_inventory"].items()
    }
    return ArchitectureTemplate(
        template_name=raw["template_name"],
        template_version=raw["template_version"],
        component=raw["component"],
        layer_topology=tuple(raw["layer_topology"]),
        layer_prefix_template=raw["layer_prefix_template"],
        convrot_suffixes=suffixes,
        scale_shape_census=scale_census,
        curve_adaln_tensors=frozenset(raw["curve_adaln_tensors"]),
        text_encoder_direct_connection=raw["text_encoder_direct_connection"],
        non_quantized_inventory=inventory,
    )


def template_digest(template: ArchitectureTemplate) -> str:
    """Return the legacy-compatible decision-table SHA-256 for one template."""

    payload = {
        "template_name": template.template_name,
        "template_version": template.template_version,
        "component": template.component,
        "layer_topology": list(template.layer_topology),
        "convrot_table": {
            prefix: [list(shape), group_size]
            for prefix, (shape, group_size) in sorted(template.convrot_table().items())
        },
        "non_quantized_inventory": {
            name: [dtype, list(shape)] for name, (dtype, shape) in sorted(template.non_quantized_inventory.items())
        },
        "curve_adaln_tensors": sorted(template.curve_adaln_tensors),
        "text_encoder_direct_connection": template.text_encoder_direct_connection,
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _load_templates() -> dict[str, ArchitectureTemplate]:
    payload = files("comfy_omni.resources.contracts").joinpath(RESOURCE_NAME).read_bytes()
    if hashlib.sha256(payload).hexdigest() != RESOURCE_SHA256:
        raise RuntimeError("contract template resource SHA-256 is not the audited value")
    document = _parse_strict(payload)
    source = document.get("source", {})
    if document.get("schema") != RESOURCE_SCHEMA or source.get("commit") != SOURCE_COMMIT:
        raise RuntimeError("contract template resource schema/source commit is not audited")
    if source.get("templates_blob") != SOURCE_BLOB:
        raise RuntimeError("contract template resource source blob is not audited")
    templates = {name: _template_from_document(raw) for name, raw in document["templates"].items()}
    observed = {name: template_digest(template) for name, template in templates.items()}
    resource_digests = {name: raw["legacy_template_sha256"] for name, raw in document["templates"].items()}
    if observed != EXPECTED_TEMPLATE_DIGESTS or resource_digests != EXPECTED_TEMPLATE_DIGESTS:
        raise RuntimeError("contract template decision tables drifted from the audited source")
    return templates


ARCHITECTURE_TEMPLATES = MappingProxyType(_load_templates())

__all__ = ["ARCHITECTURE_TEMPLATES", "ArchitectureTemplate", "template_digest"]
