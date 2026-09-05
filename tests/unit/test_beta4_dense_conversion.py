"""Fixed beta4 observations; no checkpoint payload is embedded in this fixture."""

from __future__ import annotations

from math import prod

from comfy_omni.contracts.registry import COMPILE_TIME_CATALOG
from comfy_omni.contracts.templates import ARCHITECTURE_TEMPLATES
from comfy_omni.conversion.contract_workflows.census import FileRecord, census_tensors
from comfy_omni.conversion.exporters.planning import build_native_export_plan
from comfy_omni.domain.checkpoints import TensorDescriptor

SOURCE_SHA = "54d56b15c65923b54c9ca16b494dae641bfe9455cfcb1c19c49b1008e270bbc1"
SOURCE_BYTES = 20_967_637_320
SOURCE_SCHEMA = "ae2456bc6ac904929a4b773f703f8a1baa99b6356b5a389994faf64a1a2d80f2"
MARKER = b'{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}'


def beta4_report():
    # The existing Apache-2.0 beta3 template is the independent comparison
    # inventory. The observed beta4 difference is precisely one missing grid
    # and the 200 declared INT8 ConvRot triplets.
    inventory = dict(ARCHITECTURE_TEMPLATES["h3-transformer-50l-hybrid8-bf16-plain"].non_quantized_inventory)
    inventory.pop("silu_t_emb_grid")
    quantized = ARCHITECTURE_TEMPLATES["h3-transformer-50l-convrot"].convrot_table()
    markers = {}
    for prefix, (shape, _) in quantized.items():
        inventory[f"{prefix}.weight"] = ("I8", shape)
        inventory[f"{prefix}.weight_scale"] = ("F32", (shape[0], 1))
        inventory[f"{prefix}.comfy_quant"] = ("U8", (72,))
        markers[f"{prefix}.comfy_quant"] = MARKER.ljust(72)
    widths = {"BF16": 2, "I8": 1, "U8": 1, "F32": 4}
    offset = 0
    descriptors = []
    for name, (dtype, shape) in sorted(inventory.items()):
        length = prod(shape) * widths[dtype]
        descriptors.append(TensorDescriptor(name, dtype, shape, (offset, offset + length)))
        offset += length
    report = census_tensors(
        tuple(descriptors), markers,
        files=(FileRecord("/models/primary.safetensors", SOURCE_BYTES, SOURCE_SHA),),
    )
    assert report.observed_schema_sha256 == SOURCE_SCHEMA
    assert report.tensor_count == 934
    return report


def test_fixed_beta4_can_plan_exact_534_dense_tensors_without_inventing_grid():
    report = beta4_report()
    record = COMPILE_TIME_CATALOG.resolve("transformer")
    plan = build_native_export_plan(report, record, ARCHITECTURE_TEMPLATES[record.template_name])
    assert plan.target_tensor_count == 534
    assert all(action.target_name != "silu_t_emb_grid" for action in plan.actions)
    assert plan.runtime_quant_method is None
