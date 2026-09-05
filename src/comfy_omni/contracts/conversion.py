"""Immutable H3 native-export policy contracts.

The public identifiers and runtime policy are migrated from Apache-2.0
``h3_forge.h3.profiles`` and ``h3_forge.h3.contracts.registry`` at commit
e9cb011d00b028c149db3978de246c54f6e34acc.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from comfy_omni.contracts.models import STORAGE_INT8_CONVROT

EXPORT_SCHEMA: Final = "h3-comfy-int8-export/v2"
PROFILE_DENSE_BF16_ONLINE_INT8: Final = "dense-bf16-online-int8"
PROFILE_BETA4_DENSE_BF16: Final = "beta4-dense-bf16"
QKV_SOURCE_LAYOUT: Final = "runtime-qkv"
QKV_TARGET_LAYOUT: Final = "grouped-for-official-loader"


def _transformer_ignored_layers() -> tuple[str, ...]:
    ignored = ["condition_proj", "final_layer.adaln_proj.linear"]
    ignored.extend(f"blocks.{index}.adaln_proj.linear" for index in range(50))
    for index in range(2):
        prefix = f"token_refiner.blocks.{index}"
        ignored.extend(
            (
                f"{prefix}.attn.qkv_proj",
                f"{prefix}.attn.out_proj",
                f"{prefix}.mlp.fc1",
                f"{prefix}.mlp.fc2",
            )
        )
    if len(ignored) != 60 or len(set(ignored)) != 60:
        raise RuntimeError("invalid H3 transformer ignored-layer contract")
    return tuple(ignored)


@dataclass(frozen=True)
class QkvLayoutContract:
    """The official H3 loader's disk/runtime QKV relationship."""

    source_layout: str
    target_layout: str
    num_query_groups: int
    heads_per_group: int
    head_dim: int

    @property
    def row_count(self) -> int:
        return self.num_query_groups * (self.heads_per_group + 2) * self.head_dim


@dataclass(frozen=True)
class NativeExportProfile:
    """One allowed output policy; this is policy, not an execution backend."""

    name: str
    component: str
    source_storage_kind: str
    output_weight_dtype: str
    runtime_quant_method: str | None
    runtime_ignored_layers: tuple[str, ...]
    qkv: QkvLayoutContract
    payload_semantics: str


H3_TRANSFORMER_QKV = QkvLayoutContract(
    source_layout=QKV_SOURCE_LAYOUT,
    target_layout=QKV_TARGET_LAYOUT,
    num_query_groups=56,
    heads_per_group=1,
    head_dim=128,
)

DENSE_BF16_ONLINE_INT8 = NativeExportProfile(
    name=PROFILE_DENSE_BF16_ONLINE_INT8,
    component="transformer",
    source_storage_kind=STORAGE_INT8_CONVROT,
    output_weight_dtype="BF16",
    runtime_quant_method="int8",
    runtime_ignored_layers=_transformer_ignored_layers(),
    qkv=H3_TRANSFORMER_QKV,
    payload_semantics="inverse-convrot-to-dense-bf16; runtime-int8-required; not-payload-preserving",
)

BETA4_DENSE_BF16 = NativeExportProfile(
    name=PROFILE_BETA4_DENSE_BF16,
    component="transformer",
    source_storage_kind=STORAGE_INT8_CONVROT,
    output_weight_dtype="BF16",
    runtime_quant_method=None,
    runtime_ignored_layers=(),
    qkv=H3_TRANSFORMER_QKV,
    payload_semantics="inverse-convrot-to-dense-bf16; dense-bf16-execution; not-payload-preserving",
)
NATIVE_EXPORT_PROFILES = MappingProxyType({item.name: item for item in (DENSE_BF16_ONLINE_INT8, BETA4_DENSE_BF16)})

__all__ = [
    "BETA4_DENSE_BF16",
    "PROFILE_BETA4_DENSE_BF16",
    "DENSE_BF16_ONLINE_INT8",
    "EXPORT_SCHEMA",
    "H3_TRANSFORMER_QKV",
    "NATIVE_EXPORT_PROFILES",
    "PROFILE_DENSE_BF16_ONLINE_INT8",
    "QKV_SOURCE_LAYOUT",
    "QKV_TARGET_LAYOUT",
    "NativeExportProfile",
    "QkvLayoutContract",
]
