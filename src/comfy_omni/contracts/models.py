"""Pure immutable values for native-source contract workflows.

Derived from Apache-2.0 h3-forge contract models at commit
e9cb011d00b028c149db3978de246c54f6e34acc; reshaped to remove export/runtime coupling.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

STORAGE_INT8_CONVROT = "int8-convrot"
STORAGE_BF16_PLAIN = "bf16-plain"


class ContractError(ValueError):
    """A stable fail-closed contract error carrying structured evidence."""

    def __init__(self, message: str, *, evidence: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.evidence = dict(evidence or {})


@dataclass(frozen=True)
class ArchitectureTemplate:
    """One versioned, exact H3 architecture template."""

    template_name: str
    template_version: int
    component: str
    layer_topology: tuple[int, ...]
    layer_prefix_template: str
    convrot_suffixes: Mapping[str, tuple[tuple[int, int], int]]
    scale_shape_census: Mapping[tuple[int, int], int]
    curve_adaln_tensors: frozenset[str] = frozenset()
    text_encoder_direct_connection: str | None = None
    non_quantized_inventory: Mapping[str, tuple[str, tuple[int, ...]]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "convrot_suffixes", MappingProxyType(dict(self.convrot_suffixes)))
        object.__setattr__(self, "scale_shape_census", MappingProxyType(dict(self.scale_shape_census)))
        object.__setattr__(self, "non_quantized_inventory", MappingProxyType(dict(self.non_quantized_inventory)))

    def convrot_table(self) -> dict[str, tuple[tuple[int, int], int]]:
        """Expand compact layer/suffix facts into the exact ConvRot prefix table."""

        return {
            self.layer_prefix_template.format(layer=layer, suffix=suffix): value
            for layer in self.layer_topology
            for suffix, value in self.convrot_suffixes.items()
        }

    def quantized_weight_names(self) -> frozenset[str]:
        return frozenset(f"{prefix}.weight" for prefix in self.convrot_table())


@dataclass(frozen=True)
class NativeSourceContract:
    """Flat observed/enforced source contract preserved from the legacy wire model."""

    name: str
    component: str
    tensor_count: int
    convrot_group_count: int
    schema_sha256: str | None
    include_transformer_adaln: bool = False
    transformer_adaln_group_size: int | None = None


@dataclass(frozen=True)
class ContractRecord:
    """A source contract bound to its template, storage kind, and optional snapshot."""

    contract: NativeSourceContract
    template_name: str
    storage_kind: str
    snapshot_manifest_sha256: str | None = None
    snapshot_payload: bytes | None = field(default=None, repr=False, compare=False)

    @property
    def name(self) -> str:
        return self.contract.name


@dataclass(frozen=True)
class ContractCatalog:
    """An explicit immutable catalog; no activation or process-global mutation exists."""

    records: Mapping[str, ContractRecord]
    defaults: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", MappingProxyType(dict(self.records)))
        object.__setattr__(self, "defaults", MappingProxyType(dict(self.defaults)))

    def resolve(self, component: str, name: str | None = None) -> ContractRecord:
        selected = name if name is not None else self.defaults.get(component)
        record = self.records.get(selected or "")
        if record is None or record.contract.component != component:
            raise ContractError(
                f"no native source contract for component={component!r}, name={selected!r}",
                evidence={"stage": "contract-resolution", "component": component, "name": selected},
            )
        return record

    def extend(self, external: Mapping[str, ContractRecord]) -> ContractCatalog:
        collisions = sorted(set(self.records) & set(external))
        if collisions:
            raise ContractError(
                f"external contract names collide with the catalog: {collisions}",
                evidence={"stage": "contract-catalog", "collisions": collisions},
            )
        return ContractCatalog({**self.records, **external}, self.defaults)


__all__ = [
    "ArchitectureTemplate",
    "ContractCatalog",
    "ContractError",
    "ContractRecord",
    "NativeSourceContract",
    "STORAGE_BF16_PLAIN",
    "STORAGE_INT8_CONVROT",
]
