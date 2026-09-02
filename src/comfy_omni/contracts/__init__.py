"""Pure immutable contracts and exact architecture templates owned by ComfyOmni."""

from comfy_omni.contracts.models import (
    STORAGE_BF16_PLAIN,
    STORAGE_INT8_CONVROT,
    ArchitectureTemplate,
    ContractCatalog,
    ContractError,
    ContractRecord,
    NativeSourceContract,
)
from comfy_omni.contracts.registry import COMPILE_TIME_CATALOG, COMPILE_TIME_RECORDS
from comfy_omni.contracts.templates import ARCHITECTURE_TEMPLATES, template_digest

__all__ = [
    "ARCHITECTURE_TEMPLATES",
    "COMPILE_TIME_CATALOG",
    "COMPILE_TIME_RECORDS",
    "ArchitectureTemplate",
    "ContractCatalog",
    "ContractError",
    "ContractRecord",
    "NativeSourceContract",
    "STORAGE_BF16_PLAIN",
    "STORAGE_INT8_CONVROT",
    "template_digest",
]
