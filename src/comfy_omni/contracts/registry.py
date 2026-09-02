"""Compile-time native-source contracts and explicit immutable catalog construction.

Pinned facts derive from Apache-2.0 h3-forge registry.py blob
edf6342233d86b112359abcd5e09e5c3b22028b3 at commit e9cb011d00b028c149db3978de246c54f6e34acc.
"""

from __future__ import annotations

from types import MappingProxyType

from comfy_omni.contracts.models import (
    STORAGE_INT8_CONVROT,
    ContractCatalog,
    ContractRecord,
    NativeSourceContract,
)

SOURCE_PROFILE_DASIWA_REF2VA_HYBRID = "minimax-h3-dasiwa-ref2va-hybrid-int8-convrot-v1"
SOURCE_PROFILE_REF2VA_FULL = "minimax-h3-ref2va-int8-convrot-v1"
SOURCE_PROFILE_TEXT_ENCODER_PRUNED24 = "qwen3vl-32b-minimax-h3-pruned24-int8-convrot-v1"


def _record(contract: NativeSourceContract, template_name: str) -> ContractRecord:
    return ContractRecord(contract=contract, template_name=template_name, storage_kind=STORAGE_INT8_CONVROT)


_COMPILE_TIME_RECORDS = {
    SOURCE_PROFILE_DASIWA_REF2VA_HYBRID: _record(
        NativeSourceContract(
            SOURCE_PROFILE_DASIWA_REF2VA_HYBRID,
            "transformer",
            932,
            200,
            "cc7976f678e6d4a567e718aca56c1db4aa91adfa27108db84066cce3213edf9d",
        ),
        "h3-transformer-50l-convrot",
    ),
    SOURCE_PROFILE_REF2VA_FULL: _record(
        NativeSourceContract(
            SOURCE_PROFILE_REF2VA_FULL,
            "transformer",
            1035,
            250,
            "57c6a5a9beddcc6160d0dd7a59397c9201945756f896c52fa8816f4e7d7d7bfd",
            True,
            64,
        ),
        "h3-transformer-50l-convrot-adaln64",
    ),
    SOURCE_PROFILE_TEXT_ENCODER_PRUNED24: _record(
        NativeSourceContract(
            SOURCE_PROFILE_TEXT_ENCODER_PRUNED24,
            "text_encoder",
            952,
            168,
            None,
        ),
        "h3-te-pruned24-convrot",
    ),
}

COMPILE_TIME_RECORDS = MappingProxyType(_COMPILE_TIME_RECORDS)
COMPILE_TIME_CATALOG = ContractCatalog(
    COMPILE_TIME_RECORDS,
    {
        "transformer": SOURCE_PROFILE_DASIWA_REF2VA_HYBRID,
        "text_encoder": SOURCE_PROFILE_TEXT_ENCODER_PRUNED24,
    },
)

__all__ = [
    "COMPILE_TIME_CATALOG",
    "COMPILE_TIME_RECORDS",
    "SOURCE_PROFILE_DASIWA_REF2VA_HYBRID",
    "SOURCE_PROFILE_REF2VA_FULL",
    "SOURCE_PROFILE_TEXT_ENCODER_PRUNED24",
]
