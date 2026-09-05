"""Immutable bindings for the two accepted H3 package layouts.

Legacy producer validation derives from h3-forge e9cb011, package_assembler.py
blob e64558f1d3bb6e1ee6f714b70e783d9df907f9ce (Apache-2.0).
The artifact producer is distinct from the executing ComfyOmni distribution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from comfy_omni.contracts.models import ContractError

if TYPE_CHECKING:
    from comfy_omni.runtime.h3.schedule import H3ScheduleContract

LEGACY_COMMIT = "e9cb011d00b028c149db3978de246c54f6e34acc"
HOST_COMMIT = "17285c2f55a41bf15772676121814d59a60ace35"
PROVENANCE_SCHEMA = "h3-comfy/executing-wheel/v1"
CURVE_PROFILE = "dasiwa-turbo-v4-curve-cache"
CURVE_CACHE_NAME = "curve_adaln_cache.safetensors"
CURVE_CACHE_SCHEMA = "h3-comfy/minimax-h3-curve-adaln-cache/v2"
LEGACY_COMPONENTS = ("audio_vae", "processor", "tokenizer", "video_vae", "transformer", "text_encoder")
LEGACY_TASKS = ("ref2va", "t2va", "fl2va")


class LegacyPackageError(ContractError):
    """A legacy artifact failed the compatibility contract before host loading."""


@dataclass(frozen=True)
class LegacyProducerIdentity:
    schema: str
    h3_forge_commit: str
    wheel_sha256: str
    build_context_sha256: str
    installed_payload_sha256: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


# The producer actually audited for the maintainer-approved first parity slice.
# These are artifact identities, never credentials or the executing wheel identity.
AUDITED_PRODUCER = LegacyProducerIdentity(
    PROVENANCE_SCHEMA,
    LEGACY_COMMIT,
    "d7fac7c9d49da1a3bf497fc71d6f95ad3dbf7663cb0d93934396e2c015726609",
    "d25a1e7202dda317df758f7996fd6242200017977ba72b8d7fc62d3773c7882b",
    "2df3b5b0a517fdaf64d0b7d4fea95700dc011d17832290f7ea1d1bc2df1dedc5",
)


@dataclass(frozen=True)
class CurveCacheBinding:
    cache_path: Path
    schedule: H3ScheduleContract
    sha256: str
    size: int
    source_curve_sha256: str
    producer: LegacyProducerIdentity


@dataclass(frozen=True)
class RuntimeQuantizationBinding:
    """The fixed legacy online INT8 policy; callers get a fresh host dictionary."""

    transformer_ignored_layers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "transformer": {"method": "int8", "ignored_layers": list(self.transformer_ignored_layers)},
            "text_encoder": {"method": "int8"},
            "default": None,
        }


def legacy_quantization() -> RuntimeQuantizationBinding:
    """h3-forge h3/profiles.py:277-300, with no mutable shared host config."""
    ignored = ["condition_proj", "final_layer.adaln_proj.linear"]
    ignored.extend(f"blocks.{index}.adaln_proj.linear" for index in range(50))
    for index in range(2):
        prefix = f"token_refiner.blocks.{index}"
        ignored.extend(f"{prefix}.{name}" for name in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"))
    return RuntimeQuantizationBinding(tuple(ignored))
