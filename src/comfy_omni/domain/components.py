# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: h3-forge contributors
"""Component directory values, without filesystem or runtime activation.

Derived from h3-forge e9cb011d00b028c149db3978de246c54f6e34acc,
component_catalog/catalog.py blob 322dd5b5e37722d82675d9d6c547901b296b759f.
Selection identifies a candidate; it does not identify an active model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ComponentKind(Enum):
    DIT = "dit"
    TEXT_ENCODER = "text_encoder"
    VIDEO_VAE = "video_vae"
    AUDIO_VAE = "audio_vae"
    SCHEDULE = "schedule"
    LORA = "lora"
    TOOL = "tool"


class ComponentZone(Enum):
    COMFY = "comfy"
    OFFICIAL = "official"
    SERVABLE = "servable"


SINGLE_SELECT_KINDS = frozenset(
    {
        ComponentKind.DIT,
        ComponentKind.TEXT_ENCODER,
        ComponentKind.VIDEO_VAE,
        ComponentKind.AUDIO_VAE,
        ComponentKind.SCHEDULE,
    }
)


@dataclass(frozen=True)
class Component:
    """A discovered path or a code-defined schedule, not a load proof."""

    kind: ComponentKind
    zone: ComponentZone | None
    id: str
    path: str
    selection: str | None = None
    locked: bool = False
    contract_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ComponentKind):
            raise ValueError(f"kind must be a ComponentKind, got {self.kind!r}")
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("id must be a non-empty string")
        if not isinstance(self.path, str):
            raise ValueError("path must be a string")
        if not isinstance(self.locked, bool):
            raise ValueError("locked must be a bool")
        if self.contract_digest is not None and (not isinstance(self.contract_digest, str) or not self.contract_digest):
            raise ValueError("contract_digest must be None or a non-empty string")
        if self.kind is ComponentKind.SCHEDULE:
            if self.zone is not None:
                raise ValueError("schedule entries are code-defined: zone must be None")
            if self.path:
                raise ValueError("schedule entries have no on-disk path: path must be empty")
        elif self.zone is None:
            raise ValueError(f"disk component {self.id!r} must carry a zone, got None")
        elif not isinstance(self.zone, ComponentZone):
            raise ValueError(f"zone must be a ComponentZone, got {self.zone!r}")
        if self.selection is not None:
            if self.kind not in SINGLE_SELECT_KINDS:
                raise ValueError(f"additive kind {self.kind.value!r} has no single-select slot: selection must be None")
            if self.locked:
                raise ValueError(f"locked entry {self.id!r} is not selectable: selection must be None")
            if self.selection != self.id:
                raise ValueError(f"selection must repeat the entry id {self.id!r}, got {self.selection!r}")
