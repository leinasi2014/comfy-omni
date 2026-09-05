# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: h3-forge contributors
"""H3 component directory discovery and atomic in-memory candidate snapshots.

Derived from h3-forge e9cb011d00b028c149db3978de246c54f6e34acc,
component_catalog/catalog.py blob 322dd5b5e37722d82675d9d6c547901b296b759f.
Scanning only inspects directory entries. It does not open model files,
verify formats, compute digests, select a runtime or mutate source trees.
Legacy symlink and filename discovery semantics are retained; this is not
the artifact-integrity or authorization boundary used before model loading.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path

from comfy_omni.domain.components import SINGLE_SELECT_KINDS, Component, ComponentKind, ComponentZone

COMPONENT_ROOTS_ENV = "H3_FORGE_COMPONENT_ROOTS"
_ZONE_NAMES = ("comfy", "official", "servable")
SCHEDULE_PROFILES = (
    ("turbo_4step", "turbo-distilled 4-denoise-step plan (pairs with the turbo_4step LoRAs)"),
    ("turbo_8step", "turbo-distilled 8-denoise-step plan (pairs with the turbo_8step LoRAs)"),
)
_COMFY_DIR_KINDS = {
    "diffusion_models": ComponentKind.DIT,
    "text_encoders": ComponentKind.TEXT_ENCODER,
    "loras": ComponentKind.LORA,
    "latent_upscale_models": ComponentKind.TOOL,
}
_PACKAGE_SUBDIR_KINDS = {
    "transformer": ComponentKind.DIT,
    "transformer_ref": ComponentKind.DIT,
    "text_encoder": ComponentKind.TEXT_ENCODER,
    "video_vae": ComponentKind.VIDEO_VAE,
    "audio_vae": ComponentKind.AUDIO_VAE,
}


class CatalogError(Exception):
    """Malformed roots, an unsuccessful scan or an unknown catalog key."""


class ComponentCatalog:
    """A directory snapshot whose configured roots remain fixed for its life.

    Construction reads configuration but does not scan. Static schedules
    exist immediately. A failed scan keeps the previous index; concurrent
    scans publish only complete indexes, with the last completion winning.
    """

    def __init__(
        self,
        roots: Mapping[str | ComponentZone, str | Path | Sequence[str | Path]] | None = None,
        *,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        source = os.environ if env is None else env
        configured = roots if roots is not None else source.get(COMPONENT_ROOTS_ENV)
        self._roots = _parse_roots(configured)
        self._index = {(entry.kind, entry.id): entry for entry in _schedule_entries()}

    def scan(self) -> None:
        fresh = {(entry.kind, entry.id): entry for entry in _schedule_entries()}
        for zone in ComponentZone:
            for root in self._roots.get(zone, ()):
                for entry in _scan_root(root, zone):
                    key = (entry.kind, entry.id)
                    clash = fresh.get(key)
                    if clash is not None and clash.path != entry.path:
                        raise CatalogError(
                            f"duplicate {entry.kind.value!r} id {entry.id!r}: "
                            f"{clash.path!r} and {entry.path!r} -- component ids must be unique per kind"
                        )
                    fresh[key] = entry
        with self._lock:
            self._index = fresh

    def list(self, kind: ComponentKind | None = None) -> list[Component]:
        _require_kind(kind, allow_none=True)
        with self._lock:
            snapshot = self._index.copy()
        return sorted((entry for entry in snapshot.values() if kind is None or entry.kind == kind), key=lambda e: e.id)

    def get(self, kind: ComponentKind, component_id: str) -> Component:
        _require_kind(kind, allow_none=False)
        if not isinstance(component_id, str) or not component_id:
            raise CatalogError(f"component id must be a non-empty string, got {component_id!r}")
        with self._lock:
            entry = self._index.get((kind, component_id))
            if entry is None:
                known = sorted(item.id for item in self._index.values() if item.kind == kind)
                detail = ", ".join(repr(name) for name in known) if known else "(none)"
                raise CatalogError(
                    f"unknown {kind.value!r} component {component_id!r}; cataloged {kind.value} ids: {detail}"
                )
            return entry

    def selection_candidates(self, kind: ComponentKind) -> list[str]:
        _require_kind(kind, allow_none=False)
        with self._lock:
            return sorted(entry.id for entry in self._index.values() if entry.kind == kind and not entry.locked)

    def snapshot(self, kind: ComponentKind) -> tuple[list[Component], list[str]]:
        """Read entries and candidates under one lock, never across generations."""
        _require_kind(kind, allow_none=False)
        with self._lock:
            entries = sorted((entry for entry in self._index.values() if entry.kind == kind), key=lambda e: e.id)
            candidates = sorted(entry.id for entry in self._index.values() if entry.kind == kind and not entry.locked)
        return entries, candidates

    def roots_configured(self) -> bool:
        return any(self._roots.values())


def _parse_roots(configured: object) -> dict[ComponentZone, tuple[Path, ...]]:
    if configured is None:
        return {}
    if isinstance(configured, str):
        text = configured.strip()
        if not text:
            return {}
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise CatalogError(f"{COMPONENT_ROOTS_ENV}: invalid JSON: {exc}") from exc
            if not isinstance(parsed, dict):
                raise CatalogError(
                    f"{COMPONENT_ROOTS_ENV}: the JSON form must be an object mapping zone name -> path(s), "
                    f"got {type(parsed).__name__}"
                )
            return _parse_mapping_roots(parsed)
        roots: dict[ComponentZone, list[Path]] = {}
        for item in text.split(";"):
            item = item.strip()
            if not item:
                continue
            zone_text, separator, path_text = item.partition("=")
            zone_text, path_text = zone_text.strip(), path_text.strip()
            if not separator or not zone_text or not path_text:
                raise CatalogError(
                    f"{COMPONENT_ROOTS_ENV}: item {item!r} must be 'zone=path' "
                    f"with a zone in ({', '.join(_ZONE_NAMES)})"
                )
            roots.setdefault(_zone_from_name(zone_text), []).append(Path(path_text))
        return {zone: tuple(paths) for zone, paths in roots.items()}
    if isinstance(configured, Mapping):
        return _parse_mapping_roots(configured)
    raise CatalogError(f"component roots must be None, a string or a mapping, got {type(configured).__name__}")


def _parse_mapping_roots(mapping: Mapping[object, object]) -> dict[ComponentZone, tuple[Path, ...]]:
    roots: dict[ComponentZone, list[Path]] = {}
    for raw_zone, raw_value in mapping.items():
        zone = _zone_from_name(raw_zone)
        if isinstance(raw_value, (str, bytes, Path)):
            values = [raw_value]
        elif isinstance(raw_value, Sequence):
            values = raw_value
        else:
            raise CatalogError(
                f"component roots for zone {zone.value!r} must be a path or a list of paths, "
                f"got {type(raw_value).__name__}"
            )
        collected = []
        for raw_path in values:
            if isinstance(raw_path, Path):
                collected.append(raw_path)
            elif isinstance(raw_path, str):
                stripped = raw_path.strip()
                if not stripped:
                    raise CatalogError(f"component root for zone {zone.value!r} must be a non-empty path")
                collected.append(Path(stripped))
            else:
                raise CatalogError(
                    f"component root for zone {zone.value!r} must be a path string, got {type(raw_path).__name__}"
                )
        roots.setdefault(zone, []).extend(collected)
    return {zone: tuple(paths) for zone, paths in roots.items()}


def _zone_from_name(name: object) -> ComponentZone:
    if isinstance(name, ComponentZone):
        return name
    if isinstance(name, str):
        try:
            return ComponentZone(name)
        except ValueError:
            pass
    raise CatalogError(f"unknown component zone {name!r}; expected one of {', '.join(_ZONE_NAMES)}")


def _scan_root(root: Path, zone: ComponentZone) -> list[Component]:
    if not root.is_dir():
        raise CatalogError(
            f"zone {zone.value!r} root {str(root)!r} is not a directory "
            f"(point {COMPONENT_ROOTS_ENV} at an existing path)"
        )
    return _scan_comfy_root(root) if zone is ComponentZone.COMFY else _scan_package_root(root, zone)


def _scan_comfy_root(root: Path) -> list[Component]:
    entries = []
    for dir_name, kind in _COMFY_DIR_KINDS.items():
        directory = root / dir_name
        if directory.is_dir():
            entries.extend(_file_entry(kind, ComponentZone.COMFY, file) for file in _visible_files(directory))
    vae_directory = root / "vae"
    if vae_directory.is_dir():
        for file in _visible_files(vae_directory):
            kind = ComponentKind.AUDIO_VAE if "audio" in file.stem.lower() else ComponentKind.VIDEO_VAE
            entries.append(_file_entry(kind, ComponentZone.COMFY, file))
    return entries


def _scan_package_root(root: Path, zone: ComponentZone) -> list[Component]:
    found = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        kind = _PACKAGE_SUBDIR_KINDS.get(child.name)
        if kind is not None:
            if _holds_any_file(child):
                found.append((kind, child, Path(child.name)))
            continue
        for grandchild in sorted(child.iterdir()):
            nested_kind = _PACKAGE_SUBDIR_KINDS.get(grandchild.name)
            if nested_kind is not None and grandchild.is_dir() and _holds_any_file(grandchild):
                found.append((nested_kind, grandchild, Path(child.name) / grandchild.name))
    dit_count = sum(kind is ComponentKind.DIT for kind, _, _ in found)
    locked = zone is ComponentZone.OFFICIAL
    entries = []
    for kind, directory, relative in found:
        if kind is ComponentKind.DIT:
            component_id = root.name if dit_count == 1 else f"{root.name}/{relative.parts[0]}"
        else:
            component_id = f"{root.name}/{relative.as_posix()}"
        selection = component_id if kind in SINGLE_SELECT_KINDS and not locked else None
        entries.append(Component(kind, zone, component_id, str(directory), selection=selection, locked=locked))
    return entries


def _file_entry(kind: ComponentKind, zone: ComponentZone, file: Path) -> Component:
    component_id = file.stem
    selection = component_id if kind in SINGLE_SELECT_KINDS else None
    return Component(kind, zone, component_id, str(file), selection=selection, locked=False)


def _visible_files(directory: Path) -> list[Path]:
    files = []
    for candidate in sorted(directory.rglob("*")):
        relative = candidate.relative_to(directory)
        if not any(part.startswith(".") for part in relative.parts) and candidate.is_file():
            files.append(candidate)
    return files


def _holds_any_file(directory: Path) -> bool:
    return any(candidate.is_file() for candidate in directory.rglob("*"))


def _schedule_entries() -> list[Component]:
    return [Component(ComponentKind.SCHEDULE, None, name, "", selection=name) for name, _ in SCHEDULE_PROFILES]


def _require_kind(kind: object, *, allow_none: bool) -> None:
    if kind is None and allow_none:
        return
    if not isinstance(kind, ComponentKind):
        raise CatalogError(f"unknown component kind {kind!r}; expected a ComponentKind")
