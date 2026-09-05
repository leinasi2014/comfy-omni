"""Fused single-partition serving view for a validated vLLM-Omni runtime package.

Purpose
-------
The pinned vLLM-Omni host resolves a served model directory to a pipeline class only by the
partition address it is given: the ``_minimax_h3_partition_for_task`` host behavior keys the
``ref2va`` partition off the served path's basename being ``Ref2VA`` and that directory
containing ``model_index.json``; the pipeline then loads every component through
``subfolder="<component>"`` paths relative to that partition directory. A published ComfyOmni
package keeps the frozen root-index format (``model_index.json`` and ``h3-comfy-package.json``
at the package root) with the ``Ref2VA/`` component tree below it, so an operator cannot point
``vllm serve`` at a bare partition directory the pin expects. This module fuses one validated
package root into the partition-named serving layout the host expects — a real ``Ref2VA``
directory exposing ``model_index.json`` and the six frozen components as symlinks into the
package — so ``vllm serve <work_dir>/Ref2VA --model-class-name MiniMaxH3Pipeline`` reaches a
ready orchestrator without modifying the package or the host.

Provenance
----------
The operational serving shape (serve the package ``Ref2VA`` directory together with a
``--model-class-name``) is characterized from the Apache-2.0 ``h3-forge``
``deploy/206/serve_010_native.sh`` at commit
``e9cb011d00b028c149db3978de246c54f6e34acc`` (blob
``8d266bbb07777c3808c9bb4c100261b7fbaddb2c``). The composite symlink-view layout is a ComfyOmni
addition required by the frozen root-index package format that keeps ``model_index.json`` at the
package root; see issue #39.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from comfy_omni.contracts.models import ContractError
from comfy_omni.integrations.vllm_omni.package_contract import (
    MODEL_INDEX_NAME,
    RuntimePackageContract,
    validate_runtime_package,
)

SERVING_PARTITION_NAME = "Ref2VA"
MARKER_NAME = ".comfy-omni-serving"
MARKER_CONTENT = "comfy-omni.serving-layout/v1"
_COMPONENT_NAMES = (
    "audio_vae",
    "processor",
    "text_encoder",
    "tokenizer",
    "transformer",
    "video_vae",
)


class ServingLayoutError(ContractError):
    """A refusal to prepare, verify, or clear a serving layout."""


def _fail(detail: str, stage: str, **evidence: object) -> NoReturn:
    raise ServingLayoutError(detail, evidence={"stage": stage, **evidence})


def _is_correct_layout(work: Path, package_root: Path) -> bool:
    """Return whether ``work`` holds exactly a correct serving layout for ``package_root``."""
    try:
        entries = {entry.name for entry in os.scandir(work)}
    except OSError:
        return False
    if entries != {MARKER_NAME, SERVING_PARTITION_NAME}:
        return False
    try:
        if (work / MARKER_NAME).read_text() != MARKER_CONTENT:
            return False
        view = work / SERVING_PARTITION_NAME
        if not view.is_dir() or view.is_symlink():
            return False
        view_entries = {entry.name for entry in os.scandir(view)}
        expected = {MODEL_INDEX_NAME} | set(_COMPONENT_NAMES)
        if view_entries != expected:
            return False
        if not (view / MODEL_INDEX_NAME).is_symlink():
            return False
        if (view / MODEL_INDEX_NAME).resolve(strict=True) != (package_root / MODEL_INDEX_NAME):
            return False
        for component in _COMPONENT_NAMES:
            link = view / component
            if not link.is_symlink():
                return False
            if link.resolve(strict=True) != (package_root / SERVING_PARTITION_NAME / component):
                return False
        return True
    except OSError:
        return False


def _remove_created(work: Path) -> None:
    """Best-effort remove only the entries a failed creation call made in ``work``."""
    view = work / SERVING_PARTITION_NAME
    for component in _COMPONENT_NAMES:
        try:
            os.remove(view / component)
        except OSError:
            pass
    try:
        os.remove(view / MODEL_INDEX_NAME)
    except OSError:
        pass
    try:
        os.rmdir(view)
    except OSError:
        pass
    try:
        os.remove(work / MARKER_NAME)
    except OSError:
        pass
    try:
        os.rmdir(work)
    except OSError:
        pass


def prepare_serving_layout(package_root: Path | str, work_dir: Path | str) -> Path:
    """Prepare the composite partition serving view for a validated runtime package.

    First validates ``package_root`` (any :class:`RuntimePackageContractError` from
    :func:`validate_runtime_package` propagates fail-closed), then builds ``work_dir`` holding the
    layout marker plus a REAL ``Ref2VA`` directory that exposes exactly the official partition
    shape as a link view: ``model_index.json -> <package_root>/model_index.json`` and one symlink
    per frozen component ``Ref2VA/<component> -> <package_root>/Ref2VA/<component>``. The pinned
    host requires the partition directory itself to contain ``model_index.json`` and the component
    subdirectories (it passes ``subfolder="transformer"``, ``subfolder="processor"`` and so on
    against the partition path), which the frozen root-index package layout cannot provide
    directly. An existing ``work_dir`` is accepted only if it already is a correct layout for the
    same root, in which case the path is returned idempotently; a divergent ``work_dir`` is
    refused (stage ``layout-binding``). Creation failures are refused (stage ``layout``) leaving
    no partial layout.

    Returns the servable model path ``work_dir / Ref2VA``.
    """
    contract: RuntimePackageContract = validate_runtime_package(package_root)
    if contract.layout == "h3-forge-native-v3":
        # The fully verified legacy package already has the native partition
        # index. Use those immutable bytes without creating or rewriting a view.
        assert contract.partition_path is not None
        return contract.partition_path
    package_root_resolved = contract.package_root
    work = Path(work_dir)

    if work.exists():
        if not work.is_dir():
            _fail("work_dir exists and is not a directory", "layout-binding", path=str(work))
        if not _is_correct_layout(work, package_root_resolved):
            _fail("work_dir already exists and is not a correct serving layout", "layout-binding", path=str(work))
        return work / SERVING_PARTITION_NAME

    view = work / SERVING_PARTITION_NAME
    try:
        work.mkdir(parents=True)
        (work / MARKER_NAME).write_text(MARKER_CONTENT)
        view.mkdir()
        os.symlink(package_root_resolved / MODEL_INDEX_NAME, view / MODEL_INDEX_NAME)
        for component in _COMPONENT_NAMES:
            os.symlink(
                package_root_resolved / SERVING_PARTITION_NAME / component,
                view / component,
                target_is_directory=True,
            )
    except OSError as exc:
        _remove_created(work)
        _fail("serving layout could not be created", "layout", path=str(work), cause=str(exc))

    if not (view / MODEL_INDEX_NAME).is_file():
        _remove_created(work)
        _fail("serving view does not expose the model index", "layout", path=str(view))
    for component in _COMPONENT_NAMES:
        if not (view / component).is_dir():
            _remove_created(work)
            _fail("serving view does not expose the component", "layout", path=str(view / component))
    return view


def clear_serving_layout(work_dir: Path | str) -> None:
    """Remove a serving layout created by :func:`prepare_serving_layout`.

    Refuses (stage ``layout-binding``) when the marker is absent or the ``Ref2VA`` entry is not a
    directory. Every entry inside the view is removed without following symlinks (``os.remove`` on
    links only), then the view directory, the marker file, and the now-empty ``work_dir`` (refused
    as ``layout`` if it is not empty).
    """
    work = Path(work_dir)
    marker_path = work / MARKER_NAME
    view = work / SERVING_PARTITION_NAME
    if not marker_path.is_file() or not view.is_dir() or view.is_symlink():
        _fail("directory is not a serving layout", "layout-binding", path=str(work))
    try:
        for entry in os.scandir(view):
            os.remove(entry.path)
        os.rmdir(view)
    except OSError as exc:
        _fail("serving view could not be removed", "layout", path=str(view), cause=str(exc))
    try:
        os.remove(marker_path)
    except OSError as exc:
        _fail("serving marker could not be removed", "layout", path=str(marker_path), cause=str(exc))
    try:
        os.rmdir(work)
    except OSError as exc:
        _fail("serving work_dir is not empty after clear", "layout", path=str(work), cause=str(exc))


def describe_serving_command(model_path: Path, *, host_args: Sequence[str] = ()) -> str:
    """Return the documented legacy-shape ``vllm serve`` command for a serving layout view.

    Builds ``vllm serve <model_path> --model-class-name MiniMaxH3Pipeline [host_args...]`` as a pure
    string; the model-class name is the literal key our bootstrap registers for the MiniMaxH3Pipeline
    architecture (``MiniMaxH3Pipeline``).
    """
    return " ".join(["vllm", "serve", str(model_path), "--model-class-name", "MiniMaxH3Pipeline", *host_args])


__all__ = [
    "MARKER_CONTENT",
    "MARKER_NAME",
    "SERVING_PARTITION_NAME",
    "ServingLayoutError",
    "clear_serving_layout",
    "describe_serving_command",
    "prepare_serving_layout",
]
