"""Host-facing Ref2VA routing for original files and existing verified packages.

Only the host (``vllm_omni``) imports this module; it lives inside the
sanctioned ``comfy_omni.integrations.vllm_omni`` boundary and is the only place
host subclasses may be extended with ComfyOmni package verification.

Provenance: characterized from the ``h3-forge`` legacy
``H3ComfyMiniMaxH3Pipeline`` subclass shape (curve-cache DiT swap, per-worker
latch, request locks, and schedule replay) as recorded in the ``h3-forge``
``h3/runtime_pipeline.py`` blob ``fa94f86da746ff9a11105584081464c1162d07b6`` at
commit ``e9cb011d00b028c149db3978de246c54f6e34acc``. Verified curve-cache packages
use the dedicated cache pipeline; native packages retain the official pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
    MiniMaxH3Pipeline as OfficialMiniMaxH3Pipeline,
)
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
    get_minimax_h3_post_process_func,
)

from comfy_omni.integrations.vllm_omni.package_contract import (
    MANIFEST_NAME,
    MODEL_INDEX_NAME,
    RuntimePackageContractError,
    validate_runtime_package,
)
from comfy_omni.integrations.vllm_omni.pipelines.scoped_construction import CONSTRUCTION_LOCK
from comfy_omni.integrations.vllm_omni.serving import MARKER_NAME, SERVING_PARTITION_NAME


def _raw_sources(settings):
    """Authenticate configured original files once per distinct format/path."""

    def invalid(message):
        raise RuntimePackageContractError(message, evidence={"stage": "source-binding"})

    if not isinstance(settings, Mapping):
        invalid("comfy_omni_h3 must be a mapping")
    if set(settings) == {"transformer_source"}:
        active = "initial"
        sources = {active: {"path": settings["transformer_source"], "format": "h3-beta4-convrot"}}
    elif set(settings) == {"active", "sources"}:
        active, sources = settings["active"], settings["sources"]
    else:
        invalid("comfy_omni_h3 requires transformer_source or active and sources")
    if not isinstance(sources, Mapping) or not sources:
        invalid("H3 sources must be a non-empty mapping of selection IDs")
    if not isinstance(active, str) or active not in sources:
        invalid("H3 active selection must name a configured source")
    validated = {}
    for selection, entry in sources.items():
        if not isinstance(selection, str) or not selection or len(selection) > 128:
            invalid("H3 selection IDs must contain 1 to 128 characters")
        if not isinstance(entry, Mapping) or set(entry) != {"path", "format"}:
            invalid("each H3 source requires path and format")
        source, source_format = entry["path"], entry["format"]
        if not isinstance(source, (str, Path)) or not source or not Path(source).is_absolute():
            invalid("H3 source must be an existing absolute file path")
        if source_format not in ("h3-beta4-convrot", "h3-pruned-convrot"):
            invalid(f"unsupported H3 source format: {source_format}")
        validated[selection] = (Path(source), source_format)
    authenticated, bindings = {}, {}
    for selection, key in validated.items():
        if key not in authenticated:
            path, source_format = key
            if source_format == "h3-beta4-convrot":
                from comfy_omni.runtime.h3.raw_beta4 import RawBeta4Binding

                binding = RawBeta4Binding.establish(path)
            else:
                from comfy_omni.runtime.h3.raw_standard import RawStandardBinding

                binding = RawStandardBinding.establish(path)
            authenticated[key] = binding
        bindings[selection] = authenticated[key]
    return active, bindings


def _resolve_package_root(model_path: Path) -> Path:
    """Return the real package root for a package directory or a serving view.

    A real package root carries both ``model_index.json`` and the manifest. A
    serving view (``integrations.vllm_omni.serving``) carries the model index and
    component symlinks but no manifest; its parent holds the layout marker. For
    a view, the real package root is recovered through the ``transformer``
    component symlink (``<package_root>/Ref2VA/transformer``).
    """
    if (model_path / MODEL_INDEX_NAME).is_file() and (model_path / MANIFEST_NAME).is_file():
        return model_path
    if (
        model_path.name == SERVING_PARTITION_NAME
        and (model_path / MODEL_INDEX_NAME).is_file()
        and (model_path.parent / MODEL_INDEX_NAME).is_file()
        and (model_path.parent / MANIFEST_NAME).is_file()
    ):
        return model_path.parent
    if (
        (model_path / MODEL_INDEX_NAME).is_file()
        and (model_path.parent / MARKER_NAME).is_file()
        and (model_path / "transformer").is_symlink()
    ):
        resolved = (model_path / "transformer").resolve(strict=True)
        if resolved.parent.name == SERVING_PARTITION_NAME:
            return resolved.parent.parent
    raise RuntimePackageContractError(
        "runtime package directory is missing its model index or manifest",
        evidence={"stage": "package-binding"},
    )


class H3ComfyMiniMaxH3Pipeline(OfficialMiniMaxH3Pipeline):
    """Select an explicit original source or the existing package contract."""

    def __new__(cls, *, od_config, prefix: str = ""):
        model_path = getattr(od_config, "model", None)
        if not isinstance(model_path, (str, Path)) or not model_path:
            raise RuntimePackageContractError(
                "runtime package path is not set on the od_config",
                evidence={"stage": "package-binding"},
            )
        additional = getattr(od_config, "additional_config", None) or {}
        if "comfy_omni_h3" in additional:
            component_root = Path(model_path)
            if not component_root.is_dir():
                raise RuntimePackageContractError(
                    "original H3 loading requires an existing shared component root",
                    evidence={"stage": "source-binding"},
                )
            from comfy_omni.integrations.vllm_omni.pipelines.beta4_pipeline import H3Beta4Pipeline

            active, bindings = _raw_sources(additional["comfy_omni_h3"])
            result = H3Beta4Pipeline(
                od_config=od_config, raw_binding=bindings[active], raw_selection=active, prefix=prefix
            )
            for selection, binding in bindings.items():
                if selection != active:
                    result.comfy_omni_register_h3_dit(selection, binding)
            return result
        package_root = _resolve_package_root(Path(model_path))
        package = validate_runtime_package(package_root)
        if getattr(package, "beta4", None) is not None:
            if getattr(package, "curve_cache", None) is not None:
                raise RuntimePackageContractError(
                    "runtime package binds competing DiT routes", evidence={"stage": "routing"}
                )
            from comfy_omni.integrations.vllm_omni.pipelines.beta4_pipeline import H3Beta4Pipeline

            return H3Beta4Pipeline(od_config=od_config, package=package, prefix=prefix)
        if getattr(package, "curve_cache", None) is not None:
            from comfy_omni.integrations.vllm_omni.pipelines.cache_pipeline import H3CurveCachePipeline

            return H3CurveCachePipeline(od_config=od_config, package=package, prefix=prefix)
        instance = super().__new__(cls)
        object.__setattr__(instance, "comfy_omni_package", package)
        return instance

    def __init__(self, *, od_config, prefix: str = "") -> None:
        with CONSTRUCTION_LOCK:
            super().__init__(od_config=od_config, prefix=prefix)


__all__ = ["H3ComfyMiniMaxH3Pipeline", "get_minimax_h3_post_process_func"]
