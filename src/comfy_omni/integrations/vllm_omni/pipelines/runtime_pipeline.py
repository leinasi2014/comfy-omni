"""Host-facing Ref2VA runtime pipeline bound to a verified ComfyOmni package.

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
    """Construct the runtime selected by the complete verified package contract."""

    def __new__(cls, *, od_config, prefix: str = ""):
        model_path = getattr(od_config, "model", None)
        if not isinstance(model_path, (str, Path)) or not model_path:
            raise RuntimePackageContractError(
                "runtime package path is not set on the od_config",
                evidence={"stage": "package-binding"},
            )
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
