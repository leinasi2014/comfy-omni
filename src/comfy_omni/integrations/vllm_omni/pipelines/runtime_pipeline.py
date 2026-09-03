"""Host-facing Ref2VA runtime pipeline bound to a verified ComfyOmni package.

Only the host (``vllm_omni``) imports this module; it lives inside the
sanctioned ``comfy_omni.integrations.vllm_omni`` boundary and is the only place
host subclasses may be extended with ComfyOmni package verification.

Provenance: characterized from the ``h3-forge`` legacy
``H3ComfyMiniMaxH3Pipeline`` subclass shape (curve-cache DiT swap, per-worker
latch, request locks, and schedule replay) as recorded in the ``h3-forge``
``h3/runtime_pipeline.py`` blob ``fa94f86da746ff9a11105584081464c1162d07b6`` at
commit ``e9cb011d00b028c149db3978de246c54f6e34acc``. Those runtime mechanics are
deliberately NOT migrated here yet; this slice only binds construction to a
host-free validated package contract.
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


class H3ComfyMiniMaxH3Pipeline(OfficialMiniMaxH3Pipeline):
    """A ComfyOmni package-bound Ref2VA pipeline.

    The legacy subclass shape (``h3/runtime_pipeline.py`` blob
    ``fa94f86da746ff9a11105584081464c1162d07b6``) is preserved; the curve-cache
    DiT swap, per-worker latch, request locks, and schedule replay are
    deliberately NOT migrated yet. This slice only verifies the package before
    delegating to the official pipeline.
    """

    def __init__(self, *, od_config, prefix: str = "") -> None:
        model_path = getattr(od_config, "model", None)
        if not isinstance(model_path, (str, Path)) or not model_path:
            raise RuntimePackageContractError(
                "runtime package path is not set on the od_config",
                evidence={"stage": "package-binding"},
            )
        package_dir = Path(model_path)
        if not (package_dir / MODEL_INDEX_NAME).is_file() or not (package_dir / MANIFEST_NAME).is_file():
            raise RuntimePackageContractError(
                "runtime package directory is missing its model index or manifest",
                evidence={"stage": "package-binding"},
            )
        self.comfy_omni_package = validate_runtime_package(package_dir)
        super().__init__(od_config=od_config, prefix=prefix)


__all__ = ["H3ComfyMiniMaxH3Pipeline", "get_minimax_h3_post_process_func"]
