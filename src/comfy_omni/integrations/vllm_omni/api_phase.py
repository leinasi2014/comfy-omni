# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: h3-forge contributors
"""Host mounting invoked only by the bootstrap's completed API import phase.

Derived from h3-forge e9cb011d00b028c149db3978de246c54f6e34acc
component_catalog/api.py blob 03597ba2952a6d7933fa174cdfe5b1073b234d9d.
The pinned host API module (17285c2f55a41bf15772676121814d59a60ace35,
blob 57adaad08ff28160831f503e639425f250bf4313) includes its module-level
router into its app later. No route mounting belongs to worker startup.
"""

from __future__ import annotations


def mount_components(api_server: object) -> None:
    """Mount the real component routes once; failures remain retryable."""
    from comfy_omni.api.routes.components import API_PREFIX, build_router

    target = getattr(api_server, "router", None)
    if target is None:
        target = getattr(api_server, "app", None)
    if target is None or not callable(getattr(target, "include_router", None)):
        raise RuntimeError(f"host API server has no APIRouter or FastAPI app; cannot mount components at {API_PREFIX}")
    if getattr(target, "h3_forge_components_mounted", False):
        return
    target.include_router(build_router())
    target.h3_forge_components_mounted = True
