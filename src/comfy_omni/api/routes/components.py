# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: h3-forge contributors
"""Read-only component HTTP routes; host mounting belongs to the integration.

Derived from h3-forge e9cb011d00b028c149db3978de246c54f6e34acc,
component_catalog/api.py blob 03597ba2952a6d7933fa174cdfe5b1073b234d9d.
Importing this module does not import FastAPI, read roots or scan directories.
The first actual request builds and scans the process-local catalog once.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import Any

from comfy_omni.domain.components import Component, ComponentKind
from comfy_omni.runtime.components.catalog import COMPONENT_ROOTS_ENV, CatalogError, ComponentCatalog

API_PREFIX = "/v1/h3-forge/components"
LIST_SCHEMA = "h3_forge.components.list/v1"
SCAN_SCHEMA = "h3_forge.components.scan/v1"
ERROR_SCHEMA = "h3_forge.error/v1"
ENV_ROOTS = COMPONENT_ROOTS_ENV

_BUILD_LOCK = threading.Lock()
_CATALOG: ComponentCatalog | None = None
_Error = tuple[int, str, str]


class _InitialScanError(Exception):
    def __init__(self, cause: CatalogError) -> None:
        super().__init__(str(cause))
        self.cause = cause


def _build_catalog() -> ComponentCatalog:
    catalog = ComponentCatalog()
    try:
        catalog.scan()
    except CatalogError as exc:
        raise _InitialScanError(exc) from exc
    return catalog


def _catalog() -> ComponentCatalog:
    global _CATALOG
    with _BUILD_LOCK:
        if _CATALOG is None:
            _CATALOG = _build_catalog()
        return _CATALOG


def reset_catalog() -> None:
    """Drop a catalog for tests or an explicit composition-layer reset."""
    global _CATALOG
    with _BUILD_LOCK:
        _CATALOG = None


def _not_configured_message() -> str:
    return (
        f"the h3-forge component catalog is not configured: {ENV_ROOTS} is not set; "
        "point it at the component roots as semicolon-separated zone=path pairs "
        "(zones: comfy, official, servable), for example "
        f"{ENV_ROOTS}='comfy=/data/comfy;official=/data/MiniMax-H3-official', then retry"
    )


def _configured_catalog() -> tuple[ComponentCatalog | None, _Error | None]:
    try:
        catalog = _catalog()
        configured = catalog.roots_configured()
    except _InitialScanError as exc:
        return None, (500, "components_scan_failed", str(exc.cause))
    except CatalogError as exc:
        return None, (500, "components_misconfigured", f"{ENV_ROOTS}: {exc}")
    except Exception as exc:
        return None, (500, "internal_error", f"{type(exc).__name__}: {exc}")
    if not configured:
        reset_catalog()
        return None, (503, "components_not_configured", _not_configured_message())
    return catalog, None


def _catalog_failure(exc: BaseException) -> _Error:
    if isinstance(exc, CatalogError):
        return 500, "components_scan_failed", str(exc)
    if isinstance(exc, (ValueError, TypeError)):
        return 422, "invalid_request", str(exc)
    return 500, "internal_error", f"{type(exc).__name__}: {exc}"


def _kind_key(kind: Any) -> str:
    return str(getattr(kind, "name", kind)).lower()


def _entry(component: Component) -> dict[str, Any]:
    return {
        "kind": _kind_key(component.kind),
        "zone": None if component.zone is None else str(getattr(component.zone, "name", component.zone)).lower(),
        "id": component.id,
        "path": str(component.path),
        "locked": bool(component.locked),
        "selection": component.selection,
        "contract_digest": component.contract_digest,
    }


def _group_by_kind(components: Iterable[Component]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for component in components:
        grouped.setdefault(_kind_key(component.kind), []).append(_entry(component))
    order = {member.name.lower(): index for index, member in enumerate(ComponentKind)}
    return dict(sorted(grouped.items(), key=lambda item: (order.get(item[0], len(order)), item[0])))


def _resolve_kind(kind: str) -> tuple[ComponentKind | None, _Error | None]:
    try:
        return ComponentKind[kind.upper()], None
    except KeyError:
        known = ", ".join(member.name.lower() for member in ComponentKind)
        return None, (404, "H3_COMPONENT_KIND_UNKNOWN", f"unknown component kind {kind!r}; known kinds: {known}")


def build_router():
    """Build the three legacy routes; only this explicit call imports FastAPI."""
    from fastapi import APIRouter
    from fastapi.responses import JSONResponse

    router = APIRouter(prefix=API_PREFIX, tags=["h3-forge components"])

    def error_response(error: _Error):
        status, kind, message = error
        return JSONResponse(
            status_code=status, content={"schema": ERROR_SCHEMA, "error": {"kind": kind, "message": message}}
        )

    @router.get("")
    def list_components():
        catalog, error = _configured_catalog()
        if error is not None:
            return error_response(error)
        try:
            components = catalog.list()
        except Exception as exc:
            return error_response(_catalog_failure(exc))
        return JSONResponse(content={"schema": LIST_SCHEMA, "kinds": _group_by_kind(components)})

    @router.post("/scan")
    def scan_components():
        catalog, error = _configured_catalog()
        if error is not None:
            return error_response(error)
        try:
            catalog.scan()
            components = catalog.list()
        except Exception as exc:
            return error_response(_catalog_failure(exc))
        return JSONResponse(content={"schema": SCAN_SCHEMA, "scanned": True, "kinds": _group_by_kind(components)})

    @router.get("/{kind}")
    def list_kind_components(kind: str):
        catalog, error = _configured_catalog()
        if error is not None:
            return error_response(error)
        resolved, error = _resolve_kind(kind)
        if error is not None:
            return error_response(error)
        try:
            components, candidates = catalog.snapshot(resolved)
        except Exception as exc:
            return error_response(_catalog_failure(exc))
        return JSONResponse(
            content={
                "schema": LIST_SCHEMA,
                "kind": _kind_key(resolved),
                "selection_candidates": list(candidates),
                "components": [_entry(component) for component in components],
            }
        )

    return router
