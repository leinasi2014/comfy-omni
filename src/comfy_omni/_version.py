"""Distribution-backed version access.

This standard-library-only leaf is the sole runtime reader of the package version. The canonical
value lives in ``pyproject.toml`` and is exposed through installed distribution metadata.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

_DISTRIBUTION_NAME = "comfy-omni"
_UNINSTALLED_VERSION = "0.0.0+uninstalled"


def distribution_version() -> str:
    """Return the installed ComfyOmni version or an explicit source-tree sentinel."""

    try:
        return version(_DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return _UNINSTALLED_VERSION


__all__ = ["distribution_version"]

