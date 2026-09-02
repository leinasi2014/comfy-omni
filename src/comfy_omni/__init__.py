"""Bring Comfy checkpoints to native Omni runtimes.

Only explicitly documented names are public. Importing this package is intentionally lightweight:
it does not import conversion backends, HTTP frameworks, host runtimes, or model code.
"""

from __future__ import annotations

from ._version import distribution_version

__version__ = distribution_version()

__all__ = ["__version__"]
