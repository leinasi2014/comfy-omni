"""Command-line presentation boundary for ComfyOmni.

The walking skeleton exposes project identity only. Business commands are added as independently
reviewed application slices; this module must not become an implementation layer.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from comfy_omni import __version__


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="comfy-omni",
        description="Bring Comfy checkpoints to native Omni runtimes.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the current identity-only CLI surface."""

    _parser().parse_args(argv)
    return 0


__all__ = ["main"]
