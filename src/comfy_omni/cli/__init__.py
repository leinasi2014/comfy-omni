"""Command-line presentation boundary for ComfyOmni.

Commands are thin adapters over application use cases; this module owns only root parsing and
dispatch and must not become an implementation layer.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from comfy_omni import __version__
from comfy_omni.cli.commands import inspect, normalize


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
    subparsers = parser.add_subparsers(dest="command")
    inspect_parser = subparsers.add_parser("inspect", help="inspect safetensors headers without loading tensors")
    inspect.configure_parser(inspect_parser)
    normalize_parser = subparsers.add_parser("normalize", help="apply an explicit digest-pinned normalization")
    normalize.configure_parser(normalize_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse and dispatch the current CLI surface."""

    args = _parser().parse_args(argv)
    if args.command == "inspect":
        return inspect.run(args)
    if args.command == "normalize":
        return normalize.run(args)
    return 0


__all__ = ["main"]
