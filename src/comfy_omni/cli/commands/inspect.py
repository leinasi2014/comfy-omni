"""CLI adapter for the header-only checkpoint inspection use case."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from comfy_omni.application.inspection import inspect_checkpoint_paths


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Declare the legacy-compatible inspect arguments."""

    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")


def run(args: argparse.Namespace) -> int:
    """Run inspection and render either stable JSON or concise text."""

    try:
        inspections = inspect_checkpoint_paths(args.paths)
    except (OSError, ValueError) as exc:
        print(f"comfy-omni inspect: error: {exc}", file=sys.stderr)
        return 2
    payload = [inspection.to_dict() for inspection in inspections]
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for inspection in payload:
            quantization = ", ".join(inspection["quantization"])
            print(
                f"{inspection['path']}: {inspection['component']} "
                f"tensors={inspection['tensor_count']} quant={quantization}"
            )
    return 0


__all__ = ["configure_parser", "run"]
