"""CLI adapter for explicit digest-pinned offline normalization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from comfy_omni.application.normalization import normalize_pinned_text_encoder


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Declare normalization subcommands without embedding conversion policy."""

    targets = parser.add_subparsers(dest="normalization_target", required=True)
    text_encoder = targets.add_parser(
        "text-encoder",
        help="normalize the digest-pinned ModelScope Qwen3-VL H3 text encoder",
    )
    text_encoder.add_argument("source", type=Path)
    text_encoder.add_argument("destination", type=Path)
    text_encoder.add_argument("--json", action="store_true", help="emit the normalization receipt as JSON")


def run(args: argparse.Namespace) -> int:
    """Run the selected normalization and render its committed receipt."""

    try:
        publication = normalize_pinned_text_encoder(args.source, args.destination)
    except (OSError, ValueError) as exc:
        print(f"comfy-omni normalize: error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(publication.receipt.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"normalized: {publication.artifact_path}")
        print(f"receipt: {publication.receipt_path}")
        print(f"sha256: {publication.receipt.derived.sha256}")
    return 0


__all__ = ["configure_parser", "run"]
