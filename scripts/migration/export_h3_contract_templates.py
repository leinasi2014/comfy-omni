"""Export the audited h3-forge contract templates as a ComfyOmni package resource.

This one-way migration helper is intentionally not imported by the package. It must be run against
the exact source revision recorded in ``docs/migration/contract-workflows-e9cb011.md``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SOURCE_COMMIT = "e9cb011d00b028c149db3978de246c54f6e34acc"
SOURCE_BLOB = "443a5cc9ca58891c3852079c8589fbe2f5af6484"
RESOURCE_SCHEMA = "comfy_omni.contract_templates/v1"


def _git(source: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def _validate_source(source: Path) -> None:
    observed_commit = _git(source, "rev-parse", "HEAD")
    observed_blob = _git(source, "rev-parse", "HEAD:src/h3_forge/h3/contracts/templates.py")
    if observed_commit != SOURCE_COMMIT or observed_blob != SOURCE_BLOB:
        raise RuntimeError(
            f"template export source is not the audited revision: commit={observed_commit}, blob={observed_blob}"
        )


def _template_document(template: Any, digest: str) -> dict[str, Any]:
    return {
        "template_name": template.template_name,
        "template_version": template.template_version,
        "component": template.component,
        "layer_topology": list(template.layer_topology),
        "layer_prefix_template": template.layer_prefix_template,
        "convrot_suffixes": {
            suffix: {"shape": list(shape), "group_size": group_size}
            for suffix, (shape, group_size) in sorted(template.convrot_suffixes.items())
        },
        "scale_shape_census": {
            f"{rows}x{columns}": count for (rows, columns), count in sorted(template.scale_shape_census.items())
        },
        "curve_adaln_tensors": sorted(template.curve_adaln_tensors),
        "text_encoder_direct_connection": template.text_encoder_direct_connection,
        "non_quantized_inventory": {
            name: {"dtype": dtype, "shape": list(shape)}
            for name, (dtype, shape) in sorted(template.non_quantized_inventory.items())
        },
        "legacy_template_sha256": digest,
    }


def _build_document(source: Path) -> dict[str, Any]:
    sys.path.insert(0, str(source / "src"))
    from h3_forge.h3.contracts.templates import ARCHITECTURE_TEMPLATES, template_digest

    return {
        "schema": RESOURCE_SCHEMA,
        "source": {
            "repository": "h3-forge",
            "commit": SOURCE_COMMIT,
            "templates_blob": SOURCE_BLOB,
            "license": "Apache-2.0",
        },
        "templates": {
            name: _template_document(template, template_digest(template))
            for name, template in sorted(ARCHITECTURE_TEMPLATES.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    _validate_source(source)
    document = _build_document(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    args.output.write_bytes(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
