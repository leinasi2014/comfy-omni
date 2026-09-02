"""Smoke-test the installed wheel without importing the source tree."""

from __future__ import annotations

import importlib.metadata
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def _run_cli(*arguments: str) -> None:
    executable = Path(sys.executable).with_name("comfy-omni")
    subprocess.run([str(executable), *arguments], check=True, timeout=30)


def main() -> int:
    import comfy_omni  # noqa: F401
    import comfy_omni.plugin  # noqa: F401
    from comfy_omni.artifacts.build_identity import installed_tool_identity

    entry_points = importlib.metadata.entry_points(group="vllm_omni.general_plugins")
    assert any(entry_point.name == "comfy_omni" for entry_point in entry_points)
    identity = installed_tool_identity()
    assert identity.distribution == "comfy-omni"
    assert len(identity.source_commit) == 40
    assert len(identity.wheel_sha256) == 64
    assert "torch" not in sys.modules
    assert "vllm" not in sys.modules

    _run_cli("--help")
    _run_cli("--version")
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "empty.safetensors"
        header = json.dumps({}).encode("utf-8")
        checkpoint.write_bytes(struct.pack("<Q", len(header)) + header)
        _run_cli("inspect", str(checkpoint), "--json")
    print("installed-wheel: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
