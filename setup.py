"""Setuptools command hooks for content-bound build provenance."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.sdist import sdist as _sdist

ROOT = Path(__file__).resolve().parent
SOURCE_IDENTITY_FILE = Path("src/comfy_omni/_source_identity.json")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def _validate_commit(value: str) -> str:
    commit = value.strip().lower()
    if COMMIT_PATTERN.fullmatch(commit) is None:
        raise RuntimeError("ComfyOmni build commit must be a complete lowercase 40-character Git SHA")
    return commit


def _environment_identity() -> tuple[str, bool] | None:
    value = os.environ.get("COMFY_OMNI_BUILD_COMMIT")
    if value is None:
        return None
    dirty_value = os.environ.get("COMFY_OMNI_BUILD_DIRTY", "0")
    if dirty_value not in {"0", "1"}:
        raise RuntimeError("COMFY_OMNI_BUILD_DIRTY must be 0 or 1")
    return _validate_commit(value), dirty_value == "1"


def _git_identity() -> tuple[str, bool] | None:
    if not (ROOT / ".git").exists():
        return None
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout
    return _validate_commit(commit), bool(status.strip())


def _sdist_identity() -> tuple[str, bool] | None:
    path = ROOT / SOURCE_IDENTITY_FILE
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {"source_commit", "source_dirty"} or type(payload["source_dirty"]) is not bool:
        raise RuntimeError("invalid ComfyOmni sdist source identity")
    return _validate_commit(payload["source_commit"]), payload["source_dirty"]


def _source_identity() -> tuple[str, bool]:
    identity = _environment_identity() or _git_identity() or _sdist_identity()
    if identity is None:
        raise RuntimeError("cannot prove ComfyOmni source commit for this build")
    return identity


class BuildPy(_build_py):
    """Generate an installed-only module binding the wheel to its source tree."""

    def _identity_target(self) -> Path:
        return Path(self.build_lib) / "comfy_omni" / "_build_identity.py"

    def run(self) -> None:
        super().run()
        commit, dirty = _source_identity()
        target = self._identity_target()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "\n".join(
                [
                    '"""Generated build provenance; do not edit."""',
                    "",
                    f'SOURCE_COMMIT = "{commit}"',
                    f"SOURCE_DIRTY = {dirty!r}",
                    "",
                ]
            ),
            encoding="utf-8",
            newline="\n",
        )

    def get_outputs(self, include_bytecode: int = 1) -> list[str]:
        outputs = list(super().get_outputs(include_bytecode=include_bytecode))
        outputs.append(str(self._identity_target()))
        return outputs


class Sdist(_sdist):
    """Carry the exact source identity into the release tree for wheel rebuilds."""

    def make_release_tree(self, base_dir: str, files: list[str]) -> None:
        commit, dirty = _source_identity()
        super().make_release_tree(base_dir, files)
        target = Path(base_dir) / SOURCE_IDENTITY_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps({"source_commit": commit, "source_dirty": dirty}, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )


setup(cmdclass={"build_py": BuildPy, "sdist": Sdist})
