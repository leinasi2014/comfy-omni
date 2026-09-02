"""Fail closed when repository execution escapes the Docker-first boundary."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY_MARKER = "DOCKER_FIRST_POLICY: v1"
POLICY_FILES = (
    "AGENTS.md",
    "CONTRIBUTING.md",
    "README.md",
    "README.zh-CN.md",
    "docs/development/docker-first.md",
)
WORKFLOWS = (
    ".github/workflows/documentation.yml",
    ".github/workflows/quality.yml",
)
FORBIDDEN_COMMAND = re.compile(r"(?:^|[;&|]\s*)(?:python(?:3(?:\.\d+)?)?|pip|uv|pytest|ruff|twine|conda)\b")


def _run_commands(text: str) -> tuple[str, ...]:
    commands: list[str] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        match = re.match(r"^(\s*)(?:-\s+)?run:\s*(.*)$", line)
        if match is None:
            index += 1
            continue
        indent = len(match.group(1))
        inline = match.group(2).strip()
        if inline not in {"", "|", ">", "|-", ">-"}:
            commands.append(inline)
        index += 1
        while index < len(lines):
            nested = lines[index]
            if nested.strip() and len(nested) - len(nested.lstrip()) <= indent:
                break
            if nested.strip():
                commands.append(nested.strip())
            index += 1
    return tuple(commands)


def _validate_policy_markers() -> list[str]:
    errors: list[str] = []
    for relative in POLICY_FILES:
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"missing Docker policy file: {relative}")
        elif POLICY_MARKER not in path.read_text(encoding="utf-8"):
            errors.append(f"missing {POLICY_MARKER!r} in {relative}")
    return errors


def _validate_workflows() -> list[str]:
    errors: list[str] = []
    for relative in WORKFLOWS:
        text = (ROOT / relative).read_text(encoding="utf-8")
        if "actions/setup-python" in text:
            errors.append(f"host Python setup is forbidden in {relative}")
        for command in _run_commands(text):
            if FORBIDDEN_COMMAND.search(command):
                errors.append(f"host project command is forbidden in {relative}: {command}")
    required_calls = {
        ".github/workflows/documentation.yml": ("./scripts/docker.sh docs",),
        ".github/workflows/quality.yml": ("./scripts/docker.sh quality", "./scripts/docker.sh package"),
    }
    for relative, calls in required_calls.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        for call in calls:
            if call not in text:
                errors.append(f"missing Docker workflow call in {relative}: {call}")
    return errors


def _validate_build_contract() -> list[str]:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    errors = [
        f"Dockerfile is missing target {target}"
        for target in ("documentation", "quality", "package-check", "runtime")
        if f" AS {target}" not in dockerfile
    ]
    for relative in ("scripts/docker.sh", "scripts/docker.ps1"):
        if not (ROOT / relative).is_file():
            errors.append(f"missing Docker entry point: {relative}")
    if "ARG PYTHON_REGISTRY=docker.io/library" not in dockerfile:
        errors.append("Dockerfile must default to the official Python registry namespace")
    return errors


def main() -> int:
    errors = _validate_policy_markers() + _validate_workflows() + _validate_build_contract()
    if errors:
        raise SystemExit("\n".join(f"docker-policy: {error}" for error in errors))
    print("docker-policy: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
