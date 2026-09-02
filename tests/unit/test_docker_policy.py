"""Contract tests for the repository Docker-first policy checker."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
POLICY: dict[str, Any] = runpy.run_path(str(ROOT / "scripts" / "check_docker_policy.py"))


def test_policy_parser_inspects_inline_and_multiline_run_commands() -> None:
    workflow = """
steps:
  - run: ./scripts/docker.sh quality 3.13
  - run: |
      python -m pytest
      echo complete
"""

    commands = POLICY["_run_commands"](workflow)

    assert commands == (
        "./scripts/docker.sh quality 3.13",
        "python -m pytest",
        "echo complete",
    )


def test_policy_rejects_host_python_but_allows_docker_wrapper() -> None:
    pattern = POLICY["FORBIDDEN_COMMAND"]

    assert pattern.search("python -m pytest") is not None
    assert pattern.search("./scripts/docker.sh quality 3.13") is None


def test_repository_satisfies_docker_first_policy() -> None:
    errors = (
        POLICY["_validate_policy_markers"]()
        + POLICY["_validate_workflows"]()
        + POLICY["_validate_build_contract"]()
    )

    assert errors == []
