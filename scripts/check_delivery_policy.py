from __future__ import annotations

import argparse
from pathlib import Path

MAX_AGENTS_LINES = 80


def _missing(text: str, required: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(item for item in required if item not in text)


def check_delivery_policy(repo_root: Path) -> tuple[str, ...]:
    errors: list[str] = []
    agents_path = repo_root / "AGENTS.md"
    contributing_path = repo_root / "CONTRIBUTING.md"
    delivery_path = repo_root / "docs" / "development" / "delivery.md"
    template_path = repo_root / ".github" / "pull_request_template.md"

    agents = agents_path.read_text(encoding="utf-8")
    agents_lines = len(agents.splitlines())
    if agents_lines > MAX_AGENTS_LINES:
        errors.append(f"AGENTS.md has {agents_lines} lines; maximum is {MAX_AGENTS_LINES}")
    for item in _missing(
        agents,
        (
            "docs/development/delivery.md",
            "docs/development/docker-first.md",
            "docs/post-merge-refactoring-plan.md",
            "docs/testing/model-validation-baseline.md",
            "CONTRIBUTING.md",
        ),
    ):
        errors.append(f"AGENTS.md does not link {item}")

    if not delivery_path.is_file():
        errors.append("docs/development/delivery.md is missing")
    else:
        delivery = delivery_path.read_text(encoding="utf-8")
        for item in _missing(
            delivery,
            (
                "GitHub Issues",
                "## Work in progress",
                "## Definition of Ready",
                "RED -> GREEN -> REFACTOR",
                "immutable digest",
                "## Definition of Done",
                "main push",
            ),
        ):
            errors.append(f"delivery policy is missing {item!r}")

    contributing = contributing_path.read_text(encoding="utf-8")
    if "codex/<type>-<topic>" not in contributing:
        errors.append("CONTRIBUTING.md does not define the Codex branch form")

    template = template_path.read_text(encoding="utf-8")
    for item in _missing(
        template,
        (
            "## Outcome and acceptance",
            "## Non-goals",
            "## Frozen contract",
            "## TDD evidence",
            "### RED",
            "### GREEN",
            "### REFACTOR",
            "## Verification",
            "## Migration and rollback",
        ),
    ):
        errors.append(f"pull request template is missing {item!r}")
    return tuple(errors)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the stable ComfyOmni delivery agreement")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    errors = check_delivery_policy(args.repo_root.resolve())
    if errors:
        for error in errors:
            print(f"delivery-policy: {error}")
        return 1
    print("delivery-policy: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
