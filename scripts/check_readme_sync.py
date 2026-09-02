"""Validate the structural contract shared by the English and Chinese READMEs."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
READMES = (ROOT / "README.md", ROOT / "README.zh-CN.md")
SECTION_RE = re.compile(r"<!-- README_SYNC: ([a-z0-9_-]+) -->")
MILESTONE_RE = re.compile(r"^\| (M\d+) \|", re.MULTILINE)
REQUIRED_SECTIONS = (
    "overview",
    "naming",
    "status",
    "goals",
    "architecture",
    "milestones",
    "layout",
    "development",
    "contributing",
    "license",
)
REQUIRED_MILESTONES = tuple(f"M{index}" for index in range(8))
REQUIRED_SHARED_LINKS = (
    "docs/post-merge-refactoring-plan.md",
    "docs/testing/model-validation-baseline.md",
)


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read {path.relative_to(ROOT)}: {exc}") from exc


def main() -> int:
    errors: list[str] = []
    documents: dict[Path, str] = {}

    for path in READMES:
        try:
            documents[path] = _read(path)
        except ValueError as exc:
            errors.append(str(exc))

    for path, text in documents.items():
        relative = path.relative_to(ROOT)
        sections = tuple(SECTION_RE.findall(text))
        milestones = tuple(MILESTONE_RE.findall(text))
        if sections != REQUIRED_SECTIONS:
            errors.append(f"{relative}: README_SYNC keys must be {REQUIRED_SECTIONS}, observed {sections}")
        if milestones != REQUIRED_MILESTONES:
            errors.append(f"{relative}: milestone IDs must be {REQUIRED_MILESTONES}, observed {milestones}")
        for link in REQUIRED_SHARED_LINKS:
            if link not in text:
                errors.append(f"{relative}: missing shared public link {link}")

    if len(documents) == len(READMES):
        section_sets = [tuple(SECTION_RE.findall(text)) for text in documents.values()]
        milestone_sets = [tuple(MILESTONE_RE.findall(text)) for text in documents.values()]
        if len(set(section_sets)) != 1:
            errors.append("README_SYNC section order differs between README files")
        if len(set(milestone_sets)) != 1:
            errors.append("milestone IDs differ between README files")

    if errors:
        for error in errors:
            print(f"README sync: FAIL: {error}", file=sys.stderr)
        return 1

    print("README sync: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
