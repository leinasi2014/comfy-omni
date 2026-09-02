"""Unit tests for installed-wheel provenance resolution."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from comfy_omni.artifacts import build_identity
from comfy_omni.domain.normalization import NormalizationError


class _Distribution:
    version = "0.2.0a1"

    def __init__(self, direct_url: object) -> None:
        self.direct_url = direct_url

    def read_text(self, filename: str) -> str | None:
        assert filename == "direct_url.json"
        return None if self.direct_url is None else json.dumps(self.direct_url)


def test_installed_tool_identity_binds_clean_commit_and_wheel_sha256(monkeypatch: pytest.MonkeyPatch) -> None:
    build = SimpleNamespace(SOURCE_COMMIT="a" * 40, SOURCE_DIRTY=False)
    direct_url = {
        "url": "file:///tmp/comfy_omni-0.2.0a1-py3-none-any.whl",
        "archive_info": {"hashes": {"sha256": "B" * 64}},
    }
    monkeypatch.setattr(build_identity.importlib, "import_module", lambda _: build)
    monkeypatch.setattr(build_identity, "distribution", lambda _: _Distribution(direct_url))

    identity = build_identity.installed_tool_identity()

    assert identity.source_commit == "a" * 40
    assert identity.wheel_sha256 == "b" * 64
    assert identity.version == "0.2.0a1"


def test_installed_tool_identity_rejects_dirty_build(monkeypatch: pytest.MonkeyPatch) -> None:
    build = SimpleNamespace(SOURCE_COMMIT="a" * 40, SOURCE_DIRTY=True)
    monkeypatch.setattr(build_identity.importlib, "import_module", lambda _: build)

    with pytest.raises(NormalizationError, match="normalization-dirty-build"):
        build_identity.installed_tool_identity()


def test_installed_tool_identity_rejects_install_without_wheel_archive_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build = SimpleNamespace(SOURCE_COMMIT="a" * 40, SOURCE_DIRTY=False)
    monkeypatch.setattr(build_identity.importlib, "import_module", lambda _: build)
    monkeypatch.setattr(build_identity, "distribution", lambda _: _Distribution(None))

    with pytest.raises(NormalizationError, match="normalization-unbound-wheel"):
        build_identity.installed_tool_identity()


def test_installed_tool_identity_rejects_sdist_archive_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    build = SimpleNamespace(SOURCE_COMMIT="a" * 40, SOURCE_DIRTY=False)
    direct_url = {
        "url": "file:///tmp/comfy_omni-0.2.0a1.tar.gz",
        "archive_info": {"hashes": {"sha256": "b" * 64}},
    }
    monkeypatch.setattr(build_identity.importlib, "import_module", lambda _: build)
    monkeypatch.setattr(build_identity, "distribution", lambda _: _Distribution(direct_url))

    with pytest.raises(NormalizationError, match="normalization-unbound-wheel"):
        build_identity.installed_tool_identity()
