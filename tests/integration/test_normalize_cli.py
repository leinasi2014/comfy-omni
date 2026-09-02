"""Vertical CLI tests for explicit text-encoder normalization."""

from __future__ import annotations

import json
from pathlib import Path

from pytest import CaptureFixture, MonkeyPatch

from comfy_omni.artifacts.normalization import NormalizationPublication
from comfy_omni.cli import main
from comfy_omni.cli.commands import normalize
from comfy_omni.domain.normalization import (
    ArtifactIdentity,
    NormalizationError,
    NormalizationReceipt,
    ToolIdentity,
)


def _publication(tmp_path: Path) -> NormalizationPublication:
    receipt = NormalizationReceipt(
        profile_id="test-profile",
        profile_version=1,
        source=ArtifactIdentity(2, "1" * 64),
        removed_suffix=ArtifactIdentity(1, "2" * 64),
        derived=ArtifactIdentity(1, "3" * 64),
        strict_reread_tensor_count=7,
        tool=ToolIdentity("comfy-omni", "0.2.0a1", "4" * 40, "5" * 64),
    )
    return NormalizationPublication(
        tmp_path / "derived.safetensors",
        tmp_path / "derived.safetensors.normalization.json",
        receipt,
    )


def test_normalize_text_encoder_json_renders_committed_receipt(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    publication = _publication(tmp_path)
    observed: list[tuple[Path, Path]] = []

    def run_use_case(source: Path, destination: Path) -> NormalizationPublication:
        observed.append((source, destination))
        return publication

    monkeypatch.setattr(normalize, "normalize_pinned_text_encoder", run_use_case)

    assert main(["normalize", "text-encoder", "source.safetensors", "derived.safetensors", "--json"]) == 0

    assert observed == [(Path("source.safetensors"), Path("derived.safetensors"))]
    assert json.loads(capsys.readouterr().out) == publication.receipt.to_dict()


def test_normalize_failure_is_concise_and_has_no_traceback(
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    def fail(_source: Path, _destination: Path) -> NormalizationPublication:
        raise NormalizationError("normalization-source-digest-mismatch", "unexpected source")

    monkeypatch.setattr(normalize, "normalize_pinned_text_encoder", fail)

    assert main(["normalize", "text-encoder", "source.safetensors", "derived.safetensors"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "normalization-source-digest-mismatch" in captured.err
    assert "Traceback" not in captured.err
