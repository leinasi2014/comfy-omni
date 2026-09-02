"""Unit tests for bounded, no-overwrite artifact normalization."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from comfy_omni.artifacts import normalization
from comfy_omni.artifacts.normalization import normalize_digest_pinned_prefix
from comfy_omni.domain.normalization import (
    ArtifactIdentity,
    NormalizationError,
    NormalizationProfile,
    ToolIdentity,
)


def _identity(content: bytes) -> ArtifactIdentity:
    return ArtifactIdentity(len(content), hashlib.sha256(content).hexdigest())


def _strict_safetensors() -> bytes:
    header = json.dumps({}, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(header)) + header


def _profile(
    derived: bytes,
    suffix: bytes,
    *,
    source_identity: ArtifactIdentity | None = None,
    derived_identity: ArtifactIdentity | None = None,
    suffix_identity: ArtifactIdentity | None = None,
) -> NormalizationProfile:
    return NormalizationProfile(
        profile_id="test-exact-prefix",
        profile_version=1,
        source=source_identity or _identity(derived + suffix),
        removed_suffix=suffix_identity or _identity(suffix),
        derived=derived_identity or _identity(derived),
    )


def _tool() -> ToolIdentity:
    return ToolIdentity("comfy-omni", "0.2.0a1", "1" * 40, "2" * 64)


def _assert_no_staging_files(directory: Path) -> None:
    assert not tuple(directory.glob(".*.tmp"))


def test_normalization_preserves_source_and_publishes_strict_artifact_and_receipt(tmp_path: Path) -> None:
    derived = _strict_safetensors()
    suffix = b"source-transport-marker"
    source = tmp_path / "source.safetensors"
    destination = tmp_path / "derived.safetensors"
    source.write_bytes(derived + suffix)
    source_before = source.read_bytes()

    publication = normalize_digest_pinned_prefix(source, destination, _profile(derived, suffix), _tool())

    assert source.read_bytes() == source_before
    assert destination.read_bytes() == derived
    assert publication.artifact_path == destination.resolve()
    assert publication.receipt_path == tmp_path / "derived.safetensors.normalization.json"
    receipt_payload = json.loads(publication.receipt_path.read_text(encoding="utf-8"))
    assert receipt_payload == publication.receipt.to_dict()
    assert receipt_payload["source"] == _identity(derived + suffix).to_dict()
    assert receipt_payload["removed_suffix"] == _identity(suffix).to_dict()
    assert receipt_payload["derived"] == _identity(derived).to_dict()
    assert receipt_payload["strict_reread"] == {"passed": True, "tensor_count": 0}
    assert receipt_payload["tool"] == _tool().to_dict()
    _assert_no_staging_files(tmp_path)


@pytest.mark.parametrize(
    ("profile_factory", "reason_code"),
    [
        (
            lambda derived, suffix: _profile(
                derived,
                suffix,
                source_identity=ArtifactIdentity(len(derived + suffix), "3" * 64),
            ),
            "normalization-source-digest-mismatch",
        ),
        (
            lambda derived, suffix: _profile(
                derived,
                suffix,
                suffix_identity=ArtifactIdentity(len(suffix), "4" * 64),
            ),
            "normalization-suffix-digest-mismatch",
        ),
        (
            lambda derived, suffix: _profile(
                derived,
                suffix,
                derived_identity=ArtifactIdentity(len(derived), "5" * 64),
            ),
            "normalization-derived-digest-mismatch",
        ),
    ],
)
def test_normalization_rejects_digest_mismatches_without_publication(
    tmp_path: Path,
    profile_factory: object,
    reason_code: str,
) -> None:
    derived = _strict_safetensors()
    suffix = b"suffix"
    source = tmp_path / "source.safetensors"
    destination = tmp_path / "derived.safetensors"
    source.write_bytes(derived + suffix)

    with pytest.raises(NormalizationError, match=reason_code):
        normalize_digest_pinned_prefix(source, destination, profile_factory(derived, suffix), _tool())  # type: ignore[operator]

    assert source.read_bytes() == derived + suffix
    assert not destination.exists()
    assert not (tmp_path / "derived.safetensors.normalization.json").exists()
    _assert_no_staging_files(tmp_path)


def test_normalization_rejects_source_size_before_publication(tmp_path: Path) -> None:
    derived = _strict_safetensors()
    suffix = b"suffix"
    source = tmp_path / "source.safetensors"
    destination = tmp_path / "derived.safetensors"
    source.write_bytes(derived + suffix + b"extra")

    with pytest.raises(NormalizationError, match="normalization-source-size-mismatch"):
        normalize_digest_pinned_prefix(source, destination, _profile(derived, suffix), _tool())

    assert not destination.exists()
    _assert_no_staging_files(tmp_path)


def test_normalization_rejects_linked_source_path_before_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    derived = _strict_safetensors()
    suffix = b"suffix"
    source = tmp_path / "source.safetensors"
    destination = tmp_path / "derived.safetensors"
    source.write_bytes(derived + suffix)
    monkeypatch.setattr(normalization, "_is_link_or_reparse", lambda path: path == source.absolute())

    with pytest.raises(NormalizationError, match="normalization-source-link-forbidden"):
        normalize_digest_pinned_prefix(source, destination, _profile(derived, suffix), _tool())

    assert not destination.exists()
    _assert_no_staging_files(tmp_path)


@pytest.mark.parametrize(
    ("preexisting", "reason_code"),
    [
        ("artifact", "normalization-destination-exists"),
        ("receipt", "normalization-receipt-exists"),
    ],
)
def test_normalization_never_overwrites_existing_publication(
    tmp_path: Path,
    preexisting: str,
    reason_code: str,
) -> None:
    derived = _strict_safetensors()
    suffix = b"suffix"
    source = tmp_path / "source.safetensors"
    destination = tmp_path / "derived.safetensors"
    receipt = tmp_path / "derived.safetensors.normalization.json"
    source.write_bytes(derived + suffix)
    occupied = destination if preexisting == "artifact" else receipt
    occupied.write_bytes(b"owned-by-someone-else")

    with pytest.raises(NormalizationError, match=reason_code):
        normalize_digest_pinned_prefix(source, destination, _profile(derived, suffix), _tool())

    assert occupied.read_bytes() == b"owned-by-someone-else"
    _assert_no_staging_files(tmp_path)


def test_normalization_rolls_back_artifact_if_receipt_cannot_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    derived = _strict_safetensors()
    suffix = b"suffix"
    source = tmp_path / "source.safetensors"
    destination = tmp_path / "derived.safetensors"
    source.write_bytes(derived + suffix)
    real_link = normalization.os.link
    calls = 0

    def fail_second_link(source_path: Path, destination_path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated receipt publication interruption")
        real_link(source_path, destination_path)

    monkeypatch.setattr(normalization.os, "link", fail_second_link)

    with pytest.raises(NormalizationError, match="normalization-publication-failed"):
        normalize_digest_pinned_prefix(source, destination, _profile(derived, suffix), _tool())

    assert not destination.exists()
    assert not (tmp_path / "derived.safetensors.normalization.json").exists()
    assert source.read_bytes() == derived + suffix
    _assert_no_staging_files(tmp_path)


def test_normalization_requires_derived_artifact_to_pass_strict_reread(tmp_path: Path) -> None:
    derived = b"not-a-safetensors-file"
    suffix = b"suffix"
    source = tmp_path / "source.safetensors"
    destination = tmp_path / "derived.safetensors"
    source.write_bytes(derived + suffix)

    with pytest.raises(NormalizationError, match="normalization-strict-reread-failed"):
        normalize_digest_pinned_prefix(source, destination, _profile(derived, suffix), _tool())

    assert not destination.exists()
    _assert_no_staging_files(tmp_path)
