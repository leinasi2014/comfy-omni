from __future__ import annotations

from types import SimpleNamespace

import pytest

from comfy_omni.conversion.numerics import serialization
from comfy_omni.conversion.numerics.errors import ConvRotNumericsError


def _torch(*, cuda_available: bool) -> SimpleNamespace:
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda_available))


def test_convrot_serialization_defaults_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COMFY_OMNI_CONVROT_DEVICE", raising=False)

    assert serialization._conversion_device(_torch(cuda_available=True)) == "cpu"


def test_convrot_serialization_allows_explicit_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMFY_OMNI_CONVROT_DEVICE", "cuda")

    assert serialization._conversion_device(_torch(cuda_available=True)) == "cuda"


def test_convrot_serialization_rejects_unavailable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMFY_OMNI_CONVROT_DEVICE", "cuda")

    with pytest.raises(ConvRotNumericsError, match="CUDA is unavailable"):
        serialization._conversion_device(_torch(cuda_available=False))


def test_convrot_serialization_rejects_unknown_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMFY_OMNI_CONVROT_DEVICE", "auto")

    with pytest.raises(ConvRotNumericsError, match="must be cpu or cuda"):
        serialization._conversion_device(_torch(cuda_available=True))
