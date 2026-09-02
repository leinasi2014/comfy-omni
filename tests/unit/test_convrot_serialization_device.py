from __future__ import annotations

from types import SimpleNamespace
from typing import Any

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


def test_convrot_serialization_uses_a_contiguous_buffer_copy() -> None:
    calls: list[tuple[str, Any]] = []
    uint8 = object()

    class Array:
        def tobytes(self, *, order: str) -> bytes:
            calls.append(("tobytes", order))
            return b"\x01\x02\x03\x04"

    class ByteView:
        def numpy(self) -> Array:
            calls.append(("numpy", None))
            return Array()

    class Tensor:
        def view(self, dtype: object) -> ByteView:
            calls.append(("view", dtype))
            return ByteView()

    torch = SimpleNamespace(uint8=uint8)

    assert serialization._tensor_bytes(torch, Tensor()) == b"\x01\x02\x03\x04"
    assert calls == [("view", uint8), ("numpy", None), ("tobytes", "C")]
