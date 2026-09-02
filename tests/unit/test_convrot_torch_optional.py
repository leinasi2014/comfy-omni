from __future__ import annotations

import pytest

from comfy_omni.conversion.numerics import torch_backend
from comfy_omni.conversion.numerics.errors import ConvRotNumericsError


def test_torch_backend_has_a_stable_error_when_container_omits_torch(monkeypatch) -> None:
    def missing(name: str) -> None:
        assert name == "torch"
        raise ModuleNotFoundError("no Torch in base wheel")

    monkeypatch.setattr(torch_backend.importlib, "import_module", missing)

    with pytest.raises(ConvRotNumericsError, match="conversion container"):
        torch_backend.regular_hadamard(4)
