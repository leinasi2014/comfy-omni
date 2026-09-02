"""Stable errors for optional conversion numerical backends."""

from __future__ import annotations

from comfy_omni.contracts.models import ContractError


class ConvRotNumericsError(ContractError):
    """ConvRot values or an execution backend violated the numerical contract."""


__all__ = ["ConvRotNumericsError"]
