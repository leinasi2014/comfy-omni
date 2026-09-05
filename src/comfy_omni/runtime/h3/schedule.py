# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: h3-forge contributors
# Derived from h3-forge@e9cb011d00b028c149db3978de246c54f6e34acc; Apache-2.0.
# Source: src/h3_forge/h3/h3_schedule.py; blob 961a03c4df3a3e9f3ce5b209bb48561320fe17a9.
"""Exact MiniMax H3 dual-sigma plans for cache-bound inference.

The pinned vLLM-Omni API names its argument ``num_inference_steps`` even
though the implementation consumes that many sigma *points* and therefore
performs one fewer DiT evaluations.  This module exposes the unambiguous
``denoise_steps`` contract used by h3-forge and derives ``steps + 1`` points.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

H3_SCHEDULE_SCHEMA = "h3-comfy/minimax-h3-dual-sigma/v1"
H3_SCHEDULER_ID = "simple-euler-dual-shift"
DEFAULT_VIDEO_SHIFT = 12.0
DEFAULT_AUDIO_SHIFT = 3.0
DEFAULT_IMAGE_CONDITION_TIMESTEP = 0.999
DEFAULT_AUDIO_CONDITION_TIMESTEP = 1.0

MODE_FIELDS: dict[str, tuple[str, ...]] = {
    "t2va": ("video", "audio"),
    "fl2va": ("video", "audio", "image"),
    "ref2va-image": ("video", "audio", "image"),
    "ref2va-audio": ("video", "audio", "audio_ref"),
    "ref2va-mixed": ("video", "audio", "image", "audio_ref"),
}
DEFAULT_MODE_UNION = tuple(MODE_FIELDS)


class H3ScheduleContractError(ValueError):
    """An API schedule cannot be represented by the frozen H3 contract."""


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def _float32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _require_finite_positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise H3ScheduleContractError(f"{name} must be a finite number")
    value = float(value)
    if value <= 0:
        raise H3ScheduleContractError(f"{name} must be positive")
    return value


def _require_unit_interval(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise H3ScheduleContractError(f"{name} must be a finite number")
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise H3ScheduleContractError(f"{name} must be in [0, 1]")
    return value


@dataclass(frozen=True)
class H3TimestepPlan:
    values: tuple[float, ...]
    bits: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"values": list(self.values), "float32_bits": [f"0x{value:08x}" for value in self.bits]}


@dataclass(frozen=True)
class H3ScheduleContract:
    schema: str
    scheduler_id: str
    denoise_steps: int
    api_sigma_points: int
    video_shift: float
    audio_shift: float
    image_condition_timestep: float
    audio_condition_timestep: float
    modes: tuple[str, ...]
    plans: tuple[H3TimestepPlan, ...]
    mode_plan_indices: tuple[tuple[str, tuple[int, ...]], ...]
    contract_sha256: str

    def unsigned_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("contract_sha256", None)
        value["plans"] = [plan.to_dict() for plan in self.plans]
        value["mode_plan_indices"] = {mode: list(indices) for mode, indices in self.mode_plan_indices}
        return value

    def to_dict(self) -> dict[str, Any]:
        value = self.unsigned_dict()
        value["contract_sha256"] = self.contract_sha256
        return value


def build_h3_schedule_contract(
    *,
    denoise_steps: int,
    video_shift: float = DEFAULT_VIDEO_SHIFT,
    audio_shift: float = DEFAULT_AUDIO_SHIFT,
    image_condition_timestep: float = DEFAULT_IMAGE_CONDITION_TIMESTEP,
    audio_condition_timestep: float = DEFAULT_AUDIO_CONDITION_TIMESTEP,
    modes: Iterable[str] = DEFAULT_MODE_UNION,
) -> H3ScheduleContract:
    """Build the exact plan union consumed by the pinned official denoise loop.

    Plans are deduplicated only as complete sorted-unique FP32 vectors.  Scalar
    timestep deduplication is intentionally forbidden: CUDA GEMM output can
    differ when the batch dimension changes.
    """

    if not isinstance(denoise_steps, int) or isinstance(denoise_steps, bool) or denoise_steps < 1:
        raise H3ScheduleContractError("denoise_steps must be a positive integer")
    if denoise_steps > 256:
        raise H3ScheduleContractError("denoise_steps exceeds the bounded maximum of 256")
    video_shift = _require_finite_positive("video_shift", video_shift)
    audio_shift = _require_finite_positive("audio_shift", audio_shift)
    image_condition_timestep = _require_unit_interval("image_condition_timestep", image_condition_timestep)
    audio_condition_timestep = _require_unit_interval("audio_condition_timestep", audio_condition_timestep)
    requested_modes = tuple(modes)
    if not requested_modes or len(set(requested_modes)) != len(requested_modes):
        raise H3ScheduleContractError("modes must be a non-empty unique sequence")
    unknown = sorted(set(requested_modes) - set(MODE_FIELDS))
    if unknown:
        raise H3ScheduleContractError(f"unsupported H3 cache mode: {unknown[0]!r}")

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise H3ScheduleContractError("torch is required to reproduce official float32 schedules") from exc

    base = torch.linspace(1.0, 0.0, denoise_steps + 1, dtype=torch.float32, device="cpu")

    def shifted(shift: float):
        value = float(shift) * base / (1.0 + (float(shift) - 1.0) * base)
        value, _ = torch.unique_consecutive(value, return_counts=True)
        if value.shape[0] != denoise_steps + 1:
            raise H3ScheduleContractError("shifted schedule collapsed duplicate sigma points")
        if not bool(torch.isfinite(value).all()):
            raise H3ScheduleContractError("shifted schedule contains a non-finite sigma")
        if bool((value < 0.0).any()) or bool((value > 1.0).any()):
            raise H3ScheduleContractError("shifted schedule leaves the [0, 1] sigma range")
        if float(value[0]) != 1.0 or float(value[-1]) != 0.0:
            raise H3ScheduleContractError("shifted schedule endpoints must be exactly 1 and 0")
        if not bool((value[:-1] > value[1:]).all()):
            raise H3ScheduleContractError("shifted schedule must be strictly decreasing")
        return value

    video_sigmas = shifted(video_shift)
    audio_sigmas = shifted(audio_shift)
    plan_indices: dict[tuple[int, ...], int] = {}
    plans: list[H3TimestepPlan] = []
    per_mode: list[tuple[str, tuple[int, ...]]] = []

    for mode in requested_modes:
        indices: list[int] = []
        fields = MODE_FIELDS[mode]
        for step in range(denoise_steps):
            video_t = torch.tensor(1.0, dtype=torch.float32) - video_sigmas[step]
            audio_t = torch.tensor(1.0, dtype=torch.float32) - audio_sigmas[step]
            candidates = {
                "video": video_t,
                "audio": audio_t,
                "image": torch.maximum(
                    video_t,
                    torch.tensor(image_condition_timestep, dtype=torch.float32),
                ),
                "audio_ref": torch.maximum(
                    audio_t,
                    torch.tensor(audio_condition_timestep, dtype=torch.float32),
                ),
            }
            plan_tensor = torch.stack([candidates[field] for field in fields]).unique(sorted=True)
            if not bool(torch.isfinite(plan_tensor).all()):
                raise H3ScheduleContractError("timestep plan contains a non-finite value")
            if bool((plan_tensor < 0.0).any()) or bool((plan_tensor > 1.0).any()):
                raise H3ScheduleContractError("timestep plan leaves the [0, 1] range")
            values = tuple(float(value) for value in plan_tensor.tolist())
            bits = tuple(_float32_bits(value) for value in values)
            index = plan_indices.get(bits)
            if index is None:
                index = len(plans)
                plan_indices[bits] = index
                plans.append(H3TimestepPlan(values=values, bits=bits))
            indices.append(index)
        per_mode.append((mode, tuple(indices)))

    unsigned = {
        "schema": H3_SCHEDULE_SCHEMA,
        "scheduler_id": H3_SCHEDULER_ID,
        "denoise_steps": denoise_steps,
        "api_sigma_points": denoise_steps + 1,
        "video_shift": video_shift,
        "audio_shift": audio_shift,
        "image_condition_timestep": image_condition_timestep,
        "audio_condition_timestep": audio_condition_timestep,
        "modes": list(requested_modes),
        "plans": [plan.to_dict() for plan in plans],
        "mode_plan_indices": {mode: list(indices) for mode, indices in per_mode},
    }
    digest = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    return H3ScheduleContract(
        schema=H3_SCHEDULE_SCHEMA,
        scheduler_id=H3_SCHEDULER_ID,
        denoise_steps=denoise_steps,
        api_sigma_points=denoise_steps + 1,
        video_shift=video_shift,
        audio_shift=audio_shift,
        image_condition_timestep=image_condition_timestep,
        audio_condition_timestep=audio_condition_timestep,
        modes=requested_modes,
        plans=tuple(plans),
        mode_plan_indices=tuple(per_mode),
        contract_sha256=digest,
    )


def validate_h3_schedule_contract(schedule: H3ScheduleContract) -> None:
    """Reject hand-built or mutated schedules before they reach a cache compiler."""

    if not isinstance(schedule, H3ScheduleContract):
        raise H3ScheduleContractError("schedule must be an H3ScheduleContract")
    rebuilt = build_h3_schedule_contract(
        denoise_steps=schedule.denoise_steps,
        video_shift=schedule.video_shift,
        audio_shift=schedule.audio_shift,
        image_condition_timestep=schedule.image_condition_timestep,
        audio_condition_timestep=schedule.audio_condition_timestep,
        modes=schedule.modes,
    )
    if schedule != rebuilt:
        raise H3ScheduleContractError("schedule contract canonical self-check failed")


def h3_schedule_contract_from_dict(value: Mapping[str, Any]) -> H3ScheduleContract:
    """Rebuild and strictly validate a serialized schedule contract."""

    if not isinstance(value, Mapping):
        raise H3ScheduleContractError("serialized schedule contract must be an object")
    required = {
        "schema",
        "scheduler_id",
        "denoise_steps",
        "api_sigma_points",
        "video_shift",
        "audio_shift",
        "image_condition_timestep",
        "audio_condition_timestep",
        "modes",
        "plans",
        "mode_plan_indices",
        "contract_sha256",
    }
    if set(value) != required:
        raise H3ScheduleContractError("serialized schedule contract fields are not exact")
    modes = value.get("modes")
    if not isinstance(modes, list) or not all(isinstance(mode, str) for mode in modes):
        raise H3ScheduleContractError("serialized schedule modes must be a string list")
    rebuilt = build_h3_schedule_contract(
        denoise_steps=value.get("denoise_steps"),
        video_shift=value.get("video_shift"),
        audio_shift=value.get("audio_shift"),
        image_condition_timestep=value.get("image_condition_timestep"),
        audio_condition_timestep=value.get("audio_condition_timestep"),
        modes=modes,
    )
    if _canonical_json(rebuilt.to_dict()) != _canonical_json(dict(value)):
        raise H3ScheduleContractError("serialized schedule contract is not canonical")
    return rebuilt


__all__ = [
    "DEFAULT_MODE_UNION",
    "H3ScheduleContract",
    "H3ScheduleContractError",
    "H3TimestepPlan",
    "MODE_FIELDS",
    "build_h3_schedule_contract",
    "h3_schedule_contract_from_dict",
    "validate_h3_schedule_contract",
]
