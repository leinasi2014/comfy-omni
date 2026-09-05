# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: h3-forge contributors
# Derived from h3-forge@e9cb011d00b028c149db3978de246c54f6e34acc; Apache-2.0.
# Source: src/h3_forge/h3/api_contract.py; blob dda7c5806cd8c94d022b9c9bfb16a31d15f69b0b.
"""Strict request normalization for the schedule-bound MiniMax H3 profile."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .schedule import H3ScheduleContract, validate_h3_schedule_contract

API_NAMESPACE = "h3_forge"
API_VERSION = 1
INITIAL_SCHEDULE_PROFILE = "turbo-v4-s12-a3"
SUPPORTED_TASKS = frozenset({"t2va", "fl2va", "ref2va"})
LEGACY_EXTRA_FIELDS = {
    "aspect_ratio",
    "audio_flow_shift",
    "duration",
    "duration_seconds",
    "flow_shift",
    "frame_index",
    "frame_indices",
    "short_edge",
    "start_time_seconds",
    "target",
    "task",
}
LEGACY_TARGET_FIELDS = {
    "aspect_ratio",
    "duration_seconds",
    "frame_index",
    "frame_indices",
    "short_edge",
}


class H3ApiContractError(ValueError):
    """A request is ambiguous, unsupported, or misses the immutable cache."""

    def __init__(self, message: str, *, code: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True)
class NormalizedH3Request:
    extra_args: dict[str, Any]
    api_sigma_points: int
    task: str | None
    schedule_contract_sha256: str
    #: The per-request component selection (``h3_forge.components``), passed
    #: through shape-checked only (M1); empty when the request selects none.
    components: Mapping[str, Any] = MappingProxyType({})


def _object(name: str, value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise H3ApiContractError(
            f"{name} must be an object",
            code="H3_INVALID_REQUEST",
            status_code=400,
        )
    return value


def _exact_fields(name: str, value: Mapping[str, Any], allowed: set[str]) -> None:
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise H3ApiContractError(
            f"{name} contains unsupported fields: {unexpected}",
            code="H3_INVALID_REQUEST",
            status_code=400,
        )


def _finite_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise H3ApiContractError(
            f"{name} must be a finite number",
            code="H3_INVALID_REQUEST",
            status_code=400,
        )
    result = float(value)
    if not math.isfinite(result):
        raise H3ApiContractError(
            f"{name} must be a finite number",
            code="H3_INVALID_REQUEST",
            status_code=400,
        )
    return result


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise H3ApiContractError(
            f"{name} must be a positive integer",
            code="H3_INVALID_REQUEST",
            status_code=400,
        )
    return value


def _same(name: str, left: Any, right: Any) -> None:
    if left != right:
        raise H3ApiContractError(
            f"canonical and legacy {name} values conflict",
            code="H3_PARAMETER_CONFLICT",
            status_code=400,
        )


def normalize_h3_request(
    extra_args: Mapping[str, Any] | None,
    *,
    legacy_num_inference_steps: int | None,
    schedule: H3ScheduleContract,
) -> NormalizedH3Request:
    """Normalize canonical v1 fields into the pinned official H3 aliases."""

    validate_h3_schedule_contract(schedule)
    extra = dict(_object("extra_args", extra_args))
    canonical_present = API_NAMESPACE in extra
    canonical = _object(f"extra_args.{API_NAMESPACE}", extra.pop(API_NAMESPACE, None))
    _exact_fields("extra_args", extra, LEGACY_EXTRA_FIELDS)
    target = _object("extra_args.target", extra.get("target"))
    _exact_fields("extra_args.target", target, LEGACY_TARGET_FIELDS)
    _exact_fields(
        f"extra_args.{API_NAMESPACE}",
        canonical,
        {"api_version", "task", "components", "schedule", "conditioning", "output", "acceleration"},
    )
    if canonical_present and canonical.get("api_version") != API_VERSION:
        raise H3ApiContractError(
            f"extra_args.{API_NAMESPACE}.api_version must be {API_VERSION}",
            code="H3_API_VERSION_UNSUPPORTED",
            status_code=400,
        )

    task_value = canonical.get("task")
    task = None if task_value is None else str(task_value).lower()
    if task is not None and task not in SUPPORTED_TASKS:
        raise H3ApiContractError(
            f"unsupported H3 task {task!r}",
            code="H3_TASK_UNSUPPORTED",
            status_code=400,
        )
    if task is not None and "task" in extra:
        _same("task", task, str(extra["task"]).lower())
    elif task is not None:
        extra["task"] = task

    acceleration_value = _object(f"extra_args.{API_NAMESPACE}.acceleration", canonical.get("acceleration"))
    _exact_fields(
        f"extra_args.{API_NAMESPACE}.acceleration",
        acceleration_value,
        {"profile", "spatial_scale", "upscaler", "audio_policy"},
    )
    if str(acceleration_value.get("profile", "off")).lower() != "off":
        raise H3ApiContractError(
            "this H3 curve-cache adapter supports acceleration off only",
            code="H3_ACCELERATION_UNSUPPORTED",
            status_code=400,
        )
    if (
        _finite_float("acceleration.spatial_scale", acceleration_value.get("spatial_scale", 1.0)) != 1.0
        or str(acceleration_value.get("upscaler", "none")).lower() != "none"
        or str(acceleration_value.get("audio_policy", "native")).lower() != "native"
    ):
        raise H3ApiContractError(
            "off acceleration profile does not accept acceleration controls",
            code="H3_ACCELERATION_INVALID",
            status_code=400,
        )

    schedule_value = _object(f"extra_args.{API_NAMESPACE}.schedule", canonical.get("schedule"))
    _exact_fields(
        f"extra_args.{API_NAMESPACE}.schedule",
        schedule_value,
        {
            "profile",
            "contract_sha256",
            "denoise_steps",
            "video_shift",
            "audio_shift",
            "image_condition_timestep",
            "audio_condition_timestep",
        },
    )
    legacy_denoise_steps = None
    if legacy_num_inference_steps is not None:
        legacy_denoise_steps = _positive_int("num_inference_steps", legacy_num_inference_steps) - 1
    requested = {
        "profile": schedule_value.get("profile", INITIAL_SCHEDULE_PROFILE),
        "contract_sha256": schedule_value.get("contract_sha256", schedule.contract_sha256),
        "denoise_steps": _positive_int(
            "schedule.denoise_steps",
            schedule_value.get(
                "denoise_steps",
                schedule.denoise_steps if legacy_denoise_steps is None else legacy_denoise_steps,
            ),
        ),
        "video_shift": _finite_float(
            "schedule.video_shift",
            schedule_value.get("video_shift", extra.get("flow_shift", schedule.video_shift)),
        ),
        "audio_shift": _finite_float(
            "schedule.audio_shift",
            schedule_value.get("audio_shift", extra.get("audio_flow_shift", schedule.audio_shift)),
        ),
        "image_condition_timestep": _finite_float(
            "schedule.image_condition_timestep",
            schedule_value.get("image_condition_timestep", schedule.image_condition_timestep),
        ),
        "audio_condition_timestep": _finite_float(
            "schedule.audio_condition_timestep",
            schedule_value.get("audio_condition_timestep", schedule.audio_condition_timestep),
        ),
    }
    if "video_shift" in schedule_value and "flow_shift" in extra:
        _same(
            "video shift",
            requested["video_shift"],
            _finite_float("flow_shift", extra["flow_shift"]),
        )
    if "audio_shift" in schedule_value and "audio_flow_shift" in extra:
        _same(
            "audio shift",
            requested["audio_shift"],
            _finite_float("audio_flow_shift", extra["audio_flow_shift"]),
        )
    if "denoise_steps" in schedule_value and legacy_denoise_steps is not None:
        _same("denoise steps", requested["denoise_steps"], legacy_denoise_steps)
    expected = {
        "profile": INITIAL_SCHEDULE_PROFILE,
        "contract_sha256": schedule.contract_sha256,
        "denoise_steps": schedule.denoise_steps,
        "video_shift": schedule.video_shift,
        "audio_shift": schedule.audio_shift,
        "image_condition_timestep": schedule.image_condition_timestep,
        "audio_condition_timestep": schedule.audio_condition_timestep,
    }
    if requested != expected:
        raise H3ApiContractError(
            "requested H3 schedule has no exact immutable curve-AdaLN cache",
            code="H3_SCHEDULE_NOT_COMPILED",
            status_code=409,
        )

    extra["flow_shift"] = requested["video_shift"]
    extra["audio_flow_shift"] = requested["audio_shift"]

    conditioning = _object(f"extra_args.{API_NAMESPACE}.conditioning", canonical.get("conditioning"))
    _exact_fields(
        f"extra_args.{API_NAMESPACE}.conditioning",
        conditioning,
        {"frame_indices", "video_start_time_seconds"},
    )
    for canonical_name, legacy_name in (
        ("frame_indices", "frame_indices"),
        ("video_start_time_seconds", "start_time_seconds"),
    ):
        if canonical_name not in conditioning:
            continue
        value = conditioning[canonical_name]
        if legacy_name in extra:
            _same(canonical_name, value, extra[legacy_name])
        else:
            extra[legacy_name] = value

    output = _object(f"extra_args.{API_NAMESPACE}.output", canonical.get("output"))
    _exact_fields(f"extra_args.{API_NAMESPACE}.output", output, {"include_audio"})
    if output.get("include_audio", True) is not True:
        raise H3ApiContractError(
            "the strict MiniMax H3 profile always generates joint audio",
            code="H3_OUTPUT_UNSUPPORTED",
            status_code=400,
        )

    # The per-request component selection (M1, api-comfy.md §3): shape-gated
    # here only (a JSON object); the dense runtime parses it with
    # ``ComponentSet.parse`` and owns selection/reference validation.  The
    # cache runtime passes it through untouched.
    components = _object(f"extra_args.{API_NAMESPACE}.components", canonical.get("components"))

    return NormalizedH3Request(
        extra_args=extra,
        api_sigma_points=schedule.api_sigma_points,
        task=task,
        schedule_contract_sha256=schedule.contract_sha256,
        components=components,
    )


def validate_h3_sampling_controls(sampling: Any, *, negative_prompt: Any) -> None:
    """Reject generic diffusion controls that H3 ignores or that break parity."""

    def neutral_number(value: Any, expected: float) -> bool:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        return math.isfinite(float(value)) and float(value) == expected

    neutral = {
        "negative_prompt": negative_prompt in (None, ""),
        "sigmas": getattr(sampling, "sigmas", None) is None,
        "lora_request": getattr(sampling, "lora_request", None) is None,
        "lora_scale": neutral_number(getattr(sampling, "lora_scale", 1.0), 1.0),
        "eta": neutral_number(getattr(sampling, "eta", 0.0), 0.0),
        "guidance_scale": (
            not getattr(sampling, "guidance_scale_provided", False)
            or neutral_number(getattr(sampling, "guidance_scale", None), 1.0)
        ),
        "guidance_scale_2": (
            not getattr(sampling, "guidance_scale_2_provided", False)
            or neutral_number(getattr(sampling, "guidance_scale_2", None), 1.0)
        ),
        "guidance_rescale": neutral_number(getattr(sampling, "guidance_rescale", 0.0), 0.0),
        "true_cfg_scale": getattr(sampling, "true_cfg_scale", None) in (None, 1, 1.0),
        "do_classifier_free_guidance": not getattr(sampling, "do_classifier_free_guidance", False),
        "cfg_normalize": not getattr(sampling, "cfg_normalize", False),
        "strength": getattr(sampling, "strength", None) is None,
        "timesteps": getattr(sampling, "timesteps", None) is None,
        "timestep": getattr(sampling, "timestep", None) is None,
        "step_index": getattr(sampling, "step_index", None) is None,
        "boundary_ratio": getattr(sampling, "boundary_ratio", None) is None,
        "decode_timestep": getattr(sampling, "decode_timestep", None) is None,
        "decode_noise_scale": getattr(sampling, "decode_noise_scale", None) is None,
        "extra_step_kwargs": not getattr(sampling, "extra_step_kwargs", {}),
        "quality": getattr(sampling, "quality", None) in (None, "lossless"),
    }
    rejected = sorted(name for name, is_neutral in neutral.items() if not is_neutral)
    if rejected:
        raise H3ApiContractError(
            f"controls are unavailable in the strict H3 curve-cache profile: {rejected}",
            code="H3_CONTROL_UNSUPPORTED",
            status_code=400,
        )


__all__ = [
    "API_NAMESPACE",
    "API_VERSION",
    "H3ApiContractError",
    "INITIAL_SCHEDULE_PROFILE",
    "NormalizedH3Request",
    "normalize_h3_request",
    "validate_h3_sampling_controls",
]
