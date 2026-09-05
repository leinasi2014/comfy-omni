# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: h3-forge contributors
# Derived from h3-forge@e9cb011d00b028c149db3978de246c54f6e34acc; Apache-2.0.
# Source: tests/test_api_contract.py; blob 7b6f2746d3813eb6d580234680649899ab65402a.
from __future__ import annotations

import unittest
from types import SimpleNamespace

from comfy_omni.runtime.h3.requests import (
    H3ApiContractError,
    normalize_h3_request,
    validate_h3_sampling_controls,
)
from comfy_omni.runtime.h3.schedule import build_h3_schedule_contract


class ApiContractTests(unittest.TestCase):
    def setUp(self) -> None:
        import pytest

        pytest.importorskip("torch")
        self.schedule = build_h3_schedule_contract(denoise_steps=4)

    def test_canonical_request_normalizes_to_official_aliases(self) -> None:
        result = normalize_h3_request(
            {
                "h3_forge": {
                    "api_version": 1,
                    "task": "ref2va",
                    "schedule": {
                        "profile": "turbo-v4-s12-a3",
                        "denoise_steps": 4,
                        "video_shift": 12.0,
                        "audio_shift": 3.0,
                    },
                    "conditioning": {
                        "frame_indices": [0, -1],
                        "video_start_time_seconds": [0.0],
                    },
                    "output": {"include_audio": True},
                }
            },
            legacy_num_inference_steps=None,
            schedule=self.schedule,
        )

        self.assertEqual(result.api_sigma_points, 5)
        self.assertEqual(result.extra_args["task"], "ref2va")
        self.assertEqual(result.extra_args["flow_shift"], 12.0)
        self.assertEqual(result.extra_args["audio_flow_shift"], 3.0)
        self.assertEqual(result.extra_args["frame_indices"], [0, -1])

    def test_alias_conflict_and_uncompiled_schedule_fail_closed(self) -> None:
        with self.assertRaises(H3ApiContractError) as shifted:
            normalize_h3_request(
                {
                    "flow_shift": 11.0,
                    "h3_forge": {
                        "api_version": 1,
                        "schedule": {"video_shift": 12.0},
                    },
                },
                legacy_num_inference_steps=5,
                schedule=self.schedule,
            )
        self.assertEqual(shifted.exception.code, "H3_PARAMETER_CONFLICT")
        with self.assertRaises(H3ApiContractError) as caught:
            normalize_h3_request(
                {
                    "h3_forge": {
                        "api_version": 1,
                        "schedule": {"denoise_steps": 6},
                    }
                },
                legacy_num_inference_steps=None,
                schedule=self.schedule,
            )
        self.assertEqual(caught.exception.code, "H3_SCHEDULE_NOT_COMPILED")
        self.assertEqual(caught.exception.status_code, 409)

    def test_legacy_step_semantics_and_ineffective_cfg_are_rejected(self) -> None:
        with self.assertRaises(H3ApiContractError) as steps:
            normalize_h3_request({}, legacy_num_inference_steps=4, schedule=self.schedule)
        self.assertEqual(steps.exception.code, "H3_SCHEDULE_NOT_COMPILED")
        with self.assertRaises(H3ApiContractError):
            validate_h3_sampling_controls(
                SimpleNamespace(guidance_scale_provided=True, guidance_scale=7.5),
                negative_prompt=None,
            )

    def test_unknown_legacy_extra_args_and_target_fields_are_rejected(self) -> None:
        for extra in ({"bogus": 1}, {"target": {"sampler": "ignored"}}):
            with self.subTest(extra=extra), self.assertRaises(H3ApiContractError) as caught:
                normalize_h3_request(
                    extra,
                    legacy_num_inference_steps=5,
                    schedule=self.schedule,
                )
            self.assertEqual(caught.exception.code, "H3_INVALID_REQUEST")

    def test_generic_sampling_controls_fail_closed(self) -> None:
        neutral = SimpleNamespace(
            guidance_scale_2_provided=False,
            guidance_rescale=0.0,
            quality="lossless",
        )
        validate_h3_sampling_controls(neutral, negative_prompt="")

        for field, value in (
            ("guidance_scale_2_provided", True),
            ("guidance_rescale", 0.25),
            ("strength", 0.5),
            ("timesteps", [1.0]),
            ("quality", "high"),
        ):
            sampling = SimpleNamespace(**vars(neutral))
            setattr(sampling, field, value)
            if field == "guidance_scale_2_provided":
                sampling.guidance_scale_2 = 7.5
            with self.subTest(field=field), self.assertRaises(H3ApiContractError) as caught:
                validate_h3_sampling_controls(sampling, negative_prompt=None)
            self.assertEqual(caught.exception.code, "H3_CONTROL_UNSUPPORTED")

    def test_only_explicit_or_default_off_acceleration_is_accepted(self) -> None:
        for acceleration in ({}, {"profile": "off"}, {"profile": "off", "spatial_scale": 1.0}):
            result = normalize_h3_request(
                {"h3_forge": {"api_version": 1, "acceleration": acceleration}},
                legacy_num_inference_steps=5,
                schedule=self.schedule,
            )
            self.assertEqual(result.api_sigma_points, 5)
        for profile in ("lowres-resize-v0", "latent-upscale-v0", "unknown"):
            with self.assertRaises(H3ApiContractError) as caught:
                normalize_h3_request(
                    {"h3_forge": {"api_version": 1, "acceleration": {"profile": profile}}},
                    legacy_num_inference_steps=5,
                    schedule=self.schedule,
                )
            self.assertEqual(caught.exception.code, "H3_ACCELERATION_UNSUPPORTED")
        with self.assertRaises(H3ApiContractError) as caught:
            normalize_h3_request(
                {"h3_forge": {"api_version": 1, "acceleration": {"profile": "off", "spatial_scale": 0.5}}},
                legacy_num_inference_steps=5,
                schedule=self.schedule,
            )
        self.assertEqual(caught.exception.code, "H3_ACCELERATION_INVALID")

    def test_joint_audio_cannot_be_silently_disabled(self) -> None:
        with self.assertRaises(H3ApiContractError) as caught:
            normalize_h3_request(
                {"h3_forge": {"api_version": 1, "output": {"include_audio": False}}},
                legacy_num_inference_steps=5,
                schedule=self.schedule,
            )
        self.assertEqual(caught.exception.code, "H3_OUTPUT_UNSUPPORTED")
