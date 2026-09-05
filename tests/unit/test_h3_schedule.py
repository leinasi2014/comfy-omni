# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: h3-forge contributors
# Derived from h3-forge@e9cb011d00b028c149db3978de246c54f6e34acc; Apache-2.0.
# Source: tests/test_h3_schedule.py; blob 55b9fe51104f867641cad47e0d1aae6b5219f19b.
from __future__ import annotations

import json
import struct
import unittest
from dataclasses import replace

from comfy_omni.runtime.h3.schedule import (
    H3ScheduleContractError,
    build_h3_schedule_contract,
    h3_schedule_contract_from_dict,
    validate_h3_schedule_contract,
)


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


class H3ScheduleTests(unittest.TestCase):
    def setUp(self) -> None:
        import pytest

        pytest.importorskip("torch")

    def test_turbo_four_means_five_sigma_points_and_four_forwards(self) -> None:
        contract = build_h3_schedule_contract(denoise_steps=4, modes=("t2va",))

        self.assertEqual(contract.denoise_steps, 4)
        self.assertEqual(contract.api_sigma_points, 5)
        self.assertEqual(len(dict(contract.mode_plan_indices)["t2va"]), 4)
        self.assertEqual(contract.plans[0].bits, (_bits(0.0),))

    def test_union_covers_all_api_condition_signatures_without_scalar_dedup(self) -> None:
        contract = build_h3_schedule_contract(denoise_steps=4)
        mode_indices = dict(contract.mode_plan_indices)

        self.assertEqual(set(mode_indices), {"t2va", "fl2va", "ref2va-image", "ref2va-audio", "ref2va-mixed"})
        self.assertEqual(mode_indices["fl2va"], mode_indices["ref2va-image"])
        self.assertEqual(max(len(plan.values) for plan in contract.plans), 4)
        self.assertEqual(len(mode_indices["ref2va-mixed"]), 4)
        self.assertEqual(len(contract.contract_sha256), 64)

    def test_contract_is_deterministic_and_parameter_bound(self) -> None:
        left = build_h3_schedule_contract(denoise_steps=4)
        right = build_h3_schedule_contract(denoise_steps=4)
        changed = build_h3_schedule_contract(denoise_steps=4, video_shift=11.0)

        self.assertEqual(left, right)
        self.assertNotEqual(left.contract_sha256, changed.contract_sha256)

    def test_invalid_api_schedule_fails_closed(self) -> None:
        for value in (0, -1, True, 257):
            with self.subTest(value=value), self.assertRaises(H3ScheduleContractError):
                build_h3_schedule_contract(denoise_steps=value)  # type: ignore[arg-type]
        with self.assertRaisesRegex(H3ScheduleContractError, "unsupported"):
            build_h3_schedule_contract(denoise_steps=4, modes=("combined",))
        with self.assertRaisesRegex(H3ScheduleContractError, "unique"):
            build_h3_schedule_contract(denoise_steps=4, modes=("t2va", "t2va"))
        with self.assertRaisesRegex(H3ScheduleContractError, "non-finite"):
            build_h3_schedule_contract(denoise_steps=4, video_shift=1e100)

    def test_mutated_contract_fails_canonical_self_check(self) -> None:
        contract = build_h3_schedule_contract(denoise_steps=4)
        forged = replace(contract, contract_sha256="0" * 64)
        with self.assertRaisesRegex(H3ScheduleContractError, "self-check"):
            validate_h3_schedule_contract(forged)

    def test_serialized_contract_is_rebuilt_and_forgery_is_rejected(self) -> None:
        contract = build_h3_schedule_contract(denoise_steps=4)
        serialized = json.loads(json.dumps(contract.to_dict()))

        self.assertEqual(h3_schedule_contract_from_dict(serialized), contract)
        forged = json.loads(json.dumps(contract.to_dict()))
        forged["plans"][0]["float32_bits"][0] = "0x00000001"
        with self.assertRaisesRegex(H3ScheduleContractError, "not canonical"):
            h3_schedule_contract_from_dict(forged)
