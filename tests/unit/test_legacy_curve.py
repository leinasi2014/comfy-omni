from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from test_legacy_fixtures import make_curve

from comfy_omni.runtime.h3.curve_contract import verify_curve_cache
from comfy_omni.runtime.h3.package_binding import AUDITED_PRODUCER, LegacyPackageError


@pytest.fixture
def cache(tmp_path: Path):
    pytest.importorskip("torch")
    path = tmp_path / "cache.safetensors"
    fold, claimed, schedule, payload_start = make_curve(path)
    return path, fold, claimed, schedule, payload_start


def test_legacy_curve_verifies_all_raw_payload_and_schedule_bits(cache):
    path, fold, claimed, schedule, _ = cache
    binding = verify_curve_cache(path, fold, claimed, AUDITED_PRODUCER)
    assert binding.schedule == schedule
    assert binding.sha256 == claimed["sha256"]
    assert binding.producer == AUDITED_PRODUCER
    assert binding.schedule.denoise_steps == 4


@pytest.mark.parametrize("mutation", ["offsets", "timestep", "block", "final", "fold", "producer", "claim"])
def test_legacy_curve_refuses_bit_and_binding_tamper(cache, mutation):
    path, fold, claimed, schedule, start = cache
    producer = AUDITED_PRODUCER
    if mutation == "fold":
        fold["module_count"] = 207
    elif mutation == "producer":
        producer = replace(producer, wheel_sha256="0" * 64)
    elif mutation == "claim":
        claimed["source_curve_sha256"] = "0" * 64
    else:
        offset_bytes = (len(schedule.plans) + 1) * 8
        timestep_bytes = sum(len(plan.values) for plan in schedule.plans) * 4
        location = {
            "offsets": start,
            "timestep": start + offset_bytes,
            "block": start + offset_bytes + timestep_bytes,
            "final": path.stat().st_size - 1,
        }[mutation]
        with path.open("r+b") as stream:
            stream.seek(location)
            byte = stream.read(1)
            stream.seek(location)
            stream.write(bytes([byte[0] ^ 1]))
    with pytest.raises(LegacyPackageError):
        verify_curve_cache(path, fold, claimed, producer)
