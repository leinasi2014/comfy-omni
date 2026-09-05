"""Bounded raw-byte verification of the frozen legacy curve sidecar.

Derived from h3-forge e9cb011 package_assembler.py:436-543 (blob
e64558f1d3bb6e1ee6f714b70e783d9df907f9ce) and lora_hotswap/curve_adaln_cache.py
(blob 25e12cf6ec4299b79de988b38edc2ec718f9ccad), Apache-2.0.
Raw offsets, FP32 bits and sequential BF16 hashes preserve the old Torch
verifier's byte semantics. Strict schedule reconstruction remains lazy.
"""

from __future__ import annotations

import hashlib
import re
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from comfy_omni.artifacts import fileops
from comfy_omni.artifacts.sources import SafeTensorSources
from comfy_omni.runtime.h3.package_binding import (
    CURVE_CACHE_NAME,
    CURVE_CACHE_SCHEMA,
    HOST_COMMIT,
    PROVENANCE_SCHEMA,
    CurveCacheBinding,
    LegacyPackageError,
    LegacyProducerIdentity,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LegacyPackageError(message, evidence={"stage": "curve-cache"})


def verify_curve_cache(
    path: Path,
    fold: object,
    claimed: object,
    producer: LegacyProducerIdentity,
) -> CurveCacheBinding:
    """Verify the full sidecar and its fold/producer/manifest bindings before load."""
    from comfy_omni.runtime.h3.schedule import h3_schedule_contract_from_dict

    _require(isinstance(fold, dict), "curve cache is missing its LoRA fold receipt")
    _require(isinstance(claimed, dict), "curve cache is missing its package receipt")
    expected_fold = {
        "module_count": 208,
        "quantized_module_count": 200,
        "dense_module_count": 8,
        "scale": 1.0,
        "use_adaln_cache": True,
        "profile": "h3/fl2va/dit-standard/lora-turbo-v4-full-ab/native-separate/v1",
        "alpha_policy": "PER_MODULE_RANK_FROM_ALPHA_NONE",
        "float32_matmul_precision": "highest",
        "fold_device": "cpu",
    }
    _require(
        all(fold.get(key) == value for key, value in expected_fold.items()), "curve-cache 208-module fold is invalid"
    )
    with SafeTensorSources([path]) as sources:
        metadata = sources.metadata[0]
        try:
            schedule = h3_schedule_contract_from_dict(fileops.parse_json_strict(metadata["schedule_contract"]))
        except (KeyError, TypeError, ValueError, fileops.FsopsError) as exc:
            raise LegacyPackageError("curve cache schedule is invalid", evidence={"stage": "curve-cache"}) from exc
        slots = sum(len(plan.values) for plan in schedule.plans)
        descriptors = {
            "plan_offsets": ("I64", (len(schedule.plans) + 1,)),
            "plan_timesteps": ("F32", (slots,)),
            "block_params": ("BF16", (50, slots, 96768)),
            "final_params": ("BF16", (slots, 10752)),
        }
        _require(set(sources.tensors) == set(descriptors), "curve cache tensor census is invalid")
        for name, (dtype, shape) in descriptors.items():
            descriptor = sources.tensors[name].descriptor
            _require(
                (descriptor.dtype, descriptor.shape) == (dtype, shape), f"curve cache {name} descriptor is invalid"
            )
        _verify_metadata(metadata, fold, producer, schedule.contract_sha256)
        offsets = [0]
        timesteps = []
        for plan in schedule.plans:
            timesteps.extend(plan.values)
            offsets.append(len(timesteps))
        _require(
            sources.read_raw(sources.tensors["plan_offsets"]) == struct.pack(f"<{len(offsets)}q", *offsets),
            "curve cache offsets differ from the compiled plans",
        )
        _require(
            sources.read_raw(sources.tensors["plan_timesteps"]) == struct.pack(f"<{len(timesteps)}f", *timesteps),
            "curve cache FP32 timestep bits differ from the compiled plans",
        )
        for name in ("block_params", "final_params"):
            digest = hashlib.sha256()
            for chunk in sources.iter_raw(sources.tensors[name]):
                digest.update(chunk)
            _require(digest.hexdigest() == metadata[f"{name}_sha256"], f"curve cache {name} payload hash mismatch")
        observed = {
            "schema": CURVE_CACHE_SCHEMA,
            "path": f"Ref2VA/transformer/{CURVE_CACHE_NAME}",
            "sha256": sources.hashes[0],
            "size": sources.sizes[0],
            "schedule_contract_sha256": schedule.contract_sha256,
            "plan_count": len(schedule.plans),
            "plan_slot_count": slots,
            **{key: metadata[key] for key in _DIGEST_FIELDS},
        }
        _require(observed == claimed, "curve cache differs from its package receipt")
        sources.verify_unchanged()
    return CurveCacheBinding(
        path, schedule, observed["sha256"], observed["size"], metadata["source_curve_sha256"], producer
    )


_DIGEST_FIELDS = (
    "source_curve_sha256",
    "normalized_lora_sha256",
    "silu_grid_sha256",
    "curve_payload_sha256",
    "block_params_sha256",
    "final_params_sha256",
)


def _verify_metadata(
    metadata: Mapping[str, str], fold: Mapping[str, Any], producer: LegacyProducerIdentity, schedule_sha: str
) -> None:
    required = {
        "schema": CURVE_CACHE_SCHEMA,
        "format_version": "1",
        "target_dtype": "BF16",
        "block_params_layout": "block-slot-width",
        "schedule_contract_sha256": schedule_sha,
        "normalized_lora_sha256": fold.get("normalized_lora_sha256"),
        "lora_scale": "1.0",
        "alpha_policy": "alpha-none-means-rank",
        "effective_lora_multiplier": "1.0",
        "vllm_omni_commit": HOST_COMMIT,
        "converter_commit": producer.h3_forge_commit,
        "converter_wheel_sha256": producer.wheel_sha256,
        "converter_provenance_schema": PROVENANCE_SCHEMA,
        "converter_build_context_sha256": producer.build_context_sha256,
        "converter_installed_payload_sha256": producer.installed_payload_sha256,
        "isolated_conversion_process_required": "true",
    }
    _require(
        all(metadata.get(key) == value for key, value in required.items()),
        "curve cache provenance differs from package",
    )
    _require(
        all(re.fullmatch(r"[0-9a-f]{64}", metadata.get(key, "")) for key in _DIGEST_FIELDS),
        "curve cache semantic digest is invalid",
    )
