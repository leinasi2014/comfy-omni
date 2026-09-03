# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright h3-forge contributors
#
# Provenance: wholesale migration from h3-forge@e9cb011d00b028c149db3978de246c54f6e34acc
#   source path: src/h3_forge/lora_hotswap/bake_plan.py
#   source blob: 767ffd216c1a755fe60d04082c908627f46b311b
#   license: Apache-2.0
#   attribution: h3-forge contributors
# Migrated byte-preserving except this provenance header, import retargeting, and
# mechanical line wrapping to satisfy the repository line-length (120).
"""Content-bound planning for Turbo v4 product bake and target-cast diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from comfy_omni.artifacts.safetensors import read_safetensors_header_stream
from comfy_omni.domain.checkpoints import TensorDescriptor

from .normalize import TURBO_V4_MODULE_COUNT, _expected_modules, _module_and_side

OFFICIAL_FL2VA_BF16_BAKE_PROFILE = "h3/fl2va/dit-standard/lora-turbo-v4-full-ab/official-vllm-bf16-bake/v1"
BAKE_PLAN_SCHEMA = "h3-comfy.lora-bake-plan/v1"
OFFICIAL_FL2VA_FP16_DIAGNOSTIC_PROFILE = "h3/fl2va/dit-standard/lora-turbo-v4-full-ab/diagnostic-fp16-cast/v1"
FP16_BAKE_PLAN_SCHEMA = "h3-comfy.lora-bake-plan-fp16-diagnostic/v1"
TARGET_DTYPES = ("BF16", "FP16")
COMFY_BAKED_NATIVE_USE_ADALN_CACHE = False
COMFY_BAKED_NATIVE_FOLD_MODULE_COUNT = TURBO_V4_MODULE_COUNT
COMFY_BAKED_NATIVE_PRODUCT_GATE = "comfy-reference-fold-tensor-byte-parity;use_adaln_cache=false;modules=259"
ADALN_CACHE_RUNTIME_SIDECAR_PROFILE = (
    "h3/fl2va/dit-standard/lora-turbo-v4-full-ab/runtime-sidecar-use-adaln-cache-true/v1"
)
INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
MAX_MANIFEST_BYTES = 16 * 1024 * 1024

_REQUIRED_CONFIG = {
    "_class_name": "MiniMaxH3DiTModel",
    "hidden_size": 5376,
    "num_layers": 50,
    "token_refiner_num_layers": 2,
    "num_attention_heads": 56,
    "attention_head_dim": 128,
    "ffn_hidden_size": 14336,
    "latents_dim": 24,
    "audio_latents_dim": 32,
    "patch_size": [1, 2, 2],
    "text_dim": 5120,
    "timestep_input_dim": 256,
    "time_embed_hidden_size": 5376,
    "time_embed_dim": 2688,
    "adaln_out_features": 96768,
    "final_adaln_out_features": 10752,
    "rope_inv_freq_len": 16,
    "norm_eps": 1e-5,
    "qk_norm_eps": 1e-5,
    "final_norm_eps": 1e-5,
}


@dataclass(frozen=True)
class OfficialBaseContract:
    tensor_count: int
    shard_count: int
    total_size: int
    dtype_counts: tuple[tuple[str, int], ...]
    catalog_sha256: str


OFFICIAL_FL2VA_BASE_CONTRACT = OfficialBaseContract(
    tensor_count=535,
    shard_count=13,
    total_size=66_280_430_144,
    dtype_counts=(("BF16", 522), ("F32", 13)),
    catalog_sha256="5d350cba2dd8427d9d95dd62aba4e6adaf3a244b25f755b80afee378e8db6714",
)


class _DuplicateKeyError(ValueError):
    pass


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


@dataclass(frozen=True)
class BakeOperation:
    module: str
    base_tensor: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    lora_a: str
    lora_b: str
    rank: int
    multiplier: float
    layout_operation: str
    qkv_num_query_groups: int | None
    qkv_heads_per_group: int | None
    qkv_head_dim: int | None


@dataclass(frozen=True)
class BaseShardIdentity:
    name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class LoraBakePlan:
    schema: str
    profile: str
    base_directory: str
    normalized_lora: str
    config_sha256: str
    index_sha256: str
    lora_sha256: str
    base_catalog_sha256: str
    base_shards: tuple[BaseShardIdentity, ...]
    plan_sha256: str
    operation_count: int
    operations: tuple[BakeOperation, ...]
    target_dtype: str = "BF16"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.target_dtype == "BF16":
            payload.pop("target_dtype")
        return payload


def _target_contract(target_dtype: str) -> tuple[str, str]:
    if target_dtype == "BF16":
        return OFFICIAL_FL2VA_BF16_BAKE_PROFILE, BAKE_PLAN_SCHEMA
    if target_dtype == "FP16":
        return OFFICIAL_FL2VA_FP16_DIAGNOSTIC_PROFILE, FP16_BAKE_PLAN_SCHEMA
    raise ValueError(f"unsupported bake target dtype: {target_dtype!r}")


def _read_json_object(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required regular file is missing or linked: {path}")
    size = path.stat().st_size
    if size <= 0 or size > MAX_MANIFEST_BYTES:
        raise ValueError(f"unsafe JSON manifest size for {path}: {size}")
    payload = path.read_bytes()

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _DuplicateKeyError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        decoded = json.loads(
            payload,
            object_pairs_hook=unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateKeyError) as exc:
        raise ValueError(f"invalid JSON manifest {path}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"JSON manifest must be an object: {path}")
    return decoded, payload


def _sha256_stream(stream: Any, size: int) -> str:
    digest = hashlib.sha256()
    remaining = size
    while remaining:
        chunk = stream.read(min(remaining, 8 * 1024 * 1024))
        if not chunk:
            raise ValueError("file changed or was truncated while hashing")
        digest.update(chunk)
        remaining -= len(chunk)
    if stream.read(1):
        raise ValueError("file grew while it was being hashed")
    return digest.hexdigest()


def _stat_identity(stat: os.stat_result) -> tuple[int, int, int, int]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _read_safetensors_and_hash(
    path: Path,
) -> tuple[dict[str, str], tuple[TensorDescriptor, ...], str]:
    with path.open("rb") as stream:
        snapshot = os.fstat(stream.fileno())
        metadata, tensors, _ = read_safetensors_header_stream(stream, path, snapshot.st_size)
        stream.seek(0)
        digest = _sha256_stream(stream, snapshot.st_size)
        if _stat_identity(os.fstat(stream.fileno())) != _stat_identity(snapshot) or _stat_identity(
            path.stat()
        ) != _stat_identity(snapshot):
            raise ValueError(f"file changed while it was being inspected: {path}")
    return metadata, tensors, digest


def _validate_config(config: dict[str, Any]) -> None:
    for key, expected in _REQUIRED_CONFIG.items():
        if config.get(key) != expected:
            raise ValueError(f"official FL2VA config {key!r} must be {expected!r}, found {config.get(key)!r}")
    if "quantization_config" in config:
        raise ValueError("official BF16 bake profile does not accept quantization_config")


def _validate_normalized_lora(
    metadata: dict[str, str], tensors: tuple[TensorDescriptor, ...]
) -> dict[str, dict[str, TensorDescriptor]]:
    required_metadata = {
        "application": "W_eff = W + lora_B @ lora_A",
        "base_model": "MiniMax-H3",
        "dtype": "bfloat16",
        "h3_comfy.operation": "remove-prefix:diffusion_model.",
        "h3_comfy.payload": "byte-identical",
        "h3_comfy.profile": "h3/fl2va/dit-standard/lora-turbo-v4-full-ab/native-separate/v1",
        "sampler_steps": "4",
    }
    for key, expected in required_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"normalized Turbo v4 metadata {key!r} must be {expected!r}")
    expected_modules = _expected_modules()
    modules: dict[str, dict[str, TensorDescriptor]] = {}
    for tensor in tensors:
        if tensor.dtype != "BF16":
            raise ValueError(f"normalized Turbo v4 tensor {tensor.name!r} must be BF16")
        module, side = _module_and_side(tensor.name)
        expected = expected_modules.get(module)
        if expected is None:
            raise ValueError(f"unexpected normalized Turbo v4 module: {module!r}")
        if side in modules.setdefault(module, {}):
            raise ValueError(f"duplicate normalized Turbo v4 {side} tensor for {module!r}")
        expected_shape = expected[0 if side == "A" else 1]
        if tensor.shape != expected_shape:
            raise ValueError(
                f"normalized Turbo v4 tensor {tensor.name!r} has shape {tensor.shape}, expected {expected_shape}"
            )
        modules[module][side] = tensor
    if len(modules) != TURBO_V4_MODULE_COUNT:
        raise ValueError(f"normalized Turbo v4 requires {TURBO_V4_MODULE_COUNT} modules, found {len(modules)}")
    incomplete = sorted(module for module, sides in modules.items() if set(sides) != {"A", "B"})
    if incomplete:
        raise ValueError(f"normalized Turbo v4 has incomplete A/B pair for {incomplete[0]!r}")
    return modules


def _load_base_catalog(
    base_directory: Path, weight_map: dict[str, str]
) -> tuple[dict[str, tuple[str, TensorDescriptor]], tuple[BaseShardIdentity, ...]]:
    shards = sorted(set(weight_map.values()))
    catalog: dict[str, tuple[str, TensorDescriptor]] = {}
    identities: list[BaseShardIdentity] = []
    for shard in shards:
        if not isinstance(shard, str) or not shard or Path(shard).name != shard:
            raise ValueError(f"unsafe shard name in index: {shard!r}")
        path = base_directory / shard
        if path.is_symlink() or not path.is_file() or path.resolve().parent != base_directory:
            raise ValueError(f"indexed shard is missing, linked, or outside the base directory: {shard!r}")
        with path.open("rb") as stream:
            snapshot = os.fstat(stream.fileno())
            _, tensors, _ = read_safetensors_header_stream(stream, path, snapshot.st_size)
            stream.seek(0)
            digest = _sha256_stream(stream, snapshot.st_size)
            if _stat_identity(os.fstat(stream.fileno())) != _stat_identity(snapshot) or _stat_identity(
                path.stat()
            ) != _stat_identity(snapshot):
                raise ValueError(f"indexed shard changed during planning: {shard!r}")
        identities.append(BaseShardIdentity(name=shard, size=snapshot.st_size, sha256=digest))
        for tensor in tensors:
            if tensor.name in catalog:
                raise ValueError(f"tensor appears in multiple shards: {tensor.name!r}")
            catalog[tensor.name] = (shard, tensor)
    for name, shard in weight_map.items():
        if not isinstance(name, str) or not isinstance(shard, str):
            raise ValueError("weight_map keys and values must be strings")
        actual = catalog.get(name)
        if actual is None or actual[0] != shard:
            raise ValueError(f"index/header mismatch for tensor {name!r}")
    extras = sorted(set(catalog) - set(weight_map))
    if extras:
        raise ValueError(f"shard header contains tensor absent from index: {extras[0]!r}")
    return catalog, tuple(identities)


def _catalog_sha256(catalog: dict[str, tuple[str, TensorDescriptor]]) -> str:
    rows = [(name, shard, tensor.dtype, tensor.shape) for name, (shard, tensor) in sorted(catalog.items())]
    payload = json.dumps(rows, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_official_base_contract(index: dict[str, Any], catalog: dict[str, tuple[str, TensorDescriptor]]) -> str:
    contract = OFFICIAL_FL2VA_BASE_CONTRACT
    if len(catalog) != contract.tensor_count:
        raise ValueError(f"official FL2VA requires {contract.tensor_count} tensors, found {len(catalog)}")
    shard_count = len({shard for shard, _ in catalog.values()})
    if shard_count != contract.shard_count:
        raise ValueError(f"official FL2VA requires {contract.shard_count} shards, found {shard_count}")
    metadata = index.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("total_size") != contract.total_size:
        raise ValueError(f"official FL2VA index total_size must be {contract.total_size}")
    payload_size = sum(tensor.data_offsets[1] - tensor.data_offsets[0] for _, tensor in catalog.values())
    if payload_size != contract.total_size:
        raise ValueError(f"official FL2VA tensor payload size must be {contract.total_size}, found {payload_size}")
    dtype_counts: dict[str, int] = {}
    for _, tensor in catalog.values():
        dtype_counts[tensor.dtype] = dtype_counts.get(tensor.dtype, 0) + 1
    if tuple(sorted(dtype_counts.items())) != contract.dtype_counts:
        raise ValueError(
            f"official FL2VA dtype counts must be {contract.dtype_counts}, found {tuple(sorted(dtype_counts.items()))}"
        )
    digest = _catalog_sha256(catalog)
    if digest != contract.catalog_sha256:
        raise ValueError(f"official FL2VA tensor catalog digest mismatch: {digest}")
    return digest


def _plan_sha256(
    *,
    profile: str,
    config_sha256: str,
    index_sha256: str,
    lora_sha256: str,
    base_catalog_sha256: str,
    base_shards: tuple[BaseShardIdentity, ...],
    operations: tuple[BakeOperation, ...],
    target_dtype: str = "BF16",
) -> str:
    _, expected_schema = _target_contract(target_dtype)
    identity = {
        "profile": profile,
        "config_sha256": config_sha256,
        "index_sha256": index_sha256,
        "lora_sha256": lora_sha256,
        "base_catalog_sha256": base_catalog_sha256,
        "base_shards": [asdict(shard) for shard in base_shards],
        "operations": [asdict(operation) for operation in operations],
    }
    if expected_schema == FP16_BAKE_PLAN_SCHEMA:
        identity["target_dtype"] = target_dtype
    canonical = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"bake plan {field} must be a lowercase SHA256 digest")
    return value


def _parse_base_shards(value: Any) -> tuple[BaseShardIdentity, ...]:
    if not isinstance(value, list) or len(value) != OFFICIAL_FL2VA_BASE_CONTRACT.shard_count:
        raise ValueError(f"bake plan requires {OFFICIAL_FL2VA_BASE_CONTRACT.shard_count} base shards")
    shards: list[BaseShardIdentity] = []
    names: set[str] = set()
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != {"name", "size", "sha256"}:
            raise ValueError("bake plan base shard entry has an invalid schema")
        name = raw["name"]
        size = raw["size"]
        if not isinstance(name, str) or not name or Path(name).name != name or name in names:
            raise ValueError(f"bake plan contains an unsafe or duplicate shard name: {name!r}")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError(f"bake plan shard size must be positive for {name!r}")
        names.add(name)
        shards.append(
            BaseShardIdentity(
                name=name,
                size=size,
                sha256=_require_sha256(raw["sha256"], f"base shard {name!r} sha256"),
            )
        )
    if tuple(shard.name for shard in shards) != tuple(sorted(names)):
        raise ValueError("bake plan base shards must use canonical name order")
    return tuple(shards)


def _parse_operations(value: Any, shard_names: set[str]) -> tuple[BakeOperation, ...]:
    expected_modules = _expected_modules()
    if not isinstance(value, list) or len(value) != TURBO_V4_MODULE_COUNT:
        raise ValueError(f"bake plan requires {TURBO_V4_MODULE_COUNT} operations")
    field_names = set(BakeOperation.__dataclass_fields__)
    operations: list[BakeOperation] = []
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != field_names:
            raise ValueError("bake plan operation has an invalid schema")
        string_fields = (
            "module",
            "base_tensor",
            "shard",
            "dtype",
            "lora_a",
            "lora_b",
            "layout_operation",
        )
        if any(not isinstance(raw[field], str) or not raw[field] for field in string_fields):
            raise ValueError("bake plan operation string fields must be non-empty strings")
        shape = raw["shape"]
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in shape)
        ):
            raise ValueError("bake plan operation shape must contain two positive integers")
        nullable_dimensions = (
            raw["qkv_num_query_groups"],
            raw["qkv_heads_per_group"],
            raw["qkv_head_dim"],
        )
        if any(
            item is not None and (not isinstance(item, int) or isinstance(item, bool) or item <= 0)
            for item in nullable_dimensions
        ):
            raise ValueError("bake plan QKV dimensions must be null or positive integers")
        if not isinstance(raw["rank"], int) or isinstance(raw["rank"], bool) or raw["rank"] <= 0:
            raise ValueError("bake plan operation rank must be a positive integer")
        if not isinstance(raw["multiplier"], (int, float)) or isinstance(raw["multiplier"], bool):
            raise ValueError("bake plan operation multiplier must be numeric")
        operation = BakeOperation(
            **{
                **raw,
                "shape": tuple(shape),
                "multiplier": float(raw["multiplier"]),
            }
        )
        operations.append(operation)

    if [operation.module for operation in operations] != sorted(expected_modules):
        raise ValueError("bake plan operations do not match canonical module order")
    multiplier = operations[0].multiplier
    for operation in operations:
        shape_a, shape_b = expected_modules[operation.module]
        expected_shape = (shape_b[0], shape_a[1])
        qkv = operation.module.endswith("attn.qkv_proj")
        expected_layout = "grouped-qkv-to-qkv;merge-qkv;qkv-to-grouped-qkv" if qkv else "direct-runtime-layout-merge"
        expected_qkv = (56, 1, 128) if qkv else (None, None, None)
        actual_qkv = (
            operation.qkv_num_query_groups,
            operation.qkv_heads_per_group,
            operation.qkv_head_dim,
        )
        if (
            operation.base_tensor != f"{operation.module}.weight"
            or operation.lora_a != f"{operation.module}.lora_A.weight"
            or operation.lora_b != f"{operation.module}.lora_B.weight"
            or operation.shard not in shard_names
            or operation.dtype != "BF16"
            or operation.shape != expected_shape
            or operation.rank != shape_a[0]
            or not math.isfinite(operation.multiplier)
            or operation.multiplier != multiplier
            or operation.layout_operation != expected_layout
            or actual_qkv != expected_qkv
        ):
            raise ValueError(f"bake plan operation contract mismatch for {operation.module!r}")
    return tuple(operations)


def load_lora_bake_plan_json(path: Path | str) -> tuple[LoraBakePlan, str]:
    """Strictly load and self-check a previously content-bound bake plan receipt."""

    plan_path = Path(path)
    payload, encoded = _read_json_object(plan_path)
    schema = payload.get("schema")
    if schema == BAKE_PLAN_SCHEMA:
        target_dtype = "BF16"
        expected_fields = set(LoraBakePlan.__dataclass_fields__) - {"target_dtype"}
    elif schema == FP16_BAKE_PLAN_SCHEMA:
        target_dtype = payload.get("target_dtype")
        expected_fields = set(LoraBakePlan.__dataclass_fields__)
    else:
        raise ValueError(f"unsupported bake plan schema: {schema!r}")
    if set(payload) != expected_fields:
        raise ValueError("bake plan JSON has an invalid top-level schema")
    expected_profile, expected_schema = _target_contract(target_dtype)
    if schema != expected_schema or payload["profile"] != expected_profile:
        raise ValueError(f"unsupported bake plan profile: {payload['profile']!r}")
    for field in ("base_directory", "normalized_lora"):
        if not isinstance(payload[field], str) or not payload[field]:
            raise ValueError(f"bake plan {field} must be a non-empty string")
    shards = _parse_base_shards(payload["base_shards"])
    operations = _parse_operations(payload["operations"], {shard.name for shard in shards})
    if (
        not isinstance(payload["operation_count"], int)
        or isinstance(payload["operation_count"], bool)
        or payload["operation_count"] != len(operations)
    ):
        raise ValueError("bake plan operation_count does not match operations")
    hashes = {
        field: _require_sha256(payload[field], field)
        for field in (
            "config_sha256",
            "index_sha256",
            "lora_sha256",
            "base_catalog_sha256",
            "plan_sha256",
        )
    }
    if hashes["base_catalog_sha256"] != OFFICIAL_FL2VA_BASE_CONTRACT.catalog_sha256:
        raise ValueError("bake plan base catalog is not the official FL2VA contract")
    recomputed = _plan_sha256(
        profile=payload["profile"],
        config_sha256=hashes["config_sha256"],
        index_sha256=hashes["index_sha256"],
        lora_sha256=hashes["lora_sha256"],
        base_catalog_sha256=hashes["base_catalog_sha256"],
        base_shards=shards,
        operations=operations,
        target_dtype=target_dtype,
    )
    if recomputed != hashes["plan_sha256"]:
        raise ValueError("bake plan canonical SHA256 self-check failed")
    return (
        LoraBakePlan(
            schema=expected_schema,
            profile=payload["profile"],
            base_directory=payload["base_directory"],
            normalized_lora=payload["normalized_lora"],
            config_sha256=hashes["config_sha256"],
            index_sha256=hashes["index_sha256"],
            lora_sha256=hashes["lora_sha256"],
            base_catalog_sha256=hashes["base_catalog_sha256"],
            base_shards=shards,
            plan_sha256=hashes["plan_sha256"],
            operation_count=len(operations),
            operations=operations,
            target_dtype=target_dtype,
        ),
        hashlib.sha256(encoded).hexdigest(),
    )


def plan_official_fl2va_bf16_bake(
    base_directory: Path | str,
    normalized_lora: Path | str,
    *,
    profile: str | None = None,
    scale: float = 1.0,
    target_dtype: str = "BF16",
) -> LoraBakePlan:
    """Validate complete base/adapter coverage and emit a deterministic, header-only bake plan."""

    expected_profile, schema = _target_contract(target_dtype)
    if profile is None:
        profile = expected_profile
    if profile != expected_profile:
        raise ValueError(f"unsupported bake profile: {profile!r}")
    if not math.isfinite(scale):
        raise ValueError("LoRA user scale must be finite")
    base_input = Path(base_directory)
    lora_input = Path(normalized_lora)
    if base_input.is_symlink():
        raise ValueError(f"base directory must not be linked: {base_input}")
    if lora_input.is_symlink():
        raise ValueError(f"normalized LoRA must not be linked: {lora_input}")
    base = base_input.resolve()
    lora = lora_input.resolve()
    if not base.is_dir():
        raise ValueError(f"base directory is missing or linked: {base}")
    if not lora.is_file():
        raise ValueError(f"normalized LoRA is missing or linked: {lora}")

    config, config_payload = _read_json_object(base / CONFIG_NAME)
    index, index_payload = _read_json_object(base / INDEX_NAME)
    _validate_config(config)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model index requires a non-empty weight_map")
    catalog, base_shards = _load_base_catalog(base, weight_map)
    base_catalog_sha256 = _validate_official_base_contract(index, catalog)

    lora_metadata, lora_tensors, lora_sha256 = _read_safetensors_and_hash(lora)
    lora_modules = _validate_normalized_lora(lora_metadata, lora_tensors)
    operations: list[BakeOperation] = []
    for module, (shape_a, shape_b) in sorted(_expected_modules().items()):
        base_name = f"{module}.weight"
        base_entry = catalog.get(base_name)
        if base_entry is None:
            raise ValueError(f"official FL2VA base is missing LoRA target {base_name!r}")
        shard, base_tensor = base_entry
        expected_shape = (shape_b[0], shape_a[1])
        if base_tensor.dtype != "BF16" or base_tensor.shape != expected_shape:
            raise ValueError(
                f"official FL2VA base tensor {base_name!r} must be BF16 {expected_shape}, "
                f"found {base_tensor.dtype} {base_tensor.shape}"
            )
        pair = lora_modules[module]
        layout = (
            "grouped-qkv-to-qkv;merge-qkv;qkv-to-grouped-qkv"
            if module.endswith("attn.qkv_proj")
            else "direct-runtime-layout-merge"
        )
        is_qkv = module.endswith("attn.qkv_proj")
        operations.append(
            BakeOperation(
                module=module,
                base_tensor=base_name,
                shard=shard,
                dtype="BF16",
                shape=expected_shape,
                lora_a=pair["A"].name,
                lora_b=pair["B"].name,
                rank=shape_a[0],
                multiplier=scale,
                layout_operation=layout,
                qkv_num_query_groups=int(config["num_attention_heads"]) if is_qkv else None,
                qkv_heads_per_group=1 if is_qkv else None,
                qkv_head_dim=int(config["attention_head_dim"]) if is_qkv else None,
            )
        )

    config_sha256 = hashlib.sha256(config_payload).hexdigest()
    index_sha256 = hashlib.sha256(index_payload).hexdigest()
    operation_tuple = tuple(operations)
    plan_sha256 = _plan_sha256(
        profile=profile,
        config_sha256=config_sha256,
        index_sha256=index_sha256,
        lora_sha256=lora_sha256,
        base_catalog_sha256=base_catalog_sha256,
        base_shards=base_shards,
        operations=operation_tuple,
        target_dtype=target_dtype,
    )
    return LoraBakePlan(
        schema=schema,
        profile=profile,
        base_directory=str(base),
        normalized_lora=str(lora),
        config_sha256=config_sha256,
        index_sha256=index_sha256,
        lora_sha256=lora_sha256,
        base_catalog_sha256=base_catalog_sha256,
        base_shards=base_shards,
        plan_sha256=plan_sha256,
        operation_count=len(operations),
        operations=operation_tuple,
        target_dtype=target_dtype,
    )
