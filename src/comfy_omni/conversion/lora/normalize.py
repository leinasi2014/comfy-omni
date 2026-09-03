# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright h3-forge contributors
#
# Provenance: wholesale migration from h3-forge@e9cb011d00b028c149db3978de246c54f6e34acc
#   source path: src/h3_forge/lora_hotswap/lora.py
#   source blob: 35083005d8bbc4b187b95455d7980e8a92c55f18
#   license: Apache-2.0
#   attribution: h3-forge contributors
# Migrated byte-preserving except this provenance header, import retargeting, and
# mechanical line wrapping to satisfy the repository line-length (120).
"""Strict, streaming normalization for registered MiniMax H3 LoRA profiles."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

from comfy_omni.artifacts.safetensors import read_safetensors_header_stream
from comfy_omni.domain.checkpoints import TensorDescriptor

TURBO_V4_PROFILE = "h3/fl2va/dit-standard/lora-turbo-v4-full-ab/native-separate/v1"
TURBO_V4_TENSOR_COUNT = 518
TURBO_V4_MODULE_COUNT = 259
COMBAT_V2_PROFILE = "h3/dit-standard/lora-combat-v2-backbone-full-ab/native-overlay/v1"
COMBAT_V2_TENSOR_COUNT = 416
COMBAT_V2_MODULE_COUNT = 208
SOURCE_PREFIX = "diffusion_model."
COPY_CHUNK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class TensorRewrite:
    source_name: str
    target_name: str
    dtype: str
    shape: tuple[int, ...]
    byte_length: int
    payload_sha256: str


@dataclass(frozen=True)
class LoraNormalizationReport:
    profile: str
    source: str
    output: str
    source_sha256: str
    output_sha256: str
    payload_sha256: str
    mapping_sha256: str
    tensor_count: int
    module_count: int
    tensors: tuple[TensorRewrite, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _target_name(source_name: str) -> str:
    if not source_name.startswith(SOURCE_PREFIX):
        raise ValueError(f"unexpected LoRA tensor outside {SOURCE_PREFIX!r}: {source_name!r}")
    target = source_name[len(SOURCE_PREFIX) :]
    if not target:
        raise ValueError(f"empty target tensor name after removing {SOURCE_PREFIX!r}")
    return target


def _module_and_side(name: str) -> tuple[str, str]:
    for marker, side in ((".lora_A.weight", "A"), (".lora_B.weight", "B")):
        if name.endswith(marker):
            return name[: -len(marker)], side
    raise ValueError(f"unsupported Turbo v4 LoRA tensor name: {name!r}")


def _expected_modules() -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    modules: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    block_shapes = {
        "adaln_proj.linear": ((16, 2688), (96768, 16)),
        "attn.qkv_proj": ((64, 5376), (21504, 64)),
        "attn.out_proj": ((64, 7168), (5376, 64)),
        "mlp.fc1": ((64, 5376), (28672, 64)),
        "mlp.fc2": ((64, 14336), (5376, 64)),
    }
    refiner_shapes = {
        "attn.qkv_proj": ((64, 5376), (21504, 64)),
        "attn.out_proj": ((64, 7168), (5376, 64)),
        "mlp.fc1": ((64, 5376), (28672, 64)),
        "mlp.fc2": ((64, 14336), (5376, 64)),
    }
    for block in range(50):
        for suffix, shapes in block_shapes.items():
            modules[f"blocks.{block}.{suffix}"] = shapes
    for block in range(2):
        for suffix, shapes in refiner_shapes.items():
            modules[f"token_refiner.blocks.{block}.{suffix}"] = shapes
    modules["final_layer.adaln_proj.linear"] = ((16, 2688), (10752, 16))
    return modules


def _validate_turbo_v4_contract(
    metadata: dict[str, str],
    tensors: tuple[TensorDescriptor, ...],
) -> tuple[tuple[TensorDescriptor, str], ...]:
    required_metadata = {
        "application": "W_eff = W + lora_B @ lora_A",
        "base_model": "MiniMax-H3",
        "dtype": "bfloat16",
        "sampler_steps": "4",
    }
    for key, expected in required_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Turbo v4 metadata {key!r} must be {expected!r}, found {metadata.get(key)!r}")
    if len(tensors) != TURBO_V4_TENSOR_COUNT:
        raise ValueError(f"Turbo v4 full LoRA requires {TURBO_V4_TENSOR_COUNT} tensors, found {len(tensors)}")

    rewrites: list[tuple[TensorDescriptor, str]] = []
    targets: set[str] = set()
    expected_modules = _expected_modules()
    if len(expected_modules) != TURBO_V4_MODULE_COUNT:
        raise RuntimeError("internal Turbo v4 module contract is inconsistent")
    modules: dict[str, set[str]] = {}
    for tensor in tensors:
        if tensor.dtype != "BF16":
            raise ValueError(f"Turbo v4 full LoRA requires BF16 tensors: {tensor.name!r} is {tensor.dtype}")
        target = _target_name(tensor.name)
        if target in targets:
            raise ValueError(f"multiple source tensors map to {target!r}")
        targets.add(target)
        module, side = _module_and_side(target)
        if module not in expected_modules:
            raise ValueError(f"unexpected Turbo v4 LoRA module: {module!r}")
        expected_shape = expected_modules[module][0 if side == "A" else 1]
        if tensor.shape != expected_shape:
            raise ValueError(f"Turbo v4 tensor {tensor.name!r} has shape {tensor.shape}, expected {expected_shape}")
        modules.setdefault(module, set()).add(side)
        rewrites.append((tensor, target))

    incomplete = sorted(module for module, sides in modules.items() if sides != {"A", "B"})
    if incomplete:
        raise ValueError(f"Turbo v4 LoRA has incomplete A/B pairs; first incomplete module: {incomplete[0]!r}")
    if len(modules) != TURBO_V4_MODULE_COUNT:
        raise ValueError(f"Turbo v4 full LoRA requires {TURBO_V4_MODULE_COUNT} A/B modules, found {len(modules)}")
    return tuple(sorted(rewrites, key=lambda item: item[0].data_offsets))


def _combat_v2_expected_modules() -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Return the exact H3 backbone covered by Combat V2.

    Combat V2 deliberately excludes the 51 curve-AdaLN modules.  Its rank is
    16 rather than Turbo v4's rank 64, while the base matrix dimensions are
    identical.
    """

    return {
        module: ((16, shapes[0][1]), (shapes[1][0], 16))
        for module, shapes in _expected_modules().items()
        if not module.endswith("adaln_proj.linear")
    }


def _validate_combat_v2_contract(
    metadata: dict[str, str],
    tensors: tuple[TensorDescriptor, ...],
) -> tuple[tuple[TensorDescriptor, str], ...]:
    required_metadata = {
        "ss_base_model_version": "minimax_h3",
        "ss_output_name": "doV2_copy",
        "format": "pt",
    }
    for key, expected in required_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Combat V2 metadata {key!r} must be {expected!r}, found {metadata.get(key)!r}")
    if len(tensors) != COMBAT_V2_TENSOR_COUNT:
        raise ValueError(f"Combat V2 requires {COMBAT_V2_TENSOR_COUNT} tensors, found {len(tensors)}")

    expected_modules = _combat_v2_expected_modules()
    if len(expected_modules) != COMBAT_V2_MODULE_COUNT:
        raise RuntimeError("internal Combat V2 module contract is inconsistent")
    rewrites: list[tuple[TensorDescriptor, str]] = []
    targets: set[str] = set()
    modules: dict[str, set[str]] = {}
    for tensor in tensors:
        if tensor.dtype != "BF16":
            raise ValueError(f"Combat V2 requires BF16 tensors: {tensor.name!r} is {tensor.dtype}")
        target = _target_name(tensor.name)
        if target in targets:
            raise ValueError(f"multiple source tensors map to {target!r}")
        targets.add(target)
        module, side = _module_and_side(target)
        if module not in expected_modules:
            raise ValueError(f"unexpected Combat V2 LoRA module: {module!r}")
        expected_shape = expected_modules[module][0 if side == "A" else 1]
        if tensor.shape != expected_shape:
            raise ValueError(f"Combat V2 tensor {tensor.name!r} has shape {tensor.shape}, expected {expected_shape}")
        modules.setdefault(module, set()).add(side)
        rewrites.append((tensor, target))
    incomplete = sorted(module for module, sides in modules.items() if sides != {"A", "B"})
    if incomplete:
        raise ValueError(f"Combat V2 has incomplete A/B pairs; first incomplete module: {incomplete[0]!r}")
    if set(modules) != set(expected_modules):
        raise ValueError(f"Combat V2 requires {COMBAT_V2_MODULE_COUNT} backbone modules, found {len(modules)}")
    return tuple(sorted(rewrites, key=lambda item: item[0].data_offsets))


def _encode_header(
    rewrites: tuple[tuple[TensorDescriptor, str], ...],
    *,
    profile: str = TURBO_V4_PROFILE,
) -> bytes:
    output_metadata = {
        "application": "W_eff = W + lora_B @ lora_A",
        "base_model": "MiniMax-H3",
        "dtype": "bfloat16",
        "h3_comfy.operation": f"remove-prefix:{SOURCE_PREFIX}",
        "h3_comfy.payload": "byte-identical",
        "h3_comfy.profile": profile,
        "sampler_steps": "4",
    }
    header: dict[str, object] = {"__metadata__": output_metadata}
    for tensor, target in rewrites:
        header[target] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": list(tensor.data_offsets),
        }
    encoded = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return encoded + (b" " * (-len(encoded) % 8))


def _write_all(output: BinaryIO, payload: bytes | memoryview) -> None:
    view = memoryview(payload)
    while view:
        written = output.write(view)
        if written is None or written <= 0:
            raise OSError("output stream made no progress during write")
        view = view[written:]


def _copy_exact(
    source: BinaryIO,
    output: BinaryIO,
    byte_length: int,
    source_hash: Any,
    output_hash: Any,
    payload_hash: Any,
    tensor_hash: Any,
) -> None:
    remaining = byte_length
    while remaining:
        chunk = source.read(min(remaining, COPY_CHUNK_BYTES))
        if not chunk:
            raise ValueError("source changed or was truncated during conversion")
        _write_all(output, chunk)
        source_hash.update(chunk)
        output_hash.update(chunk)
        payload_hash.update(chunk)
        tensor_hash.update(chunk)
        remaining -= len(chunk)


def _snapshot_stat(stat: os.stat_result) -> tuple[int, int, int, int]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _snapshot(path: Path) -> tuple[int, int, int, int]:
    return _snapshot_stat(path.stat())


def _verify_output(
    path: Path,
    expected: tuple[TensorRewrite, ...],
    expected_output_sha256: str,
) -> None:
    whole_file_hasher = hashlib.sha256()
    with path.open("rb") as stream:
        descriptor_snapshot = _snapshot_stat(os.fstat(stream.fileno()))
        _, tensors, header_length = read_safetensors_header_stream(stream, path, descriptor_snapshot[2])
        ordered = tuple(sorted(tensors, key=lambda tensor: tensor.data_offsets))
        if len(ordered) != len(expected):
            raise ValueError("output tensor count changed during verification")
        stream.seek(0)
        preamble = stream.read(8 + header_length)
        if len(preamble) != 8 + header_length:
            raise ValueError("output was truncated during verification")
        whole_file_hasher.update(preamble)
        for actual, wanted in zip(ordered, expected, strict=True):
            if (actual.name, actual.dtype, actual.shape) != (wanted.target_name, wanted.dtype, wanted.shape):
                raise ValueError(f"output descriptor mismatch for {wanted.target_name!r}")
            tensor_hasher = hashlib.sha256()
            remaining = wanted.byte_length
            while remaining:
                chunk = stream.read(min(remaining, COPY_CHUNK_BYTES))
                if not chunk:
                    raise ValueError(f"output payload was truncated at {wanted.target_name!r}")
                whole_file_hasher.update(chunk)
                tensor_hasher.update(chunk)
                remaining -= len(chunk)
            if tensor_hasher.hexdigest() != wanted.payload_sha256:
                raise ValueError(f"output payload digest mismatch for {wanted.target_name!r}")
        if stream.read(1):
            raise ValueError("output contains trailing bytes after verification")
        if _snapshot_stat(os.fstat(stream.fileno())) != descriptor_snapshot:
            raise ValueError("output changed during verification")
    if whole_file_hasher.hexdigest() != expected_output_sha256:
        raise ValueError("output file digest changed during verification")


def normalize_registered_lora(
    source: Path | str, output: Path | str, *, profile: str = TURBO_V4_PROFILE
) -> LoraNormalizationReport:
    """Normalize a registered Comfy LoRA without materializing tensor payloads."""

    if profile not in {TURBO_V4_PROFILE, COMBAT_V2_PROFILE}:
        raise ValueError(f"unsupported LoRA profile: {profile!r}")
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    if source_path == output_path:
        raise ValueError("source and output must be different paths")
    if source_path.suffix.lower() != ".safetensors" or output_path.suffix.lower() != ".safetensors":
        raise ValueError("source and output must use the .safetensors extension")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")

    temporary_path: Path | None = None
    tensor_reports: list[TensorRewrite] = []
    source_hasher = hashlib.sha256()
    payload_hasher = hashlib.sha256()
    output_hasher = hashlib.sha256()
    mapping_sha256 = ""
    module_count = 0

    try:
        with source_path.open("rb") as source_stream:
            source_snapshot = _snapshot(source_path)
            descriptor_snapshot = _snapshot_stat(os.fstat(source_stream.fileno()))
            if descriptor_snapshot != source_snapshot:
                raise ValueError("source path changed while it was being opened")
            metadata, tensors, source_header_length = read_safetensors_header_stream(
                source_stream, source_path, descriptor_snapshot[2]
            )
            if profile == TURBO_V4_PROFILE:
                rewrites = _validate_turbo_v4_contract(metadata, tensors)
                module_count = TURBO_V4_MODULE_COUNT
            else:
                rewrites = _validate_combat_v2_contract(metadata, tensors)
                module_count = COMBAT_V2_MODULE_COUNT
            encoded_header = _encode_header(rewrites, profile=profile)
            output_preamble = struct.pack("<Q", len(encoded_header)) + encoded_header
            output_hasher.update(output_preamble)
            mapping_payload = json.dumps(
                [(tensor.name, target, tensor.dtype, tensor.shape) for tensor, target in rewrites],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            mapping_sha256 = hashlib.sha256(mapping_payload).hexdigest()

            source_stream.seek(0)
            source_preamble = source_stream.read(8 + source_header_length)
            if len(source_preamble) != 8 + source_header_length:
                raise ValueError("source changed or was truncated during conversion")
            source_hasher.update(source_preamble)

            output_path.parent.mkdir(parents=True, exist_ok=True)

            with tempfile.NamedTemporaryFile(
                mode="x+b", prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent, delete=False
            ) as output_stream:
                temporary_path = Path(output_stream.name)
                _write_all(output_stream, output_preamble)
                cursor = 0
                for tensor, target in rewrites:
                    start, end = tensor.data_offsets
                    if start != cursor:
                        raise ValueError(f"unexpected payload cursor before {tensor.name!r}")
                    tensor_hasher = hashlib.sha256()
                    _copy_exact(
                        source_stream,
                        output_stream,
                        end - start,
                        source_hasher,
                        output_hasher,
                        payload_hasher,
                        tensor_hasher,
                    )
                    tensor_reports.append(
                        TensorRewrite(
                            source_name=tensor.name,
                            target_name=target,
                            dtype=tensor.dtype,
                            shape=tensor.shape,
                            byte_length=end - start,
                            payload_sha256=tensor_hasher.hexdigest(),
                        )
                    )
                    cursor = end
                if source_stream.read(1):
                    raise ValueError("source contains trailing data not covered by the validated header")
                output_stream.flush()
                os.fsync(output_stream.fileno())

            if _snapshot_stat(os.fstat(source_stream.fileno())) != descriptor_snapshot:
                raise ValueError("source changed during conversion")

        if _snapshot(source_path) != source_snapshot:
            raise ValueError("source changed during conversion")
        if temporary_path is None:
            raise RuntimeError("conversion staging output was not created")
        _verify_output(temporary_path, tuple(tensor_reports), output_hasher.hexdigest())
        os.link(temporary_path, output_path)
        published_temporary_path = temporary_path
        temporary_path = None
        try:
            published_temporary_path.unlink()
        except OSError:
            # The verified output is already atomically published. A stale same-inode
            # staging link is recoverable and must not turn success into a false failure.
            pass
    except BaseException:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise

    return LoraNormalizationReport(
        profile=profile,
        source=str(source_path),
        output=str(output_path),
        source_sha256=source_hasher.hexdigest(),
        output_sha256=output_hasher.hexdigest(),
        payload_sha256=payload_hasher.hexdigest(),
        mapping_sha256=mapping_sha256,
        tensor_count=len(tensor_reports),
        module_count=module_count,
        tensors=tuple(tensor_reports),
    )


def normalize_turbo_v4_lora(
    source: Path | str, output: Path | str, *, profile: str = TURBO_V4_PROFILE
) -> LoraNormalizationReport:
    """Backward-compatible wrapper for registered LoRA normalization."""

    return normalize_registered_lora(source, output, profile=profile)
