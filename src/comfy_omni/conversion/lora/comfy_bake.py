# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright h3-forge contributors
#
# Provenance: wholesale migration from h3-forge@e9cb011d00b028c149db3978de246c54f6e34acc
#   source path: src/h3_forge/lora_hotswap/comfy_lora_bake.py
#   source blob: 323794f758692eb5549d285dec69c7e3ff6591a9
#   license: Apache-2.0
#   attribution: h3-forge contributors
# Migrated byte-preserving except this provenance header, import retargeting, and
# mechanical line wrapping to satisfy the repository line-length (120).
"""Bounded offline Turbo v4 LoRA folding for decoded H3 weights.

This module deliberately operates on the runtime ``Q|K|V`` layout.  The
native exporter owns the later runtime-to-grouped serialization transform.
No function in this module is registered in, or called from, the denoising
path.
"""

from __future__ import annotations

import hashlib
import math
import os
import stat
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from comfy_omni.artifacts.safetensors import read_safetensors_header_stream
from comfy_omni.domain.checkpoints import TensorDescriptor

from .bake_plan import _validate_normalized_lora
from .normalize import (
    COMBAT_V2_MODULE_COUNT,
    COMBAT_V2_PROFILE,
    COMBAT_V2_TENSOR_COUNT,
    TURBO_V4_MODULE_COUNT,
    TURBO_V4_PROFILE,
    TURBO_V4_TENSOR_COUNT,
    _combat_v2_expected_modules,
    _expected_modules,
    _module_and_side,
)

DEFAULT_CHUNK_ROWS = 64
MAX_CHUNK_ROWS = 256


class ComfyLoraBakeContractError(ValueError):
    """The LoRA, decoded base, or requested fold violates the frozen profile."""


def _module_sets() -> tuple[frozenset[str], frozenset[str]]:
    expected = frozenset(_expected_modules())
    quantized = frozenset(
        module
        for module in expected
        if module.endswith(("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"))
        and not module.startswith("token_refiner.")
    )
    dense = expected - quantized
    if len(expected) != 259 or len(quantized) != 200 or len(dense) != 59:
        raise RuntimeError("internal Turbo v4 bake module census is inconsistent")
    return quantized, dense


TURBO_V4_QUANTIZED_MODULES, TURBO_V4_DENSE_MODULES = _module_sets()
TURBO_V4_ALL_MODULES = TURBO_V4_QUANTIZED_MODULES | TURBO_V4_DENSE_MODULES
TURBO_V4_ADALN_MODULES = frozenset(module for module in TURBO_V4_ALL_MODULES if module.endswith("adaln_proj.linear"))
TURBO_V4_BACKBONE_MODULES = TURBO_V4_ALL_MODULES - TURBO_V4_ADALN_MODULES
if len(TURBO_V4_ADALN_MODULES) != 51 or len(TURBO_V4_BACKBONE_MODULES) != 208:
    raise RuntimeError("internal Turbo v4 backbone/AdaLN module census is inconsistent")


class DecodedWeightFoldCallback(Protocol):
    """Callback shape consumed by the offline native exporter."""

    def __call__(
        self,
        module: str,
        decoded_runtime_weight: Any,
        *,
        source_was_convrot: bool,
    ) -> Any: ...


@dataclass(frozen=True)
class ComfyLoraBakeReport:
    module_count: int
    quantized_module_count: int
    dense_module_count: int
    folded_modules: tuple[str, ...]
    alpha: float | None
    scale: float
    use_adaln_cache: bool
    in_place: bool = True
    qkv_input_layout: str = "runtime-qkv"
    qkv_output_layout: str = "runtime-qkv"
    normalized_lora_path: str = ""
    normalized_lora_size: int = 0
    normalized_lora_sha256: str = ""
    profile: str = TURBO_V4_PROFILE
    alpha_policy: str = "PER_MODULE_RANK_FROM_ALPHA_NONE"
    float32_matmul_precision: str = "highest"
    isolated_conversion_process_required: bool = True
    fold_device: str = "cpu"
    torch_version: str = ""


def validate_turbo_v4_descriptors(
    metadata: Mapping[str, str],
    tensors: Sequence[TensorDescriptor],
) -> dict[str, dict[str, TensorDescriptor]]:
    """Reuse and tighten the normalized 259-module/518-tensor contract."""

    tensor_tuple = tuple(tensors)
    if len(tensor_tuple) != TURBO_V4_TENSOR_COUNT:
        raise ComfyLoraBakeContractError(
            f"normalized Turbo v4 requires {TURBO_V4_TENSOR_COUNT} tensors, found {len(tensor_tuple)}"
        )
    try:
        modules = _validate_normalized_lora(dict(metadata), tensor_tuple)
    except ValueError as exc:
        raise ComfyLoraBakeContractError(str(exc)) from exc
    if set(modules) != TURBO_V4_ALL_MODULES or len(modules) != TURBO_V4_MODULE_COUNT:
        raise ComfyLoraBakeContractError("normalized Turbo v4 module coverage is not exact")
    tensor_names = {descriptor.name for sides in modules.values() for descriptor in sides.values()}
    if len(tensor_names) != TURBO_V4_TENSOR_COUNT:
        raise ComfyLoraBakeContractError("normalized Turbo v4 tensor names are not unique")
    return modules


def validate_combat_v2_descriptors(
    metadata: Mapping[str, str],
    tensors: Sequence[TensorDescriptor],
) -> dict[str, dict[str, TensorDescriptor]]:
    """Validate the normalized 208-module Combat V2 overlay contract."""

    if metadata.get("h3_comfy.profile") != COMBAT_V2_PROFILE:
        raise ComfyLoraBakeContractError("normalized Combat V2 profile metadata is invalid")
    if (
        metadata.get("application") != "W_eff = W + lora_B @ lora_A"
        or metadata.get("base_model") != "MiniMax-H3"
        or metadata.get("dtype") != "bfloat16"
        or metadata.get("h3_comfy.payload") != "byte-identical"
    ):
        raise ComfyLoraBakeContractError("normalized Combat V2 metadata is invalid")
    if len(tensors) != COMBAT_V2_TENSOR_COUNT:
        raise ComfyLoraBakeContractError(
            f"normalized Combat V2 requires {COMBAT_V2_TENSOR_COUNT} tensors, found {len(tensors)}"
        )
    expected = _combat_v2_expected_modules()
    modules: dict[str, dict[str, TensorDescriptor]] = {}
    for descriptor in tensors:
        if descriptor.dtype != "BF16":
            raise ComfyLoraBakeContractError(f"normalized Combat V2 tensor {descriptor.name!r} is not BF16")
        try:
            module, side = _module_and_side(descriptor.name)
        except ValueError as exc:
            raise ComfyLoraBakeContractError(str(exc)) from exc
        if module not in expected:
            raise ComfyLoraBakeContractError(f"unexpected normalized Combat V2 module: {module!r}")
        shape = expected[module][0 if side == "A" else 1]
        if descriptor.shape != shape:
            raise ComfyLoraBakeContractError(
                f"normalized Combat V2 tensor {descriptor.name!r} has shape {descriptor.shape}, expected {shape}"
            )
        pair = modules.setdefault(module, {})
        if side in pair:
            raise ComfyLoraBakeContractError(f"duplicate normalized Combat V2 {side} tensor for {module!r}")
        pair[side] = descriptor
    if set(modules) != set(expected) or len(modules) != COMBAT_V2_MODULE_COUNT:
        raise ComfyLoraBakeContractError("normalized Combat V2 module coverage is not exact")
    incomplete = sorted(module for module, pair in modules.items() if set(pair) != {"A", "B"})
    if incomplete:
        raise ComfyLoraBakeContractError(f"normalized Combat V2 A/B pair is incomplete: {incomplete[0]!r}")
    return modules


def _validate_scalar_contract(alpha: float | None, scale: float) -> float | None:
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale):
        raise ComfyLoraBakeContractError("LoRA scale must be a finite number")
    if float(scale) != 1.0:
        raise ComfyLoraBakeContractError("the Turbo v4 product slice requires scale=1")
    if alpha is None:
        return None
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha):
        raise ComfyLoraBakeContractError("LoRA alpha must be a finite number or None")
    return float(alpha)


def fold_lora_weight_rows(
    decoded_runtime_weight: Any,
    lora_a: Any,
    lora_b_runtime: Any,
    *,
    alpha: float | None = None,
    scale: float = 1.0,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    in_place: bool = False,
    torch_module: Any | None = None,
) -> Any:
    """Compute ``W + scale * alpha/rank * (B @ A)`` with bounded FP32 rows.

    ``alpha=None`` has the Comfy meaning ``alpha=rank``.  The base and LoRA B
    are both runtime-layout tensors; in particular, QKV B is never permuted.
    Each output row is cast exactly once back to the decoded base dtype.
    """

    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:  # pragma: no cover - deployment dependent
            raise ComfyLoraBakeContractError("torch is required for LoRA folding") from exc
    torch = torch_module
    if not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool) or not 1 <= chunk_rows <= MAX_CHUNK_ROWS:
        raise ComfyLoraBakeContractError(f"chunk_rows must be an integer in 1..{MAX_CHUNK_ROWS}")
    alpha_value = _validate_scalar_contract(alpha, scale)
    if not isinstance(decoded_runtime_weight, torch.Tensor):
        raise ComfyLoraBakeContractError("decoded runtime weight must be a torch tensor")
    if not isinstance(lora_a, torch.Tensor) or not isinstance(lora_b_runtime, torch.Tensor):
        raise ComfyLoraBakeContractError("LoRA A and B must be torch tensors")
    if decoded_runtime_weight.ndim != 2 or lora_a.ndim != 2 or lora_b_runtime.ndim != 2:
        raise ComfyLoraBakeContractError("base, LoRA A, and LoRA B must all be rank-2")
    supported_dtypes = {torch.bfloat16, torch.float16, torch.float32}
    if decoded_runtime_weight.dtype not in supported_dtypes:
        raise ComfyLoraBakeContractError("decoded runtime weight must be BF16, FP16, or FP32")
    rank = int(lora_a.shape[0])
    expected_a = (rank, int(decoded_runtime_weight.shape[1]))
    expected_b = (int(decoded_runtime_weight.shape[0]), rank)
    if rank <= 0 or tuple(lora_a.shape) != expected_a:
        raise ComfyLoraBakeContractError(f"LoRA A shape {tuple(lora_a.shape)} does not match {expected_a}")
    if tuple(lora_b_runtime.shape) != expected_b:
        raise ComfyLoraBakeContractError(f"LoRA B shape {tuple(lora_b_runtime.shape)} does not match {expected_b}")
    multiplier = float(scale) * (float(rank) if alpha_value is None else alpha_value) / float(rank)
    result = decoded_runtime_weight if in_place else decoded_runtime_weight.clone()
    a_fp32 = lora_a.detach().to(device=result.device, dtype=torch.float32)
    if not bool(torch.isfinite(a_fp32).all()):
        raise ComfyLoraBakeContractError("LoRA A contains non-finite values")
    with torch.no_grad():
        for start in range(0, result.shape[0], chunk_rows):
            end = min(start + chunk_rows, result.shape[0])
            b_rows = lora_b_runtime[start:end].detach().to(device=result.device, dtype=torch.float32)
            base_rows = result[start:end].to(dtype=torch.float32)
            if not bool(torch.isfinite(base_rows).all()):
                raise ComfyLoraBakeContractError("decoded runtime weight contains non-finite values")
            if not bool(torch.isfinite(b_rows).all()):
                raise ComfyLoraBakeContractError("LoRA B contains non-finite values")
            delta = torch.matmul(b_rows, a_fp32)
            base_rows.add_(delta, alpha=multiplier)
            cast_rows = base_rows.to(dtype=result.dtype)
            if not bool(torch.isfinite(cast_rows.float()).all()):
                raise ComfyLoraBakeContractError("LoRA fold produced non-finite values")
            result[start:end].copy_(cast_rows)
    return result


def _is_link_or_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    junction = getattr(path, "is_junction", None)
    if junction is not None and junction():
        return True
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


class _NormalizedLoraReader:
    def __init__(
        self,
        path: Path,
        *,
        expected_sha256: str,
        profile: str = TURBO_V4_PROFILE,
    ) -> None:
        if len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256):
            raise ComfyLoraBakeContractError("expected LoRA SHA256 must be 64 lowercase hex characters")
        requested = Path(os.path.abspath(path))
        if _is_link_or_reparse(requested) or not requested.is_file():
            raise ComfyLoraBakeContractError(f"normalized LoRA must be a non-linked regular file: {requested}")
        self.path = requested
        self._stream = requested.open("rb", buffering=0)
        self._lock = threading.Lock()
        opened = os.fstat(self._stream.fileno())
        before = requested.stat()
        self._identity = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        if self._identity != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns):
            self._stream.close()
            raise ComfyLoraBakeContractError("normalized LoRA changed while opening")
        try:
            metadata, tensors, header_length = read_safetensors_header_stream(self._stream, requested, opened.st_size)
            if profile == TURBO_V4_PROFILE:
                self.modules = validate_turbo_v4_descriptors(metadata, tensors)
            elif profile == COMBAT_V2_PROFILE:
                self.modules = validate_combat_v2_descriptors(metadata, tensors)
            else:
                raise ComfyLoraBakeContractError(f"unsupported normalized LoRA reader profile: {profile!r}")
            self.profile = profile
            self._payload_offset = 8 + header_length
            self.size = opened.st_size
            self.sha256 = self._hash_open_file()
            if self.sha256 != expected_sha256:
                raise ComfyLoraBakeContractError(
                    f"normalized LoRA SHA256 mismatch: expected {expected_sha256}, found {self.sha256}"
                )
        except Exception:
            self._stream.close()
            raise

    def _hash_open_file(self) -> str:
        digest = hashlib.sha256()
        with self._lock:
            self._stream.seek(0)
            while chunk := self._stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def _read_exact(self, offset: int, byte_count: int) -> bytes:
        if hasattr(os, "pread"):
            raw = os.pread(self._stream.fileno(), byte_count, offset)
        else:  # pragma: no cover - legacy Windows Python
            with self._lock:
                self._stream.seek(offset)
                raw = self._stream.read(byte_count)
        if len(raw) != byte_count:
            raise ComfyLoraBakeContractError("normalized LoRA changed or was truncated")
        return raw

    def read_bf16_rows(
        self,
        descriptor: TensorDescriptor,
        start: int,
        end: int,
        *,
        torch: Any,
        device: Any,
    ) -> Any:
        if descriptor.dtype != "BF16" or len(descriptor.shape) != 2:
            raise ComfyLoraBakeContractError(f"LoRA tensor {descriptor.name!r} is not a BF16 matrix")
        rows, columns = descriptor.shape
        if not 0 <= start <= end <= rows:
            raise ComfyLoraBakeContractError("LoRA row request is outside the tensor")
        row_bytes = columns * 2
        begin = self._payload_offset + descriptor.data_offsets[0] + start * row_bytes
        raw = self._read_exact(begin, (end - start) * row_bytes)
        value = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(end - start, columns)
        return value.to(device=device)

    def verify_unchanged(self) -> None:
        opened = os.fstat(self._stream.fileno())
        current = self.path.stat()
        opened_identity = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        path_identity = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
        if _is_link_or_reparse(self.path) or opened_identity != self._identity or path_identity != self._identity:
            raise ComfyLoraBakeContractError("normalized LoRA identity changed during folding")
        if self._hash_open_file() != self.sha256:
            raise ComfyLoraBakeContractError("normalized LoRA contents changed during folding")

    def close(self) -> None:
        self._stream.close()


class TurboV4ComfyLoraBake:
    """Strict stateful callback for one complete offline native export."""

    def __init__(
        self,
        normalized_lora: Path | str,
        *,
        expected_sha256: str,
        alpha: float | None = None,
        scale: float = 1.0,
        chunk_rows: int = DEFAULT_CHUNK_ROWS,
        use_adaln_cache: bool = False,
    ) -> None:
        if not isinstance(use_adaln_cache, bool):
            raise ComfyLoraBakeContractError("use_adaln_cache must be a boolean")
        if not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool) or not 1 <= chunk_rows <= MAX_CHUNK_ROWS:
            raise ComfyLoraBakeContractError(f"chunk_rows must be an integer in 1..{MAX_CHUNK_ROWS}")
        self.alpha = _validate_scalar_contract(alpha, scale)
        if self.alpha is not None:
            raise ComfyLoraBakeContractError(
                "the Turbo v4 product profile requires alpha=None so each module uses its own rank"
            )
        self.scale = float(scale)
        self.chunk_rows = chunk_rows
        self.use_adaln_cache = use_adaln_cache
        torch = __import__("torch")
        self._previous_matmul_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        if torch.get_float32_matmul_precision() != "highest":
            raise ComfyLoraBakeContractError("failed to freeze float32 matmul precision to highest")
        try:
            self._reader = _NormalizedLoraReader(Path(normalized_lora), expected_sha256=expected_sha256)
        except Exception:
            torch.set_float32_matmul_precision(self._previous_matmul_precision)
            raise
        self._folded: set[str] = set()
        self._closed = False
        self._poisoned = False

    @property
    def pending_modules(self) -> tuple[str, ...]:
        target = TURBO_V4_BACKBONE_MODULES if self.use_adaln_cache else TURBO_V4_ALL_MODULES
        return tuple(sorted(target - self._folded))

    def source_was_convrot(self, module: str) -> bool:
        target = TURBO_V4_BACKBONE_MODULES if self.use_adaln_cache else TURBO_V4_ALL_MODULES
        if module not in target:
            raise ComfyLoraBakeContractError(f"unexpected Turbo v4 module: {module!r}")
        return module in TURBO_V4_QUANTIZED_MODULES

    def __call__(
        self,
        module: str,
        decoded_runtime_weight: Any,
        *,
        source_was_convrot: bool,
    ) -> Any:
        if self._closed:
            raise ComfyLoraBakeContractError("LoRA bake callback is closed")
        if self._poisoned:
            raise ComfyLoraBakeContractError("LoRA bake callback is poisoned after a prior fold failure")
        target = TURBO_V4_BACKBONE_MODULES if self.use_adaln_cache else TURBO_V4_ALL_MODULES
        if module not in target:
            raise ComfyLoraBakeContractError(f"unexpected Turbo v4 module: {module!r}")
        if module in self._folded:
            raise ComfyLoraBakeContractError(f"Turbo v4 module was folded more than once: {module!r}")
        expected_convrot = module in TURBO_V4_QUANTIZED_MODULES
        if source_was_convrot is not expected_convrot:
            expected_kind = "ConvRot-decoded" if expected_convrot else "non-quantized"
            raise ComfyLoraBakeContractError(f"Turbo v4 module {module!r} must use a {expected_kind} base")
        torch = __import__("torch")
        if not isinstance(decoded_runtime_weight, torch.Tensor):
            raise ComfyLoraBakeContractError("decoded runtime weight must be a torch tensor")
        if decoded_runtime_weight.dtype is not torch.bfloat16:
            raise ComfyLoraBakeContractError("the Turbo v4 product base weight must be BF16")
        if decoded_runtime_weight.device.type != "cpu":
            raise ComfyLoraBakeContractError("the frozen Turbo v4 conversion process requires CPU weight folding")
        shape_a, shape_b = _expected_modules()[module]
        expected_base = (shape_b[0], shape_a[1])
        if tuple(getattr(decoded_runtime_weight, "shape", ())) != expected_base:
            raise ComfyLoraBakeContractError(
                f"base tensor for {module!r} has shape "
                f"{tuple(getattr(decoded_runtime_weight, 'shape', ()))}, expected {expected_base}"
            )
        self._poisoned = True
        try:
            if torch.get_float32_matmul_precision() != "highest":
                raise ComfyLoraBakeContractError("float32 matmul precision changed during the isolated conversion")
            pair = self._reader.modules[module]
            a = self._reader.read_bf16_rows(pair["A"], 0, shape_a[0], torch=torch, device=decoded_runtime_weight.device)
            # The decoded tensor is exporter-owned and ephemeral. Mutating it
            # avoids a second full H3 matrix; only A and one row window are extra.
            result = decoded_runtime_weight
            a_fp32 = a.to(dtype=torch.float32)
            if not bool(torch.isfinite(a_fp32).all()):
                raise ComfyLoraBakeContractError(f"LoRA A for {module!r} contains non-finite values")
            with torch.no_grad():
                for start in range(0, expected_base[0], self.chunk_rows):
                    end = min(start + self.chunk_rows, expected_base[0])
                    b_rows = self._reader.read_bf16_rows(pair["B"], start, end, torch=torch, device=result.device).to(
                        dtype=torch.float32
                    )
                    rows = result[start:end].to(dtype=torch.float32)
                    if not bool(torch.isfinite(rows).all()):
                        raise ComfyLoraBakeContractError(
                            f"decoded runtime weight for {module!r} contains non-finite values"
                        )
                    if not bool(torch.isfinite(b_rows).all()):
                        raise ComfyLoraBakeContractError(f"LoRA B for {module!r} contains non-finite values")
                    rows.add_(torch.matmul(b_rows, a_fp32), alpha=self.scale)
                    cast_rows = rows.to(dtype=result.dtype)
                    if not bool(torch.isfinite(cast_rows.float()).all()):
                        raise ComfyLoraBakeContractError(f"LoRA fold for {module!r} produced non-finite values")
                    result[start:end].copy_(cast_rows)
        except Exception:
            raise
        else:
            self._poisoned = False
        self._folded.add(module)
        return result

    def assert_complete(self) -> ComfyLoraBakeReport:
        if self._closed:
            raise ComfyLoraBakeContractError("LoRA bake callback is closed")
        if self._poisoned:
            raise ComfyLoraBakeContractError("LoRA bake callback is poisoned")
        if self.pending_modules:
            raise ComfyLoraBakeContractError(
                f"Turbo v4 bake is incomplete; first missing module: {self.pending_modules[0]!r}"
            )
        self._reader.verify_unchanged()
        torch = __import__("torch")
        return ComfyLoraBakeReport(
            module_count=len(self._folded),
            quantized_module_count=len(self._folded & TURBO_V4_QUANTIZED_MODULES),
            dense_module_count=len(self._folded & TURBO_V4_DENSE_MODULES),
            folded_modules=tuple(sorted(self._folded)),
            alpha=self.alpha,
            scale=self.scale,
            use_adaln_cache=self.use_adaln_cache,
            normalized_lora_path=str(self._reader.path),
            normalized_lora_size=self._reader.size,
            normalized_lora_sha256=self._reader.sha256,
            torch_version=str(torch.__version__),
        )

    def close(self) -> None:
        if not self._closed:
            self._reader.close()
            torch = __import__("torch")
            torch.set_float32_matmul_precision(self._previous_matmul_precision)
            self._closed = True

    def __enter__(self) -> TurboV4ComfyLoraBake:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if exc_type is None:
                self.assert_complete()
        finally:
            self.close()


def open_turbo_v4_comfy_lora_bake(
    normalized_lora: Path | str,
    *,
    expected_sha256: str,
    alpha: float | None = None,
    scale: float = 1.0,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    use_adaln_cache: bool = False,
) -> TurboV4ComfyLoraBake:
    """Create the callback passed to the offline native checkpoint exporter."""

    return TurboV4ComfyLoraBake(
        normalized_lora,
        expected_sha256=expected_sha256,
        alpha=alpha,
        scale=scale,
        chunk_rows=chunk_rows,
        use_adaln_cache=use_adaln_cache,
    )
