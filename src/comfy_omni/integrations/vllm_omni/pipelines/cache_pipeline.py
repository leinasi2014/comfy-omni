# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: h3-forge contributors
# Derived from h3-forge@e9cb011d00b028c149db3978de246c54f6e34acc; Apache-2.0.
# Source: src/h3_forge/h3/runtime_pipeline.py; blob fa94f86da746ff9a11105584081464c1162d07b6.
"""Cache-only MiniMax H3 runtime for the pinned official vLLM-Omni host."""

from __future__ import annotations

import copy
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import torch
import torch.nn as nn
from vllm_omni.diffusion.forward_context import get_forward_context
from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as official_transformer
from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as official_pipeline
from vllm_omni.diffusion.models.minimax_h3.denoise_loop import (
    MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    MINIMAX_H3_IMGVID_COND_TIMESTEP,
)
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
    MiniMaxH3DiTModel as OfficialMiniMaxH3DiTModel,
)
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
    MiniMaxH3Pipeline as OfficialMiniMaxH3Pipeline,
)
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
    get_minimax_h3_post_process_func,
)
from vllm_omni.diffusion.models.minimax_h3.time_request import (
    minimax_h3_time_shift_sigmas,
)
from vllm_omni.errors import OmniClientError

from comfy_omni.runtime.h3.package_binding import CurveCacheBinding
from comfy_omni.runtime.h3.requests import (
    H3ApiContractError,
    normalize_h3_request,
    validate_h3_sampling_controls,
)
from comfy_omni.runtime.h3.schedule import MODE_FIELDS, H3ScheduleContract

_CONSTRUCTION_LOCK = threading.RLock()
_REQUEST_LOCK = threading.RLock()
_BUILD_STATE: _CurveCacheState | None = None
_PACKAGE_BINDING: CurveCacheBinding | None = None
_MODEL_CONSTRUCTED = False
_PIPELINE_CONSTRUCTED = False


def _verify_official_runtime_schedule(schedule: H3ScheduleContract) -> None:
    """Replay the pinned official CPU scheduler once and compare every FP32 plan bit."""

    if (
        schedule.image_condition_timestep != MINIMAX_H3_IMGVID_COND_TIMESTEP
        or schedule.audio_condition_timestep != MINIMAX_H3_AUDIO_REF_COND_TIMESTEP
    ):
        raise ValueError("curve cache conditioning anchors differ from the official H3 runtime")
    video_sigmas = minimax_h3_time_shift_sigmas(
        num_steps=schedule.api_sigma_points,
        shift_scale=schedule.video_shift,
    )
    audio_sigmas = minimax_h3_time_shift_sigmas(
        num_steps=schedule.api_sigma_points,
        shift_scale=schedule.audio_shift,
    )
    if len(video_sigmas) != schedule.api_sigma_points or len(audio_sigmas) != schedule.api_sigma_points:
        raise ValueError("official H3 scheduler changed the compiled sigma-point count")
    mode_indices = dict(schedule.mode_plan_indices)
    for mode, fields in MODE_FIELDS.items():
        indices = mode_indices.get(mode)
        if indices is None or len(indices) != schedule.denoise_steps:
            raise ValueError(f"official H3 schedule mode {mode!r} is not fully compiled")
        for step, plan_index in enumerate(indices):
            video_t = 1.0 - video_sigmas[step]
            audio_t = 1.0 - audio_sigmas[step]
            candidates = {
                "video": torch.tensor(video_t, dtype=torch.float32),
                "audio": torch.tensor(audio_t, dtype=torch.float32),
                "image": torch.tensor(
                    max(video_t, MINIMAX_H3_IMGVID_COND_TIMESTEP),
                    dtype=torch.float32,
                ),
                "audio_ref": torch.tensor(
                    max(audio_t, MINIMAX_H3_AUDIO_REF_COND_TIMESTEP),
                    dtype=torch.float32,
                ),
            }
            observed = torch.stack([candidates[field] for field in fields]).unique(sorted=True)
            expected = torch.tensor(schedule.plans[plan_index].values, dtype=torch.float32)
            if not torch.equal(observed.view(torch.int32), expected.view(torch.int32)):
                raise ValueError(f"official H3 runtime schedule differs from cache mode={mode!r} step={step}")


class _CurveCacheState:
    def __init__(self, binding: CurveCacheBinding) -> None:
        self.cache_path = binding.cache_path
        self.schedule = binding.schedule
        _verify_official_runtime_schedule(self.schedule)
        self._offsets = [0]
        for plan in self.schedule.plans:
            self._offsets.append(self._offsets[-1] + len(plan.values))
        self._span_by_offset = {
            self._offsets[index]: self._offsets[index + 1] - self._offsets[index]
            for index in range(len(self._offsets) - 1)
        }
        self._mode_indices = dict(self.schedule.mode_plan_indices)
        self.time_embedder: _CacheTimeEmbedder | None = None
        self.block_modules: dict[int, _CacheAdalnProj] = {}
        self.final_module: _CacheAdalnProj | None = None
        self.active_plan_indices: tuple[int, ...] | None = None
        self.loaded = False

    def register_time_embedder(self, module: _CacheTimeEmbedder) -> None:
        if self.time_embedder is not None:
            raise RuntimeError("curve cache constructed more than one time embedder")
        self.time_embedder = module

    def register_adaln(self, prefix: str, module: _CacheAdalnProj) -> None:
        if prefix == "final_layer.adaln_proj":
            if self.final_module is not None:
                raise RuntimeError("curve cache constructed more than one final AdaLN")
            self.final_module = module
            return
        parts = prefix.split(".")
        if len(parts) != 3 or parts[0] != "blocks" or not parts[1].isdigit() or parts[2] != "adaln_proj":
            raise RuntimeError(f"unsupported cache AdaLN prefix {prefix!r}")
        index = int(parts[1])
        if index in self.block_modules:
            raise RuntimeError(f"duplicate cache AdaLN block {index}")
        self.block_modules[index] = module

    def assert_constructed(self) -> None:
        if self.time_embedder is None or set(self.block_modules) != set(range(50)) or self.final_module is None:
            raise RuntimeError("cache-only H3 model module census is incomplete")

    def load(self, device: torch.device) -> None:
        if self.loaded:
            return
        self.assert_constructed()
        from safetensors import safe_open

        with safe_open(self.cache_path, framework="pt", device="cpu") as source:
            offsets = source.get_tensor("plan_offsets")
            timesteps = source.get_tensor("plan_timesteps")
            blocks = source.get_tensor("block_params")
            final = source.get_tensor("final_params")
        if offsets.tolist() != self._offsets:
            raise RuntimeError("curve cache offsets changed after startup validation")
        self.time_embedder.install(
            offsets.to(device=device, non_blocking=False),
            timesteps.to(device=device, non_blocking=False),
        )
        blocks = blocks.to(device=device, non_blocking=False)
        final = final.to(device=device, non_blocking=False)
        for index, module in self.block_modules.items():
            rows = blocks[index]
            if not rows.is_contiguous():
                raise RuntimeError("curve cache block-first layout did not produce contiguous block rows")
            module.install(rows)
        self.final_module.install(final)
        self.loaded = True

    @contextmanager
    def activate(self, mode: str, *, step_indices: tuple[int, ...] | None = None) -> Iterator[None]:
        if not self.loaded:
            raise RuntimeError("curve AdaLN cache is not resident before denoise")
        try:
            compiled_indices = self._mode_indices[mode]
        except KeyError as exc:
            raise OmniClientError(
                f"H3 schedule mode {mode!r} is not compiled",
                status_code=409,
                error_type="H3_SCHEDULE_NOT_COMPILED",
            ) from exc
        if step_indices is None:
            indices = compiled_indices
        else:
            if not step_indices or any(
                isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(compiled_indices)
                for index in step_indices
            ):
                raise ValueError("H3 curve-cache step_indices are invalid")
            indices = tuple(compiled_indices[index] for index in step_indices)
        if self.active_plan_indices is not None:
            raise RuntimeError("overlapping H3 curve-cache request scopes are forbidden")
        self.active_plan_indices = indices
        try:
            yield
        finally:
            self.active_plan_indices = None

    def selection(self, actual: torch.Tensor) -> torch.Tensor:
        if self.time_embedder is None or not self.loaded:
            raise RuntimeError("curve AdaLN cache is not loaded")
        indices = self.active_plan_indices
        if indices is None:
            raise RuntimeError("curve AdaLN cache has no active request schedule")
        step = get_forward_context().denoise_step_idx
        if step is None or not 0 <= step < len(indices):
            raise RuntimeError(f"invalid H3 denoise step index {step!r}")
        plan_index = indices[step]
        start, end = self._offsets[plan_index], self._offsets[plan_index + 1]
        expected = self.time_embedder.plan_timesteps.narrow(0, start, end - start)
        actual = actual.view(-1)
        if actual.device != expected.device or actual.dtype != torch.float32:
            raise RuntimeError("runtime H3 timesteps have the wrong device or dtype")
        if actual.numel() != expected.numel():
            raise RuntimeError("runtime H3 timestep vector has the wrong compiled width")
        return expected

    def token_span(self, token: torch.Tensor) -> tuple[int, int]:
        if self.time_embedder is None or token.dtype != torch.float32 or token.dim() != 1:
            raise RuntimeError("invalid curve-cache timestep token")
        source = self.time_embedder.plan_timesteps
        if token.device != source.device or token.untyped_storage().data_ptr() != source.untyped_storage().data_ptr():
            raise RuntimeError("curve-cache timestep token has the wrong storage")
        start = int(token.storage_offset())
        length = int(token.numel())
        if self._span_by_offset.get(start) != length:
            raise RuntimeError("curve-cache timestep token has an invalid span")
        return start, length


class _CacheTimeEmbedder(nn.Module):
    def __init__(self, arch: Any, *, prefix: str) -> None:
        super().__init__()
        del arch
        if prefix != "time_embedder" or _BUILD_STATE is None:
            raise RuntimeError("cache time embedder constructed outside the H3 cache-only scope")
        self._state = _BUILD_STATE
        self.register_buffer("plan_offsets", torch.empty(0, dtype=torch.int64), persistent=False)
        self.register_buffer("plan_timesteps", torch.empty(0, dtype=torch.float32), persistent=False)
        self._state.register_time_embedder(self)

    def install(self, offsets: torch.Tensor, timesteps: torch.Tensor) -> None:
        self.plan_offsets = offsets
        self.plan_timesteps = timesteps

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self._state.selection(timesteps)


class _CacheAdalnProj(nn.Module):
    def __init__(
        self,
        arch: Any,
        out_features: int,
        quant_config: Any,
        *,
        expand_ratio: int,
        modality_num: int,
        prefix: str,
    ) -> None:
        super().__init__()
        del quant_config
        if _BUILD_STATE is None:
            raise RuntimeError("cache AdaLN constructed outside the H3 cache-only scope")
        if out_features != expand_ratio * arch.hidden_size * modality_num:
            raise ValueError("cache AdaLN output shape differs from the official H3 architecture")
        self.expand_ratio = int(expand_ratio)
        self.modality_num = int(modality_num)
        self.hidden_size = int(arch.hidden_size)
        self._state = _BUILD_STATE
        self.register_buffer("cache_rows", torch.empty(0, dtype=torch.bfloat16), persistent=False)
        self._state.register_adaln(prefix, self)

    def install(self, rows: torch.Tensor) -> None:
        self.cache_rows = rows

    def forward(self, token: torch.Tensor) -> tuple[torch.Tensor, ...]:
        start, length = self._state.token_span(token)
        rows = self.cache_rows.narrow(0, start, length)
        if not rows.is_contiguous():
            raise RuntimeError("curve AdaLN cache slice is not contiguous")
        rows = rows.view(length * self.modality_num, self.expand_ratio * self.hidden_size)
        return tuple(rows.chunk(self.expand_ratio, dim=-1))


class H3ComfyCacheDiTModel(OfficialMiniMaxH3DiTModel):
    def __init__(self, od_config: Any, quant_config: Any = None) -> None:
        with _CONSTRUCTION_LOCK:
            self._initialize_cache_model(od_config, quant_config)

    def _initialize_cache_model(self, od_config: Any, quant_config: Any) -> None:
        global _BUILD_STATE, _MODEL_CONSTRUCTED
        if _MODEL_CONSTRUCTED:
            raise RuntimeError("h3-forge permits exactly one H3 model per worker process")
        if _PACKAGE_BINDING is None:
            raise RuntimeError("cache model constructed without a verified package binding")
        state = _CurveCacheState(_PACKAGE_BINDING)
        if _BUILD_STATE is not None:
            raise RuntimeError("nested H3 cache-only model construction is forbidden")
        original_time = official_transformer.MiniMaxH3TimeEmbedder
        original_adaln = official_transformer.MiniMaxH3AdalnProj
        _BUILD_STATE = state
        official_transformer.MiniMaxH3TimeEmbedder = _CacheTimeEmbedder
        official_transformer.MiniMaxH3AdalnProj = _CacheAdalnProj
        try:
            super().__init__(od_config, quant_config=quant_config)
        finally:
            official_transformer.MiniMaxH3TimeEmbedder = original_time
            official_transformer.MiniMaxH3AdalnProj = original_adaln
            _BUILD_STATE = None
        state.assert_constructed()
        self.h3_forge_curve_cache = state
        _MODEL_CONSTRUCTED = True


def _cache_mode(task: str, ref_blocks: list[dict[str, Any]] | None) -> str:
    if task == "t2va":
        return "t2va"
    if task == "fl2va":
        return "fl2va"
    blocks = ref_blocks or []
    visual = any(block.get("kind") in {"image", "video", "video_audio"} for block in blocks)
    audio = any(block.get("kind") in {"audio", "video_audio"} for block in blocks)
    if visual and audio:
        return "ref2va-mixed"
    if audio:
        return "ref2va-audio"
    return "ref2va-image"


class H3CurveCachePipeline(OfficialMiniMaxH3Pipeline):
    """Official H3 pipeline with cache-only time/AdaLN construction."""

    def __init__(self, *, od_config: Any, package: Any, prefix: str = "") -> None:
        global _PIPELINE_CONSTRUCTED, _PACKAGE_BINDING
        with _CONSTRUCTION_LOCK:
            if _PIPELINE_CONSTRUCTED:
                raise RuntimeError("h3-forge permits exactly one H3 pipeline per worker process")
            cache_backend = str(getattr(od_config, "cache_backend", "none") or "none").lower()
            if cache_backend != "none":
                raise RuntimeError("h3-forge strict parity forbids approximate diffusion cache backends")
            if package.curve_cache is None or _PACKAGE_BINDING is not None:
                raise RuntimeError("cache pipeline requires one verified package binding")
            local_config = copy.copy(od_config)
            local_config.model = str(package.partition_path)
            original_model = official_pipeline.MiniMaxH3DiTModel
            _PACKAGE_BINDING = package.curve_cache
            official_pipeline.MiniMaxH3DiTModel = H3ComfyCacheDiTModel
            try:
                super().__init__(od_config=local_config, prefix=prefix)
            finally:
                official_pipeline.MiniMaxH3DiTModel = original_model
                _PACKAGE_BINDING = None
        if not isinstance(self.transformer, H3ComfyCacheDiTModel):
            raise RuntimeError("official H3 pipeline did not construct the cache-only DiT")
        if self.partition != "ref2va" or hasattr(self, "transformers_ref"):
            raise RuntimeError("h3-forge v1 requires one Ref2VA-primary transformer")
        self.comfy_omni_package = package
        _PIPELINE_CONSTRUCTED = True

    def eval(self) -> H3CurveCachePipeline:
        super().eval()
        self.transformer.h3_forge_curve_cache.load(torch.device(self.device))
        return self

    def forward(self, request: Any) -> Any:
        with _REQUEST_LOCK:
            if len(request.requests) != 1:
                raise OmniClientError("MiniMax H3 supports one request at a time")
            original = request.requests[0]
            sampling = original.sampling_params
            state = self.transformer.h3_forge_curve_cache
            prompt = original.prompt
            negative = prompt.get("negative_prompt") if isinstance(prompt, Mapping) else None
            try:
                normalized = normalize_h3_request(
                    sampling.extra_args,
                    legacy_num_inference_steps=sampling.num_inference_steps,
                    schedule=state.schedule,
                )
                validate_h3_sampling_controls(sampling, negative_prompt=negative)
            except H3ApiContractError as exc:
                raise OmniClientError(
                    str(exc),
                    status_code=exc.status_code,
                    error_type=exc.code,
                ) from exc
            copied_sampling = replace(
                sampling,
                extra_args=normalized.extra_args,
                num_inference_steps=normalized.api_sigma_points,
            )
            copied_request = replace(original, sampling_params=copied_sampling)
            copied_batch = replace(request, requests=[copied_request])
            return super().forward(copied_batch)

    def diffuse(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.transformer.h3_forge_curve_cache
        if (
            kwargs.get("base_schedule") is not None
            or kwargs.get("num_steps") != state.schedule.api_sigma_points
            or kwargs.get("video_shift") != state.schedule.video_shift
            or kwargs.get("audio_shift") != state.schedule.audio_shift
        ):
            raise OmniClientError(
                "requested H3 sigma schedule is not compiled",
                status_code=409,
                error_type="H3_SCHEDULE_NOT_COMPILED",
            )
        mode = _cache_mode(str(kwargs.get("task")), kwargs.get("ref_blocks"))
        with state.activate(mode):
            return super().diffuse(**kwargs)


__all__ = [
    "H3ComfyCacheDiTModel",
    "H3CurveCachePipeline",
    "get_minimax_h3_post_process_func",
]
