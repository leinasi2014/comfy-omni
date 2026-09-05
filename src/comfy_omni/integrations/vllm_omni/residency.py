"""Worker-local H3 DiT weight residency transactions for vLLM-Omni.

The pipeline owns the trusted registry and source readers.  RPC callers pass a
registered selection ID, never a path or callable.  This extension mutates the
already-live transformer through its real ``load_weights`` lifecycle.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from typing import Any

from comfy_omni.integrations.vllm_omni.h3_memory import H3WeightMemory, H3WeightMemoryError
from comfy_omni.runtime.hotel import (
    H3DitResidency,
    H3ResidencyError,
    H3ResidencyPhase,
    PreparedH3DitSelection,
)


class H3ResidencyWorkerExtension:
    """Mixin installed through vLLM-Omni's ``worker_extension_cls`` seam."""

    _COMFY_OMNI_H3_STATUS_KEYS = (
        "phase",
        "active_selection",
        "active_identity",
        "execution_profile",
        "transaction_id",
        "target_selection",
        "target_identity",
        "poison_reason",
        "weight_residency",
    )

    def comfy_omni_prepare_h3_dit(
        self,
        transaction_id: str,
        selection: str,
        cpu_cache_budget_bytes: int = 0,
    ) -> dict[str, object]:
        pipeline, state = self._comfy_omni_h3_context()
        self._comfy_omni_require_quiescent()
        if not isinstance(selection, str) or not selection:
            raise H3ResidencyError("selection must be a non-empty registered ID")

        prepare = getattr(pipeline, "comfy_omni_prepare_h3_dit", None)
        if not callable(prepare):
            raise H3ResidencyError("pipeline does not expose the trusted H3 DiT prepare hook")
        target = prepare(selection)
        state.prepare(
            transaction_id,
            target,
            cpu_cache_budget_bytes=cpu_cache_budget_bytes,
        )
        return self._comfy_omni_h3_local_status()

    def comfy_omni_commit_h3_dit(self, transaction_id: str) -> dict[str, object]:
        pipeline, state = self._comfy_omni_h3_context()
        memory = self._comfy_omni_h3_memory_context(pipeline)
        self._comfy_omni_require_quiescent()
        state._require_transaction(transaction_id, H3ResidencyPhase.PREPARED)
        assert state.target is not None

        try:
            if state.target.identity != state.active.identity:
                memory.begin_source_load()
                cached = getattr(self, "_comfy_omni_h3_cached_weights", None)
                cached_identity = getattr(self, "_comfy_omni_h3_cached_identity", None)
                if cached is not None and cached_identity == state.target.identity:
                    source = iter(cached)
                elif state.cpu_cache_budget_bytes > 0:
                    self._comfy_omni_clear_h3_cache()
                    cached, cached_bytes = self._comfy_omni_materialize_h3_weights(
                        pipeline,
                        state.target,
                        state.cpu_cache_budget_bytes,
                    )
                    self._comfy_omni_h3_cached_weights = cached
                    self._comfy_omni_h3_cached_identity = state.target.identity
                    self._comfy_omni_h3_cached_bytes = cached_bytes
                    source = iter(cached)
                else:
                    source = self._comfy_omni_open_h3_weights(pipeline, state.target)
                self._comfy_omni_load_h3_weights(pipeline, state.target, source)
                memory.mark_loaded()
            elif memory.residency == "cpu":
                memory.begin_cpu_restore()
                pipeline.transformer.post_load_weights()
                memory.mark_loaded()
            elif memory.residency == "released":
                memory.begin_source_load()
                source = self._comfy_omni_open_h3_weights(pipeline, state.active)
                self._comfy_omni_load_h3_weights(pipeline, state.active, source)
                memory.mark_loaded()
            elif memory.residency != "loaded":
                raise H3WeightMemoryError(f"cannot commit while H3 weights are {memory.residency}")
            state.mark_committed(transaction_id)
        except Exception as commit_error:
            try:
                self._comfy_omni_reload_active_h3(pipeline, state)
                state.mark_rolled_back(transaction_id)
            except Exception as rollback_error:
                reason = f"commit failed ({commit_error}); rollback failed ({rollback_error})"
                state.poison(reason)
                self._comfy_omni_clear_h3_cache()
                raise H3ResidencyError(f"H3 DiT {reason}; worker is poisoned") from rollback_error
            self._comfy_omni_clear_h3_cache()
            raise H3ResidencyError(
                f"H3 DiT commit failed ({commit_error}); rollback restored active selection {state.active.selection!r}"
            ) from commit_error
        return self._comfy_omni_h3_local_status()

    def comfy_omni_rollback_h3_dit(self, transaction_id: str) -> dict[str, object]:
        pipeline, state = self._comfy_omni_h3_context()
        self._comfy_omni_require_quiescent()
        if state.phase is H3ResidencyPhase.ROLLED_BACK:
            state.mark_rolled_back(transaction_id)
            return self._comfy_omni_h3_local_status()
        state._require_transaction(
            transaction_id,
            H3ResidencyPhase.PREPARED,
            H3ResidencyPhase.COMMITTED,
        )
        try:
            if state.phase is H3ResidencyPhase.COMMITTED and state.target is not None:
                if state.target.identity != state.active.identity:
                    self._comfy_omni_reload_active_h3(pipeline, state)
            state.mark_rolled_back(transaction_id)
        except Exception as rollback_error:
            state.poison(f"rollback failed ({rollback_error})")
            self._comfy_omni_clear_h3_cache()
            raise H3ResidencyError(f"H3 DiT rollback failed; worker is poisoned: {rollback_error}") from rollback_error
        self._comfy_omni_clear_h3_cache()
        return self._comfy_omni_h3_local_status()

    def comfy_omni_finalize_h3_dit(self, transaction_id: str) -> dict[str, object]:
        pipeline, state = self._comfy_omni_h3_context()
        memory = self._comfy_omni_h3_memory_context(pipeline)
        if memory.residency != "loaded":
            raise H3ResidencyError(f"cannot finalize while H3 weights are {memory.residency}")
        state.finalize(transaction_id)
        pipeline.comfy_omni_active_h3_dit = state.active
        return self._comfy_omni_h3_local_status()

    def comfy_omni_h3_residency_status(self) -> dict[str, object]:
        """Return this rank's status; the controller follows with a bool consensus RPC."""
        return self._comfy_omni_h3_local_status()

    def comfy_omni_check_h3_dit_status(self, expected: dict[str, object]) -> bool:
        """Compare all critical local fields for vLLM-Omni's all-rank bool AND."""
        if not isinstance(expected, dict) or set(expected) != set(self._COMFY_OMNI_H3_STATUS_KEYS):
            return False
        try:
            local = self._comfy_omni_h3_local_status()
        except Exception:
            return False
        return all(local.get(key) == expected[key] for key in self._COMFY_OMNI_H3_STATUS_KEYS)

    def _comfy_omni_h3_local_status(self) -> dict[str, object]:
        pipeline, state = self._comfy_omni_h3_context()
        result = state.status()
        result["cpu_cached_bytes"] = getattr(self, "_comfy_omni_h3_cached_bytes", 0)
        result["cpu_cached_identity"] = getattr(self, "_comfy_omni_h3_cached_identity", None)
        result.update(self._comfy_omni_h3_memory_context(pipeline).status())
        result.update(
            {
                "worker_pid": os.getpid(),
                "pipeline_id": id(pipeline),
                "transformer_id": id(pipeline.transformer),
                "shared_object_ids": {
                    name: id(value)
                    for name in ("text_encoder", "video_vae", "audio_vae", "tokenizer", "processor")
                    if (value := getattr(pipeline, name, None)) is not None
                },
            }
        )
        return result

    def comfy_omni_unload_h3_dit(
        self,
        mode: str = "release",
        cpu_budget_bytes: int = 0,
    ) -> dict[str, object]:
        pipeline, state = self._comfy_omni_h3_context()
        self._comfy_omni_require_quiescent()
        state._require_healthy()
        if state.transaction_id is not None or state.phase not in {
            H3ResidencyPhase.IDLE,
            H3ResidencyPhase.ROLLED_BACK,
        }:
            raise H3ResidencyError("cannot unload H3 weights during a residency transaction")
        memory = self._comfy_omni_h3_memory_context(pipeline)
        self._comfy_omni_clear_h3_cache()
        memory.unload(mode, cpu_budget_bytes=cpu_budget_bytes)
        return self._comfy_omni_h3_local_status()

    def comfy_omni_load_h3_dit(self) -> dict[str, object]:
        pipeline, state = self._comfy_omni_h3_context()
        self._comfy_omni_require_quiescent()
        state._require_healthy()
        if state.transaction_id is not None or state.phase not in {
            H3ResidencyPhase.IDLE,
            H3ResidencyPhase.ROLLED_BACK,
        }:
            raise H3ResidencyError("cannot load H3 weights during a residency transaction")
        memory = self._comfy_omni_h3_memory_context(pipeline)
        if memory.residency == "loaded":
            return self._comfy_omni_h3_local_status()
        try:
            if memory.residency == "cpu":
                memory.begin_cpu_restore()
                pipeline.transformer.post_load_weights()
            elif memory.residency == "released":
                memory.begin_source_load()
                source = self._comfy_omni_open_h3_weights(pipeline, state.active)
                self._comfy_omni_load_h3_weights(pipeline, state.active, source)
            else:
                raise H3WeightMemoryError(f"cannot load while H3 weights are {memory.residency}")
            memory.mark_loaded()
        except Exception as error:
            memory.fail_closed_release()
            raise H3ResidencyError(f"H3 weight load failed; weight storage remains released: {error}") from error
        return self._comfy_omni_h3_local_status()

    def _comfy_omni_h3_context(self) -> tuple[Any, H3DitResidency]:
        runner = getattr(self, "model_runner", None)
        pipeline = getattr(runner, "pipeline", None)
        if pipeline is None:
            raise H3ResidencyError("diffusion worker has no live pipeline")
        transformer = getattr(pipeline, "transformer", None)
        if transformer is None or not callable(getattr(transformer, "load_weights", None)):
            raise H3ResidencyError("live pipeline transformer does not expose load_weights")
        if not callable(getattr(transformer, "post_load_weights", None)):
            raise H3ResidencyError("live pipeline transformer does not expose post_load_weights")

        state = getattr(self, "_comfy_omni_h3_residency", None)
        if state is None:
            active = getattr(pipeline, "comfy_omni_active_h3_dit", None)
            if not isinstance(active, PreparedH3DitSelection):
                raise H3ResidencyError("pipeline has no validated active H3 DiT selection")
            state = H3DitResidency(active=active)
            self._comfy_omni_h3_residency = state
            self._comfy_omni_h3_cached_weights = None
            self._comfy_omni_h3_cached_identity = None
            self._comfy_omni_h3_cached_bytes = 0
        return pipeline, state

    def _comfy_omni_require_quiescent(self) -> None:
        runner = getattr(self, "model_runner", None)
        if runner is None:
            raise H3ResidencyError("diffusion worker has no model runner")
        if getattr(runner, "state_cache", None):
            raise H3ResidencyError("diffusion worker is not quiescent: state_cache is not empty")
        if getattr(runner, "input_batch", None) is not None:
            raise H3ResidencyError("diffusion worker is not quiescent: input_batch is active")
        if getattr(self, "_step_lora_state", None):
            raise H3ResidencyError("diffusion worker is not quiescent: step LoRA state is active")
        if getattr(runner, "cache_backend", None) is not None:
            raise H3ResidencyError("H3 DiT residency does not support an active cache backend")
        if getattr(runner, "offload_backend", None) is not None:
            raise H3ResidencyError("H3 DiT residency does not support an active offload backend")

    def _comfy_omni_h3_memory_context(self, pipeline: Any) -> H3WeightMemory:
        memory = getattr(self, "_comfy_omni_h3_weight_memory", None)
        if memory is None:
            memory = H3WeightMemory(pipeline.transformer)
            self._comfy_omni_h3_weight_memory = memory
        elif memory.module is not pipeline.transformer:
            raise H3ResidencyError("live H3 transformer changed after residency metadata capture")
        return memory

    def _comfy_omni_open_h3_weights(
        self,
        pipeline: Any,
        prepared: PreparedH3DitSelection,
    ) -> Iterator[tuple[str, Any]]:
        open_weights = getattr(pipeline, "comfy_omni_iter_h3_dit", None)
        if not callable(open_weights):
            raise H3ResidencyError("pipeline does not expose the trusted H3 DiT iterator hook")
        source = open_weights(prepared)
        if not isinstance(source, Iterable):
            raise H3ResidencyError("trusted H3 DiT iterator hook did not return an iterable")
        return iter(source)

    def _comfy_omni_validated_h3_weights(
        self,
        prepared: PreparedH3DitSelection,
        source: Iterator[tuple[str, Any]],
        seen: set[str],
    ) -> Iterator[tuple[str, Any]]:
        descriptors = {item.name: item for item in prepared.tensors}
        for item in source:
            if not isinstance(item, tuple) or len(item) != 2:
                raise H3ResidencyError("H3 DiT source must yield (logical_name, tensor) pairs")
            name, tensor = item
            if not isinstance(name, str) or name not in descriptors:
                raise H3ResidencyError(f"H3 DiT source yielded unknown tensor {name!r}")
            if name in seen:
                raise H3ResidencyError(f"H3 DiT source yielded duplicate tensor {name!r}")
            descriptor = descriptors[name]
            shape = tuple(getattr(tensor, "shape", ()))
            dtype = str(getattr(tensor, "dtype", ""))
            if shape != descriptor.shape or dtype != descriptor.dtype:
                raise H3ResidencyError(
                    f"H3 DiT tensor {name!r} does not match descriptor: shape={shape}, dtype={dtype}"
                )
            seen.add(name)
            yield name, tensor

    def _comfy_omni_load_h3_weights(
        self,
        pipeline: Any,
        prepared: PreparedH3DitSelection,
        source: Iterator[tuple[str, Any]],
    ) -> None:
        bind = getattr(pipeline, "comfy_omni_bind_h3_dit", None)
        if bind is not None:
            if not callable(bind):
                raise H3ResidencyError("pipeline H3 DiT binding hook is not callable")
            bind(prepared)
        seen: set[str] = set()
        validated = self._comfy_omni_validated_h3_weights(prepared, source, seen)
        pipeline.transformer.load_weights(validated)
        expected = {item.name for item in prepared.tensors}
        if seen != expected:
            missing = sorted(expected - seen)
            raise H3ResidencyError(f"H3 DiT loader did not consume the complete source; missing={missing}")
        pipeline.transformer.post_load_weights()

    def _comfy_omni_reload_active_h3(self, pipeline: Any, state: H3DitResidency) -> None:
        memory = self._comfy_omni_h3_memory_context(pipeline)
        memory.begin_source_load()
        source = self._comfy_omni_open_h3_weights(pipeline, state.active)
        self._comfy_omni_load_h3_weights(pipeline, state.active, source)
        memory.mark_loaded()

    def _comfy_omni_materialize_h3_weights(
        self,
        pipeline: Any,
        prepared: PreparedH3DitSelection,
        budget_bytes: int,
    ) -> tuple[tuple[tuple[str, Any], ...], int]:
        seen: set[str] = set()
        source = self._comfy_omni_open_h3_weights(pipeline, prepared)
        cached: list[tuple[str, Any]] = []
        cached_bytes = 0
        for name, tensor in self._comfy_omni_validated_h3_weights(prepared, source, seen):
            try:
                retained = tensor.detach().to(device="cpu").clone()
                tensor_bytes = int(retained.numel()) * int(retained.element_size())
            except (AttributeError, TypeError, ValueError) as error:
                raise H3ResidencyError(f"H3 DiT tensor {name!r} cannot be retained in the CPU cache") from error
            cached_bytes += tensor_bytes
            if cached_bytes > budget_bytes:
                raise H3ResidencyError(
                    f"CPU cache budget exceeded while retaining {name!r}: {cached_bytes} > {budget_bytes}"
                )
            cached.append((name, retained))
        expected = {item.name for item in prepared.tensors}
        if seen != expected:
            missing = sorted(expected - seen)
            raise H3ResidencyError(f"H3 DiT source is incomplete; missing={missing}")
        return tuple(cached), cached_bytes

    def _comfy_omni_clear_h3_cache(self) -> None:
        self._comfy_omni_h3_cached_weights = None
        self._comfy_omni_h3_cached_identity = None
        self._comfy_omni_h3_cached_bytes = 0


__all__ = ["H3ResidencyWorkerExtension"]
