"""In-place storage residency for one live H3 DiT module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class H3WeightMemoryError(RuntimeError):
    """An invalid or failed persistent-slot residency transition."""


@dataclass(frozen=True, slots=True)
class H3PersistentSlot:
    """Small restoration record; ``tensor`` remains the original Parameter/buffer."""

    name: str
    tensor: Any
    shape: tuple[int, ...]
    dtype: Any
    original_device: Any


class H3WeightMemory:
    """Move or release slot storage without replacing registered tensor objects."""

    def __init__(self, module: Any) -> None:
        if not hasattr(module, "beta4_ready"):
            raise H3WeightMemoryError("H3 explicit residency requires the beta4_ready forward guard")
        state_dict = getattr(module, "state_dict", None)
        if not callable(state_dict):
            raise H3WeightMemoryError("H3 transformer does not expose state_dict")
        slots = state_dict(keep_vars=True)
        if not isinstance(slots, dict) or not slots:
            raise H3WeightMemoryError("H3 transformer has no persistent slots")

        unique: list[H3PersistentSlot] = []
        seen_objects: set[int] = set()
        for name, tensor in slots.items():
            identity = id(tensor)
            if identity in seen_objects:
                continue
            seen_objects.add(identity)
            unique.append(
                H3PersistentSlot(
                    name=name,
                    tensor=tensor,
                    shape=tuple(tensor.shape),
                    dtype=tensor.dtype,
                    original_device=tensor.device,
                )
            )
        self.module = module
        self.slots = tuple(unique)
        self.slot_name_count = len(slots)
        self.residency = "loaded"

    def unload(self, mode: str, *, cpu_budget_bytes: int) -> None:
        if self.residency != "loaded":
            raise H3WeightMemoryError(f"H3 weights are already {self.residency}")
        if mode not in {"release", "cpu"}:
            raise H3WeightMemoryError("unload mode must be 'release' or 'cpu'")
        if not isinstance(cpu_budget_bytes, int) or isinstance(cpu_budget_bytes, bool) or cpu_budget_bytes < 0:
            raise H3WeightMemoryError("CPU unload budget must be a non-negative integer")
        self._validate_loaded_shapes()
        required = self._resident_bytes()
        if mode == "cpu" and cpu_budget_bytes < required:
            raise H3WeightMemoryError(
                f"CPU unload budget {cpu_budget_bytes} cannot hold {required} bytes of H3 weights"
            )

        self.module.beta4_ready = False
        try:
            if mode == "cpu":
                for slot in self.slots:
                    slot.tensor.data = slot.tensor.data.to(device="cpu")
                self.residency = "cpu"
            else:
                self._empty_all_slots()
                self.residency = "released"
            self._release_cuda_allocator_cache()
        except Exception as error:
            self.fail_closed_release()
            raise H3WeightMemoryError(f"H3 {mode} unload failed; all weight storage was released") from error

    def begin_source_load(self) -> None:
        """Ensure original-sized slots exist before calling the real weight loader."""
        if self.residency == "cpu":
            self._empty_all_slots()
        if self.residency in {"cpu", "released"}:
            try:
                for slot in self.slots:
                    slot.tensor.data = slot.tensor.data.new_empty(
                        slot.shape,
                        dtype=slot.dtype,
                        device=slot.original_device,
                    )
            except Exception as error:
                self.fail_closed_release()
                raise H3WeightMemoryError("H3 slot allocation failed; all weight storage was released") from error
        elif self.residency not in {"loaded", "loading"}:
            raise H3WeightMemoryError(f"cannot begin source load while weights are {self.residency}")
        self.module.beta4_ready = False
        self.residency = "loading"

    def begin_cpu_restore(self) -> None:
        if self.residency != "cpu":
            raise H3WeightMemoryError(f"cannot restore CPU weights while residency is {self.residency}")
        try:
            for slot in self.slots:
                slot.tensor.data = slot.tensor.data.to(device=slot.original_device)
        except Exception as error:
            self.fail_closed_release()
            raise H3WeightMemoryError("H3 CPU restore failed; all weight storage was released") from error
        self.module.beta4_ready = True
        self.residency = "loading"

    def mark_loaded(self) -> None:
        if self.residency != "loading":
            raise H3WeightMemoryError(f"cannot mark H3 weights loaded from {self.residency}")
        self._validate_loaded_shapes()
        if not bool(self.module.beta4_ready):
            raise H3WeightMemoryError("H3 loader did not restore beta4_ready")
        self.residency = "loaded"

    def fail_closed_release(self) -> None:
        self.module.beta4_ready = False
        self._empty_all_slots()
        self.residency = "released"
        self._release_cuda_allocator_cache()

    def status(self) -> dict[str, object]:
        cpu_bytes = 0
        device_bytes = 0
        for slot in self.slots:
            tensor = slot.tensor
            size = int(tensor.numel()) * int(tensor.element_size())
            if tensor.device.type == "cpu":
                cpu_bytes += size
            else:
                device_bytes += size
        cuda_allocated_bytes, cuda_reserved_bytes = self._cuda_allocator_bytes()
        return {
            "weight_residency": self.residency,
            "weight_slot_count": self.slot_name_count,
            "cpu_weight_bytes": cpu_bytes,
            "device_weight_bytes": device_bytes,
            "resident_weight_bytes": cpu_bytes + device_bytes,
            "cuda_memory_allocated_bytes": cuda_allocated_bytes,
            "cuda_memory_reserved_bytes": cuda_reserved_bytes,
        }

    def _resident_bytes(self) -> int:
        return sum(int(slot.tensor.numel()) * int(slot.tensor.element_size()) for slot in self.slots)

    def _validate_loaded_shapes(self) -> None:
        for slot in self.slots:
            if tuple(slot.tensor.shape) != slot.shape or slot.tensor.dtype != slot.dtype:
                raise H3WeightMemoryError(f"H3 persistent slot {slot.name!r} no longer matches its loaded metadata")

    def _empty_all_slots(self) -> None:
        for slot in self.slots:
            slot.tensor.data = slot.tensor.data.new_empty((0,), dtype=slot.dtype, device="cpu")

    def _cuda_devices(self) -> tuple[Any, ...]:
        devices: dict[str, Any] = {}
        for slot in self.slots:
            if slot.original_device.type == "cuda":
                devices[str(slot.original_device)] = slot.original_device
        return tuple(devices.values())

    def _release_cuda_allocator_cache(self) -> None:
        devices = self._cuda_devices()
        if not devices:
            return
        import torch

        if not torch.cuda.is_available():
            return
        for device in devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

    def _cuda_allocator_bytes(self) -> tuple[int, int]:
        devices = self._cuda_devices()
        if not devices:
            return 0, 0
        import torch

        if not torch.cuda.is_available():
            return 0, 0
        allocated = sum(int(torch.cuda.memory_allocated(device)) for device in devices)
        reserved = sum(int(torch.cuda.memory_reserved(device)) for device in devices)
        return allocated, reserved


__all__ = ["H3PersistentSlot", "H3WeightMemory", "H3WeightMemoryError"]
