# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: h3-forge contributors
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Three local beta4 adaptations around the pinned official DiT implementation.

Math: h3-forge e9cb011 dense_pipeline.py, blob
6ddd34c49d532d56f568ec0010e925dbd86e5a2a. Host constructors/final forward:
vLLM-Omni 17285c2f55a41bf15772676121814d59a60ace35 transformer blob
91e03c865b22ffaaa5dbb3bf3ceeaf804a5564c8. Both Apache-2.0.
"""

from __future__ import annotations

import copy
import hashlib
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as official
from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipeline

from comfy_omni.contracts.beta4 import BETA4_TARGET_INVENTORY
from comfy_omni.integrations.vllm_omni.pipelines import scoped_construction as construction
from comfy_omni.runtime.h3.beta4_binding import BETA4_RUNTIME_ARCHITECTURE, verify_beta4_binding_unchanged
from comfy_omni.runtime.h3.raw_beta4 import RawBeta4Binding
from comfy_omni.runtime.h3.raw_standard import RawStandardBinding
from comfy_omni.runtime.hotel import H3TensorDescriptor, PreparedH3DitSelection

if TYPE_CHECKING:
    from comfy_omni.runtime.h3.beta4_binding import Beta4ComponentBinding

_PACKAGE_BINDING: Beta4ComponentBinding | RawBeta4Binding | None = None
_BUFFERS = frozenset({"adaln_t_table", "adaln_basis", "adaln_mean", "rope.inv_freq"})
_FP32_HEADS = ("video_patch_proj.", "audio_patch_proj.", "final_layer.video_out.", "final_layer.audio_out.")
_AUXILIARY_BUFFERS = frozenset({"adaln_basis", "adaln_mean"})


def _verify_binding_unchanged(binding):
    if isinstance(binding, RawBeta4Binding):
        binding.verify_unchanged()
    else:
        verify_beta4_binding_unchanged(binding)


def _runtime_name(source_name):
    return "time_embedder.adaln_t_table" if source_name == "adaln_t_table" else source_name


def _check_host():
    for module, expected in (
        (official, "91e03c865b22ffaaa5dbb3bf3ceeaf804a5564c8"),
        (pipeline, "62d10488ce0a82e9d3f72b3d0b23550565ff4db9"),
    ):
        payload = Path(module.__file__).read_bytes()
        observed = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
        if observed != expected:
            raise RuntimeError("beta4 requires the exact pinned H3 host source")


def _runtime_dtype(name):
    return (
        torch.float32
        if (name in {"adaln_t_table", "rope.inv_freq"} or ".adaln_proj.linear." in name or name.startswith(_FP32_HEADS))
        else torch.bfloat16
    )


def _local_shape(name, shape, tp_size):
    shape = list(shape)
    if name in _BUFFERS or "norm" in name:
        return tuple(shape)
    axis = 1 if name.endswith((".attn.out_proj.weight", ".mlp.fc2.weight")) else 0
    if shape[axis] % tp_size:
        raise ValueError(f"beta4 tensor cannot be partitioned: {name}")
    shape[axis] //= tp_size
    return tuple(shape)


class _TableTimeEmbedder(nn.Module):
    table_shape = (1025, 8)

    def __init__(self, arch, *, prefix):
        super().__init__()
        if prefix != "time_embedder" or arch.time_embed_dim != self.table_shape[1]:
            raise ValueError("beta4 table conditioning geometry differs")
        self.register_buffer("adaln_t_table", torch.empty(self.table_shape, dtype=torch.float32))

    def forward(self, timesteps):
        table = self.adaln_t_table
        if table.dtype != torch.float32 or timesteps.ndim != 1:
            raise ValueError("beta4 requires an FP32 table and a one-dimensional timestep vector")
        rows = table.shape[0]
        if not timesteps.dtype.is_floating_point:
            if bool(((timesteps < 0) | (timesteps >= rows)).any()):
                raise ValueError("beta4 timestep row is outside the table")
            return table.index_select(0, timesteps.long())
        values = timesteps.to(torch.float32)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("beta4 timesteps must be finite")
        positions = values.clamp(0.0, 1.0) * (rows - 1)
        lower = positions.floor().long().clamp(max=rows - 2)
        fraction = (positions - lower.float()).unsqueeze(1)
        return torch.lerp(table.index_select(0, lower), table.index_select(0, lower + 1), fraction)


class _Beta4AdalnProj(nn.Module):
    def __init__(self, arch, out_features, quant_config, *, expand_ratio, modality_num, prefix):
        super().__init__()
        if quant_config is not None or arch.time_embed_dim != 8:
            raise ValueError("beta4 AdaLN requires dense rank-eight conditioning")
        if out_features != expand_ratio * arch.hidden_size * modality_num:
            raise ValueError("beta4 AdaLN output geometry differs")
        if (expand_ratio, modality_num) not in {(6, 3), (2, 1)}:
            raise ValueError("beta4 AdaLN modulation layout differs")
        self.expand_ratio, self.modality_num, self.hidden_size = expand_ratio, modality_num, arch.hidden_size
        self.linear = ColumnParallelLinear(
            8,
            out_features,
            bias=True,
            gather_output=True,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=f"{prefix}.linear",
        )

    def forward(self, t_emb):
        values, _ = self.linear(t_emb.float())
        values = values.view(values.shape[0] * self.modality_num, self.expand_ratio * self.hidden_size)
        if self.expand_ratio == 6:
            values = values.to(torch.bfloat16)
        return tuple(values.chunk(self.expand_ratio, dim=-1))


class _Beta4FinalLayer(official.MiniMaxH3FinalLayer):
    def forward(self, x, *, t_emb, inverse_indices):
        shift, scale = self.adaln_proj(t_emb)
        hidden = self.norm(x).float()
        hidden = official.indexed_scale_shift_(hidden, shift, scale, inverse_indices)
        video, _ = self.video_out(hidden)
        audio, _ = self.audio_out(hidden)
        return video, audio


class H3Beta4DiTModel(official.MiniMaxH3DiTModel):
    """Strict 534-source load; official block stack, TP loaders and forward."""

    _inventory = BETA4_TARGET_INVENTORY
    _architecture = BETA4_RUNTIME_ARCHITECTURE

    def __init__(self, od_config, quant_config=None, *, binding=None):
        with construction.construction("model"):
            _check_host()
            binding = binding if binding is not None else _PACKAGE_BINDING
            if binding is None or quant_config is not None:
                raise ValueError("beta4 model requires a verified dense component binding")
            if any(
                getattr(od_config, name, False)
                for name in (
                    "enable_cpu_offload",
                    "enable_layerwise_offload",
                    "enable_distributed_layerwise_offload",
                )
            ) or getattr(od_config.parallel_config, "use_hsdp", False):
                raise ValueError("beta4 first runtime supports native TP construction without host offload/HSDP")
            _verify_binding_unchanged(binding)
            self.beta4_binding = binding
            self.beta4_ready = False
            self._load_started = False
            self.beta4_loaded_sources = frozenset()
            local = copy.copy(od_config)
            raw = od_config.tf_model_config
            config = raw.to_dict() if hasattr(raw, "to_dict") else dict(raw)
            for name, expected in self._architecture.items():
                actual = config.get(name, expected)
                if name == "patch_size":
                    actual = tuple(actual)
                # The official MLP input width describes a removed module.
                if name != "time_embed_dim" and actual != expected:
                    raise ValueError(f"beta4 host geometry differs: {name}")
            config.update(self._architecture)
            local.tf_model_config = config
            table_type = type(
                "Beta4BoundTimeEmbedder", (_TableTimeEmbedder,), {"table_shape": self._inventory["adaln_t_table"][1]}
            )
            with construction.substitute(
                official,
                MiniMaxH3TimeEmbedder=table_type,
                MiniMaxH3AdalnProj=_Beta4AdalnProj,
                MiniMaxH3FinalLayer=_Beta4FinalLayer,
            ):
                super().__init__(local, quant_config=None)
            for name in ("adaln_basis", "adaln_mean"):
                self.register_buffer(name, torch.empty(self._inventory[name][1], dtype=torch.bfloat16))
            self._check_layout()

    def _check_layout(self):
        expected_names = {_runtime_name(name) for name in self._inventory}
        actual = self.state_dict(keep_vars=True)
        if set(actual) != expected_names:
            raise ValueError("beta4 persistent slot census differs from the exact 534-source map")
        tp_size = get_tensor_model_parallel_world_size()
        for source, (_, shape) in self._inventory.items():
            tensor = actual[_runtime_name(source)]
            if tuple(tensor.shape) != _local_shape(source, shape, tp_size) or tensor.dtype != _runtime_dtype(source):
                raise ValueError(f"beta4 runtime shape/dtype differs: {source}")
        if len(dict(self.named_parameters())) != 530 or len(actual) != 534:
            raise ValueError("beta4 requires 530 parameters and four persistent buffers")

    def load_weights(self, weights):
        if self._load_started and not isinstance(self.beta4_binding, RawBeta4Binding):
            raise RuntimeError("beta4 permits one complete load per constructed model")
        self._load_started = True
        self.beta4_ready = False
        _verify_binding_unchanged(self.beta4_binding)
        seen = set()

        def checked():
            for name, tensor in weights:
                expected = self._inventory.get(name)
                if expected is None or name in seen:
                    raise ValueError(f"beta4 unknown or duplicate source slot: {name}")
                accepted_dtype = (
                    {torch.bfloat16, _runtime_dtype(name)}
                    if isinstance(self.beta4_binding, RawBeta4Binding)
                    else {torch.bfloat16}
                )
                if tensor.dtype not in accepted_dtype or tuple(tensor.shape) != expected[1]:
                    raise ValueError(f"beta4 source dtype/shape differs: {name}")
                seen.add(name)
                yield _runtime_name(name), tensor
            if seen != set(self._inventory):
                raise ValueError(f"beta4 missing source slots: {sorted(set(self._inventory) - seen)}")

        loaded = super().load_weights(checked())
        if loaded != {_runtime_name(name) for name in self._inventory}:
            raise ValueError("beta4 official loader left unconsumed persistent slots")
        self._check_layout()
        _verify_binding_unchanged(self.beta4_binding)
        if isinstance(self.beta4_binding, RawBeta4Binding):
            self.beta4_loaded_sources = frozenset(item.name for item in self.beta4_binding.target_descriptors)
        else:
            self.beta4_loaded_sources = frozenset(seen)
        self.beta4_ready = True
        return loaded

    def loading_receipt(self):
        self.post_load_weights()
        parameters = dict(self.named_parameters())
        slots = self.state_dict(keep_vars=True)
        ledger = []
        for source, (_, shape) in sorted(self._inventory.items()):
            target = _runtime_name(source)
            tensor = slots[target]
            ledger.append(
                {
                    "source": source,
                    "target": target,
                    "source_shape": list(shape),
                    "local_shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "bytes": tensor.numel() * tensor.element_size(),
                    "kind": "parameter" if target in parameters else "buffer",
                    "numerical_forward_input": source not in {"adaln_basis", "adaln_mean"},
                }
            )
        return {
            "status": "READY",
            "source_slots": len(self.beta4_loaded_sources),
            "runtime_initializers": len(self._inventory) - len(self.beta4_loaded_sources),
            "runtime_slots": len(slots),
            "tensor_parallel_size": get_tensor_model_parallel_world_size(),
            "parameter_bytes": sum(x["bytes"] for x in ledger if x["kind"] == "parameter"),
            "buffer_bytes": sum(x["bytes"] for x in ledger if x["kind"] == "buffer"),
            "ledger": ledger,
        }

    def post_load_weights(self):
        self._check_layout()
        if not self.beta4_ready:
            raise RuntimeError("beta4 model is not ready after a complete verified load")

    def forward(self, **kwargs):
        if not self.beta4_ready:
            raise RuntimeError("beta4 forward requires all 534 source slots")
        return super().forward(**kwargs)


class H3Beta4Pipeline(pipeline.MiniMaxH3Pipeline):
    def __init__(self, *, od_config, package=None, raw_binding=None, raw_selection="initial", prefix=""):
        global _PACKAGE_BINDING
        with construction.construction("pipeline"):
            if raw_binding is not None and package is not None:
                raise ValueError("beta4 requires exactly one original-file or package binding")
            binding = raw_binding if raw_binding is not None else getattr(package, "beta4", None)
            if binding is None or _PACKAGE_BINDING is not None:
                raise ValueError("beta4 pipeline requires one verified component binding")
            if raw_binding is not None and not isinstance(raw_binding, RawBeta4Binding):
                raise ValueError("beta4 original source requires a RawBeta4Binding")
            quantization = getattr(od_config, "quantization_config", None)
            transformer_quantization = (
                pipeline._resolve_component_quant_config(quantization, "transformer")
                if raw_binding is not None
                else quantization
            )
            if transformer_quantization is not None:
                raise ValueError("beta4 transformer requires explicit dense quantization_config=None")
            if str(getattr(od_config, "cache_backend", "none") or "none").lower() != "none":
                raise ValueError("beta4 pipeline forbids approximate diffusion caches")
            local = copy.copy(od_config)
            # Native v3 uses an existing serving view for the partition index.
            # Its source binding was resolved/verified by the package router.
            source_path = Path(od_config.model)
            if raw_binding is not None:
                if not source_path.is_dir() or not any(
                    (candidate / "model_index.json").is_file() for candidate in (source_path, source_path / "Ref2VA")
                ):
                    raise ValueError("raw beta4 requires an existing read-only H3 shared component root")
                local.model = str(source_path)
            else:
                partition = package.partition_path
                if (partition / "model_index.json").is_file():
                    local.model = str(partition)
                elif (source_path / "transformer").is_symlink() and (source_path / "transformer").resolve(
                    strict=True
                ) == package.beta4.component_root:
                    local.model = str(source_path)
                else:
                    raise ValueError("beta4 native package requires its prepared Ref2VA serving view")
            _PACKAGE_BINDING = binding
            try:
                with construction.substitute(pipeline, MiniMaxH3DiTModel=H3Beta4DiTModel):
                    super().__init__(od_config=local, prefix=prefix)
            finally:
                _PACKAGE_BINDING = None
            if (
                not isinstance(self.transformer, H3Beta4DiTModel)
                or self.partition != "ref2va"
                or hasattr(self, "transformers_ref")
            ):
                raise RuntimeError("beta4 pipeline requires one Ref2VA-primary transformer")
            self.comfy_omni_package = package
            self._raw_beta4_binding = raw_binding
            self._raw_beta4_loaded = False
            if raw_binding is not None:
                sources = [source for source in self.weights_sources if source.prefix == "transformer."]
                if len(sources) != 1:
                    raise RuntimeError("raw beta4 requires exactly one host transformer source")
                self.weights_sources = [source for source in self.weights_sources if source.prefix != "transformer."]
                self._comfy_omni_h3_sources = {}
                self.comfy_omni_active_h3_dit = self.comfy_omni_register_h3_dit(raw_selection, raw_binding)

    def comfy_omni_register_h3_dit(self, selection, binding):
        """Register an already authenticated source at the trusted composition boundary."""
        if self._raw_beta4_binding is None or not isinstance(binding, RawBeta4Binding):
            raise ValueError("H3 source registration requires the original-file runtime")
        if not isinstance(selection, str) or not selection:
            raise ValueError("H3 selection must be a non-empty identifier")
        if selection in self._comfy_omni_h3_sources:
            raise ValueError(f"H3 selection is already registered: {selection}")
        binding.verify_unchanged()
        inventory = {item.name: (item.dtype, tuple(item.shape)) for item in binding.target_descriptors}
        canonical = dict(self.transformer._inventory)
        expected = (
            {name: value for name, value in canonical.items() if name not in _AUXILIARY_BUFFERS}
            if isinstance(binding, RawStandardBinding)
            else canonical
        )
        if set(inventory) != set(expected) or any(
            shape != expected[name][1]
            or (
                dtype not in {"BF16", "F16", "F32"}
                if isinstance(binding, RawStandardBinding)
                else dtype != expected[name][0]
            )
            for name, (dtype, shape) in inventory.items()
        ):
            raise ValueError("H3 source geometry differs from the active transformer")
        descriptors = tuple(
            H3TensorDescriptor(
                name,
                tuple(shape),
                str(_runtime_dtype(name)),
                "buffer" if name in _BUFFERS else "parameter",
            )
            for name, (_, shape) in sorted(canonical.items())
        )
        prepared = PreparedH3DitSelection(
            selection=selection,
            identity=f"sha256:{binding.trusted_identity.sha256}",
            execution_profile="minimax-h3-beta4-table/v1",
            tensors=descriptors,
            logical_bytes=sum(prod(item.shape) * (4 if item.dtype == "torch.float32" else 2) for item in descriptors),
        )
        self._comfy_omni_h3_sources[selection] = (binding, prepared)
        return prepared

    def comfy_omni_prepare_h3_dit(self, selection):
        """Check registered metadata without reading weights or mutating model memory."""
        entry = getattr(self, "_comfy_omni_h3_sources", {}).get(selection)
        if entry is None:
            raise ValueError(f"H3 selection is not registered: {selection}")
        binding, prepared = entry
        binding.verify_unchanged()
        return prepared

    def comfy_omni_bind_h3_dit(self, prepared):
        """Select authenticated metadata, also when payloads come from the CPU cache."""
        current = self.comfy_omni_prepare_h3_dit(prepared.selection)
        if current != prepared:
            raise ValueError("H3 source selection changed after preparation")
        binding, _ = self._comfy_omni_h3_sources[prepared.selection]
        self.transformer.beta4_binding = binding
        return binding

    def comfy_omni_iter_h3_dit(self, prepared):
        """Stream real source weights and explicit non-forward auxiliary initializers."""
        binding = self.comfy_omni_bind_h3_dit(prepared)
        for name, tensor in binding.open_weights():
            # Preserve FP32 table/conditioning values. The backbone uses the
            # same explicit BF16 execution dtype as the existing host model.
            yield name, tensor.to(dtype=_runtime_dtype(name))
        if isinstance(binding, RawStandardBinding):
            for name in sorted(_AUXILIARY_BUFFERS):
                yield name, torch.zeros(self.transformer._inventory[name][1], dtype=torch.bfloat16)

    def load_weights(self, weights):
        binding = self._raw_beta4_binding
        if binding is None:
            return super().load_weights(weights)
        if self._raw_beta4_loaded:
            raise RuntimeError("raw beta4 pipeline weights have already been loaded")
        binding.verify_unchanged()

        def shared_weights():
            for name, tensor in weights:
                if name.startswith("transformer."):
                    raise ValueError("raw beta4 cannot also consume a disk transformer shard")
                yield name, tensor

        loaded = super().load_weights(shared_weights())
        source = self.comfy_omni_iter_h3_dit(self.comfy_omni_active_h3_dit)
        loaded.update(super().load_weights((f"transformer.{name}", tensor) for name, tensor in source))
        binding.verify_unchanged()
        self._raw_beta4_loaded = True
        return loaded
