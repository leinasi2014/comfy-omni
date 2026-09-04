"""Host-only hybrid8 dense DiT model tree for the pinned vLLM-Omni host.

Provenance: extracted and adapted from ``h3-forge`` at commit
``e9cb011d00b028c149db3978de246c54f6e34acc`` (Apache-2.0), file
``src/h3_forge/h3/dense_pipeline.py``, blob
``6ddd34c49d532d56f568ec0010e925dbd86e5a2a``. Only the model tree, assembly,
conditioning and load/forward paths are ported here; the legacy component
runtime / VRAM / package-binding machinery is replaced by plan-derived
validation in :func:`discover_hybrid8_dit_form` (the package contract itself is
already verified by the integration bootstrap). The ported code supports both
the 535-tensor ``h3-transformer-50l-hybrid8-bf16-plain`` form and the
532-tensor Ref2VA ConvRot export form (``adaln_t_table`` only, mixed
BF16/F32 dtypes). See ``docs/migration/dense-hybrid8-runtime-port-e9cb011.md``.

This module is importable only when the host (``vllm_omni``) is resident: it
imports vLLM / vllm_omni at module scope.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from vllm.distributed import (
    get_tensor_model_parallel_world_size as _get_tp_world_size,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
    MiniMaxH3DiTModel as OfficialMiniMaxH3DiTModel,
)
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
    MiniMaxH3Pipeline as OfficialMiniMaxH3Pipeline,
)

from comfy_omni.runtime.h3.hybrid8.contracts import (
    Hybrid8DitForm,
    block_indices,
    family_shapes,
    manifest_schema_sha256,
)
from comfy_omni.runtime.h3.hybrid8.contracts import (
    Hybrid8StructureError as DenseHybridStructureError,
)

logger = logging.getLogger(__name__)

#: od_config fields probed (in order) for the pipeline model root.
_MODEL_ROOT_FIELD_CANDIDATES: tuple[str, ...] = (
    "model",
    "model_name_or_path",
    "model_path",
    "model_dir",
)

#: Directories probed for the transformer checkpoint when only the pipeline
#: root is known: the conventional ``transformer/`` subdirectory first, then
#: the root itself (single-file deployments / dense-export layout).
_TRANSFORMER_DIR_CANDIDATES: tuple[str, ...] = ("Ref2VA/transformer", "transformer", ".")
_HYBRID8_T_TABLE_NAME = "adaln_t_table"
_HYBRID8_QKV_GROUPED_LAYOUT = "grouped-for-official-loader"


try:
    from vllm_omni.diffusion.attention.ops.minimax_h3_modulation import (
        indexed_gate as _official_indexed_gate,
    )
    from vllm_omni.diffusion.attention.ops.minimax_h3_modulation import (
        indexed_gate_rms_norm_scale_shift as _official_indexed_gate_rms_norm_scale_shift,
    )
    from vllm_omni.diffusion.attention.ops.minimax_h3_modulation import (
        indexed_scale_shift_ as _official_indexed_scale_shift_,
    )
    from vllm_omni.diffusion.attention.ops.minimax_h3_modulation import (
        rms_norm_indexed_scale_shift as _official_rms_norm_indexed_scale_shift,
    )

    _indexed_gate = _official_indexed_gate
    _indexed_gate_rms_norm_scale_shift = _official_indexed_gate_rms_norm_scale_shift
    _indexed_scale_shift_ = _official_indexed_scale_shift_
    _rms_norm_indexed_scale_shift = _official_rms_norm_indexed_scale_shift
except Exception:  # noqa: BLE001 - the offline mirror below is math-identical

    def _rms_norm_indexed_scale_shift(
        x: torch.Tensor,
        weight: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        indices: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        input_dtype = x.dtype
        normalized = x.float()
        variance = normalized.pow(2).mean(-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + eps)
        normalized = (weight.float() * normalized).to(input_dtype)
        return (normalized * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)).to(input_dtype)

    def _indexed_gate(
        x: torch.Tensor,
        gate: torch.Tensor,
        other: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        return (x + gate.index_select(0, indices) * other).to(x.dtype)

    def _indexed_gate_rms_norm_scale_shift(
        residual: torch.Tensor,
        gate: torch.Tensor,
        branch: torch.Tensor,
        weight: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        indices: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_dtype = residual.dtype
        residual_out = (residual + gate.index_select(0, indices) * branch).to(input_dtype)
        normalized = residual_out.float()
        variance = normalized.pow(2).mean(-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + eps)
        normalized = (weight.float() * normalized).to(input_dtype)
        modulated_out = (normalized * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)).to(
            input_dtype
        )
        return residual_out, modulated_out

    def _indexed_scale_shift_(
        x: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        x.copy_((x * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)).to(x.dtype))
        return x


MINIMAX_H3_DENSE_PIPELINE_ARCH = "MiniMaxH3DensePipeline"

# Safetensors header key of the raw pruned Qwen3-VL checkpoints ("24/50
# retained" Comfy pruning export) and the fields that switch the strict

_QUANT_MARKER_SUFFIXES = (".comfy_quant", ".weight_scale")
# Hybrid-side artifacts that must never appear in an *official-form* dense DiT
# checkpoint.  A census that carries them switches host-load onto the
# fail-closed hybrid8 route instead (see :func:`discover_hybrid8_dit_form`);
# on the official route they stay rejected exactly as before.
_DIT_FORBIDDEN_NAMES = frozenset({"adaln_t_table"})

# -- hybrid8 dense form (10Eros TURBO hybrid beta3; template ---------------
#    ``h3-transformer-50l-hybrid8-bf16-plain``) ------------------------------
#: Top-level tensors that exist ONLY in the hybrid8 dense form.  Their
#: presence in a transformer census is the route switch: it demands the
#: complete pinned 535-tensor manifest, item by item, before any module is
#: assembled.  None of these names appears in any other pinned inventory or
#: in an official-shaped export, so the switch cannot misfire on the
#: official route.
_HYBRID8_SIGNATURE_NAMES = frozenset(
    {
        "adaln_t_table",
        "adaln_basis",
        "adaln_mean",
        "silu_t_emb_grid",
    }
)
#: The top-level timestep code table: one 8-dim conditioning code per
#: timestep row (the hybrid8 replacement for the online ``time_embedder``).
_HYBRID8_T_TABLE_NAME = "adaln_t_table"
#: Tensor-name suffixes of one hybrid8 transformer block (10 tensors).
_HYBRID8_BLOCK_SUFFIXES = (
    "adaln_proj.linear.bias",
    "adaln_proj.linear.weight",
    "attn.k_norm.weight",
    "attn.out_proj.weight",
    "attn.q_norm.weight",
    "attn.qkv_proj.weight",
    "mlp.fc1.weight",
    "mlp.fc2.weight",
    "norm1.weight",
    "norm2.weight",
)
#: Token-refiner blocks keep the official block skeleton minus the AdaLN
#: conditioning projection (8 tensors).
_HYBRID8_TOKEN_REFINER_SUFFIXES = (
    "attn.k_norm.weight",
    "attn.out_proj.weight",
    "attn.q_norm.weight",
    "attn.qkv_proj.weight",
    "mlp.fc1.weight",
    "mlp.fc2.weight",
    "norm1.weight",
    "norm2.weight",
)
#: Manifest dtype spellings this runtime can assemble, mapped to torch dtypes.
_HYBRID8_TENSOR_DTYPES: dict[str, torch.dtype] = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
}
_TORCH_DTYPE_NAMES: dict[str, str] = {str(dtype): name for name, dtype in _HYBRID8_TENSOR_DTYPES.items()}
#: Tensor-name suffixes of the final layer family (7 tensors).
_FINAL_LAYER_SUFFIXES = (
    "adaln_proj.linear.bias",
    "adaln_proj.linear.weight",
    "audio_out.bias",
    "audio_out.weight",
    "norm.weight",
    "video_out.bias",
    "video_out.weight",
)
#: Directories probed (in order) for the transformer checkpoint when only the
#: pipeline root is known: the conventional ``transformer/`` subdirectory of
#: a deployed package first, then the root itself (single-file deployments
#: and the dense-export directory layout).
_TRANSFORMER_DIR_CANDIDATES = ("transformer", ".")
#: The only QKV checkpoint row layout the hybrid8 route serves: the dense-bf16
#: export contract's grouped rows (``[Q0;K0;V0;Q1;K1;V1;...]``, one
#: ``(heads_per_group+2)*head_dim`` row block per query group), declared by
#: the export manifest the exporter writes next to the shards.  The official
#: loader reorders exactly this layout into the runtime ``[Q;K;V]`` rows its
#: ``QKVParallelLinear`` shards (official ``load_weights`` :1015-1024), and
#: this runtime mirrors that reorder bit for bit before sharding.
_HYBRID8_QKV_GROUPED_LAYOUT = "grouped-for-official-loader"
#: The export component this route serves: the transformer dense-bf16 export
#: (the ``component`` value ``export_dense_bf16_checkpoint`` publishes for
#: the 10Eros transformer package).
_HYBRID8_EXPORT_COMPONENT = "transformer"
#: A lowercase 64-hex digest -- the spelling of every ``shards[*].sha256``
#: and ``index_sha256`` field of the dense-bf16 export receipt.
_HEX64_PATTERN = re.compile(r"[0-9a-f]{64}")
#: AdaLN modality count of the official skeleton (token tags carry -1 for
#: padding and 0/1/2 for video/text/audio; padding clamps to 0).
_HYBRID8_ADALN_MODALITY_NUM = 3

_CONSTRUCTION_LOCK = threading.RLock()
#: Serializes whole requests through the dense pipeline (the same pattern as
#: ``runtime_pipeline._REQUEST_LOCK``), so a text-encoder swap and the
#: generation that follows it can never interleave with another request.
_REQUEST_LOCK = threading.RLock()

logger = logging.getLogger(__name__)

_MODEL_ROOT_FIELD_CANDIDATES: tuple[str, ...] = (
    "model",
    "model_name_or_path",
    "model_path",
    "model_dir",
)


def _od_config_model_root(od_config: Any) -> tuple[str | Path, str] | None:
    """Resolve the model root from an od_config-like object, alias-proof.

    Returns the first non-empty ``str``/``Path`` value among the known field
    spellings together with the field it came from, or ``None`` when the
    object carries no usable root (absent fields, empty strings, non-string
    values).
    """

    for field in _MODEL_ROOT_FIELD_CANDIDATES:
        value = getattr(od_config, field, None)
        if isinstance(value, str | Path) and str(value).strip():
            return value, field
    return None


def _reorder_grouped_qkv_to_qkv(
    weight: torch.Tensor,
    *,
    num_query_groups: int,
    heads_per_group: int,
    head_dim: int,
) -> torch.Tensor:
    """Official ``_reorder_grouped_qkv_to_qkv``, verbatim math.

    Line-for-line the pinned official loader's reorder
    (``vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py``
    :151-180, applied to every ``.attn.qkv_proj.weight`` at :1015-1024): the
    checkpoint's grouped rows ``[Q_g;K_g;V_g]`` per query group become the
    runtime ``[Q_all;K_all;V_all]`` rows the ``QKVParallelLinear``
    ``weight_loader`` shards per head section.  The official loader is the
    single authority for this permutation -- this mirror never diverges.
    """

    per_group = (heads_per_group + 2) * head_dim
    expected_out = num_query_groups * per_group
    if weight.shape[0] != expected_out:
        raise DenseHybridStructureError(
            "hybrid8 qkv weight has incompatible output dim for the grouped "
            f"checkpoint layout: got {tuple(weight.shape)}, expected first dim {expected_out}."
        )

    rest_shape = weight.shape[1:]
    grouped = weight.reshape(num_query_groups, per_group, *rest_shape)
    q, k, v = torch.split(
        grouped,
        [heads_per_group * head_dim, head_dim, head_dim],
        dim=1,
    )
    return torch.cat(
        [
            q.reshape(num_query_groups * heads_per_group * head_dim, *rest_shape),
            k.reshape(num_query_groups * head_dim, *rest_shape),
            v.reshape(num_query_groups * head_dim, *rest_shape),
        ],
        dim=0,
    )


def _hybrid8_tp_uninitialized() -> bool:
    """True when the vLLM tensor-parallel group cannot be queried yet.

    The official transformer asks ``get_tensor_model_parallel_world_size()``
    unconditionally at construction (:922 of the official
    ``minimax_h3_transformer.py``); serve always initializes the group first
    (engine distributed init precedes model construction), but a real-host
    process that merely imports the module without vLLM distributed state
    has no group to ask.  vLLM's own escape hatch for exactly that is the
    ``disable_tp`` layer flag -- the *same* layer classes built with
    ``tp_size == 1`` -- so the hybrid8 tree takes that sanctioned no-TP
    construction instead of growing a second math path.
    """

    try:
        _get_tp_world_size()
    except Exception:  # noqa: BLE001 - any failure means "no group in this process"
        return True
    return False


def _hybrid8_tp_state() -> tuple[bool, int]:
    """``(disable_tp, tp_world_size)`` for one hybrid8 assembly.

    ``H3_FORGE_HYBRID8_FORCE_DISABLE_TP=1`` forces the no-TP construction
    even inside an initialized TP group: the equivalence tests build their
    full-quantity reference replica this way (same classes, tp_size=1), and
    operators can use it for single-rank introspection of a TP-serving
    tree.  It never changes the math, only the sharding.
    """

    import os

    if os.environ.get("H3_FORGE_HYBRID8_FORCE_DISABLE_TP") == "1":
        return True, 1
    if _hybrid8_tp_uninitialized():
        return True, 1
    return False, int(_get_tp_world_size())


def _validate_hybrid8_tp_config(*, geometry: _Hybrid8Geometry, tp_size: int) -> None:
    """The official ``MiniMaxH3DiTModel._validate_tp_config`` (:882), hybrid8.

    Same two divisibility invariants the official transformer enforces
    (heads and ffn over ``tensor_parallel_size``); every other sharded
    dimension of the pinned manifest (hidden, patch widths, AdaLN rows,
    final heads) is an integer multiple of one of these two, and the vLLM
    layer constructors themselves ``divide``-assert their own dimensions.
    """

    if tp_size < 1:
        raise DenseHybridStructureError(f"tensor_parallel_size must be positive, got {tp_size}")
    if geometry.num_heads % tp_size:
        raise DenseHybridStructureError(
            f"num_attention_heads must be divisible by tensor_parallel_size: {geometry.num_heads} % {tp_size} != 0"
        )
    if geometry.ffn_hidden_size % tp_size:
        raise DenseHybridStructureError(
            f"ffn_hidden_size must be divisible by tensor_parallel_size: {geometry.ffn_hidden_size} % {tp_size} != 0"
        )


#: RMSNorm epsilon of the official ``MiniMaxH3DiTArchConfig`` defaults
#: (``norm_eps``/``qk_norm_eps``/``final_norm_eps`` are all ``1e-5``).
_HYBRID8_NORM_EPS = 1e-5


class _Hybrid8RMSNorm(nn.Module):
    """RMSNorm with fp32 accumulation over the last dimension.

    Parameter-census twin of the official ``layers.norm.RMSNorm`` (one
    ``weight`` of the normalized size) with its ``forward_native`` math --
    the same accumulation semantics the official fused modulation kernels
    implement internally.
    """

    def __init__(self, size: int, *, eps: float = _HYBRID8_NORM_EPS, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size, dtype=dtype or torch.get_default_dtype()))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        normalized = x.to(torch.float32)
        variance = normalized.pow(2).mean(-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.to(torch.float32) * normalized).to(input_dtype)


def _hybrid8_layer_norm(shape: tuple[int, ...], dtype: torch.dtype) -> _Hybrid8RMSNorm:
    """A manifest-shaped RMSNorm (name kept for the host-load call sites)."""

    if len(shape) != 1:
        raise DenseHybridStructureError(f"hybrid8 norm shape {shape} is not 1-D")
    return _Hybrid8RMSNorm(shape[0], dtype=dtype)


def _hybrid8_rope_freqs(inv_freq: torch.Tensor, img_position_ids: torch.Tensor) -> torch.Tensor:
    """3D rope angles over (t, h, w): ``[1, S, 3]`` -> ``[S, rot_dim]`` fp32.

    Verbatim mirror of the official ``MiniMaxH3Rope.forward``: per-axis
    products with ``inv_freq``, concatenated as ``(t, h, w)`` twice.  The
    manifest pins ``rope.inv_freq`` BF16 (the official checkpoint carries it
    fp32); angles are computed in fp32 either way.
    """

    if img_position_ids.dim() != 3 or img_position_ids.shape[0] != 1:
        raise DenseHybridStructureError(
            f"hybrid8 rope img_position_ids must be [1, S, 3], got {list(img_position_ids.shape)}"
        )
    pos = img_position_ids[0].to(torch.float32)
    per_axis = pos.unsqueeze(-1) * inv_freq.to(torch.float32).view(1, 1, -1)
    t_f, h_f, w_f = per_axis.unbind(dim=1)
    half = torch.cat((t_f, h_f, w_f), dim=-1)
    return torch.cat((half, half), dim=-1)


def _hybrid8_rope_rotate(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Split-half RoPE on the first ``rot_dim`` head dims; the rest pass.

    Mirror of the official ``MiniMaxH3Attention._apply_rope`` with its
    ``RotaryEmbedding(is_neox_style=True, half_head_dim=False)``: pair dim
    ``i`` with dim ``i + rot_dim/2``; cos/sin of the unique angle half,
    cast to the activation dtype before the elementwise math.
    """

    rot_dim = freqs.shape[-1]
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    half = rot_dim // 2
    cos = torch.cos(freqs[..., :half]).to(x.dtype).unsqueeze(1)
    sin = torch.sin(freqs[..., :half]).to(x.dtype).unsqueeze(1)
    x0, x1 = x_rot[..., :half], x_rot[..., half:]
    rotated = torch.cat((cos * x0 - sin * x1, sin * x0 + cos * x1), dim=-1)
    return torch.cat((rotated, x_pass), dim=-1)


def _hybrid8_segment_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    softmax_scale: float,
) -> torch.Tensor:
    """Segment-wise non-causal SDPA over packed ``cu_seqlens`` documents.

    Mirror of the official ``_sdpa_varlen_attention``: attention never
    crosses a packed-document boundary (padding rows attend only themselves,
    so their garbage never reaches real rows).
    """

    out = torch.empty_like(q)
    bounds = cu_seqlens.tolist()
    for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
        if stop == start:
            continue
        seg_q = q[start:stop].transpose(0, 1).unsqueeze(0)
        seg_k = k[start:stop].transpose(0, 1).unsqueeze(0)
        seg_v = v[start:stop].transpose(0, 1).unsqueeze(0)
        seg_out = F.scaled_dot_product_attention(seg_q, seg_k, seg_v, scale=softmax_scale)
        out[start:stop] = seg_out.squeeze(0).transpose(0, 1)
    return out


class _Hybrid8Attention(nn.Module):
    """Packed attention over the official vLLM tensor-parallel projections.

    Official ``MiniMaxH3Attention`` math and TP layout, mirrored member for
    member: fused ``QKVParallelLinear`` slices q/k/v by *local heads*
    (``total_num_heads = total_num_kv_heads`` for the MHA manifest), the
    per-head q/k RMSNorm and split-half RoPE run identically on every
    rank's head shard, segment SDPA attends the full packed sequence with
    ``num_heads`` local heads (per-head attention is head-independent, so
    the local shard is exactly the matching slice of the full-head output),
    and ``RowParallelLinear(input_is_parallel=True)`` consumes the local
    attention columns and ``all_reduce``s the partial sums -- column shards
    + one reduction reproduce the full ``out_proj`` sum.
    """

    def __init__(self, shapes: Mapping[str, tuple[int, ...]], dtype: torch.dtype, *, disable_tp: bool = False) -> None:
        super().__init__()
        head_dim = shapes["attn.q_norm.weight"][0]
        qkv_rows, hidden_size = shapes["attn.qkv_proj.weight"]
        if qkv_rows % (3 * head_dim):
            raise DenseHybridStructureError(f"hybrid8 qkv rows {qkv_rows} are not 3 * heads * head_dim {head_dim}")
        total_heads = qkv_rows // (3 * head_dim)
        if total_heads < 1:
            raise DenseHybridStructureError("hybrid8 attention needs at least one head")
        self.head_dim = head_dim
        self.total_num_heads = total_heads
        self.softmax_scale = head_dim**-0.5
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=total_heads,
            total_num_kv_heads=total_heads,
            bias=False,
            params_dtype=dtype,
            quant_config=None,
            return_bias=True,
            disable_tp=disable_tp,
        )
        # Local heads on this rank (official :343: ``self.num_heads =
        # self.qkv_proj.num_heads``) -- with tp_size == 1 they are the full
        # roster and the layer math is the plain full projection.
        self.num_heads = self.qkv_proj.num_heads
        self.num_kv_heads = self.qkv_proj.num_kv_heads
        self.q_norm = _hybrid8_layer_norm(shapes["attn.q_norm.weight"], dtype)
        self.k_norm = _hybrid8_layer_norm(shapes["attn.k_norm.weight"], dtype)
        self.out_proj = RowParallelLinear(
            total_heads * head_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            params_dtype=dtype,
            quant_config=None,
            disable_tp=disable_tp,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_freqs: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        total = x.shape[0]
        qkv, _ = self.qkv_proj(x)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(total, self.num_heads, self.head_dim)
        k = k.view(total, self.num_kv_heads, self.head_dim)
        v = v.view(total, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope_freqs is not None:
            if rope_freqs.shape[-1] > self.head_dim:
                raise DenseHybridStructureError(
                    f"hybrid8 rope width {rope_freqs.shape[-1]} exceeds head dim {self.head_dim}"
                )
            q = _hybrid8_rope_rotate(q, rope_freqs)
            k = _hybrid8_rope_rotate(k, rope_freqs)
        out = _hybrid8_segment_sdpa(q, k, v, cu_seqlens, softmax_scale=self.softmax_scale)
        out = out.reshape(total, q_size)
        out, _ = self.out_proj(out)
        return out


class _Hybrid8MLP(nn.Module):
    """Fused gate/up ``fc1`` + SiLU-gated ``fc2`` (official ``MiniMaxH3MLP``).

    ``MergedColumnParallelLinear([ffn, ffn], gather_output=False)`` holds this
    rank's ``[gate_local; up_local]`` rows (the loader chunks the fused
    checkpoint rows into gate/up with shard ids 0/1, exactly like the official
    ``load_weights``); the elementwise ``silu(gate) * up`` runs on the local
    slice (value-preserving per column), and ``RowParallelLinear`` reduces the
    partial ``fc2`` products with one ``all_reduce``.
    """

    def __init__(self, shapes: Mapping[str, tuple[int, ...]], dtype: torch.dtype, *, disable_tp: bool = False) -> None:
        super().__init__()
        fc1_rows, hidden_size = shapes["mlp.fc1.weight"]
        if fc1_rows % 2:
            raise DenseHybridStructureError(f"hybrid8 fc1 rows {fc1_rows} do not split into gate/up")
        ffn_hidden_size = fc1_rows // 2
        self.fc1 = MergedColumnParallelLinear(
            hidden_size,
            [ffn_hidden_size, ffn_hidden_size],
            bias=False,
            gather_output=False,
            params_dtype=dtype,
            quant_config=None,
            disable_tp=disable_tp,
        )
        self.fc2 = RowParallelLinear(
            ffn_hidden_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            params_dtype=dtype,
            quant_config=None,
            disable_tp=disable_tp,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.fc1(x)
        gate, up = hidden.chunk(2, dim=-1)
        out, _ = self.fc2(F.silu(gate) * up)
        return out


class _Hybrid8AdalnProj(nn.Module):
    """The narrow hybrid8 AdaLN projection: linear over the table code.

    The hybrid8 twin of the official ``MiniMaxH3AdalnProj`` with two pinned
    differences (both from the reference runtime,
    ``ref/.recovery-…/tests/calibrate_ckpt850_closed_loop.py``):

    * **no SiLU** -- the online time embedder's SiLU is already baked into
      ``adaln_t_table`` (the 207 evidence grid
      ``silu_t_emb_grid = silu(time_embedder(t))`` is aligned with the table
      rows), so the linear consumes the raw conditioning code;
    * **fp32 compute** -- ``AdalnProj(8, H, expand, modalities,
      apply_silu=False, dtype=torch.float32)``: the BF16-manifest weight is
      widened once at load into the layer's fp32 parameters (an exact
      widening, replacing the per-call upcast with the same values; a
      ``[96768, 8]`` copy either way, bandwidth-trivial next to the
      attention it feeds).

    The layer class is the official ``ColumnParallelLinear`` (official
    :573) with ``gather_output=True``: the sharded rows are ``all_gather``ed
    back in rank order, so every rank indexes the *full* six-segment
    modulations over ``combined_indices`` (conditioning is replicated).
    """

    def __init__(
        self,
        shapes: Mapping[str, tuple[int, ...]],
        dtype: torch.dtype,
        *,
        expand_ratio: int,
        modality_num: int,
        hidden_size: int,
        disable_tp: bool = False,
    ) -> None:
        super().__init__()
        weight_shape = shapes["adaln_proj.linear.weight"]
        if weight_shape[0] != expand_ratio * hidden_size * modality_num:
            raise DenseHybridStructureError(
                f"hybrid8 adaln out-features {weight_shape[0]} != {expand_ratio}*{hidden_size}*{modality_num}"
            )
        self.expand_ratio = expand_ratio
        self.modality_num = modality_num
        self.hidden_size = hidden_size
        # Official layer class (``ColumnParallelLinear(gather_output=True)``,
        # official :573) with **fp32 parameters**: the manifest's BF16 weights
        # are widened once at load (``copy_`` into the fp32 shard, an exact
        # widening), which replaces the per-call upcast of the reference
        # runtime with the same values and keeps the pinned fp32 compute.
        # ``gather_output=True`` re-assembles the sharded rows in rank order
        # (``all_gather``), so the modulations each rank indexes are the full
        # ones -- conditioning stays replicated, never sharded by head.
        self.linear = ColumnParallelLinear(
            weight_shape[1],
            weight_shape[0],
            bias=True,
            gather_output=True,
            params_dtype=torch.float32,
            quant_config=None,
            disable_tp=disable_tp,
        )

    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """``t_emb [M, cond]`` -> ``expand_ratio`` tensors of ``[M*modality, H]`` fp32."""

        x, _ = self.linear(t_emb.to(torch.float32))
        m = x.shape[0]
        x = x.view(m * self.modality_num, self.expand_ratio * self.hidden_size)
        return tuple(x.chunk(self.expand_ratio, dim=-1))


class _Hybrid8DiTBlock(nn.Module):
    """One hybrid8 block: official skeleton plus the narrow AdaLN projection.

    The DiT-block forward is the official ``MiniMaxH3DiTBlock.forward``
    sequence verbatim -- six modulation segments per (timestep, modality)
    row, RMSNorm+indexed scale/shift before attention and MLP, gated
    residuals via the fused modulation ops -- with ``with_adaln=False``
    reducing the module to the official token-refiner block (pre-norm
    attention + MLP, no AdaLN, no RoPE).
    """

    def __init__(
        self,
        shapes: Mapping[str, tuple[int, ...]],
        dtype: torch.dtype,
        *,
        with_adaln: bool,
        disable_tp: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = shapes["norm1.weight"][0]
        self.norm1 = _hybrid8_layer_norm(shapes["norm1.weight"], dtype)
        self.norm2 = _hybrid8_layer_norm(shapes["norm2.weight"], dtype)
        self.attn = _Hybrid8Attention(shapes, dtype, disable_tp=disable_tp)
        self.mlp = _Hybrid8MLP(shapes, dtype, disable_tp=disable_tp)
        if with_adaln:
            self.adaln_proj = _Hybrid8AdalnProj(
                shapes,
                dtype,
                expand_ratio=6,
                modality_num=_HYBRID8_ADALN_MODALITY_NUM,
                hidden_size=self.hidden_size,
                disable_tp=disable_tp,
            )

    def forward(
        self,
        x: torch.Tensor,
        *,
        t_emb: torch.Tensor | None = None,
        combined_indices: torch.Tensor | None = None,
        rope_freqs: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        if not hasattr(self, "adaln_proj"):
            # Token-refiner block: the official pre-norm block without AdaLN
            # or RoPE (rope_freqs must not be provided).
            if rope_freqs is not None or t_emb is not None:
                raise DenseHybridStructureError("a hybrid8 token-refiner block takes no conditioning")
            x = x + self.attn(self.norm1(x), rope_freqs=None, cu_seqlens=cu_seqlens)
            return x + self.mlp(self.norm2(x))

        if t_emb is None or combined_indices is None:
            raise DenseHybridStructureError("a hybrid8 DiT block requires t_emb and combined_indices")
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaln_proj(t_emb)
        # The reference runtime applies the modulations in the activation
        # dtype (its cache path casts the baked mods to the model dtype);
        # the official path produces bf16 mods for the same reason.
        shift_msa, scale_msa, gate_msa = shift_msa.to(x.dtype), scale_msa.to(x.dtype), gate_msa.to(x.dtype)
        shift_mlp, scale_mlp, gate_mlp = shift_mlp.to(x.dtype), scale_mlp.to(x.dtype), gate_mlp.to(x.dtype)

        residual = x
        h = _rms_norm_indexed_scale_shift(
            x,
            self.norm1.weight,
            shift_msa,
            scale_msa,
            combined_indices,
            self.norm1.variance_epsilon,
        )
        h = self.attn(h, rope_freqs=rope_freqs, cu_seqlens=cu_seqlens)
        x, h = _indexed_gate_rms_norm_scale_shift(
            residual,
            gate_msa,
            h,
            self.norm2.weight,
            shift_mlp,
            scale_mlp,
            combined_indices,
            self.norm2.variance_epsilon,
        )
        residual = x
        h = self.mlp(h)
        return _indexed_gate(residual, gate_mlp, h, combined_indices)


class _Hybrid8FinalLayer(nn.Module):
    """Final layer with its own narrow AdaLN projection and output heads.

    Official ``MiniMaxH3FinalLayer.forward`` sequence: single-modality
    shift/scale AdaLN over ``inverse_indices``, fp32 activations, both
    output heads applied to all rows.  The official heads are fp32
    parameters; the pinned hybrid8 manifest ships them BF16, so they are
    upcast per call to keep the official full-precision head math.  The
    final modulation itself stays fp32 end to end (QA 009): the narrow
    ``adaln_proj`` produces fp32 shift/scale, and the fused indexed op
    runs over an fp32 copy of the normalized rows -- the reference
    FinalLayer's ``(norm(x) * (1.0 + scale) + shift).to(fp32)`` -- instead
    of narrowing the mods to the bf16 sequence dtype (the official final
    layer likewise feeds its ``adaln_proj`` outputs to ``indexed_scale_shift_``
    uncast).
    """

    def __init__(
        self,
        shapes: Mapping[str, tuple[int, ...]],
        dtype: torch.dtype,
        *,
        hidden_size: int,
        disable_tp: bool = False,
    ) -> None:
        super().__init__()
        self.norm = _hybrid8_layer_norm(shapes["norm.weight"], dtype)
        self.adaln_proj = _Hybrid8AdalnProj(
            shapes,
            dtype,
            expand_ratio=2,
            modality_num=1,
            hidden_size=hidden_size,
            disable_tp=disable_tp,
        )
        # Official final heads (``ColumnParallelLinear(..., gather_output=True)``
        # with fp32 parameters, official :773-790): the BF16 manifest weights
        # widen once at load, ``all_gather`` restores the full logits on every
        # rank, and the fp32 head math the reference runtime pins is kept.
        self.video_out = ColumnParallelLinear(
            hidden_size,
            shapes["video_out.weight"][0],
            bias=True,
            gather_output=True,
            params_dtype=torch.float32,
            quant_config=None,
            disable_tp=disable_tp,
        )
        self.audio_out = ColumnParallelLinear(
            hidden_size,
            shapes["audio_out.weight"][0],
            bias=True,
            gather_output=True,
            params_dtype=torch.float32,
            quant_config=None,
            disable_tp=disable_tp,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        t_emb: torch.Tensor,
        inverse_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale = self.adaln_proj(t_emb)
        h = self.norm(x).to(torch.float32)
        h = _indexed_scale_shift_(h, shift, scale, inverse_indices)
        video, _ = self.video_out(h)
        audio, _ = self.audio_out(h)
        return video, audio


@dataclass(frozen=True)
class _Hybrid8Geometry:
    """The forward geometry of the pinned hybrid8 manifest (all derived).

    Every field is derived from the validated 535-tensor manifest itself and
    cross-checked for the official skeleton's coherence (no dimension is
    assumed): the hidden size from the block norms, the head layout from the
    q/k norms and the fused qkv rows, the rotary width from ``rope.inv_freq``
    (``rot_dim = 6 * inv_freq_len``, official ``MiniMaxH3Attention``), and
    the video/audio/text input widths from the patch/condition projections.
    """

    hidden_size: int
    num_heads: int
    head_dim: int
    rot_dim: int
    ffn_hidden_size: int
    video_patch_dim: int
    audio_patch_dim: int
    text_dim: int


def _derive_hybrid8_geometry(
    inventory: Mapping[str, tuple[str, tuple[int, ...]]],
    *,
    num_blocks: int,
) -> _Hybrid8Geometry:
    """Derive and structurally validate the forward geometry, fail-closed."""

    def _shape(name: str) -> tuple[int, ...]:
        return inventory[name][1]

    hidden = _shape("blocks.0.norm1.weight")[0]
    head_dim = _shape("blocks.0.attn.q_norm.weight")[0]
    qkv_rows, qkv_cols = _shape("blocks.0.attn.qkv_proj.weight")
    if qkv_cols != hidden or qkv_rows % (3 * head_dim):
        raise DenseHybridStructureError(
            f"hybrid8 block 0 qkv {qkv_rows}x{qkv_cols} does not fit 3*heads*{head_dim} over hidden {hidden}"
        )
    heads = qkv_rows // (3 * head_dim)
    if heads < 1:
        raise DenseHybridStructureError("hybrid8 attention needs at least one head")
    out_rows, out_cols = _shape("blocks.0.attn.out_proj.weight")
    if out_rows != hidden or out_cols != heads * head_dim:
        raise DenseHybridStructureError(
            f"hybrid8 block 0 out_proj {out_rows}x{out_cols} != hidden x heads*head_dim ({hidden}x{heads * head_dim})"
        )
    fc1_rows, fc1_cols = _shape("blocks.0.mlp.fc1.weight")
    ffn = _shape("blocks.0.mlp.fc2.weight")[1]
    if fc1_cols != hidden or fc1_rows != 2 * ffn or _shape("blocks.0.mlp.fc2.weight")[0] != hidden:
        raise DenseHybridStructureError(
            f"hybrid8 block 0 mlp fc1 {fc1_rows}x{fc1_cols}/fc2 disagrees with hidden {hidden}, ffn {ffn}"
        )
    inv_freq = _shape("rope.inv_freq")
    if len(inv_freq) != 1:
        raise DenseHybridStructureError(f"hybrid8 rope.inv_freq must be 1-D, got {inv_freq}")
    rot_dim = 6 * inv_freq[0]
    if rot_dim > head_dim:
        raise DenseHybridStructureError(f"hybrid8 rope width {rot_dim} (6*{inv_freq[0]}) exceeds head dim {head_dim}")
    video_patch_dim = _shape("video_patch_proj.weight")[1]
    audio_patch_dim = _shape("audio_patch_proj.weight")[1]
    text_dim = _shape("condition_proj.weight")[1]
    for name, dim in (
        ("video_patch_proj", video_patch_dim),
        ("audio_patch_proj", audio_patch_dim),
        ("condition_proj", text_dim),
    ):
        rows, cols = _shape(f"{name}.weight")
        if rows != hidden or cols != dim:
            raise DenseHybridStructureError(f"hybrid8 {name}.weight {rows}x{cols} disagrees with hidden {hidden}")
    if _shape("final_layer.norm.weight")[0] != hidden:
        raise DenseHybridStructureError("hybrid8 final layer norm disagrees with the block hidden size")
    if _shape("final_layer.video_out.weight") != (video_patch_dim, hidden):
        raise DenseHybridStructureError("hybrid8 final video_out disagrees with the video patch width")
    if _shape("final_layer.audio_out.weight") != (audio_patch_dim, hidden):
        raise DenseHybridStructureError("hybrid8 final audio_out disagrees with the audio patch width")
    if _shape("token_refiner.final_norm.weight")[0] != hidden:
        raise DenseHybridStructureError("hybrid8 token refiner final norm disagrees with the block hidden size")
    # Every DiT and refiner block must carry the same geometry (the census
    # already pins each shape; this proves the pinned families are mutually
    # coherent for the forward).
    for index in range(num_blocks):
        for prefix, _family in (
            (f"blocks.{index}", _HYBRID8_BLOCK_SUFFIXES),
            (f"token_refiner.blocks.{index}", _HYBRID8_TOKEN_REFINER_SUFFIXES),
        ):
            if f"{prefix}.norm1.weight" not in inventory:
                break  # refiner family is shorter than the DiT stack
            if _shape(f"{prefix}.norm1.weight")[0] != hidden:
                raise DenseHybridStructureError(f"{prefix} hidden size deviates from block 0")
            if _shape(f"{prefix}.attn.q_norm.weight")[0] != head_dim:
                raise DenseHybridStructureError(f"{prefix} head dim deviates from block 0")
            if _shape(f"{prefix}.attn.qkv_proj.weight") != (qkv_rows, qkv_cols):
                raise DenseHybridStructureError(f"{prefix} qkv geometry deviates from block 0")
        if _shape(f"blocks.{index}.adaln_proj.linear.weight")[0] != 18 * hidden:
            raise DenseHybridStructureError(f"blocks.{index} adaln rows != 18*hidden ({18 * hidden})")
    if _shape("final_layer.adaln_proj.linear.weight")[0] != 2 * hidden:
        raise DenseHybridStructureError(f"final adaln rows != 2*hidden ({2 * hidden})")
    return _Hybrid8Geometry(
        hidden_size=hidden,
        num_heads=heads,
        head_dim=head_dim,
        rot_dim=rot_dim,
        ffn_hidden_size=ffn,
        video_patch_dim=video_patch_dim,
        audio_patch_dim=audio_patch_dim,
        text_dim=text_dim,
    )


class _Hybrid8TokenRefiner(nn.Module):
    """Text token refiner: pre-norm blocks over the text-only packed document."""

    def __init__(self, blocks: nn.ModuleList, final_norm: _Hybrid8RMSNorm) -> None:
        super().__init__()
        self.blocks = blocks
        self.final_norm = final_norm

    def forward(self, x: torch.Tensor, *, cu_seqlens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, rope_freqs=None, cu_seqlens=cu_seqlens)
        return self.final_norm(x)


def _assemble_hybrid8_dit(dit: DenseHybridDiT, form: Hybrid8DitForm) -> None:
    """Build the hybrid8 module tree, census-locked to the manifest 1:1.

    The tree mirrors the official skeleton exactly where the hybrid8 form
    keeps it (attention, MLP, norms, token refiner, patch/condition
    projections, final heads, ``rope.inv_freq``) and replaces the online
    time embedder with the four top-level conditioning buffers.  After
    construction the produced ``named_parameters``/``named_buffers`` census
    must equal the manifest name set exactly -- the assembly and the pinned
    535-entry roster cannot drift apart silently.
    """

    inventory = dict(form.inventory)
    # Mixed-dtype support (E4 Ref2VA ConvRot export: BF16 blocks, F32 top-level &
    # patch projections). The block family determines the sequence dtype; each
    # buffer/projection honors its declared dtype.
    dtype = _HYBRID8_TENSOR_DTYPES[inventory["blocks.0.norm1.weight"][0]]

    def _zeros(name: str) -> torch.Tensor:
        _dtype_string, shape = inventory[name]
        return torch.zeros(shape, dtype=_HYBRID8_TENSOR_DTYPES[_dtype_string])

    # Official TP entry point (issue 010): validate the geometry against the
    # tensor-parallel world exactly like ``MiniMaxH3DiTModel._validate_tp_config``
    # (:882/:918), then build every projection from the same vLLM parallel
    # layer classes the official transformer uses.  ``disable_tp`` only covers
    # real-host processes without an initialized TP group (vLLM's sanctioned
    # no-TP construction); serve initializes the group first, so tp_size=2
    # halves every rank's projection weights (~30 GiB instead of the full
    # ~60 GiB replica) while tp_size=1 keeps the plain full math.
    disable_tp, tp_world_size = _hybrid8_tp_state()
    geometry = _derive_hybrid8_geometry(inventory, num_blocks=form.num_blocks)
    _validate_hybrid8_tp_config(geometry=geometry, tp_size=tp_world_size)
    local_heads = geometry.num_heads // tp_world_size

    # Top-level hybrid8 conditioning state (the online time embedder's
    # replacement): the per-timestep code table plus the basis/mean/grid it
    # was derived from, all checkpoint-loaded, all persistent buffers.
    # Conditioning is replicated (official: small tensors like rope.inv_freq
    # and the AdaLN modulation inputs never shard).
    dit.register_buffer(_HYBRID8_T_TABLE_NAME, _zeros(_HYBRID8_T_TABLE_NAME))
    # Optional top-level conditioning evidence (10Eros 535 form only; the
    # E4/Ref2VA 532 form carries adaln_t_table alone).
    if "adaln_basis" in inventory:
        dit.register_buffer("adaln_basis", _zeros("adaln_basis"))
    if "adaln_mean" in inventory:
        dit.register_buffer("adaln_mean", _zeros("adaln_mean"))
    if "silu_t_emb_grid" in inventory:
        dit.register_buffer("silu_t_emb_grid", _zeros("silu_t_emb_grid"))
    dit.rope = nn.Module()
    dit.rope.register_buffer("inv_freq", _zeros("rope.inv_freq"))
    # Official patch/condition projections (:933-959): ColumnParallel with
    # gather_output=True.  The latent embedders keep the official fp32
    # parameters (BF16 manifest rows widen once at load); the text condition
    # projection stays in the bf16 sequence dtype like the official layer.
    dit.video_patch_proj = ColumnParallelLinear(
        inventory["video_patch_proj.weight"][1][1],
        inventory["video_patch_proj.weight"][1][0],
        bias=True,
        gather_output=True,
        params_dtype=torch.float32,
        quant_config=None,
        disable_tp=disable_tp,
    )
    dit.audio_patch_proj = ColumnParallelLinear(
        inventory["audio_patch_proj.weight"][1][1],
        inventory["audio_patch_proj.weight"][1][0],
        bias=True,
        gather_output=True,
        params_dtype=torch.float32,
        quant_config=None,
        disable_tp=disable_tp,
    )
    dit.condition_proj = ColumnParallelLinear(
        inventory["condition_proj.weight"][1][1],
        inventory["condition_proj.weight"][1][0],
        bias=True,
        gather_output=True,
        params_dtype=dtype,
        quant_config=None,
        disable_tp=disable_tp,
    )

    def _block_shapes(prefix: str, suffixes: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
        return family_shapes(inventory, prefix, suffixes)

    dit.blocks = nn.ModuleList(
        _Hybrid8DiTBlock(
            _block_shapes(f"blocks.{index}", _HYBRID8_BLOCK_SUFFIXES),
            dtype,
            with_adaln=True,
            disable_tp=disable_tp,
        )
        for index in range(form.num_blocks)
    )
    dit.token_refiner = _Hybrid8TokenRefiner(
        nn.ModuleList(
            _Hybrid8DiTBlock(
                _block_shapes(f"token_refiner.blocks.{index}", _HYBRID8_TOKEN_REFINER_SUFFIXES),
                dtype,
                with_adaln=False,
                disable_tp=disable_tp,
            )
            for index in range(len(block_indices(inventory, "token_refiner.blocks")))
        ),
        _hybrid8_layer_norm(inventory["token_refiner.final_norm.weight"][1], dtype),
    )
    dit.final_layer = _Hybrid8FinalLayer(
        _block_shapes("final_layer", _FINAL_LAYER_SUFFIXES),
        dtype,
        hidden_size=geometry.hidden_size,
        disable_tp=disable_tp,
    )
    # Plain attribute (not a parameter/buffer): the derived geometry never
    # enters the census, it only serves the forward.
    dit._hybrid8_geometry = geometry
    # Plain attribute likewise: the TP world the tree was assembled under
    # (``disable_tp`` construction stores 1 -- that tree holds the full
    # replica, which is exactly what the VRAM fuse must budget for).
    dit._hybrid8_tp_world_size = tp_world_size

    produced = {name for name, _ in dit.named_parameters()}
    produced.update(name for name, _ in dit.named_buffers())
    if produced != set(inventory):
        unseen = sorted(set(inventory) - produced)
        undeclared = sorted(produced - set(inventory))
        raise DenseHybridStructureError(
            "the assembled hybrid8 module census does not match the pinned 535-tensor "
            f"manifest: missing={unseen[:8]}{'...' if len(unseen) > 8 else ''} "
            f"extra={undeclared[:8]}{'...' if len(undeclared) > 8 else ''}"
        )
    logger.info(
        "h3-forge dense hybrid8 DiT host-load: %d blocks, %d-dim conditioning, %d tensors "
        "(tensor-parallel world %d: %d/%d local attention heads%s), "
        "validated against the pinned manifest from %s (schema %s)",
        form.num_blocks,
        form.cond_dim,
        len(inventory),
        tp_world_size,
        local_heads,
        geometry.num_heads,
        ", vLLM TP layers disabled (no TP group in this process)" if disable_tp else "",
        form.source,
        form.observed_schema_sha256,
    )


_HYBRID8_FORWARD_KWARGS = frozenset(
    {
        "x",
        "audio_x",
        "img_position_ids",
        "unique_timesteps",
        "inverse_indices",
        "update_mask",
        "update_audio_mask",
        "token_tags",
        "skip_mask_out_condition",
        "prompt_embeds",
        "img_pos_info",
        "audio_pos_info",
        "text_pos_info",
        "img_pos_for_infer_output_info",
        "packed_seq_params",
        "refiner_packed_seq_params",
        "video_token_layout",
    }
)


def _hybrid8_required_kwarg(kwargs: dict[str, Any], key: str) -> Any:
    if key not in kwargs or kwargs[key] is None:
        raise ValueError(f"hybrid8 forward requires kwarg {key!r}")
    return kwargs[key]


def _hybrid8_pos_ids(pos_info: Any, key: str) -> torch.Tensor:
    """Position-id extraction, mirroring ``MiniMaxH3DiTModel._pos_ids``."""

    if isinstance(pos_info, dict):
        ids = pos_info.get("position_ids")
    else:
        ids = getattr(pos_info, "position_ids", None)
    if ids is None:
        raise ValueError(f"{key}.position_ids is required")
    return ids.view(-1).to(torch.long)


def _hybrid8_psp_field(psp: Any, key: str, field: str) -> Any:
    if isinstance(psp, dict):
        value = psp.get(field)
    else:
        value = getattr(psp, field, None)
    if value is None:
        raise ValueError(f"{key}.{field} is required")
    return value


class DenseHybridDiT(OfficialMiniMaxH3DiTModel):
    """Official H3 DiT host with fail-closed dense assembly in two forms.

    **Official old form** (default; no hybrid8 signature in the transformer
    census): the served dense checkpoint is the official-isomorphic bf16
    export of the full ``minimax_h3_ref2va_int8_convrot`` source --
    official online-AdaLN naming end to end (the time embedder ships as
    official ``time_embedder.*`` tensors, every block keeps a full
    ``time_embed_dim``-input AdaLN projection, ``rope.inv_freq`` is
    included), so construction and weight mapping are exactly the official
    path -- grouped-QKV layout reorder, fused ``mlp.fc1`` gate/up split and
    all -- plus the full-coverage census below.  Hybrid8/curve-side
    artifacts stay rejected on this route.

    **Hybrid8 dense form** (10Eros shape): when the transformer census
    carries the hybrid8 top-level signature, construction validates the
    census against the pinned 535-tensor manifest of
    ``h3-transformer-50l-hybrid8-bf16-plain`` item by item
    (:func:`discover_hybrid8_dit_form`, fail-closed on any deviation) and
    assembles the hybrid8 tree instead: per-block narrow
    ``adaln_proj.linear`` ``[96768, cond]`` projections fed by the shared
    top-level ``adaln_t_table`` conditioning, no online time embedder, the
    token refiner / attention / MLP / final-layer skeleton unchanged.  The
    module census is locked 1:1 to the manifest and the weight stream must
    cover it exactly.  The hybrid8 form implements the official
    ``MiniMaxH3DiTModel.forward`` kwargs contract (issue 009): the packed
    multimodal embedding, per-block six-segment modulation over
    ``combined_indices``, packed non-causal attention over ``cu_seqlens``,
    split-half RoPE, and the dual final heads -- all with the timestep
    conditioning lerped from the top-level table instead of an online time
    embedder (see :meth:`hybrid8_conditioning`).  Since issue 010 the
    hybrid8 tree is tensor-parallel exactly like the official transformer:
    every projection is a vLLM ``ColumnParallelLinear``/``RowParallelLinear``
    /``QKVParallelLinear``/``MergedColumnParallelLinear`` with local
    attention heads, so each rank holds its shard (~30 GiB at tp_size=2
    instead of the full ~60 GiB replica issues 008/009 carried) while the
    gather-output heads still return the full logits on every rank -- the
    same external contract, now with the official internal layout.

    Despite the "hybrid" in the class name (the registered arch), the
    Dasiwa *curve-cache* source -- ``adaln_t_table`` over serialized INT8
    ConvRot payloads -- remains cache-only and is rejected in both forms.
    """

    def __init__(self, od_config: Any, *args: Any, **kwargs: Any) -> None:
        # Serve-activation (issue 008): the model root is resolved from every
        # known od_config field spelling, and a resolved root ALWAYS probes the
        # transformer census -- the worker must never take the official branch
        # merely because one field's spelling or value shape drifted.  Each
        # outcome is logged so the serve log shows which form was built and
        # from which module copy.
        root = _od_config_model_root(od_config)
        form: Hybrid8DitForm | None = None
        if root is None:
            logger.warning(
                "h3-forge dense DiT construction found no model root on od_config "
                "(probed fields: %s); constructing the official dense form -- a "
                "hybrid8 checkpoint cannot be detected without a root",
                ", ".join(_MODEL_ROOT_FIELD_CANDIDATES),
            )
        else:
            model_path, root_field = root
            try:
                form = discover_hybrid8_dit_form(model_path)
            except DenseHybridStructureError:
                logger.exception(
                    "hybrid8 dense-form validation failed for the transformer "
                    "checkpoint under %s (od_config.%s); failing closed instead "
                    "of falling back to the official form",
                    model_path,
                    root_field,
                )
                raise
            if form is None:
                logger.info(
                    "h3-forge dense DiT: no hybrid8 signature under %s (od_config.%s); "
                    "constructing the official dense form",
                    model_path,
                    root_field,
                )
        if form is None:
            super().__init__(od_config, *args, **kwargs)
            self.hybrid8_dit_form: Hybrid8DitForm | None = None
            return
        logger.info(
            "h3-forge dense DiT activates the hybrid8 form from %s (od_config.%s, "
            "module %s): %d blocks, %d-dim conditioning, schema %s",
            form.source,
            root[1] if root is not None else "?",
            Path(__file__).parent,
            form.num_blocks,
            form.cond_dim,
            form.observed_schema_sha256,
        )
        # Hybrid8 form: the official module tree (time embedder, full-width
        # AdaLN) has no counterpart in the 535-tensor manifest, so the tree
        # is assembled from the validated manifest instead of the official
        # constructor.  ``arch`` is deliberately left unset: official-host
        # attributes (``arch.time_embed_dim``) must never be consulted for a
        # form that has no online time embedder.
        nn.Module.__init__(self)
        self.od_config = od_config
        self.hybrid8_dit_form = form
        # The package root this form was discovered under (issue 011): the
        # VRAM fuse scans it for the trusted non-DiT lower bound and, when
        # the serve pre-flight exported a package fingerprint, recomputes it
        # here so pre-flight numbers about a different package fail closed.
        assert root is not None
        self._hybrid8_model_root = Path(model_path)
        _assemble_hybrid8_dit(self, form)

    def hybrid8_conditioning(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Route timesteps through the top-level conditioning table (fp32).

        **Float timesteps** (the denoise-loop contract: ``unique_timesteps``
        carries the float ``t = 1 - sigma`` values in ``[0, 1]``) are routed
        exactly like the pinned reference runtime
        (``ref/ComfyUI-MiniMaxH3/models/model.py::velocity``,
        ``use_adaln_curves`` branch): ``pos = clamp(t, 0, 1) * (rows - 1)``,
        then a linear interpolation between the two adjacent table rows --
        the table replaces the online time embedder, and its rows already
        carry the embedder's SiLU (207 evidence:
        ``silu_t_emb_grid = silu(time_embedder(t))`` aligned with the table
        rows), so no activation is applied on top.  Out-of-range values are
        clamped onto the grid like the reference; non-finite input fails
        closed.

        **Integer timesteps** select table rows directly (the host-load
        route); an out-of-range index is rejected.
        """

        form = getattr(self, "hybrid8_dit_form", None)
        if form is None:
            raise DenseHybridStructureError("hybrid8_conditioning is only defined for the hybrid8 dense form")
        table = getattr(self, _HYBRID8_T_TABLE_NAME, None)
        if table is None:
            raise DenseHybridStructureError("the hybrid8 conditioning table is not loaded")
        if timesteps.ndim != 1:
            raise DenseHybridStructureError(
                f"hybrid8 conditioning timesteps must be 1-D, got shape {tuple(timesteps.shape)}"
            )
        rows = table.shape[0]
        table = table.to(torch.float32)
        if not timesteps.dtype.is_floating_point:
            if bool(((timesteps < 0) | (timesteps >= rows)).any()):
                raise DenseHybridStructureError(f"hybrid8 conditioning timesteps out of range [0, {rows - 1}]")
            return table.index_select(0, timesteps.long())
        values = timesteps.to(torch.float32)
        if not bool(torch.isfinite(values).all()):
            raise DenseHybridStructureError("hybrid8 conditioning timesteps must be finite")
        pos = values.clamp(0.0, 1.0) * (rows - 1)
        i0 = pos.floor().to(torch.long).clamp(max=rows - 2)
        frac = (pos - i0.to(torch.float32)).unsqueeze(1)
        return torch.lerp(table.index_select(0, i0), table.index_select(0, i0 + 1), frac)

    def forward(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """The official packed forward contract, dispatched per form.

        The hybrid8 form runs :meth:`_hybrid8_forward` (official kwargs and
        output semantics, table-routed conditioning); the official form
        keeps the host implementation untouched.
        """

        if getattr(self, "hybrid8_dit_form", None) is not None:
            return self._hybrid8_forward(**kwargs)
        return super().forward(**kwargs)

    def _hybrid8_embed(
        self,
        *,
        x: torch.Tensor,
        audio_x: torch.Tensor,
        text_embeddings_selected: torch.Tensor,
        unique_timesteps: torch.Tensor,
        img_pos: torch.Tensor,
        audio_pos: torch.Tensor,
        text_pos: torch.Tensor,
        refiner_cu_seqlens: torch.Tensor,
        seq_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the packed multimodal embedding rows (official ``_embed``).

        Returns ``(decoder_input [S, H] bf16, t_emb [M, cond] fp32)``.  The
        latent patch projections keep the official fp32 compute (the
        BF16-manifest weights are upcast per call -- the official
        parameters are fp32 natively); the text path runs the condition
        projection and the token refiner in bf16 exactly like the official
        path.
        """

        # Latent embedders stay fp32 in and out (official fp32 parameters);
        # their gathered outputs are cast to the bf16 sequence dtype only
        # during indexed scattering.  gather_output=True restores the full
        # [rows, hidden] embedding on every rank.
        x_rows = x.view(-1, x.shape[-1]).index_select(0, img_pos).to(torch.float32)
        video_embed, _ = self.video_patch_proj(x_rows)
        audio_rows = audio_x.view(-1, audio_x.shape[-1])
        audio_rows = audio_rows.index_select(0, audio_pos).to(torch.float32)
        audio_embed, _ = self.audio_patch_proj(audio_rows)

        text_rows = text_embeddings_selected.to(device=device, dtype=self.condition_proj.weight.dtype)
        if text_rows.shape[-1] != self._hybrid8_geometry.text_dim:
            raise DenseHybridStructureError(
                f"hybrid8 text rows {tuple(text_rows.shape)} do not match the condition "
                f"projection input width {self._hybrid8_geometry.text_dim}"
            )
        text_embed, _ = self.condition_proj(text_rows)
        text_embed = self.token_refiner(text_embed, cu_seqlens=refiner_cu_seqlens)

        hidden_size = self._hybrid8_geometry.hidden_size
        sequence_dtype = self.condition_proj.weight.dtype
        embeddings = torch.zeros(
            (seq_len, hidden_size),
            device=device,
            dtype=sequence_dtype,
        )
        embeddings.index_add_(
            0,
            text_pos,
            text_embed.to(sequence_dtype)[: text_pos.shape[0]],
        )
        embeddings.index_add_(
            0,
            img_pos,
            video_embed.to(sequence_dtype)[: img_pos.shape[0]],
        )
        embeddings.index_add_(
            0,
            audio_pos,
            audio_embed.to(sequence_dtype)[: audio_pos.shape[0]],
        )
        t_emb = self.hybrid8_conditioning(unique_timesteps)
        return embeddings, t_emb

    def _hybrid8_forward(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Packed inference forward for the hybrid8 form (issue 009).

        Mirrors ``MiniMaxH3DiTModel.forward`` line for line -- strict kwargs
        contract, packed multimodal embedding, ``combined_indices``
        modulation addressing, block stack, dual final heads, output-row
        selection and update masking -- with the timestep embedding coming
        from the top-level conditioning table and without the
        sequence-parallel prepare/gather hooks (tensor parallelism shards
        the projections per rank but sequence parallelism is not configured
        for the hybrid8 tree; every rank still computes the whole packed
        sequence and the gather-output heads return the full logits).
        Returns ``(video_logits, audio_logits)``.

        Entry order (issue 011 QA): the per-rank VRAM fuse runs after the
        raw-kwarg extraction and the cheap shape checks but BEFORE any
        dtype/device conversion -- every ``.to()`` allocates a fresh tensor,
        so an over-cap request must die with the readable refusal instead of
        an OOM inside a conversion.
        """

        # Strict keyword contract: refuse any kwarg forward does not consume.
        unexpected = sorted(set(kwargs) - _HYBRID8_FORWARD_KWARGS)
        if unexpected:
            raise TypeError(
                "hybrid8 forward received unexpected kwargs: "
                f"{unexpected}; supported kwargs: {sorted(_HYBRID8_FORWARD_KWARGS)}"
            )

        # Extract every required input RAW first -- no dtype/device
        # conversion, no tensor work: each ``.to()`` below would allocate a
        # fresh GPU tensor, and the VRAM fuse must run before the first one
        # so an over-cap request dies with the readable refusal instead of
        # an OOM inside a conversion (issue 011 QA).
        x = _hybrid8_required_kwarg(kwargs, "x")
        audio_x = _hybrid8_required_kwarg(kwargs, "audio_x")
        img_position_ids = _hybrid8_required_kwarg(kwargs, "img_position_ids")
        unique_timesteps = _hybrid8_required_kwarg(kwargs, "unique_timesteps")
        inverse_indices_raw = _hybrid8_required_kwarg(kwargs, "inverse_indices")
        update_mask = _hybrid8_required_kwarg(kwargs, "update_mask")
        token_tags_raw = _hybrid8_required_kwarg(kwargs, "token_tags")
        skip_mask_out_condition = bool(kwargs.get("skip_mask_out_condition", False))
        text_selected = _hybrid8_required_kwarg(kwargs, "prompt_embeds")
        psp = _hybrid8_required_kwarg(kwargs, "packed_seq_params")
        cu_seqlens_raw = _hybrid8_psp_field(psp, "packed_seq_params", "cu_seqlens_q")
        refiner_psp = _hybrid8_required_kwarg(kwargs, "refiner_packed_seq_params")
        refiner_cu_raw = _hybrid8_psp_field(refiner_psp, "refiner_packed_seq_params", "cu_seqlens_q")
        if kwargs.get("video_token_layout") is not None:
            pass  # block-sparse attention hint: the segment-SDPA path ignores it

        geometry = self._hybrid8_geometry
        if x.dim() != 3 or x.shape[0] != 1:
            raise ValueError(f"x must be [1, S, C], got {list(x.shape)}")
        if audio_x.shape[:2] != x.shape[:2]:
            raise ValueError(f"audio_x must share the packed layout of x, got {list(audio_x.shape)} vs {list(x.shape)}")
        if x.shape[-1] != geometry.video_patch_dim:
            raise ValueError(
                f"x carries {x.shape[-1]} video channels, the pinned patch width is {geometry.video_patch_dim}"
            )
        if audio_x.shape[-1] != geometry.audio_patch_dim:
            raise ValueError(
                f"audio_x carries {audio_x.shape[-1]} audio channels, the pinned width is {geometry.audio_patch_dim}"
            )
        seq_len = int(x.shape[1])
        # VRAM fuse (issue 011): with the real packed length known from the
        # raw ``x`` shape alone, refuse an over-cap request BEFORE any
        # conversion or allocation -- a readable error instead of the
        # 009-style qkv CUDA OOM that poisoned the whole worker process.
        _hybrid8_enforce_vram_budget(self, stage="forward", seq_len=seq_len)

        # Only now the dtype/device work (each ``.to`` may allocate).
        inverse_indices = inverse_indices_raw.view(-1).to(torch.long)
        token_tags = token_tags_raw.view(-1).to(torch.long)
        img_pos = _hybrid8_pos_ids(_hybrid8_required_kwarg(kwargs, "img_pos_info"), "img_pos_info")
        audio_pos = _hybrid8_pos_ids(_hybrid8_required_kwarg(kwargs, "audio_pos_info"), "audio_pos_info")
        text_pos = _hybrid8_pos_ids(_hybrid8_required_kwarg(kwargs, "text_pos_info"), "text_pos_info")
        infer_out_pos = _hybrid8_pos_ids(
            _hybrid8_required_kwarg(kwargs, "img_pos_for_infer_output_info"),
            "img_pos_for_infer_output_info",
        )
        cu_seqlens = cu_seqlens_raw.to(torch.int32)
        refiner_cu = refiner_cu_raw.to(torch.int32)
        if token_tags.shape[0] != seq_len:
            raise ValueError(f"token_tags must cover the full packed sequence ({seq_len}), got {token_tags.shape[0]}.")
        if inverse_indices.shape[0] != seq_len:
            raise ValueError(f"inverse_indices must be [{seq_len}], got {list(inverse_indices.shape)}")
        device = x.device
        cu_seqlens = cu_seqlens.to(device)
        refiner_cu = refiner_cu.to(device)
        rope_freqs = _hybrid8_rope_freqs(self.rope.inv_freq, img_position_ids).to(device)

        decoder_input, t_emb = self._hybrid8_embed(
            x=x,
            audio_x=audio_x,
            text_embeddings_selected=text_selected,
            unique_timesteps=unique_timesteps.view(-1).to(device),
            img_pos=img_pos.to(device),
            audio_pos=audio_pos.to(device),
            text_pos=text_pos.to(device),
            refiner_cu_seqlens=refiner_cu,
            seq_len=seq_len,
            device=device,
        )

        combined_indices = (inverse_indices * _HYBRID8_ADALN_MODALITY_NUM + token_tags.clamp(min=0)).to(device)
        inverse_indices = inverse_indices.to(device)

        hidden = decoder_input
        for block in self.blocks:
            hidden = block(
                hidden,
                t_emb=t_emb,
                combined_indices=combined_indices,
                rope_freqs=rope_freqs,
                cu_seqlens=cu_seqlens,
            )
        video_logits, audio_logits = self.final_layer(
            hidden,
            t_emb=t_emb,
            inverse_indices=inverse_indices,
        )

        # Select target and condition rows at inference-output positions, then
        # zero the condition rows.
        video_logits = video_logits.index_select(0, infer_out_pos.to(device))
        audio_logits = audio_logits.index_select(0, audio_pos.to(device))
        if not skip_mask_out_condition:
            update_mask = update_mask.view(-1).to(device)
            if update_mask.shape[0] != video_logits.shape[0]:
                raise ValueError(f"update_mask length mismatch: {update_mask.shape[0]} != {video_logits.shape[0]}")
            video_logits = video_logits * update_mask.unsqueeze(-1)
            # Audio has no condition rows in the supported tasks, so its
            # derived update mask is all ones. Honor an explicit mask when
            # provided.
            update_audio_mask = kwargs.get("update_audio_mask")
            if update_audio_mask is not None:
                audio_logits = audio_logits * update_audio_mask.view(-1).unsqueeze(-1)
        return video_logits, audio_logits

    def _load_hybrid8_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load the hybrid8 stream against the pinned 535-tensor manifest.

        Entry order (both fail-closed, neither allocates): the per-rank VRAM
        fuse (:meth:`_hybrid8_enforce_vram_budget`, issue 011) runs FIRST --
        an over-cap deployment must not copy a single checkpoint byte -- and
        then the trusted fused-QKV row-layout declaration below.

        Fail-closed per tensor: an unknown name (extra tensor), a duplicate,
        or a dtype/shape deviation from the manifest aborts the load (no
        silent ``copy_`` cast), and a stream that does not cover the whole
        manifest is an incomplete checkpoint -- the official export's
        complete-roster/missing-refused semantics, continued.

        Rank-aware sharding follows the official ``load_weights`` (:1001)
        exactly: every projection parameter carries its vLLM parallel
        layer's ``weight_loader``, which narrows the checkpoint tensor to
        this rank's shard (column layers slice output rows, row layers
        slice input columns, the fused qkv splits per head section); the
        fused ``mlp.fc1`` rows are chunked into gate/up with shard ids 0/1
        like the official loader.  Replicated parameters and buffers (norms,
        conditioning tables, ``rope.inv_freq``) take the plain ``copy_``.

        QKV row layout (the official contract, issue 010 QA blocker): the
        checkpoint's fused ``qkv_proj.weight`` rows are
        **grouped-for-official-loader** -- the dense-bf16 export publishes
        ``[Q0;K0;V0;Q1;K1;V1;...]`` (``native_export.py`` applies the exact
        inverse permutation at export), and the official loader reorders
        them into the runtime ``[Q;K;V]`` rows *before* the parallel layer
        shards (official :1015-1024).  This loader mirrors that reorder bit
        for bit first; handing the grouped tensor straight to
        ``QKVParallelLinear.weight_loader`` would silently shard mixed
        Q/K/V rows as this rank's q/k/v heads.  The declaration must be
        trusted, never guessed: ``form.qkv_layout`` comes from the export
        manifest (``grouped-for-official-loader``) whose receipt
        :func:`_hybrid8_checkpoint_qkv_layout` already bound to exactly
        these payload files (shard digests + weight-map index +
        output_tensor_count -- a manifest copied next to a raw runtime-qkv
        source is refused at discovery), and an undeclared checkpoint
        (e.g. a raw Comfy runtime-qkv source -- the same name/dtype/shape
        census as a grouped export, indistinguishable by content) fails
        closed here instead of loading under an assumption.
        """

        form = self.hybrid8_dit_form
        assert form is not None
        # VRAM fuse (issue 011), BEFORE the stream is consumed: if the
        # estimated per-rank peak already exceeds the cap, no checkpoint byte
        # is copied and no GPU allocation happens -- the container-side
        # pre-flight is the first line of defense, this is the second.
        _hybrid8_enforce_vram_budget(self, stage="load")
        if form.qkv_layout is None:
            raise DenseHybridStructureError(
                f"the dense hybrid8 H3 DiT refuses to load the checkpoint under {form.source}: "
                "its fused-QKV row layout is undeclared.  The hybrid8 route serves dense-bf16 "
                "export packages whose qkv rows are grouped-for-official-loader (export via "
                "`export-native --profile dense-bf16` / export_dense_bf16_checkpoint, manifest "
                "h3-comfy-native-export.json).  A raw Comfy runtime-qkv source carries no "
                "declaration and cannot be told from a grouped export by census -- export it "
                "first (fail-closed, no layout guessing)."
            )
        geometry = self._hybrid8_geometry
        inventory = dict(form.inventory)
        slots = dict(self.named_parameters())
        slots.update(dict(self.named_buffers()))
        if set(slots) != set(inventory):
            raise DenseHybridStructureError("the assembled hybrid8 module census no longer matches the pinned manifest")
        loaded: set[str] = set()
        for name, tensor in weights:
            if name.endswith(_QUANT_MARKER_SUFFIXES):
                raise DenseHybridStructureError(
                    f"dense hybrid8 H3 DiT rejects serialized-quantization checkpoint tensor "
                    f"{name!r}; serve the marker-free dense bf16 hybrid8 checkpoint instead"
                )
            if name not in inventory:
                raise DenseHybridStructureError(
                    f"dense hybrid8 H3 DiT rejects checkpoint tensor {name!r}: it is not part "
                    "of the pinned hybrid8 535-tensor manifest"
                )
            if name in loaded:
                raise DenseHybridStructureError(f"dense hybrid8 H3 DiT received duplicate checkpoint tensor {name!r}")
            expected_dtype, expected_shape = inventory[name]
            actual_dtype = _TORCH_DTYPE_NAMES.get(str(tensor.dtype))
            if tuple(tensor.shape) != expected_shape or actual_dtype != expected_dtype:
                raise DenseHybridStructureError(
                    f"dense hybrid8 H3 DiT tensor {name!r} is {actual_dtype}{tuple(tensor.shape)}, "
                    f"the pinned manifest requires {expected_dtype}{expected_shape}"
                )
            param = slots[name]
            weight_loader = getattr(param, "weight_loader", None)
            if weight_loader is None:
                # Replicated parameter/buffer: the manifest shape is the
                # parameter shape, a plain copy is the whole job.
                param.data.copy_(tensor)
            elif name.endswith("attn.qkv_proj.weight"):
                # Official ``load_weights`` (:1015-1024) reorders the grouped
                # checkpoint rows into the runtime [q; k; v] layout before
                # the QKV layer shards q/k/v per rank; the hybrid8 checkpoint
                # is the same grouped-for-official-loader export payload, so
                # the same reorder runs first (``form.qkv_layout`` is the
                # manifest-backed declaration that the rows ARE grouped).
                weight_loader(
                    param,
                    _reorder_grouped_qkv_to_qkv(
                        tensor,
                        num_query_groups=geometry.num_heads,
                        heads_per_group=1,
                        head_dim=geometry.head_dim,
                    ),
                )
            elif name.endswith("mlp.fc1.weight"):
                gate, up = tensor.chunk(2, dim=0)
                weight_loader(param, gate, 0)
                weight_loader(param, up, 1)
            else:
                weight_loader(param, tensor)
            loaded.add(name)
        missing = sorted(set(inventory) - loaded)
        if missing:
            raise RuntimeError(
                f"dense hybrid8 H3 DiT checkpoint is incomplete: {len(missing)} tensors not "
                f"loaded: {missing[:8]}{'...' if len(missing) > 8 else ''}"
            )
        return loaded

    def post_load_weights(self) -> None:
        """Post-load guards, per form (issue 008).

        The official pipeline's ``load_weights`` calls this on the transformer
        after the stream is consumed.  For the **hybrid8 form** the official
        fp32-parameter invariants (``MINIMAX_H3_FP32_PARAM_NAMES``:
        ``video_patch_proj.*``, ``final_layer.video_out.*`` ...) are not the
        contract: the pinned 535-tensor manifest pins the *checkpoint*
        tensors **BF16**, and every loaded tensor's dtype was already
        enforced item by item against that manifest in
        :meth:`_load_hybrid8_weights` (the parallel layers widen their BF16
        shards once at load; issue 008's ``must stay fp32`` abort came from
        checking *checkpoint* dtype against *official* parameter dtype
        before this override existed).  The **official form** keeps the
        official check.
        """

        if getattr(self, "hybrid8_dit_form", None) is not None:
            logger.debug("h3-forge dense hybrid8 DiT post-load: manifest dtypes already enforced, skipping fp32 guards")
            return
        official_post_load = getattr(OfficialMiniMaxH3DiTModel, "post_load_weights", None)
        if official_post_load is not None:
            official_post_load(self)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        if getattr(self, "hybrid8_dit_form", None) is not None:
            return self._load_hybrid8_weights(weights)
        time_embed_dim = self.arch.time_embed_dim
        inspected: list[tuple[str, torch.Tensor]] = []
        for name, tensor in weights:
            if name in _DIT_FORBIDDEN_NAMES:
                raise DenseHybridStructureError(
                    f"dense hybrid H3 DiT (official form) rejects checkpoint tensor {name!r}; "
                    "the hybrid8 dense form must be served through its pinned-manifest "
                    "host-load route, not the official-form loader"
                )
            if name.endswith(_QUANT_MARKER_SUFFIXES):
                raise DenseHybridStructureError(
                    f"dense hybrid H3 DiT rejects serialized-quantization checkpoint tensor "
                    f"{name!r}; serve the dequantized dense bf16 hybrid checkpoint instead"
                )
            if name.endswith(".adaln_proj.linear.weight"):
                width = int(tensor.shape[-1])
                if width != time_embed_dim:
                    raise DenseHybridStructureError(
                        f"{name} has AdaLN input width {width}, expected {time_embed_dim}; a narrow "
                        "AdaLN projection is hybrid8/curve-shaped and requires the hybrid8 "
                        "dense host-load route (pinned 535-tensor manifest)"
                    )
            inspected.append((name, tensor))
        loaded = super().load_weights(inspected)
        expected = {name for name, _ in self.named_parameters()}
        expected.update(name for name, _ in self.named_buffers())
        missing = sorted(expected - loaded)
        if missing:
            raise RuntimeError(
                f"dense hybrid H3 DiT checkpoint is incomplete: {len(missing)} params not loaded: {missing}"
            )
        return loaded


def _hybrid8_rank_state_bytes(dit: nn.Module) -> int:
    """Parameter + buffer bytes the assembled tree holds (arithmetic, no alloc)."""
    total = 0
    for tensor in list(dit.parameters()) + list(dit.buffers()):
        total += tensor.numel() * tensor.element_size()
    return total


def _hybrid8_enforce_vram_budget(dit: nn.Module, *, stage: str, seq_len: int | None = None) -> None:
    """Optional per-rank VRAM guard (issue 011 fuse, simplified).

    The authoritative VRAM gate runs in the server-side preflight (Docker
    container, real card capacity). This in-process guard is armed only when
    ``COMFY_OMNI_HYBRID8_VRAM_CAP_MIB`` is set: it refuses a load/forward whose
    assembled parameter+buffer bytes exceed the cap, so the container fails
    with a readable refusal instead of a CUDA OOM poisoning the worker.
    """
    raw = os.environ.get("COMFY_OMNI_HYBRID8_VRAM_CAP_MIB")
    if raw is None:
        logger.info("dense hybrid8 VRAM fuse disabled (COMFY_OMNI_HYBRID8_VRAM_CAP_MIB unset) at %s", stage)
        return
    cap_mib = int(raw)
    state_bytes = _hybrid8_rank_state_bytes(dit)
    if state_bytes > cap_mib * 1024 * 1024:
        raise DenseHybridStructureError(
            f"dense hybrid8 {stage} refuses {state_bytes} bytes of assembled state against the VRAM cap {cap_mib} MiB"
        )


def discover_hybrid8_dit_form(model_path: str | Path) -> Hybrid8DitForm | None:
    """Detect and validate the hybrid8 dense form under a package root.

    The E4/Ref2VA ConvRot export and the 10Eros hybrid8 plain export both
    publish ``export.plan.json`` next to the shards; its target census
    (``actions[*].target_name/target_dtype/target_shape``) is the authoritative
    inventory for the runtime, and the payload header census must match it
    name/dtype/shape one to one (fail-closed). ``grouped-for-official-loader``
    is the only servable QKV row layout. A census without the ``adaln_t_table``
    signature returns ``None`` so the official old form keeps its route; any
    declared-but-unmatched census raises instead of guessing.
    """
    root = Path(model_path)
    transformer_dir = None
    for candidate in _TRANSFORMER_DIR_CANDIDATES:
        probe = root / candidate
        if (probe / "export.plan.json").is_file():
            transformer_dir = probe
            break
    if transformer_dir is None:
        return None
    plan = json.loads((transformer_dir / "export.plan.json").read_text(encoding="utf-8"))
    if plan.get("component") != "transformer" or plan.get("output_schema") != "h3-comfy-int8-export/v2":
        logger.info("dense hybrid8 discovery: export plan is not the authorized convrot schema; official route")
        return None
    qkv = plan.get("qkv_layout")
    if not isinstance(qkv, dict) or qkv.get("target_layout") != _HYBRID8_QKV_GROUPED_LAYOUT:
        raise DenseHybridStructureError(f"dense hybrid8 refuses undeclared fused-QKV layout under {transformer_dir}")
    inventory: dict[str, tuple[str, tuple[int, ...]]] = {}
    for action in plan.get("actions", []):
        name = action.get("target_name")
        if name is None or action.get("target_dtype") is None:
            continue
        inventory.setdefault(name, (action["target_dtype"], tuple(action["shape"])))
    if _HYBRID8_T_TABLE_NAME not in inventory:
        logger.info("dense hybrid8 discovery: no adaln_t_table signature under %s; official route", transformer_dir)
        return None
    if any(name.startswith("time_embedder.") for name in inventory):
        raise DenseHybridStructureError(f"dense hybrid8 census under {transformer_dir} carries time_embedder.* tensors")
    index_path = transformer_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise DenseHybridStructureError(f"dense hybrid8 checkpoint index missing under {transformer_dir}")
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    census: dict[str, tuple[str, tuple[int, ...]]] = {}
    _TORCH_TO_CENSUS = {
        torch.bfloat16: "BF16",
        torch.float16: "F16",
        torch.float32: "F32",
        torch.uint8: "U8",
    }
    for shard in sorted(set(weight_map.values())):
        with safe_open(str(transformer_dir / shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                ctype = _TORCH_TO_CENSUS.get(handle.get_slice(key).get_dtype())
                if ctype is None:
                    raise DenseHybridStructureError(
                        f"dense hybrid8 tensor {key!r} carries unsupported dtype {handle.get_slice(key).get_dtype()}"
                    )
                census[key] = (ctype, tuple(int(v) for v in handle.get_slice(key).get_shape()))
    missing = sorted(set(inventory) - set(census))
    extra = sorted(set(census) - set(inventory))
    if missing or extra:
        raise DenseHybridStructureError(
            f"transformer census under {transformer_dir} does not match the export plan: "
            f"missing={missing[:8]} extra={extra[:8]}"
        )
    wrong = sorted(name for name in inventory if inventory[name] != census[name])
    if wrong:
        details = "; ".join(f"{name}: {census[name]} != {inventory[name]}" for name in wrong[:4])
        raise DenseHybridStructureError(
            f"transformer census under {transformer_dir} deviates from the export plan on "
            f"{len(wrong)} tensors: {details}"
        )
    if any(name.endswith((".comfy_quant", ".weight_scale")) for name in census):
        raise DenseHybridStructureError(
            f"dense hybrid8 census under {transformer_dir} carries serialized-quantization markers"
        )
    table_shape = census[_HYBRID8_T_TABLE_NAME][1]
    cond_dim = table_shape[1]
    return Hybrid8DitForm(
        inventory=inventory,
        observed_schema_sha256=manifest_schema_sha256(inventory),
        num_blocks=len(block_indices(inventory, "blocks")),
        cond_dim=cond_dim,
        source=str(transformer_dir),
        qkv_layout=str(qkv.get("target_layout")),
        transformer_dir=str(transformer_dir),
    )


class _ScopedAttribute:
    """Temporarily replace a module attribute and restore it afterwards."""

    def __init__(self, module: Any, name: str, value: Any) -> None:
        self.module = module
        self.name = name
        self.value = value
        self.previous: Any = None
        self.applied = False

    def apply(self) -> None:
        self.previous = getattr(self.module, self.name, None)
        setattr(self.module, self.name, self.value)
        self.applied = True

    def restore(self) -> None:
        if not self.applied:
            return
        setattr(self.module, self.name, self.previous)
        self.applied = False

    def __enter__(self) -> _ScopedAttribute:
        self.apply()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.restore()


def construct_dense_pipeline(od_config: Any, prefix: str = "") -> Any:
    """Construct the official H3 pipeline with the DiT class scoped to DenseHybridDiT.

    The official pipeline builds ``MiniMaxH3DiTModel`` through the name it
    imported from the transformer module; swapping that module attribute for
    the construction duration makes the official constructor instantiate the
    hybrid8 tree in place (no wasted double build). Returns the constructed
    official pipeline instance; the caller adopts its state when the package
    served a hybrid8 form.
    """
    import vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 as official_pipeline_module

    with _ScopedAttribute(official_pipeline_module, "MiniMaxH3DiTModel", DenseHybridDiT):
        return OfficialMiniMaxH3Pipeline(od_config=od_config, prefix=prefix)
