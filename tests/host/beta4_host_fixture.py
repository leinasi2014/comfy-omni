"""Synthetic 534-name fixture; never an authority for production beta4 assets.

Independent shapes follow the closed inventory of Apache-2.0 h3-forge
e9cb011d00b028c149db3978de246c54f6e34acc, templates.py blob
443a5cc9ca58891c3852079c8589fbe2f5af6484, minus the unused beta3 grid.
Constructor and packed-input signatures follow vLLM-Omni
17285c2f55a41bf15772676121814d59a60ace35 transformer blob
91e03c865b22ffaaa5dbb3bf3ceeaf804a5564c8. No product beta4 module is imported.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

HIDDEN, HEADS, HEAD_DIM, FFN = 32, 2, 128, 64
CONDITION, TABLE_ROWS, BASIS_WIDTH = 8, 5, 16
VIDEO_WIDTH, AUDIO_WIDTH, TEXT_WIDTH = 8, 4, 16


def tiny_arch_config():
    """Keep head_dim=128/rotary=96 so later GPU tests can use the real kernel."""
    return {
        "num_layers": 50,
        "token_refiner_num_layers": 2,
        "hidden_size": HIDDEN,
        "num_attention_heads": HEADS,
        "attention_head_dim": HEAD_DIM,
        "ffn_hidden_size": FFN,
        "latents_dim": VIDEO_WIDTH,
        "audio_latents_dim": AUDIO_WIDTH,
        "patch_size": (1, 1, 1),
        "text_dim": TEXT_WIDTH,
        "timestep_input_dim": 16,
        "time_embed_hidden_size": HIDDEN,
        "time_embed_dim": CONDITION,
        "adaln_out_features": 18 * HIDDEN,
        "final_adaln_out_features": 2 * HIDDEN,
        "rope_inv_freq_len": 16,
        "norm_eps": 1e-5,
        "qk_norm_eps": 1e-5,
        "final_norm_eps": 1e-5,
    }


def tiny_inventory():
    """Closed 50+2-layer source roster, all BF16; exactly four buffers."""
    block = {
        "norm1.weight": (HIDDEN,),
        "norm2.weight": (HIDDEN,),
        "attn.q_norm.weight": (HEAD_DIM,),
        "attn.k_norm.weight": (HEAD_DIM,),
        "attn.qkv_proj.weight": (3 * HEADS * HEAD_DIM, HIDDEN),
        "attn.out_proj.weight": (HIDDEN, HEADS * HEAD_DIM),
        "mlp.fc1.weight": (2 * FFN, HIDDEN),
        "mlp.fc2.weight": (HIDDEN, FFN),
    }
    shapes = {
        "adaln_t_table": (TABLE_ROWS, CONDITION),
        "adaln_basis": (CONDITION, BASIS_WIDTH),
        "adaln_mean": (BASIS_WIDTH,),
        "rope.inv_freq": (16,),
        "token_refiner.final_norm.weight": (HIDDEN,),
        "final_layer.norm.weight": (HIDDEN,),
        "final_layer.adaln_proj.linear.weight": (2 * HIDDEN, CONDITION),
        "final_layer.adaln_proj.linear.bias": (2 * HIDDEN,),
        "final_layer.video_out.weight": (VIDEO_WIDTH, HIDDEN),
        "final_layer.video_out.bias": (VIDEO_WIDTH,),
        "final_layer.audio_out.weight": (AUDIO_WIDTH, HIDDEN),
        "final_layer.audio_out.bias": (AUDIO_WIDTH,),
    }
    for prefix, width in (
        ("video_patch_proj", VIDEO_WIDTH),
        ("audio_patch_proj", AUDIO_WIDTH),
        ("condition_proj", TEXT_WIDTH),
    ):
        shapes[f"{prefix}.weight"] = (HIDDEN, width)
        shapes[f"{prefix}.bias"] = (HIDDEN,)
    for layer in range(50):
        shapes.update({f"blocks.{layer}.{suffix}": shape for suffix, shape in block.items()})
        shapes[f"blocks.{layer}.adaln_proj.linear.weight"] = (18 * HIDDEN, CONDITION)
        shapes[f"blocks.{layer}.adaln_proj.linear.bias"] = (18 * HIDDEN,)
    for layer in range(2):
        shapes.update({f"token_refiner.blocks.{layer}.{suffix}": shape for suffix, shape in block.items()})
    assert len(shapes) == 534
    return {name: ("BF16", shape) for name, shape in sorted(shapes.items())}


def tiny_tensor_stream():
    """Reconstruct fresh deterministic BF16 source tensors in grouped QKV order.

    Values are small for a stable 50-block forward. QKV/fc1 carry distinct
    per-row BF16 values so loader reorder and gate/up sharding are traceable.
    """
    import torch

    for ordinal, (name, (_, shape)) in enumerate(tiny_inventory().items()):
        if name == "adaln_t_table":
            values = torch.arange(TABLE_ROWS, dtype=torch.float32).unsqueeze(1) / 2 - 1
            values = values + torch.arange(CONDITION, dtype=torch.float32).unsqueeze(0) / 32
        elif name == "rope.inv_freq":
            values = 10000.0 ** (-torch.arange(16, dtype=torch.float32) / 16)
        elif "norm" in name and name.endswith(".weight"):
            values = torch.ones(shape, dtype=torch.float32)
        elif name.endswith(("attn.qkv_proj.weight", "mlp.fc1.weight")):
            # Reinterpret safe finite positive BF16 patterns, avoiding row-id
            # collisions caused by casting integer rows greater than 256.
            rows = (torch.arange(shape[0], dtype=torch.int32) + 0x3000).to(torch.int16)
            values = rows.view(torch.bfloat16).unsqueeze(1).expand(shape).contiguous()
        else:
            count = 1
            for dim in shape:
                count *= dim
            values = ((torch.arange(count, dtype=torch.float32) + ordinal) % 17 - 8).reshape(shape) / 512
        yield name, values.to(torch.bfloat16).contiguous()


def packed_inputs():
    """One real document plus one padding row, all three modalities and masks."""
    import torch

    def positions(values):
        return {"position_ids": torch.tensor(values, dtype=torch.long)}

    return {
        "x": torch.arange(10 * VIDEO_WIDTH, dtype=torch.float32).reshape(1, 10, VIDEO_WIDTH) / 128,
        "audio_x": torch.arange(10 * AUDIO_WIDTH, dtype=torch.float32).reshape(1, 10, AUDIO_WIDTH) / 64,
        "img_position_ids": torch.arange(30, dtype=torch.float32).reshape(1, 10, 3) / 16,
        "unique_timesteps": torch.tensor([0.25, 0.5, 1.0], dtype=torch.float32),
        "inverse_indices": torch.tensor([0, 0, 0, 0, 0, 0, 2, 1, 1, 0], dtype=torch.long),
        "token_tags": torch.tensor([1, 1, 1, 0, 0, 0, 0, 2, 2, -1], dtype=torch.long),
        "update_mask": torch.tensor([1, 1, 1, 0], dtype=torch.float32),
        "update_audio_mask": torch.tensor([1, 0], dtype=torch.float32),
        "prompt_embeds": torch.arange(3 * TEXT_WIDTH, dtype=torch.float32).reshape(3, TEXT_WIDTH).to(torch.bfloat16) / 64,
        "img_pos_info": positions([3, 4, 5, 6]),
        "audio_pos_info": positions([7, 8]),
        "text_pos_info": positions([0, 1, 2]),
        "img_pos_for_infer_output_info": positions([3, 4, 5, 6]),
        "packed_seq_params": {"cu_seqlens_q": torch.tensor([0, 9, 10], dtype=torch.int32), "max_seqlen_q": 9},
        "refiner_packed_seq_params": {"cu_seqlens_q": torch.tensor([0, 3], dtype=torch.int32), "max_seqlen_q": 3},
    }


@contextmanager
def actual_cpu_host():
    """Real TP1 Gloo and official modules; a disclosed CPU selector adaptation.

    17285's UnspecifiedOmniPlatform lacks its attention selector. Only that
    selection method is adapted to return the real official SDPABackend.
    No module import, attention calculation, linear, loader, or collective
    is mocked. The installed vLLM's actual CpuPlatform supplies CPU groups.
    Caller must use an isolated test process with no existing process group.
    """
    with TemporaryDirectory(prefix="beta4-host-cache-") as cache, patch.dict(
        os.environ,
        {
            "USER": "beta4-host-test",
            "TORCHINDUCTOR_CACHE_DIR": str(Path(cache, "inductor")),
            "TRITON_CACHE_DIR": str(Path(cache, "triton")),
            "VLLM_CACHE_ROOT": str(Path(cache, "vllm")),
            "XDG_CACHE_HOME": cache,
        },
    ):
        with _actual_cpu_host() as host:
            yield host


@contextmanager
def _actual_cpu_host():
    import torch
    import vllm.platforms as vllm_platforms
    from vllm.platforms.cpu import CpuPlatform
    from vllm_omni.platforms.interface import UnspecifiedOmniPlatform

    if torch.distributed.is_initialized():
        raise RuntimeError("actual_cpu_host requires an isolated process with no existing distributed group")
    old_threads = torch.get_num_threads()
    with ExitStack() as stack:
        stack.callback(torch.set_num_threads, old_threads)
        torch.set_num_threads(1)
        stack.enter_context(patch.object(vllm_platforms, "_current_platform", CpuPlatform()))
        stack.enter_context(
            patch.object(
                UnspecifiedOmniPlatform,
                "get_diffusion_attn_backend_cls",
                classmethod(lambda cls, **kwargs: "vllm_omni.diffusion.attention.backends.sdpa.SDPABackend"),
            )
        )
        from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
        from vllm.distributed import (
            destroy_distributed_environment,
            destroy_model_parallel,
            init_distributed_environment,
            initialize_model_parallel,
        )
        from vllm_omni.diffusion.config import set_current_diffusion_config
        from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
        from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as official

        od_config = OmniDiffusionConfig(
            model=None,
            model_class_name="MiniMaxH3Pipeline",
            tf_model_config=TransformerConfig.from_dict(tiny_arch_config()),
            diffusion_attention_config={"default": "sdpa"},
        )
        stack.enter_context(set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))))
        stack.enter_context(set_current_diffusion_config(od_config))
        temporary = stack.enter_context(TemporaryDirectory(prefix="beta4-gloo-"))
        stack.callback(destroy_distributed_environment)
        stack.callback(destroy_model_parallel)
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=Path(temporary, "init").as_uri(),
            backend="gloo",
        )
        initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, backend="gloo")
        yield SimpleNamespace(torch=torch, official=official, od_config=od_config, cpu_selector_adapted=True)
