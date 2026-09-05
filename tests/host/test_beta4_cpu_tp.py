"""Actual-host TP partition evidence and disclosed CPU forward observations.

No TP1/TP2 whole-forward tolerance is authorized here. Only source-slot
loading, rank replication, and finite outputs are pass/fail assertions.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import multiprocessing
import struct
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from beta4_host_fixture import (
    HEAD_DIM,
    HEADS,
    actual_cpu_host,
    packed_inputs,
    tiny_arch_config,
    tiny_inventory,
    tiny_tensor_stream,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("vllm_omni") is None,
    reason="the actual pinned host image is required",
)


def _expected_partition(name, full, rank, world_size):
    """Independent source-to-rank ordering, without adapter/host loader helpers."""
    if name.endswith(".attn.qkv_proj.weight"):
        heads = HEADS // world_size
        grouped = full.reshape(HEADS, 3, HEAD_DIM, -1)
        return grouped[rank * heads : (rank + 1) * heads].transpose(0, 1).reshape(-1, full.shape[1])
    if name.endswith(".mlp.fc1.weight"):
        width = full.shape[0] // (2 * world_size)
        gate_up = full.reshape(2, full.shape[0] // 2, full.shape[1])
        return gate_up[:, rank * width : (rank + 1) * width].reshape(-1, full.shape[1])
    if name in {"adaln_t_table", "adaln_basis", "adaln_mean", "rope.inv_freq"} or "norm" in name:
        return full
    axis = 1 if name.endswith((".attn.out_proj.weight", ".mlp.fc2.weight")) else 0
    width = full.shape[axis] // world_size
    return full.narrow(axis, rank * width, width)


def _tensor_digest(value):
    import torch

    return hashlib.sha256(value.contiguous().view(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _worker(rank, world_size, root):
    root = Path(root)
    with actual_cpu_host(world_size=world_size, rank=rank, init_method=(root / "gloo-init").as_uri()) as host:
        from comfy_omni.integrations.vllm_omni.pipelines import beta4_pipeline as adapter

        with patch.object(adapter.construction, "state", adapter.construction.WorkerConstructionState()):

            class TinyDiT(adapter.H3Beta4DiTModel):
                _inventory = tiny_inventory()
                _architecture = tiny_arch_config()

            empty_component = root / f"synthetic-binding-{rank}"
            empty_component.mkdir()
            model = TinyDiT(host.od_config, binding=SimpleNamespace(component_root=empty_component, files=()))
            weights = dict(tiny_tensor_stream())
            model.load_weights(weights.items())
            state = model.state_dict()
            exact_slots = 0
            for name, full in weights.items():
                target = "time_embedder.adaln_t_table" if name == "adaln_t_table" else name
                expected = _expected_partition(name, full, rank, world_size).to(state[target].dtype)
                assert host.torch.equal(state[target], expected), f"rank {rank}: {name}"
                exact_slots += 1
            assert exact_slots == 534
            inputs = packed_inputs()
            input_digest = hashlib.sha256()

            def bind_inputs(value, path=""):
                if isinstance(value, dict):
                    for name, child in sorted(value.items()):
                        bind_inputs(child, path + "/" + name)
                elif isinstance(value, host.torch.Tensor):
                    input_digest.update(path.encode())
                    input_digest.update(str((str(value.dtype), list(value.shape))).encode())
                    input_digest.update(bytes.fromhex(_tensor_digest(value)))
                else:
                    input_digest.update(str((path, value)).encode())

            bind_inputs(inputs)
            with host.torch.inference_mode():
                outputs = model(**inputs)
            assert all(host.torch.isfinite(value).all() for value in outputs)
            (root / f"rank-{rank}.json").write_text(
                json.dumps(
                    {
                        "rank": rank,
                        "world_size": world_size,
                        "exact_source_slots": exact_slots,
                        "input_sha256": input_digest.hexdigest(),
                        "outputs": [value.tolist() for value in outputs],
                        "output_sha256": [_tensor_digest(value) for value in outputs],
                        "loading": model.loading_receipt(),
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )


def _run(world_size, root):
    root.mkdir()
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_worker, args=(rank, world_size, str(root))) for rank in range(world_size)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=90)
            assert process.exitcode == 0, f"actual CPU TP worker failed: {process.exitcode}"
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    return [json.loads((root / f"rank-{rank}.json").read_text(encoding="utf-8")) for rank in range(world_size)]


def _fp32_order(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return 0x80000000 - (bits & 0x7FFFFFFF) if bits & 0x80000000 else bits + 0x80000000


def test_actual_cpu_tp_partition_bits_and_forward_characterization(tmp_path):
    one = _run(1, tmp_path / "tp1")[0]
    two = _run(2, tmp_path / "tp2")
    assert one["input_sha256"] == two[0]["input_sha256"] == two[1]["input_sha256"]
    assert two[0]["output_sha256"] == two[1]["output_sha256"], "TP ranks must replicate gathered output"
    observations = []
    for name, left, right in zip(("video", "audio"), one["outputs"], two[0]["outputs"], strict=True):
        pairs = [(a, b) for row_a, row_b in zip(left, right, strict=True) for a, b in zip(row_a, row_b, strict=True)]
        errors = [abs(a - b) for a, b in pairs]
        observations.append(
            {
                "modality": name,
                "elements": len(pairs),
                "max_abs": max(errors),
                "rms": (sum(error * error for error in errors) / len(errors)) ** 0.5,
                "max_fp32_ulp": max(abs(_fp32_order(a) - _fp32_order(b)) for a, b in pairs),
                "exact_bits": all(struct.pack("<f", a) == struct.pack("<f", b) for a, b in pairs),
            }
        )
    print(
        "TP_NUMERICS_OBSERVED "
        + json.dumps(
            {
                "status": "TP_NUMERICS_OBSERVED",
                "fixture_sha256": hashlib.sha256(
                    Path(__file__).with_name("beta4_host_fixture.py").read_bytes()
                ).hexdigest(),
                "input_sha256": one["input_sha256"],
                "exact_source_slots_per_rank": [
                    one["exact_source_slots"],
                    *(rank["exact_source_slots"] for rank in two),
                ],
                "cpu_sdpa_dispatch_adapted": True,
                "numerical_pass_authorized": False,
                "observations": observations,
            },
            sort_keys=True,
        )
    )
