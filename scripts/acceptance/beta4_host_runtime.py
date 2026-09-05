#!/usr/bin/env python3
"""Externally torchrun-launched installed-wheel beta4 DiT acceptance.

No server, subprocess, model conversion, CPU offload or GPU launch is owned
here. Run inside the pinned host image with the designated devices exposed.
Tiny geometry comes only from a separately SHA-bound synthetic fixture;
real mode has no geometry override. TP comparison is observational until a
separate numerical budget is frozen: finite output is not numerical parity.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import resource
import struct
import sys
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

from comfy_omni.artifacts import fileops
from comfy_omni.artifacts.build_identity import installed_tool_identity
from comfy_omni.runtime.h3.beta4_binding import (
    BETA4_RUNTIME_ARCHITECTURE,
    verify_beta4_binding_unchanged,
    verify_beta4_component,
)

MEMORY_LIMIT = 4 * 1024**3
HOST_COMMIT = "17285c2f55a41bf15772676121814d59a60ace35"
SCHEMA = "comfy_omni.beta4.host_acceptance/v1"
COMPARISON_POLICY = {"status": "UNFROZEN", "automatic_pass": False, "legacy_bitwise_claim": False}
SEED = 20260905
COMMON_IDENTITY = (
    "candidate",
    "image_id",
    "harness_sha256",
    "host_commit",
    "mode",
    "fixture_sha256",
    "binding",
    "input_sha256",
    "inputs",
    "numerical_policy",
    "libraries",
    "architecture",
    "cuda_runtime",
    "nccl_version",
    "attention_backend",
    "tf32",
)


def _require(condition, detail):
    if not condition:
        raise ValueError(detail)


def _digest(value):
    return hashlib.sha256(fileops.canonical_json(value)).hexdigest()


def _write(path, document):
    document = {**document, "schema": SCHEMA}
    document.pop("receipt_sha256", None)
    document["receipt_sha256"] = _digest(document)
    fileops.write_exclusive(path, fileops.canonical_json(document))
    return document


def _read_receipt(path):
    _require(path.stat().st_size <= 4 * 1024**2, "receipt exceeds the bounded size")
    payload, _ = fileops.read_file_pinned(path)
    doc = fileops.parse_json_strict(payload)
    _require(isinstance(doc, dict) and fileops.canonical_json(doc) == payload, "receipt must be canonical")
    _require(doc.get("schema") == SCHEMA, "receipt schema differs")
    _require(
        doc.get("receipt_sha256") == _digest({k: v for k, v in doc.items() if k != "receipt_sha256"}),
        "receipt digest differs",
    )
    return doc


def _memory():
    for root, maximum, current, peak in (
        (Path("/sys/fs/cgroup"), "memory.max", "memory.current", "memory.peak"),
        (Path("/sys/fs/cgroup/memory"), "memory.limit_in_bytes", "memory.usage_in_bytes", "memory.max_usage_in_bytes"),
    ):
        path = root / maximum
        if not path.is_file():
            continue
        raw = path.read_text().strip()
        if raw.isdecimal() and 0 < int(raw) <= MEMORY_LIMIT:
            values = {
                "limit_bytes": int(raw),
                "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            }
            for key, name in (("current_bytes", current), ("peak_bytes", peak)):
                observed = root / name
                values[key] = int(observed.read_text().strip()) if observed.is_file() else None
            _require(values["max_rss_bytes"] <= int(raw), "process RSS exceeds the acceptance limit")
            return values
    raise ValueError("acceptance requires an enforced container memory limit of at most 4 GiB")


def _validate_args(args, environ):
    _require(re.fullmatch(r"[0-9a-f]{40}", args.commit) is not None, "candidate commit must be exact")
    _require(re.fullmatch(r"[0-9a-f]{64}", args.wheel_sha256) is not None, "candidate wheel must be exact")
    _require(re.fullmatch(r"[0-9a-f]{64}", args.harness_sha256) is not None, "acceptance script must be exact")
    _require(re.fullmatch(r"sha256:[0-9a-f]{64}", args.image_id) is not None, "resolved image ID is required")
    world, rank, local = (int(environ.get(key, "-1")) for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK"))
    _require(
        world == args.tp and world in {1, 2} and 0 <= rank < world and local == rank,
        "requires external single-node torchrun",
    )
    _require(len(args.device_uuids) == world and len(set(args.device_uuids)) == world, "device UUID roster differs")
    _require(
        all(re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", value) for value in args.device_uuids), "full GPU UUIDs are required"
    )
    visible = environ.get("NVIDIA_VISIBLE_DEVICES", "").split(",")
    _require(
        visible in [args.device_uuids, [str(i) for i in range(world)]],
        "container device exposure must be explicit and exclude GPU 2",
    )
    if args.mode == "real":
        _require(world == 2 and args.component is not None, "real A requires exactly TP2 and its component")
        _require(args.fixture_file is None and args.fixture_sha256 is None, "real A forbids synthetic geometry")
        _require(
            all(
                isinstance(x, str) and re.fullmatch(r"[0-9a-f]{64}", x)
                for x in (args.manifest_sha256, args.manifest_file_sha256)
            ),
            "real A requires both manifest identities",
        )
    else:
        _require(
            args.fixture_file is not None
            and isinstance(args.fixture_sha256, str)
            and re.fullmatch(r"[0-9a-f]{64}", args.fixture_sha256),
            "tiny mode requires a pinned fixture file",
        )
        _require(
            args.component is None and args.manifest_sha256 is None and args.manifest_file_sha256 is None,
            "tiny mode cannot claim a real component",
        )
    return rank, local


def _fixture(path, expected):
    _require(path.stat().st_size <= 128 * 1024, "fixture exceeds the bounded size")
    payload, _ = fileops.read_file_pinned(path)
    _require(hashlib.sha256(payload).hexdigest() == expected, "fixture source differs")
    module = ModuleType("beta4_acceptance_fixture")
    # Execute exactly the bytes checked above; do not reopen via an import loader.
    exec(compile(payload, "<beta4-synthetic-fixture>", "exec"), module.__dict__)
    _require(
        all(
            callable(getattr(module, name, None))
            for name in ("tiny_arch_config", "tiny_inventory", "tiny_tensor_stream", "packed_inputs")
        ),
        "fixture interface differs",
    )
    arch, inventory = module.tiny_arch_config(), module.tiny_inventory()
    _require(
        len(inventory) == 534
        and (
            arch["hidden_size"],
            arch["num_layers"],
            arch["token_refiner_num_layers"],
            arch["num_attention_heads"],
            arch["attention_head_dim"],
            arch["ffn_hidden_size"],
            arch["time_embed_dim"],
            arch["rope_inv_freq_len"],
        )
        == (32, 50, 2, 2, 128, 64, 8, 16),
        "tiny geometry differs from the frozen fixture",
    )
    _require(
        all(
            dtype == "BF16" and shape and all(type(x) is int and 0 < x <= 768 for x in shape)
            for dtype, shape in inventory.values()
        )
        and sum(math.prod(shape) for _, shape in inventory.values()) <= 8_000_000,
        "synthetic tensor allocation exceeds the tiny boundary",
    )
    return module


def _real_inputs(torch):
    def positions(values):
        return {"position_ids": torch.tensor(values, dtype=torch.long)}

    def values(rows, width):
        return ((torch.arange(rows * width, dtype=torch.float32) % 31 - 15) / 128).reshape(rows, width)

    return {
        "x": values(10, 96).unsqueeze(0),
        "audio_x": values(10, 32).unsqueeze(0),
        "prompt_embeds": values(3, 5120).to(torch.bfloat16),
        "img_position_ids": torch.arange(30, dtype=torch.float32).reshape(1, 10, 3) / 16,
        "unique_timesteps": torch.tensor([0.25, 0.5, 1.0]),
        "inverse_indices": torch.tensor([0, 0, 0, 0, 0, 0, 2, 1, 1, 0]),
        "token_tags": torch.tensor([1, 1, 1, 0, 0, 0, 0, 2, 2, -1]),
        "update_mask": torch.tensor([1, 1, 1, 0], dtype=torch.float32),
        "update_audio_mask": torch.tensor([1, 0], dtype=torch.float32),
        "img_pos_info": positions([3, 4, 5, 6]),
        "audio_pos_info": positions([7, 8]),
        "text_pos_info": positions([0, 1, 2]),
        "img_pos_for_infer_output_info": positions([3, 4, 5, 6]),
        "packed_seq_params": {"cu_seqlens_q": torch.tensor([0, 9, 10], dtype=torch.int32), "max_seqlen_q": 9},
        "refiner_packed_seq_params": {"cu_seqlens_q": torch.tensor([0, 3], dtype=torch.int32), "max_seqlen_q": 3},
    }


def _tensor_record(tensor, *, include_bytes=False):
    import torch

    values = tensor.detach().contiguous().cpu()
    raw = values.view(-1).view(dtype=torch.uint8).numpy().tobytes()
    result = {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }
    if include_bytes:
        result["fp32_hex"] = raw.hex()
    return result


def _input_tree(value, device=None):
    if isinstance(value, dict):
        return {key: _input_tree(item, device) for key, item in sorted(value.items())}
    if isinstance(value, int):
        return value
    return value.to(device) if device is not None else _tensor_record(value)


@contextmanager
def _gpu_host(args, rank, local, architecture):
    import torch
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig, TransformerConfig

    _require(
        not torch.distributed.is_initialized() and torch.cuda.device_count() == args.tp,
        "requires an isolated process and exact device count",
    )
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    props = torch.cuda.get_device_properties(device)
    _require(str(props.uuid) == args.device_uuids[local], "actual CUDA device UUID differs")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = OmniDiffusionConfig(
        model=None,
        model_class_name="MiniMaxH3Pipeline",
        num_gpus=args.tp,
        tf_model_config=TransformerConfig.from_dict(dict(architecture)),
        parallel_config=DiffusionParallelConfig(
            tensor_parallel_size=args.tp,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            sequence_parallel_size=1,
            use_hsdp=False,
        ),
        diffusion_attention_config={"default": "sdpa"},
        quantization_config=None,
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        enable_distributed_layerwise_offload=False,
        cache_backend="none",
    )
    try:
        with (
            set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cuda"))),
            set_current_diffusion_config(config),
        ):
            init_distributed_environment(
                world_size=args.tp,
                rank=rank,
                local_rank=local,
                distributed_init_method="env://",
                backend="nccl",
                timeout=timedelta(minutes=20),
            )
            initialize_model_parallel(
                tensor_model_parallel_size=args.tp, pipeline_model_parallel_size=1, backend="nccl"
            )
            yield SimpleNamespace(torch=torch, device=device, config=config, properties=props)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def _component(args, host, rank):
    box = [None]
    if rank == 0:
        value = verify_beta4_component(args.component)
        _require(
            value.manifest_sha256 == args.manifest_sha256 and value.manifest_file_sha256 == args.manifest_file_sha256,
            "expected export manifest identity differs",
        )
        box[0] = value
    host.torch.distributed.broadcast_object_list(box, src=0, device=host.device)
    verify_beta4_binding_unchanged(box[0])
    return box[0]


def _tiny_exact_slots(model, fixture, tp, rank, torch):
    """Independent full local slices imply exact reconstruction of every shard.

    Grouped QKV [head, section, 128] becomes three independently TP-split
    sections. Gate/up are two independently split halves. All other row or
    column partitions are checked in full, never sampled.
    """
    state, records = model.state_dict(), []
    buffers = {"adaln_basis", "adaln_mean", "adaln_t_table", "rope.inv_freq"}
    qkv_count = gate_count = 0
    for source, values in fixture.tiny_tensor_stream():
        target = "time_embedder.adaln_t_table" if source == "adaln_t_table" else source
        if source.endswith(".attn.qkv_proj.weight"):
            indices = [
                head * 384 + section * 128 + offset
                for section in range(3)
                for head in range(2)
                for offset in range(128)
            ]
            sections = values[indices].chunk(3, dim=0)
            values = torch.cat([part.chunk(tp, dim=0)[rank] for part in sections], dim=0)
            qkv_count += 1
        elif source.endswith(".mlp.fc1.weight"):
            values = torch.cat([part.chunk(tp, dim=0)[rank] for part in values.chunk(2, dim=0)], dim=0)
            gate_count += 1
        elif source not in buffers and "norm" not in source:
            axis = 1 if source.endswith((".attn.out_proj.weight", ".mlp.fc2.weight")) else 0
            values = values.chunk(tp, dim=axis)[rank]
        actual = state[target]
        expected = values.to(device=actual.device, dtype=actual.dtype).contiguous()
        _require(
            torch.equal(actual.contiguous().view(torch.uint8), expected.view(torch.uint8)),
            "synthetic official-loader mapping differs in exact bits",
        )
        records.append({"source": source, **_tensor_record(actual)})
    _require(
        len(records) == len({x["source"] for x in records}) == 534 and qkv_count == gate_count == 52,
        "synthetic exact-slot coverage differs",
    )
    return {
        "status": "EXACT_LOCAL_SLICES",
        "slots": 534,
        "qkv_slots": qkv_count,
        "gate_up_slots": gate_count,
        "all_elements_checked": True,
        "local_state_sha256": _digest(records),
    }


def _forward(args, host, rank, fixture):
    from comfy_omni.integrations.vllm_omni.pipelines.beta4_pipeline import H3Beta4DiTModel

    torch, device = host.torch, host.device
    if args.mode == "real":
        from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator

        binding = _component(args, host, rank)
        model_class = H3Beta4DiTModel
        weights = safetensors_weights_iterator([str(path) for path in binding.shard_paths], False, "lazy")
        inputs = _real_inputs(torch)
        identity = binding.to_dict()
    else:
        binding = SimpleNamespace(component_root=args.result_dir / "synthetic-empty", files=())
        model_class = type(
            "SyntheticBeta4DiT",
            (H3Beta4DiTModel,),
            {"_inventory": fixture.tiny_inventory(), "_architecture": fixture.tiny_arch_config()},
        )
        weights, inputs = fixture.tiny_tensor_stream(), fixture.packed_inputs()
        identity = {"synthetic": True, "fixture_sha256": args.fixture_sha256}
    input_records = _input_tree(inputs)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    # Construct directly on each rank's CUDA device; never model.to from CPU.
    with torch.device(device):
        model = model_class(host.config, quant_config=None, binding=binding)
    loaded = model.load_weights(weights)
    model.eval()
    ledger = model.loading_receipt()
    _require(len(loaded) == ledger["source_slots"] == ledger["runtime_slots"] == 534, "loading census differs")
    _require(
        all(tensor.device == device for tensor in model.state_dict().values()),
        "persistent state is not resident on its rank GPU",
    )
    exact_slots = _tiny_exact_slots(model, fixture, args.tp, rank, torch) if fixture else None
    torch.cuda.synchronize(device)
    loading_seconds = time.monotonic() - start
    start = time.monotonic()
    with torch.inference_mode():
        outputs = model(**_input_tree(inputs, device))
    torch.cuda.synchronize(device)
    forward_seconds = time.monotonic() - start
    expected_widths = (96, 32) if args.mode == "real" else (8, 4)
    _require(len(outputs) == 2, "forward output count differs")
    result = {}
    for name, output, rows, width in zip(("video", "audio"), outputs, (4, 2), expected_widths, strict=True):
        _require(
            output.dtype == torch.float32 and tuple(output.shape) == (rows, width),
            "forward output geometry or precision differs",
        )
        _require(
            bool(torch.isfinite(output).all())
            and bool(torch.count_nonzero(output[:-1]))
            and not bool(torch.count_nonzero(output[-1])),
            "output is nonfinite, empty or violates the condition mask",
        )
        result[name] = _tensor_record(output, include_bytes=True)
    verify_beta4_binding_unchanged(binding)
    return {
        "binding": identity,
        "input_sha256": _digest(input_records),
        "inputs": input_records,
        "outputs": result,
        "loading": ledger,
        "loading_seconds": loading_seconds,
        "forward_seconds": forward_seconds,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "tiny_exact_slots": exact_slots,
    }


def _aggregate(reports):
    _require(reports and len(reports) in {1, 2}, "rank report census differs")
    first = reports[0]
    _require(
        {r["rank"] for r in reports} == set(range(first["tp"])) and len(reports) == first["tp"],
        "rank report roster differs",
    )
    for report in reports:
        _require(report["status"] == "FORWARD_COMPLETED", "a rank did not complete")
        _require(
            report["host_commit"] == HOST_COMMIT and report["numerical_policy"] == COMPARISON_POLICY,
            "host or numerical policy differs",
        )
        _require(report["input_sha256"] == _digest(report["inputs"]), "rank input digest differs")
        for key in (*COMMON_IDENTITY, "tp", "outputs"):
            _require(report[key] == first[key], "rank identity, input or output replica differs")
        _require(
            report["loading"]["source_slots"] == report["loading"]["runtime_slots"] == 534, "rank coverage differs"
        )
        slots = report["loading"]["ledger"]
        _require(
            len(slots) == len({x["source"] for x in slots}) == len({x["target"] for x in slots}) == 534,
            "loading ledger is incomplete",
        )
        _require(
            sum(x["kind"] == "parameter" for x in slots) == 530
            and sum(x["kind"] == "buffer" for x in slots) == 4
            and sum(x["numerical_forward_input"] is True for x in slots) == 532,
            "loading ledger roles differ",
        )
        for kind in ("parameter", "buffer"):
            _require(
                report["loading"][kind + "_bytes"] == sum(x["bytes"] for x in slots if x["kind"] == kind),
                "loading ledger allocation differs",
            )
        if report["mode"] == "tiny":
            exact = report["tiny_exact_slots"]
            _require(
                exact["status"] == "EXACT_LOCAL_SLICES"
                and exact["slots"] == 534
                and exact["qkv_slots"] == exact["gate_up_slots"] == 52
                and exact["all_elements_checked"] is True,
                "tiny exact loader evidence differs",
            )
        widths = (96, 32) if report["mode"] == "real" else (8, 4)
        _require(set(report["outputs"]) == {"video", "audio"}, "output roster differs")
        for name, rows, width in zip(("video", "audio"), (4, 2), widths, strict=True):
            _require(report["outputs"][name]["shape"] == [rows, width], "output shape differs")
            _float_metrics(report["outputs"][name], report["outputs"][name])
    return {
        **{k: first[k] for k in (*COMMON_IDENTITY, "tp", "outputs")},
        "status": "FORWARD_COMPLETED",
        "replicas_bitwise_equal": True,
        "ranks": reports,
    }


def _run(args, rank, local):
    tool = installed_tool_identity()
    _require(
        tool.source_commit == args.commit and tool.wheel_sha256 == args.wheel_sha256,
        "installed candidate identity differs",
    )
    import comfy_omni

    installed = Path(importlib.metadata.distribution("comfy-omni").locate_file("comfy_omni")).resolve()
    _require(Path(comfy_omni.__file__).resolve().parent == installed, "source checkout shadows the installed wheel")
    before = _memory()
    fixture = _fixture(args.fixture_file, args.fixture_sha256) if args.mode == "tiny" else None
    architecture = fixture.tiny_arch_config() if fixture else BETA4_RUNTIME_ARCHITECTURE
    with _gpu_host(args, rank, local, architecture) as host:
        args.joined = True
        report = _forward(args, host, rank, fixture)
        script, _ = fileops.read_file_pinned(Path(__file__))
        _require(hashlib.sha256(script).hexdigest() == args.harness_sha256, "harness changed during acceptance")
        report.update(
            {
                "status": "FORWARD_COMPLETED",
                "candidate": tool.to_dict(),
                "image_id": args.image_id,
                "image_identity_source": "external-container-orchestrator",
                "harness_sha256": args.harness_sha256,
                "host_commit": HOST_COMMIT,
                "mode": args.mode,
                "fixture_sha256": args.fixture_sha256,
                "tp": args.tp,
                "rank": rank,
                "local_rank": local,
                "pid": os.getpid(),
                "device_uuid": str(host.properties.uuid),
                "device_name": host.properties.name,
                "device_total_bytes": host.properties.total_memory,
                "architecture": dict(architecture),
                "libraries": {
                    name: importlib.metadata.version(name)
                    for name in ("torch", "vllm", "vllm-omni", "safetensors", "triton")
                },
                "cuda_runtime": host.torch.version.cuda,
                "nccl_version": host.torch.cuda.nccl.version(),
                "attention_backend": "actual-official-sdpa-cuda",
                "tf32": False,
                "memory_before": before,
                "memory_after": _memory(),
                "numerical_policy": dict(COMPARISON_POLICY),
            }
        )
        reports = [None] * args.tp
        host.torch.distributed.all_gather_object(reports, report)
        aggregate = _aggregate(reports)
        _write(args.result_dir / f"rank-{rank}.json", report)
        host.torch.distributed.barrier()
        if rank == 0:
            _write(args.result_dir / "aggregate.json", aggregate)


def _float_metrics(left, right):
    _require(
        left["dtype"] == right["dtype"] == "torch.float32" and left["shape"] == right["shape"],
        "compared output descriptors differ",
    )
    raw_a, raw_b = bytes.fromhex(left["fp32_hex"]), bytes.fromhex(right["fp32_hex"])
    size = math.prod(left["shape"]) * 4
    _require(
        0 < size <= 4096 and len(raw_a) == len(raw_b) == size == left["bytes"] == right["bytes"],
        "compared output byte sizes differ",
    )
    _require(
        hashlib.sha256(raw_a).hexdigest() == left["sha256"] and hashlib.sha256(raw_b).hexdigest() == right["sha256"],
        "compared output bytes differ from their digests",
    )
    a, b = struct.unpack(f"<{size // 4}f", raw_a), struct.unpack(f"<{size // 4}f", raw_b)
    _require(all(math.isfinite(x) for x in (*a, *b)), "compared outputs must be finite")
    differences = [abs(x - y) for x, y in zip(a, b, strict=True)]

    def ordered(word):
        return 0x80000000 - (word & 0x7FFFFFFF) if word & 0x80000000 else 0x80000000 + word

    words_a, words_b = struct.unpack(f"<{size // 4}I", raw_a), struct.unpack(f"<{size // 4}I", raw_b)
    return {
        "elements": len(a),
        "max_abs": max(differences),
        "rms": math.sqrt(math.fsum(x * x for x in differences) / len(a)),
        "max_fp32_ulp": max(abs(ordered(x) - ordered(y)) for x, y in zip(words_a, words_b, strict=True)),
        "bitwise_equal": raw_a == raw_b,
        "equal_bits_elements": sum(x == y for x, y in zip(words_a, words_b, strict=True)),
    }


def compare_receipts(left, right):
    _require(
        left.get("status") == right.get("status") == "FORWARD_COMPLETED"
        and left.get("tp") == 1
        and right.get("tp") == 2,
        "comparison requires completed TP1 and TP2 aggregates",
    )
    _require(left.get("mode") == right.get("mode") == "tiny", "TP comparison requires the synthetic fixture")
    for report in (left, right):
        _require(
            _aggregate(report["ranks"]) == {k: v for k, v in report.items() if k not in {"schema", "receipt_sha256"}},
            "aggregate differs from rank evidence",
        )
    for key in COMMON_IDENTITY:
        _require(left[key] == right[key], "TP comparison identities or inputs differ")
    _require(left["input_sha256"] == _digest(left["inputs"]), "input tree digest differs")
    _require(left["numerical_policy"] == COMPARISON_POLICY, "comparison policy differs")
    return {
        "status": "TP_NUMERICS_OBSERVED",
        "numerical_policy": dict(COMPARISON_POLICY),
        "tp1_receipt_sha256": left["receipt_sha256"],
        "tp2_receipt_sha256": right["receipt_sha256"],
        "metrics": {key: _float_metrics(left["outputs"][key], right["outputs"][key]) for key in ("video", "audio")},
    }


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run")
    run.add_argument("--mode", choices=("tiny", "real"), required=True)
    run.add_argument("--tp", type=int, choices=(1, 2), required=True)
    for key in ("commit", "wheel-sha256", "harness-sha256", "image-id"):
        run.add_argument(f"--{key}", required=True)
    run.add_argument("--device-uuids", nargs="+", required=True)
    run.add_argument("--result-dir", type=Path, required=True)
    run.add_argument("--component", type=Path)
    run.add_argument("--manifest-sha256")
    run.add_argument("--manifest-file-sha256")
    run.add_argument("--fixture-file", type=Path)
    run.add_argument("--fixture-sha256")
    compare = sub.add_parser("compare")
    compare.add_argument("--tp1-receipt", type=Path, required=True)
    compare.add_argument("--tp2-receipt", type=Path, required=True)
    compare.add_argument("--result", type=Path, required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    rank, owned = -1, False
    try:
        if args.action == "compare":
            _write(args.result, compare_receipts(_read_receipt(args.tp1_receipt), _read_receipt(args.tp2_receipt)))
            return 0
        rank, local = _validate_args(args, os.environ)
        script, _ = fileops.read_file_pinned(Path(__file__))
        _require(
            hashlib.sha256(script).hexdigest() == args.harness_sha256, "acceptance script differs from the candidate"
        )
        args.result_dir = fileops.reject_linked_ancestors(args.result_dir, allow_missing_final=True).absolute()
        for source in (args.component, args.fixture_file):
            if source is not None:
                resolved = fileops.reject_linked_ancestors(source).resolve(strict=True)
                _require(
                    not args.result_dir.is_relative_to(resolved) and not resolved.is_relative_to(args.result_dir),
                    "evidence and input roots must be separate",
                )
        if rank == 0:
            args.result_dir.mkdir()
            owned = True
            if args.mode == "tiny":
                (args.result_dir / "synthetic-empty").mkdir(mode=0o555)
        # The rank-zero directory creation precedes NCCL initialization; every
        # rank reaches the first collective before attempting to use it.
        _run(args, rank, local)
        return 0
    except Exception as exc:
        # Preserve failure without private input paths or library tracebacks.
        if owned or getattr(args, "joined", False):
            try:
                _write(
                    args.result_dir / f"failed-rank-{rank}.json",
                    {"status": "FAILED", "rank": rank, "error_type": type(exc).__name__},
                )
            except (OSError, fileops.FsopsError):
                pass
        print(json.dumps({"status": "FAILED", "rank": rank, "error_type": type(exc).__name__}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
