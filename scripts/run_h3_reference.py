"""Run or compare a fixed H3 reference case inside the pinned Docker host.

Characterized from h3-forge e9cb011's successful AsyncOmni request and its
public schedule contract. Inputs and quantization are supplied explicitly;
the same runner serves both the legacy and migrated plugin environments.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def generate(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    distributions = {d.metadata["Name"].lower().replace("_", "-") for d in importlib.metadata.distributions()}
    expected = "h3-forge" if args.plugin == "legacy" else "comfy-omni"
    unexpected = "comfy-omni" if args.plugin == "legacy" else "h3-forge"
    if expected not in distributions or unexpected in distributions:
        raise RuntimeError(f"reference environment must contain only the {expected} H3 plugin")
    if args.out.exists():
        raise FileExistsError(f"reference output already exists: {args.out}")
    args.out.mkdir(parents=True)
    for device in (0, 1):
        with torch.cuda.device(device):
            value = torch.ones(1, device=f"cuda:{device}")
            if value.item() != 1:
                raise RuntimeError(f"CUDA device {device} initialization failed")
    print("CUDA-READY: both TP ranks initialized", flush=True)

    from vllm_omni.entrypoints.async_omni import AsyncOmni

    model = args.model / "Ref2VA"
    if not (model / "model_index.json").is_file():
        raise RuntimeError("the reference case requires the native Ref2VA package partition")
    quantization = json.loads(args.quantization.read_text(encoding="utf-8"))
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    engine_args = {
        "model": str(model),
        "trust_remote_code": True,
        "num_gpus": 2,
        "usp": 1,
        "ring": 1,
        "vae_parallel_mode": "tile",
        "vae_use_tiling": True,
        "diffusion_attention_backend": "CUDNN_ATTN",
        "request_batch_max_wait_ms": 200,
        "enforce_eager": True,
        "stage_init_timeout": 1800,
        "init_timeout": 1800,
        "tensor_parallel_size": 2,
        "data_parallel_size": 1,
        "text_encoder_tp_size": 2,
        "vae_patch_parallel_size": 2,
        "enable_distributed_layerwise_offload": False,
        "diffusion_quantization_config": quantization,
    }
    case = {
        "task": "ref2va",
        "seed": args.seed,
        "width": 864,
        "height": 480,
        "fps": 24,
        "frame_count": 124,
        "duration_seconds": 5.0,
        "num_inference_steps": 5,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "reference_sha256": file_sha256(args.reference),
        "quantization_sha256": file_sha256(args.quantization),
        "manifest_sha256": file_sha256(args.model / "h3-comfy-package.json"),
        "runner_sha256": file_sha256(Path(__file__)),
        "schedule": {"profile": "turbo-v4-s12-a3", "denoise_steps": 4, "video_shift": 12.0, "audio_shift": 3.0},
    }

    async def run() -> dict[str, object]:
        started = time.monotonic()
        engine = AsyncOmni(**engine_args)
        initialized = time.monotonic()
        try:
            params = copy.deepcopy(engine.default_sampling_params_list)
            diffusion = params[0]
            for name in ("width", "height", "fps", "seed", "num_inference_steps"):
                setattr(diffusion, name, case[name])
            diffusion.extra_args = {
                "task": "ref2va",
                "duration": 5.0,
                "aspect_ratio": "16:9",
                "flow_shift": 12.0,
                "audio_flow_shift": 3.0,
                "h3_forge": {"api_version": 1, "schedule": case["schedule"], "output": {"include_audio": True}},
            }
            result = None
            async for output in engine.generate(
                prompt={"prompt": prompt, "multi_modal_data": {"image": str(args.reference)}},
                request_id=f"comfy-omni-reference-{args.seed}",
                sampling_params_list=params,
            ):
                if output.finished:
                    result = output
            generated = time.monotonic()
            if result is None:
                raise RuntimeError("reference request finished without output")
            frames = np.asarray(result.images[0])
            multimodal = getattr(result, "_multimodal_output", None) or {}
            audio = np.asarray(multimodal.get("audio"))
            rate = multimodal.get("audio_sample_rate")
            if frames.shape != (124, 480, 864, 3) or not np.isfinite(frames).all():
                raise RuntimeError(f"invalid video output: {frames.shape}")
            if audio.ndim != 3 or audio.shape[:2] != (1, 2) or audio.shape[2] < 160000 or rate != 32000:
                raise RuntimeError(f"invalid audio output: {audio.shape}, rate={rate}")
            if not np.isfinite(audio).all() or float(np.square(audio.astype(np.float64)).mean()) == 0:
                raise RuntimeError("reference audio is non-finite or silent")
            np.save(args.out / "frames.npy", frames, allow_pickle=False)
            np.save(args.out / "audio.npy", audio, allow_pickle=False)
            return {
                "schema": "comfy-omni.h3-reference/v1",
                "status": "GENERATED",
                "plugin": args.plugin,
                "case": case,
                "engine_construction_seconds": round(initialized - started, 3),
                "request_seconds": round(generated - initialized, 3),
                "total_seconds": round(generated - started, 3),
                "output_validation_and_save_seconds": round(time.monotonic() - generated, 3),
                "frame_dtype": str(frames.dtype),
                "audio_dtype": str(audio.dtype),
                "frame_shape": list(frames.shape),
                "audio_shape": list(audio.shape),
                "audio_sample_rate": rate,
                "frames_sha256": file_sha256(args.out / "frames.npy"),
                "audio_sha256": file_sha256(args.out / "audio.npy"),
                "versions": {
                    name: importlib.metadata.version(name) for name in (expected, "vllm-omni", "vllm", "torch")
                },
            }
        finally:
            engine.shutdown()

    record = asyncio.run(run())
    write_json(args.out / "result.json", record)
    print("REFERENCE-GENERATED", json.dumps(record, sort_keys=True), flush=True)
    return 0


def compare(args: argparse.Namespace) -> int:
    import numpy as np

    old = json.loads((args.baseline / "result.json").read_text(encoding="utf-8"))
    new = json.loads((args.candidate / "result.json").read_text(encoding="utf-8"))
    if old.get("plugin") != "legacy" or new.get("plugin") != "comfy-omni":
        raise RuntimeError("comparison requires a legacy baseline and a comfy-omni candidate")
    if old.get("status") != "GENERATED" or new.get("status") != "GENERATED" or old["case"] != new["case"]:
        raise RuntimeError("comparison requires two successful runs with identical reference case identities")
    metrics = {}
    for name in ("frames", "audio"):
        for directory, record in ((args.baseline, old), (args.candidate, new)):
            if file_sha256(directory / f"{name}.npy") != record[f"{name}_sha256"]:
                raise RuntimeError(f"{name} output differs from its generation receipt")
        a = np.load(args.baseline / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        b = np.load(args.candidate / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        if a.shape != b.shape:
            raise RuntimeError(f"{name} shape mismatch")
        error_sum = 0.0
        max_error = 0.0
        peak = 0.0
        for index in range(a.shape[0]):
            x = np.asarray(a[index], dtype=np.float64)
            y = np.asarray(b[index], dtype=np.float64)
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise RuntimeError(f"{name} contains non-finite values")
            difference = x - y
            error_sum += float(np.square(difference).sum())
            max_error = max(max_error, float(np.abs(difference).max()))
            peak = max(peak, float(np.abs(x).max()))
        mse = error_sum / a.size
        metrics[name] = {
            "exact": mse == 0,
            "mse": mse,
            "max_absolute_error": max_error,
            "psnr_db": None if mse == 0 else 10 * math.log10(max(peak * peak, 1e-30) / mse),
        }
    report = {
        "schema": "comfy-omni.h3-parity/v1",
        "case": old["case"],
        "media": metrics,
        "baseline_seconds": old["total_seconds"],
        "candidate_seconds": new["total_seconds"],
        "requires_visual_review": True,
    }
    write_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("generate")
    run.add_argument("--plugin", choices=("legacy", "comfy-omni"), required=True)
    for name in ("model", "prompt", "reference", "quantization", "out"):
        run.add_argument(f"--{name}", type=Path, required=True)
    run.add_argument("--seed", type=int, default=0)
    run.set_defaults(action=generate)
    comparison = commands.add_parser("compare")
    for name in ("baseline", "candidate", "out"):
        comparison.add_argument(f"--{name}", type=Path, required=True)
    comparison.set_defaults(action=compare)
    args = parser.parse_args()
    return args.action(args)


if __name__ == "__main__":
    raise SystemExit(main())
