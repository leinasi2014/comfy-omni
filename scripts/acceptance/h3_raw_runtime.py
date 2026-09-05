"""Run one bounded, single-process H3 raw-source residency acceptance case.

This runner is intentionally an acceptance harness, not a model loader.  It
uses the pinned AsyncOmni host and existing read-only H3 components, produces
only JSON receipts and optional thumbnail PNGs under a fresh output directory,
and never writes a converted model or checkpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib.metadata as metadata
import json
import os
import sys
import traceback
from pathlib import Path
from time import monotonic
from typing import Any

COMPONENT_ROOT = Path("/data/models/comfy-omni/h3-forge-output/converted-dasiwa-turbo-v4/Ref2VA")
SOURCE_A = Path(
    "/data/models/comfy-omni/h3-10eros-max-beta4-int8-convrot/diffusion_models/"
    "10Eros_Max_h3_TURBO-hybrid_beta4_int8_convrot.safetensors"
)
SOURCE_B = Path(
    "/data/models/comfy-omni/h3-10eros-max-beta4-int8-convrot/diffusion_models/"
    "minimax_h3_ref2va_pruned_zs05_int8_convrot.safetensors"
)
IDENTITY_KEYS = ("worker_pid", "pipeline_id", "transformer_id", "shared_object_ids")


class PluginPreflightError(RuntimeError):
    """The acceptance image cannot prove that its ComfyOmni plugin is active."""


def _plugin_preflight() -> None:
    """Reject an image that would route H3 through the official host pipeline."""
    try:
        metadata.distribution("comfy-omni")
    except metadata.PackageNotFoundError as error:
        raise PluginPreflightError("comfy-omni distribution is not installed") from error
    try:
        metadata.distribution("h3-forge")
    except metadata.PackageNotFoundError:
        pass
    else:
        raise PluginPreflightError("h3-forge distribution is installed; the acceptance image must use ComfyOmni only")

    entries = metadata.entry_points()
    candidates = (
        entries.select(group="vllm_omni.general_plugins")
        if hasattr(entries, "select")
        else entries.get("vllm_omni.general_plugins", ())
    )
    if not any(item.name == "comfy_omni" and item.value == "comfy_omni.plugin:register" for item in candidates):
        raise PluginPreflightError("vllm_omni.general_plugins lacks comfy_omni=comfy_omni.plugin:register")

    configured = os.environ.get("VLLM_PLUGINS")
    if configured is not None and "comfy_omni" not in {item.strip() for item in configured.split(",") if item.strip()}:
        raise PluginPreflightError("explicit VLLM_PLUGINS does not include comfy_omni")


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _path_receipt(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _assert_output_root(output: Path, roots: tuple[Path, ...]) -> None:
    if output.exists():
        raise FileExistsError(f"acceptance output directory already exists: {output}")
    resolved = output.resolve()
    for root in roots:
        source = root.resolve()
        if resolved == source or source in resolved.parents:
            raise ValueError(f"acceptance output must be outside read-only model/source tree: {source}")


def _engine_arguments(args: argparse.Namespace) -> dict[str, object]:
    return {
        "model": str(args.component_root),
        "trust_remote_code": True,
        "num_gpus": 2,
        "usp": 1,
        "ring": 1,
        "vae_parallel_mode": "tile",
        "vae_use_tiling": True,
        "diffusion_attention_backend": "CUDNN_ATTN",
        "request_batch_max_wait_ms": 200,
        "enforce_eager": True,
        "stage_init_timeout": args.init_timeout,
        "init_timeout": args.init_timeout,
        "tensor_parallel_size": 2,
        "data_parallel_size": 1,
        "text_encoder_tp_size": 2,
        "vae_patch_parallel_size": 2,
        "enable_distributed_layerwise_offload": False,
        "worker_extension_cls": "comfy_omni.integrations.vllm_omni.residency.H3ResidencyWorkerExtension",
        "diffusion_quantization_config": {"default": None, "text_encoder": {"method": "int8"}},
        "additional_config": {
            "comfy_omni_h3": {
                "active": "a",
                "sources": {
                    "a": {"path": str(args.source_a), "format": "h3-beta4-convrot"},
                    "b": {"path": str(args.source_b), "format": "h3-pruned-convrot"},
                },
            }
        },
    }


def _status_identity(status: dict[str, object], *, require: bool) -> dict[str, object]:
    routes = status.get("routes")
    if not isinstance(routes, list) or not routes or not all(isinstance(item, dict) for item in routes):
        raise RuntimeError("residency status has no per-route worker records")
    missing = {
        str(index): [key for key in IDENTITY_KEYS if route.get(key) is None] for index, route in enumerate(routes)
    }
    missing = {route: keys for route, keys in missing.items() if keys}
    if missing and require:
        raise RuntimeError(
            "residency status cannot prove worker/component continuity; missing=" + json.dumps(missing, sort_keys=True)
        )
    worker_pids = status.get("worker_pids_by_replica")
    if require and (status.get("worker_pid_scope") != "parent-owned-all-ranks" or not worker_pids):
        raise RuntimeError("TP acceptance requires the parent-owned process inventory for every worker rank")
    return {
        "proven": not missing,
        "missing": missing,
        "routes": [{key: route.get(key) for key in IDENTITY_KEYS} for route in routes],
        "worker_pids_by_replica": worker_pids,
    }


def _same_identity(before: dict[str, object], after: dict[str, object]) -> None:
    if not before["proven"] or not after["proven"] or before["routes"] != after["routes"]:
        raise RuntimeError("worker/pipeline/shared-component identity changed or is unproven across raw H3 switch")
    if before["worker_pids_by_replica"] != after["worker_pids_by_replica"]:
        raise RuntimeError("the parent-owned worker PID inventory changed across raw H3 switch")


def _content_sha256(array: Any, np: Any, *, chunk_axis: int) -> str:
    """Hash contiguous views one frame/channel at a time without a whole-output copy."""
    digest = hashlib.sha256()
    for item in np.moveaxis(array, chunk_axis, 0):
        digest.update(memoryview(np.ascontiguousarray(item)).cast("B"))
    return digest.hexdigest()


def _thumbnail_pixels(frame: Any, np: Any) -> Any:
    """Return an RGB uint8 thumbnail input without changing generation pixels."""
    pixels = np.asarray(frame)
    if pixels.ndim != 3 or pixels.shape[-1] != 3:
        raise RuntimeError(f"thumbnail frame must be HWC RGB, got {pixels.shape}")
    if pixels.dtype == np.uint8:
        return np.ascontiguousarray(pixels)
    if not np.issubdtype(pixels.dtype, np.floating):
        raise RuntimeError(f"thumbnail frame dtype must be uint8 or floating RGB, got {pixels.dtype}")
    if not np.isfinite(pixels).all():
        raise RuntimeError("thumbnail frame contains non-finite values")
    return np.rint(np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8)


def _assert_same_media(first: dict[str, object], second: dict[str, object], *, label: str) -> None:
    if any(first[name] != second[name] for name in ("frame_sha256", "audio_sha256")):
        raise RuntimeError(f"{label}: repeated A generation is not byte-identical; the receipt records both digests")


def _assert_changed_media(first: dict[str, object], second: dict[str, object]) -> None:
    if all(first[name] == second[name] for name in ("frame_sha256", "audio_sha256")):
        raise RuntimeError("B generation matches A in both media digests; raw selection effect was not observed")


def _release_receipt(pre_release: dict[str, object], released: dict[str, object]) -> dict[str, int]:
    """Require release to reclaim reported H3 storage without calling it a TP total."""
    required = ("device_weight_bytes", "cpu_weight_bytes", "resident_weight_bytes", "cuda_memory_allocated_bytes")
    if any(type(pre_release.get(key)) is not int or type(released.get(key)) is not int for key in required):
        raise RuntimeError("residency status is missing integer release-memory evidence")
    nonzero = {key: released[key] for key in required[:3] if released[key] != 0}
    if nonzero:
        raise RuntimeError(f"release unload retained H3 weight bytes: {nonzero}")
    expected = pre_release["device_weight_bytes"]
    observed = pre_release["cuda_memory_allocated_bytes"] - released["cuda_memory_allocated_bytes"]
    tolerance = 16 * 1024 * 1024
    if observed < expected - tolerance:
        raise RuntimeError(
            "release unload did not reclaim reported device weight storage: "
            f"allocated_drop={observed}, reported_device_weights={expected}, tolerance={tolerance}"
        )
    return {"allocated_drop_bytes": observed, "reported_device_weight_bytes": expected, "tolerance_bytes": tolerance}


async def _generate(engine: Any, args: argparse.Namespace, output: Path, label: str) -> dict[str, object]:
    import numpy as np

    params = copy.deepcopy(engine.default_sampling_params_list)
    diffusion = params[0]
    for name, value in (
        ("width", args.width),
        ("height", args.height),
        ("fps", args.fps),
        ("seed", args.seed),
        ("num_inference_steps", args.steps),
    ):
        setattr(diffusion, name, value)
    diffusion.extra_args = {
        "task": "ref2va",
        "duration": args.duration,
        "aspect_ratio": "16:9",
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
    }
    result = None
    async for item in engine.generate(
        prompt={
            "prompt": args.prompt.read_text(encoding="utf-8").strip(),
            "multi_modal_data": {"image": str(args.reference)},
        },
        request_id=f"comfy-omni-h3-raw-{label}-{args.seed}",
        sampling_params_list=params,
    ):
        if item.finished:
            result = item
    if result is None:
        raise RuntimeError(f"{label}: AsyncOmni request finished without output")
    frames = np.asarray(result.images[0])
    multimodal = getattr(result, "_multimodal_output", None) or {}
    audio = np.asarray(multimodal.get("audio"))
    rate = multimodal.get("audio_sample_rate")
    if frames.shape != (args.frame_count, args.height, args.width, 3) or not np.isfinite(frames).all():
        raise RuntimeError(f"{label}: invalid frame output {frames.shape}")
    if audio.ndim != 3 or audio.shape[:2] != (1, 2) or audio.shape[2] < 160000 or rate != 32000:
        raise RuntimeError(f"{label}: invalid audio output {audio.shape}, rate={rate}")
    if not np.isfinite(audio).all() or float(np.square(audio.astype(np.float64)).mean()) == 0:
        raise RuntimeError(f"{label}: audio is non-finite or silent")
    frame_variance = float(np.var(frames, dtype=np.float64))
    if frame_variance == 0:
        raise RuntimeError(f"{label}: image output is degenerate")
    thumbnail = output / f"{label}-frame0.png"
    from PIL import Image

    Image.fromarray(_thumbnail_pixels(frames[0], np)).resize((216, 120)).save(thumbnail)
    return {
        "label": label,
        "frame_shape": list(frames.shape),
        "frame_dtype": str(frames.dtype),
        "frame_sha256": _content_sha256(frames, np, chunk_axis=0),
        "frame_variance": frame_variance,
        "audio_shape": list(audio.shape),
        "audio_dtype": str(audio.dtype),
        "audio_sha256": _content_sha256(audio, np, chunk_axis=0),
        "audio_sample_rate": rate,
        "thumbnail": thumbnail.name,
    }


async def _run(args: argparse.Namespace, output: Path) -> dict[str, object]:
    started = monotonic()
    phase = "construct"
    engine = None
    record: dict[str, object] = {
        "schema": "comfy-omni.h3-raw-runtime/v1",
        "control_pid": os.getpid(),
        "stage": args.stage,
        "inputs": {
            "components": _path_receipt(args.component_root),
            "source_a": _path_receipt(args.source_a),
            "source_b": _path_receipt(args.source_b),
            "prompt": _path_receipt(args.prompt),
            "reference": _path_receipt(args.reference),
        },
        "case": {
            "task": "ref2va",
            "seed": args.seed,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "frame_count": args.frame_count,
            "duration_seconds": args.duration,
            "num_inference_steps": args.steps,
        },
        "events": [],
    }
    try:
        phase = "plugin-preflight"
        _plugin_preflight()
        from vllm_omni.entrypoints.async_omni import AsyncOmni

        from comfy_omni.integrations.vllm_omni.residency_control import H3ResidencyCoordinator

        engine = AsyncOmni(**_engine_arguments(args))
        coordinator = H3ResidencyCoordinator(engine, stage_id=0, rpc_timeout_seconds=args.rpc_timeout)
        phase = "initial-status"
        initial = await coordinator.status()
        if initial["active_selection"] != "a" or initial["weight_residency"] != "loaded":
            raise RuntimeError(f"unexpected initial raw H3 status: {initial}")
        identity_a = _status_identity(initial, require=args.stage in {"aba", "full"})
        record["events"].append({"phase": phase, "status": initial, "identity": identity_a})
        if args.stage == "load":
            record["status"] = "LOADED"
            return record

        phase = "forward-a"
        a_initial = await _generate(engine, args, output, "a-initial")
        record["events"].append({"phase": phase, "media": a_initial})
        if args.stage == "forward":
            record["status"] = "FORWARDED_A"
            return record

        phase = "switch-b"
        record["events"].append({"phase": phase, "switch": await coordinator.switch("b")})
        phase = "status-b"
        status_b = await coordinator.status()
        if status_b["active_selection"] != "b" or status_b["weight_residency"] != "loaded":
            raise RuntimeError(f"B switch did not reach loaded selection: {status_b}")
        identity_b = _status_identity(status_b, require=True)
        _same_identity(identity_a, identity_b)
        record["events"].append({"phase": phase, "status": status_b, "identity": identity_b})
        phase = "forward-b"
        b_media = await _generate(engine, args, output, "b")
        record["events"].append({"phase": phase, "media": b_media})
        _assert_changed_media(a_initial, b_media)

        phase = "switch-a"
        record["events"].append({"phase": phase, "switch": await coordinator.switch("a")})
        phase = "status-a-restored"
        restored = await coordinator.status()
        if restored["active_selection"] != "a" or restored["weight_residency"] != "loaded":
            raise RuntimeError(f"A restore did not reach loaded selection: {restored}")
        identity_restored = _status_identity(restored, require=True)
        _same_identity(identity_a, identity_restored)
        record["events"].append({"phase": phase, "status": restored, "identity": identity_restored})
        phase = "forward-a-restored"
        a_restored = await _generate(engine, args, output, "a-restored")
        record["events"].append({"phase": phase, "media": a_restored})
        _assert_same_media(a_initial, a_restored, label="A→B→A")
        if args.stage == "aba":
            record["status"] = "ABA_COMPLETED"
            return record

        phase = "pre-release-status"
        pre_release = await coordinator.status()
        record["events"].append({"phase": phase, "status": pre_release})
        phase = "unload-release"
        released = await coordinator.unload(mode="release")
        if released["weight_residency"] != "released":
            raise RuntimeError(f"release unload did not release weights: {released}")
        release_event = {"phase": phase, "status": released}
        record["events"].append(release_event)
        release_event["reclaim"] = _release_receipt(pre_release, released)
        phase = "reload"
        reloaded = await coordinator.load()
        if reloaded["weight_residency"] != "loaded":
            raise RuntimeError(f"reload did not restore weights: {reloaded}")
        identity_reloaded = _status_identity(reloaded, require=True)
        _same_identity(identity_a, identity_reloaded)
        record["events"].append({"phase": phase, "status": reloaded, "identity": identity_reloaded})
        phase = "forward-a-reloaded"
        a_reloaded = await _generate(engine, args, output, "a-reloaded")
        record["events"].append({"phase": phase, "media": a_reloaded})
        _assert_same_media(a_initial, a_reloaded, label="A release/load")
        record["status"] = "FULL_COMPLETED"
        return record
    except BaseException as error:
        record.update(
            {
                "status": "FAILED",
                "failed_phase": phase,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        record["elapsed_seconds"] = round(monotonic() - started, 3)
        _json(output / "result.json", record)
        if engine is not None:
            engine.shutdown()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("load", "forward", "aba", "full"), default="load")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--component-root", type=Path, default=COMPONENT_ROOT)
    parser.add_argument("--source-a", type=Path, default=SOURCE_A)
    parser.add_argument("--source-b", type=Path, default=SOURCE_B)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--frame-count", type=int, default=124)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--init-timeout", type=float, default=1800)
    parser.add_argument("--rpc-timeout", type=float, default=600)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    roots = (args.component_root, args.source_a.parent, args.source_b.parent)
    _assert_output_root(args.out, roots)
    for path in (args.component_root, args.source_a, args.source_b, args.prompt, args.reference):
        if not path.exists():
            raise FileNotFoundError(path)
    args.out.mkdir(parents=True)
    try:
        asyncio.run(_run(args, args.out))
    except BaseException as error:
        print(f"h3 raw runtime failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
