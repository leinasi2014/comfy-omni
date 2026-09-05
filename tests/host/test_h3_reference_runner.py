from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_h3_reference.py"
_SPEC = importlib.util.spec_from_file_location("h3_reference_runner", _PATH)
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def _case(directory: Path, offset: float = 0, plugin: str = "legacy") -> None:
    directory.mkdir()
    for name in ("frames", "audio"):
        np.save(directory / f"{name}.npy", np.ones((2, 3), dtype=np.float32) + offset)
    runner.write_json(
        directory / "result.json",
        {
            "status": "GENERATED",
            "plugin": plugin,
            "case": {"seed": 0, "reference_sha256": "fixture"},
            "total_seconds": 10,
            "frames_sha256": runner.file_sha256(directory / "frames.npy"),
            "audio_sha256": runner.file_sha256(directory / "audio.npy"),
        },
    )


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(baseline=tmp_path / "old", candidate=tmp_path / "new", out=tmp_path / "parity.json")


def test_comparison_reports_both_media_and_preserves_visual_review(tmp_path):
    args = _args(tmp_path)
    _case(args.baseline)
    _case(args.candidate, plugin="comfy-omni")
    assert runner.compare(args) == 0
    result = json.loads(args.out.read_text())
    assert result["media"]["frames"]["exact"] is True
    assert result["media"]["audio"]["exact"] is True
    assert result["requires_visual_review"] is True


def test_comparison_rejects_modified_media_even_if_both_arrays_now_match(tmp_path):
    args = _args(tmp_path)
    _case(args.baseline)
    _case(args.candidate, plugin="comfy-omni")
    for directory in (args.baseline, args.candidate):
        np.save(directory / "audio.npy", np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(RuntimeError, match="generation receipt"):
        runner.compare(args)
    assert not args.out.exists()


def test_comparison_rejects_different_inputs(tmp_path):
    args = _args(tmp_path)
    _case(args.baseline)
    _case(args.candidate, plugin="comfy-omni")
    path = args.candidate / "result.json"
    record = json.loads(path.read_text())
    record["case"]["seed"] = 1
    runner.write_json(path, record)
    with pytest.raises(RuntimeError, match="identical reference case"):
        runner.compare(args)


def test_comparison_rejects_two_legacy_runs(tmp_path):
    args = _args(tmp_path)
    _case(args.baseline)
    _case(args.candidate)
    with pytest.raises(RuntimeError, match="legacy baseline and a comfy-omni candidate"):
        runner.compare(args)


def test_comparison_keeps_differences_that_fp16_would_hide(tmp_path):
    args = _args(tmp_path)
    _case(args.baseline)
    _case(args.candidate, offset=0.00001, plugin="comfy-omni")
    a = np.load(args.baseline / "frames.npy")
    b = np.load(args.candidate / "frames.npy")
    assert np.array_equal(a.astype(np.float16), b.astype(np.float16))
    runner.compare(args)
    result = json.loads(args.out.read_text())
    assert result["media"]["frames"]["exact"] is False
    assert result["media"]["audio"]["exact"] is False
