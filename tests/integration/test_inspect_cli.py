"""Vertical acceptance tests for header-only checkpoint inspection."""

from __future__ import annotations

import json
import struct
from pathlib import Path

from pytest import CaptureFixture

from comfy_omni.cli import main


def _write_safetensors(path: Path, header: dict[str, object], payload: bytes = b"") -> None:
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _tensor(dtype: str, shape: list[int], start: int, end: int) -> dict[str, object]:
    return {"dtype": dtype, "shape": shape, "data_offsets": [start, end]}


def test_inspect_json_crosses_cli_to_strict_header_classification(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    checkpoint = tmp_path / "h3.safetensors"
    _write_safetensors(
        checkpoint,
        {
            "__metadata__": {"model_type": "minimax-h3"},
            "diffusion_model.video_patch_proj.weight": _tensor("F16", [1], 0, 2),
            "diffusion_model.audio_patch_proj.weight": _tensor("F16", [1], 2, 4),
        },
        b"\0" * 4,
    )

    assert main(["inspect", str(checkpoint), "--json"]) == 0

    captured = capsys.readouterr()
    assert json.loads(captured.out) == [
        {
            "component": "transformer",
            "evidence": ["H3 diffusion transformer tensor naming"],
            "metadata": {"model_type": "minimax-h3"},
            "path": str(checkpoint.resolve()),
            "quantization": ["unquantized-or-unspecified"],
            "tensor_count": 2,
        }
    ]


def test_inspect_directory_recurses_in_sorted_order_and_renders_text(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    first = tmp_path / "a.safetensors"
    second = nested / "b.safetensors"
    for checkpoint in (second, first):
        _write_safetensors(checkpoint, {})
    (tmp_path / "ignored.bin").write_bytes(b"not a checkpoint")

    assert main(["inspect", str(tmp_path)]) == 0

    assert capsys.readouterr().out.splitlines() == [
        f"{first.resolve()}: unknown tensors=0 quant=unquantized-or-unspecified",
        f"{second.resolve()}: unknown tensors=0 quant=unquantized-or-unspecified",
    ]


def test_inspect_rejects_unindexed_trailing_bytes_without_a_traceback(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    checkpoint = tmp_path / "transport-marked.safetensors"
    _write_safetensors(
        checkpoint,
        {"weight": _tensor("U8", [1], 0, 1)},
        b"\0\ntransport-marker\n",
    )

    assert main(["inspect", str(checkpoint), "--json"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "safetensors-unindexed-trailing-bytes" in captured.err
    assert "Traceback" not in captured.err
