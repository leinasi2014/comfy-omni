"""Characterize strict inspection behavior migrated from h3-forge at e9cb011d."""

from __future__ import annotations

import io
import json
import struct
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from comfy_omni.artifacts import safetensors
from comfy_omni.artifacts.safetensors import (
    MAX_DIMENSION,
    MAX_HEADER_BYTES,
    MAX_TENSOR_RANK,
    read_safetensors_header,
    read_safetensors_header_stream,
)
from comfy_omni.conversion.inspection import inspect_safetensors
from comfy_omni.domain.checkpoints import ArtifactInspection


def _write_header(path: Path, header: dict[str, object], payload: bytes = b"") -> None:
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _tensor(dtype: str, shape: list[int], start: int, end: int) -> dict[str, object]:
    return {"dtype": dtype, "shape": shape, "data_offsets": [start, end]}


@pytest.mark.parametrize(
    ("namespace", "component"),
    (
        ("minimax_h3_video_vae", "video_vae"),
        ("minimax_h3_audio_vae", "audio_vae"),
    ),
)
def test_real_h3_vae_metadata_namespaces_classify_without_filename_tokens(
    tmp_path: Path, namespace: str, component: str
) -> None:
    path = tmp_path / f"{component}.safetensors"
    _write_header(
        path,
        {
            "__metadata__": {namespace: "{}"},
            "decoder.weight": _tensor("F32", [1], 0, 4),
        },
        b"\0" * 4,
    )

    inspection = inspect_safetensors(path)

    assert inspection.component == component
    assert "metadata namespace" in " ".join(inspection.evidence)


def test_rejects_contradictory_h3_vae_metadata_namespaces(tmp_path: Path) -> None:
    path = tmp_path / "contradictory.safetensors"
    _write_header(
        path,
        {
            "__metadata__": {
                "minimax_h3_video_vae": "{}",
                "minimax_h3_audio_vae": "{}",
            }
        },
    )

    with pytest.raises(ValueError, match="contradictory H3 VAE"):
        inspect_safetensors(path)


def test_inspects_lora_and_comfy_quant_without_tensor_payload(tmp_path: Path) -> None:
    path = tmp_path / "adapter.safetensors"
    _write_header(
        path,
        {
            "__metadata__": {
                "format": "comfy_quant",
                "model_type": "minimax-h3",
                "quantization": "int8_convrot",
            },
            "diffusion_model.video_patch_proj.lora_A.weight": _tensor("F16", [1], 0, 2),
        },
        b"\0\0",
    )

    inspection = inspect_safetensors(path)

    assert inspection.component == "lora"
    assert inspection.tensor_count == 1
    assert set(inspection.quantization) >= {"comfy_quant", "convrot", "int8"}


def test_inspection_metadata_is_detached_immutable_and_serializes_as_a_copy() -> None:
    source = {"model_type": "minimax-h3"}
    inspection = ArtifactInspection(
        path="checkpoint.safetensors",
        component="transformer",
        quantization=("int8",),
        tensor_count=1,
        metadata=source,
        evidence=("header",),
    )

    source["model_type"] = "changed"
    assert inspection.metadata["model_type"] == "minimax-h3"
    with pytest.raises(TypeError):
        inspection.metadata["model_type"] = "changed"  # type: ignore[index]

    serialized = inspection.to_dict()
    assert serialized["quantization"] == ("int8",)
    assert serialized["evidence"] == ("header",)
    serialized["metadata"]["model_type"] = "serialized-change"
    assert inspection.metadata["model_type"] == "minimax-h3"


def test_rejects_unsafe_header_length(tmp_path: Path) -> None:
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", MAX_HEADER_BYTES + 1))

    with pytest.raises(ValueError, match="unsafe safetensors header length"):
        inspect_safetensors(path)


def test_rejects_non_safetensors_extension(tmp_path: Path) -> None:
    path = tmp_path / "model.bin"
    path.write_bytes(b"ignored")

    with pytest.raises(ValueError, match=r"expected a \.safetensors file"):
        inspect_safetensors(path)


def test_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.safetensors"
    header = b'{"tensor":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"tensor":{}}'
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        inspect_safetensors(path)


@pytest.mark.parametrize(("name", "value"), (("nan", b"NaN"), ("infinity", b"Infinity")))
def test_rejects_non_standard_json_constants(tmp_path: Path, name: str, value: bytes) -> None:
    path = tmp_path / f"{name}.safetensors"
    header = b'{"tensor":{"dtype":"U8","shape":[1],"data_offsets":[0,1],"junk":' + value + b"}}"
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0")

    with pytest.raises(ValueError, match="non-standard JSON constant"):
        inspect_safetensors(path)


def test_rejects_unpaired_surrogates_and_non_utf8_headers(tmp_path: Path) -> None:
    surrogate = tmp_path / "surrogate.safetensors"
    header = b'{"__metadata__":{"model":"\\ud800"}}'
    surrogate.write_bytes(struct.pack("<Q", len(header)) + header)
    with pytest.raises(ValueError, match="Unicode surrogates"):
        inspect_safetensors(surrogate)

    utf16 = tmp_path / "utf16.safetensors"
    header = "{}".encode("utf-16")
    utf16.write_bytes(struct.pack("<Q", len(header)) + header)
    with pytest.raises(ValueError, match="invalid safetensors JSON header"):
        inspect_safetensors(utf16)


def test_rejects_non_string_metadata(tmp_path: Path) -> None:
    path = tmp_path / "metadata.safetensors"
    _write_header(path, {"__metadata__": {"version": 1}})

    with pytest.raises(ValueError, match="keys and values must be strings"):
        inspect_safetensors(path)


def test_rejects_invalid_dtype_and_boolean_dimension(tmp_path: Path) -> None:
    bad_dtype = tmp_path / "dtype.safetensors"
    _write_header(bad_dtype, {"tensor": _tensor("F4_UNKNOWN", [1], 0, 1)}, b"\0")
    with pytest.raises(ValueError, match="incomplete descriptor"):
        inspect_safetensors(bad_dtype)

    boolean_shape = tmp_path / "shape.safetensors"
    _write_header(boolean_shape, {"tensor": _tensor("U8", [True], 0, 1)}, b"\0")
    with pytest.raises(ValueError, match="invalid shape or offsets"):
        inspect_safetensors(boolean_shape)


def test_accepts_all_pinned_host_fp8_dtypes(tmp_path: Path) -> None:
    path = tmp_path / "fp8_fnuz.safetensors"
    _write_header(
        path,
        {
            "__metadata__": {"model_type": "minimax-h3"},
            "diffusion_model.video_patch_proj.first": _tensor("F8_E4M3FNUZ", [1], 0, 1),
            "diffusion_model.video_patch_proj.second": _tensor("F8_E5M2FNUZ", [1], 1, 2),
        },
        b"\0\0",
    )

    inspection = inspect_safetensors(path)

    assert inspection.component == "transformer"
    assert inspection.quantization == ("fp8",)


def test_accepts_current_complex_and_packed_dtypes(tmp_path: Path) -> None:
    path = tmp_path / "packed.safetensors"
    _write_header(
        path,
        {
            "complex": _tensor("C64", [1], 0, 8),
            "exponent": _tensor("F8_E8M0", [1], 8, 9),
            "four_bit": _tensor("F4", [2], 9, 10),
            "six_bit_e2m3": _tensor("F6_E2M3", [4], 10, 13),
            "six_bit_e3m2": _tensor("F6_E3M2", [4], 13, 16),
        },
        b"\0" * 16,
    )

    assert inspect_safetensors(path).tensor_count == 5


def test_rejects_excessive_tensor_count_rank_and_dimension(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    count = tmp_path / "count.safetensors"
    _write_header(
        count,
        {
            "one": _tensor("U8", [1], 0, 1),
            "two": _tensor("U8", [1], 1, 2),
        },
        b"\0\0",
    )
    monkeypatch.setattr(safetensors, "MAX_TENSOR_COUNT", 1)
    with pytest.raises(ValueError, match="tensor-count limit"):
        inspect_safetensors(count)

    rank = tmp_path / "rank.safetensors"
    _write_header(rank, {"tensor": _tensor("U8", [1] * (MAX_TENSOR_RANK + 1), 0, 1)}, b"\0")
    with pytest.raises(ValueError, match="rank limit"):
        inspect_safetensors(rank)

    dimension = tmp_path / "dimension.safetensors"
    _write_header(dimension, {"tensor": _tensor("U8", [MAX_DIMENSION + 1], 0, 0)})
    with pytest.raises(ValueError, match="dimension limit"):
        inspect_safetensors(dimension)


def test_rejects_shape_size_mismatch_and_out_of_file_range(tmp_path: Path) -> None:
    mismatch = tmp_path / "mismatch.safetensors"
    _write_header(mismatch, {"tensor": _tensor("F16", [2], 0, 2)}, b"\0\0")
    with pytest.raises(ValueError, match="byte range does not match"):
        inspect_safetensors(mismatch)

    truncated = tmp_path / "truncated.safetensors"
    _write_header(truncated, {"tensor": _tensor("F16", [1], 0, 2)}, b"\0")
    with pytest.raises(ValueError, match="extends beyond end of file"):
        inspect_safetensors(truncated)

    absurd_offset = tmp_path / "absurd_offset.safetensors"
    _write_header(absurd_offset, {"tensor": _tensor("U8", [1], 0, (1 << 64) - 1)})
    with pytest.raises(ValueError, match="extends beyond end of file"):
        inspect_safetensors(absurd_offset)


def test_rejects_huge_json_numbers_and_non_finite_floats(tmp_path: Path) -> None:
    huge = tmp_path / "huge.safetensors"
    header = b'{"tensor":{"dtype":"U8","shape":[' + (b"9" * 10_000) + b'],"data_offsets":[0,0]}}'
    huge.write_bytes(struct.pack("<Q", len(header)) + header)
    with pytest.raises(ValueError, match="JSON integer exceeds"):
        inspect_safetensors(huge)

    non_finite = tmp_path / "non_finite.safetensors"
    header = b'{"tensor":{"dtype":"U8","shape":[1e1000000],"data_offsets":[0,0]}}'
    non_finite.write_bytes(struct.pack("<Q", len(header)) + header)
    with pytest.raises(ValueError, match="finite range"):
        inspect_safetensors(non_finite)


@pytest.mark.parametrize("dtype", ("F4", "F6_E2M3", "F6_E3M2"))
def test_rejects_non_integral_packed_dtype_byte_spans(tmp_path: Path, dtype: str) -> None:
    path = tmp_path / f"{dtype}.safetensors"
    _write_header(path, {"tensor": _tensor(dtype, [1], 0, 1)}, b"\0")

    with pytest.raises(ValueError, match="byte range does not match"):
        inspect_safetensors(path)


def test_rejects_offset_gap_overlap_and_unindexed_payload(tmp_path: Path) -> None:
    gap = tmp_path / "gap.safetensors"
    _write_header(gap, {"tensor": _tensor("U8", [1], 1, 2)}, b"\0\0")
    with pytest.raises(ValueError, match="offset gap"):
        inspect_safetensors(gap)

    overlap = tmp_path / "overlap.safetensors"
    _write_header(
        overlap,
        {
            "first": _tensor("U8", [2], 0, 2),
            "second": _tensor("U8", [1], 1, 2),
        },
        b"\0\0",
    )
    with pytest.raises(ValueError, match="offset overlap"):
        inspect_safetensors(overlap)

    trailing = tmp_path / "trailing.safetensors"
    _write_header(trailing, {"tensor": _tensor("U8", [1], 0, 1)}, b"\0\0")
    with pytest.raises(ValueError, match="unindexed bytes"):
        inspect_safetensors(trailing)


def test_fp8_dtype_and_scale_marker_do_not_imply_int8(tmp_path: Path) -> None:
    path = tmp_path / "fp8.safetensors"
    _write_header(
        path,
        {
            "diffusion_model.video_patch_proj.weight": _tensor("F8_E4M3", [1], 0, 1),
            "diffusion_model.video_patch_proj.weight_scale_inv": _tensor("F32", [1], 1, 5),
        },
        b"\0" * 5,
    )

    inspection = inspect_safetensors(path)

    assert "fp8" in inspection.quantization
    assert "int8" not in inspection.quantization


def test_generic_lora_and_transformer_are_not_classified_as_h3(tmp_path: Path) -> None:
    lora = tmp_path / "generic_lora.safetensors"
    _write_header(lora, {"unet.block.lora_A.weight": _tensor("F16", [1], 0, 2)}, b"\0\0")
    assert inspect_safetensors(lora).component == "unknown"

    transformer = tmp_path / "generic_transformer.safetensors"
    _write_header(
        transformer,
        {"diffusion_model.blocks.0.attn.weight": _tensor("F16", [1], 0, 2)},
        b"\0\0",
    )
    assert inspect_safetensors(transformer).component == "unknown"


@pytest.mark.parametrize(
    "tensor_name",
    (
        "text_model.layers.0.weight",
        "visual.blocks.0.weight",
        "audio_decoder.conv.weight",
        "video_vae.decoder.weight",
        "diffusion_model.video_patch_proj.flora_analysis.weight",
    ),
)
def test_generic_component_substrings_are_not_classified_as_h3(tmp_path: Path, tensor_name: str) -> None:
    path = tmp_path / "generic.safetensors"
    _write_header(path, {tensor_name: _tensor("F16", [1], 0, 2)}, b"\0\0")

    assert inspect_safetensors(path).component == "unknown"


@pytest.mark.parametrize("marker", ("curve_model", "token_refiner", "video_patch_proj", "audio_patch_proj"))
def test_single_generic_h3_like_marker_is_not_a_signature(tmp_path: Path, marker: str) -> None:
    path = tmp_path / f"{marker}.safetensors"
    _write_header(
        path,
        {f"diffusion_model.{marker}.weight": _tensor("F16", [1], 0, 2)},
        b"\0\0",
    )

    assert inspect_safetensors(path).component == "unknown"


def test_coherent_h3_transformer_markers_form_a_signature(tmp_path: Path) -> None:
    path = tmp_path / "h3_transformer.safetensors"
    _write_header(
        path,
        {
            "diffusion_model.video_patch_proj.weight": _tensor("F16", [1], 0, 2),
            "diffusion_model.audio_patch_proj.weight": _tensor("F16", [1], 2, 4),
        },
        b"\0" * 4,
    )

    assert inspect_safetensors(path).component == "transformer"


@pytest.mark.parametrize(
    ("metadata", "expected"),
    (
        ({"quantization": "mxfp8"}, ("mxfp8",)),
        ({"quantization": "nvfp4"}, ("nvfp4",)),
        ({"notes": "this is not fp8; ordinary weights"}, ("unquantized-or-unspecified",)),
        ({"quantization": "not fp8"}, ("unquantized-or-unspecified",)),
    ),
)
def test_quantization_uses_structured_evidence_without_false_positives(
    tmp_path: Path, metadata: dict[str, str], expected: tuple[str, ...]
) -> None:
    path = tmp_path / "quantization.safetensors"
    _write_header(
        path,
        {
            "__metadata__": metadata,
            "model.weight_scale": _tensor("F32", [1], 0, 4),
        },
        b"\0" * 4,
    )

    assert inspect_safetensors(path).quantization == expected


def test_rejects_contradictory_component_evidence(tmp_path: Path) -> None:
    path = tmp_path / "mixed.safetensors"
    _write_header(
        path,
        {
            "diffusion_model.video_patch_proj.weight": _tensor("F16", [1], 0, 2),
            "text_model.layers.0.weight": _tensor("F16", [1], 2, 4),
        },
        b"\0" * 4,
    )

    with pytest.raises(ValueError, match="contradictory component evidence"):
        inspect_safetensors(path)


def test_parses_space_padded_header_and_leaves_stream_at_payload(tmp_path: Path) -> None:
    path = tmp_path / "padded.safetensors"
    header = json.dumps(
        {
            "__metadata__": {"model_type": "minimax-h3"},
            "diffusion_model.video_patch_proj.weight": _tensor("F16", [1], 0, 2),
            "diffusion_model.audio_patch_proj.weight": _tensor("F16", [1], 2, 4),
        },
        separators=(",", ":"),
    ).encode("utf-8")
    padded_length = (len(header) + 7) // 8 * 8
    padded_header = header.ljust(padded_length, b" ")
    path.write_bytes(struct.pack("<Q", padded_length) + padded_header + b"\0" * 4)

    with path.open("rb") as stream:
        metadata, descriptors, consumed = read_safetensors_header_stream(stream, path, path.stat().st_size)
        assert stream.tell() == 8 + padded_length

    assert consumed == padded_length
    assert metadata == {"model_type": "minimax-h3"}
    assert len(descriptors) == 2


def test_parses_header_entries_in_non_offset_order(tmp_path: Path) -> None:
    path = tmp_path / "shuffled.safetensors"
    _write_header(
        path,
        {
            "__metadata__": {"model_type": "minimax-h3"},
            "diffusion_model.video_patch_proj.late": _tensor("F16", [1], 2, 4),
            "diffusion_model.audio_patch_proj.early": _tensor("F16", [1], 0, 2),
        },
        b"\0" * 4,
    )

    metadata, descriptors = read_safetensors_header(path)

    assert metadata == {"model_type": "minimax-h3"}
    assert {descriptor.name: descriptor.data_offsets for descriptor in descriptors} == {
        "diffusion_model.video_patch_proj.late": (2, 4),
        "diffusion_model.audio_patch_proj.early": (0, 2),
    }
    assert inspect_safetensors(path).component == "transformer"


class _HeaderOnlyStream(io.BytesIO):
    def __init__(self, value: bytes, maximum_read_position: int) -> None:
        super().__init__(value)
        self._maximum_read_position = maximum_read_position

    def read(self, size: int = -1) -> bytes:
        if self.tell() >= self._maximum_read_position:
            raise AssertionError("inspection attempted to read tensor payload bytes")
        result = super().read(size)
        if self.tell() > self._maximum_read_position:
            raise AssertionError("inspection read beyond the safetensors header")
        return result


def test_stream_reader_never_reads_tensor_payload() -> None:
    header = b'{"tensor":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}'
    boundary = 8 + len(header)
    stream = _HeaderOnlyStream(struct.pack("<Q", len(header)) + header + b"DATA", boundary)

    _, descriptors, consumed = read_safetensors_header_stream(stream, Path("guarded.safetensors"), boundary + 4)

    assert consumed == len(header)
    assert len(descriptors) == 1
    assert stream.tell() == boundary
