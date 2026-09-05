"""Contract tests for comfy_quant NVFP4/int8 decoding helpers.

Pure marker parsing is always tested; decode math tests are torch-guarded
(the quality gate image has no torch; the decode is validated on the server
against the real checkpoint).
"""

import pytest

from comfy_omni.conversion.nvfp4 import (
    E2M1_LUT,
    F4_E2M1_MAX,
    SUPPORTED_FORMATS,
    parse_comfy_marker,
)


def test_supported_formats() -> None:
    assert SUPPORTED_FORMATS == {"nvfp4", "int8_tensorwise"}
    assert F4_E2M1_MAX == 6.0
    assert len(E2M1_LUT) == 16


def test_parse_nvfp4_marker() -> None:
    conf = parse_comfy_marker(b'{"format": "nvfp4"}')
    assert conf == {"format": "nvfp4"}


def test_parse_int8_tensorwise_marker() -> None:
    conf = parse_comfy_marker(b'{"format": "int8_tensorwise"}')
    assert conf == {"format": "int8_tensorwise"}


def test_parse_rejects_unknown_format() -> None:
    with pytest.raises(ValueError, match="unsupported comfy_quant format"):
        parse_comfy_marker(b'{"format": "mxfp4"}')


def test_parse_rejects_malformed_json() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_comfy_marker(b"{oops")


def test_parse_rejects_non_object() -> None:
    with pytest.raises(ValueError, match="must be a JSON object"):
        parse_comfy_marker(b'["nvfp4"]')


def test_decode_math_hand_checked() -> None:
    torch = pytest.importorskip("torch")
    from comfy_omni.conversion.nvfp4 import decode_nvfp4

    # natural grid: logical columns must be a multiple of 16.  8 stored bytes ->
    # 16 logical columns; the first three bytes carry the hand-checked values and
    # the remaining five are zeros.
    q = torch.tensor([[0x10, 0x2A, 0x73, 0x00, 0x00, 0x00, 0x00, 0x00]], dtype=torch.uint8)
    scale2 = torch.tensor([2.0], dtype=torch.float32)
    bs = torch.tensor([[1.0]], dtype=torch.float32)  # [R, C/16] = [1, 1]
    out = decode_nvfp4(q, scale2, bs, output_dtype=torch.float32)
    # hi/lo per byte (hi_first): 0x10 -> [+0.5, +0.0]; 0x2A -> [+1.0, -1.0];
    # 0x73 -> [+6.0, +1.5]; 5 x 0x00 -> [0.0, 0.0] each.
    expected = (
        torch.tensor(
            [[0.5, 0.0, 1.0, -1.0, 6.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        * 2.0
    )
    assert torch.allclose(out, expected, atol=1e-6)


def test_decode_int8_tensorwise_hand_checked() -> None:
    torch = pytest.importorskip("torch")
    from comfy_omni.conversion.nvfp4 import decode_int8_tensorwise

    q = torch.tensor([[-2, 3]], dtype=torch.int8)
    scale = torch.tensor([1.5], dtype=torch.float32)
    out = decode_int8_tensorwise(q, scale, output_dtype=torch.float32)
    assert torch.allclose(out, torch.tensor([[-3.0, 4.5]], dtype=torch.float32), atol=1e-6)
