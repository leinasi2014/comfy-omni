"""Independent verifier must bind the newly authorized beta4 artifact."""

import hashlib
import importlib.util
import json
import math
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest


def _verifier():
    path = Path(__file__).resolve().parents[2] / "scripts/acceptance/verify_beta4_dense_conversion.py"
    spec = importlib.util.spec_from_file_location("beta4_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_beta4_verifier_rejects_old_asset_authority():
    verifier = _verifier()
    assert verifier.SOURCE_SHA256 == "54d56b15c65923b54c9ca16b494dae641bfe9455cfcb1c19c49b1008e270bbc1"
    assert verifier.SOURCE_SCHEMA_SHA256 == "ae2456bc6ac904929a4b773f703f8a1baa99b6356b5a389994faf64a1a2d80f2"
    assert verifier.TARGET_PAYLOAD_BYTES == 40_222_925_872


def _canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _bf16(values):
    # Small fixture values are exactly representable; no production converter.
    return b"".join(struct.pack("<H", struct.unpack("<I", struct.pack("<f", value))[0] >> 16) for value in values)


def _write_tensors(path, tensors):
    header, payload, records = {}, bytearray(), {}
    for name, dtype, shape, raw in sorted(tensors):
        start = len(payload)
        payload.extend(raw)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(payload)]}
        records[name] = {"dtype": dtype, "shape": shape, "start": start, "end": len(payload)}
    encoded = _canonical(header)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return records, 8 + len(encoded)


def _schema(records):
    return _digest([{"name": n, "dtype": r["dtype"], "shape": r["shape"]} for n, r in sorted(records.items())])


def _fixture(tmp_path, monkeypatch, *, row_scale=0.5, group_size=4):
    verifier = _verifier()
    source, output = tmp_path / "source.safetensors", tmp_path / "output"
    output.mkdir()
    source_tensors, target_tensors = [], []
    permutation = [0, 2, 4, 1, 3, 5]
    columns = 2 * group_size
    h4 = ((1, 1, 1, -1), (1, 1, -1, 1), (1, -1, 1, 1), (-1, 1, 1, 1))
    levels = 1 if group_size == 4 else 4
    matrix = tuple(
        tuple(
            math.prod(h4[(row // 4**level) % 4][(column // 4**level) % 4] for level in range(levels))
            for column in range(group_size)
        )
        for row in range(group_size)
    )
    # Independent integer Kronecker oracle, two complete groups in every row.
    for name, rows in (("blocks.0.attn.out_proj.weight", 4), ("blocks.0.attn.qkv_proj.weight", 6)):
        prefix = name.removesuffix(".weight")
        qvalues = [((row * columns + column) % 11) - 5 for row in range(rows) for column in range(columns)]
        source_tensors.append((name, "I8", [rows, columns], struct.pack(f"<{len(qvalues)}b", *qvalues)))
        source_tensors.append(
            (prefix + ".weight_scale", "F32", [rows, 1], struct.pack(f"<{rows}f", *([row_scale] * rows)))
        )
        marker = _canonical({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": group_size})
        source_tensors.append((prefix + ".comfy_quant", "U8", [len(marker)], marker))
        dense = []
        for row in permutation if "qkv" in name else range(rows):
            for block in (0, group_size):
                vector = qvalues[row * columns + block : row * columns + block + group_size]
                dense.extend(
                    sum(coef * value for coef, value in zip(coefficients, vector, strict=True))
                    * row_scale
                    / math.sqrt(group_size)
                    for coefficients in matrix
                )
        target_tensors.append((name, "BF16", [rows, columns], _bf16(dense)))
    for block in range(2):
        name = f"token_refiner.blocks.{block}.attn.qkv_proj.weight"
        values = [float(i + block) for i in range(48)]
        source_tensors.append((name, "BF16", [6, 8], _bf16(values)))
        target_tensors.append(
            (name, "BF16", [6, 8], _bf16([values[row * 8 + col] for row in permutation for col in range(8)]))
        )
    source_tensors.append(("adaln_basis", "BF16", [1, 4], _bf16([1, 2, 3, 4])))
    target_tensors.append(source_tensors[-1])
    source_records, source_offset = _write_tensors(source, source_tensors)
    shard = "model-00001-of-00001.safetensors"
    target_records, target_offset = _write_tensors(output / shard, target_tensors)
    operations = {
        "copy-raw": 1,
        "copy-runtime-qkv-to-grouped": 2,
        "inverse-convrot-to-bf16": 1,
        "inverse-convrot-to-bf16-runtime-qkv-to-grouped": 1,
        "omit-comfy-quant-marker": 2,
        "omit-source-rowwise-scale": 2,
    }
    overrides = {
        "SOURCE_SIZE": source.stat().st_size,
        "SOURCE_SHA256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "SOURCE_SCHEMA_SHA256": _schema(source_records),
        "TARGET_SCHEMA_SHA256": _schema(target_records),
        "TARGET_PAYLOAD_BYTES": sum(len(t[3]) for t in target_tensors),
        "SOURCE_TENSOR_COUNT": 9,
        "TARGET_TENSOR_COUNT": 5,
        "SOURCE_DTYPES": {"BF16": 3, "I8": 2, "F32": 2, "U8": 2},
        "GROUP_COUNT": 2,
        "GROUP_SIZE": group_size,
        "SHARD_COUNT": 1,
        "QKV_DIMENSIONS": (2, 1, 1),
        "EXPECTED_OPERATIONS": operations,
    }
    for name, value in overrides.items():
        monkeypatch.setattr(verifier, name, value)
    actions = []
    for name, record in sorted(source_records.items()):
        dtype, shape, source_bytes = record["dtype"], record["shape"], record["end"] - record["start"]
        target_name, target_dtype, target_bytes = name, "BF16", math.prod(shape) * 2
        prefix, group = None, None
        if dtype in {"I8", "F32", "U8"}:
            prefix = name.rsplit(".", 1)[0]
            group = group_size
            if dtype == "I8":
                operation = "inverse-convrot-to-bf16" + ("-runtime-qkv-to-grouped" if "qkv" in name else "")
            else:
                target_name, target_dtype, target_bytes = None, None, 0
                operation = "omit-source-rowwise-scale" if dtype == "F32" else "omit-comfy-quant-marker"
        elif "qkv" in name:
            operation, prefix = "copy-runtime-qkv-to-grouped", name.removesuffix(".weight")
        else:
            operation = "copy-raw"
        actions.append(
            {
                "source_name": name,
                "target_name": target_name,
                "source_dtype": dtype,
                "target_dtype": target_dtype,
                "shape": shape,
                "source_bytes": source_bytes,
                "target_bytes": target_bytes,
                "operation": operation,
                "group_prefix": prefix,
                "group_size": group,
            }
        )
    layout = {
        "source_layout": "runtime-qkv",
        "target_layout": "grouped-for-official-loader",
        "num_query_groups": 2,
        "heads_per_group": 1,
        "head_dim": 1,
        "row_count": 6,
        "permutation_sha256": _digest(permutation),
    }
    plan = {
        "schema": "comfy_omni.native_export.plan/v2",
        "status": "AUTHORIZED_PLAN",
        "profile": verifier.PROFILE,
        "output_schema": verifier.OUTPUT_SCHEMA,
        "component": "transformer",
        "source_contract": {
            "name": verifier.SOURCE_CONTRACT,
            "origin": "compile-time",
            "schema_sha256": verifier.SOURCE_SCHEMA_SHA256,
            "snapshot_manifest_sha256": None,
            "snapshot_file_sha256": None,
        },
        "architecture_template": {
            "name": verifier.SOURCE_TEMPLATE,
            "version": 1,
            "sha256": verifier.SOURCE_TEMPLATE_SHA256,
        },
        "source_files": [{"path": str(source), "size": verifier.SOURCE_SIZE, "sha256": verifier.SOURCE_SHA256}],
        "target": {
            "tensor_count": 5,
            "payload_bytes": verifier.TARGET_PAYLOAD_BYTES,
            "contract": verifier.TARGET_CONTRACT,
            "schema_sha256": verifier.TARGET_SCHEMA_SHA256,
        },
        "runtime_quantization": {
            "required": False,
            "method": None,
            "ignored_layers": [],
            "checkpoint_int8_serialized": False,
        },
        "qkv_layout": layout,
        "actions": actions,
        "shards": [
            {"name": shard, "tensor_names": sorted(target_records), "payload_bytes": verifier.TARGET_PAYLOAD_BYTES}
        ],
        "resource_envelope": {
            "max_rows": 128,
            "max_shard_bytes": verifier.MAX_SHARD_BYTES,
            "largest_target_tensor_bytes": max(96, 6 * columns * 2),
        },
    }
    args = SimpleNamespace(
        source=source,
        output=output,
        preflight_plan=tmp_path / "preflight.plan.json",
        preflight_result=tmp_path / "preflight.json",
        result=tmp_path / "result.json",
        expected_commit="a" * 40,
        expected_wheel_sha256="b" * 64,
        verification=tmp_path / "verification.json",
    )
    fixture = SimpleNamespace(
        verifier=verifier,
        args=args,
        plan=plan,
        shard=shard,
        target_records=target_records,
        target_offset=target_offset,
        source_records=source_records,
        source_offset=source_offset,
    )
    _refresh(fixture)
    return fixture


def _refresh(fixture, *, bad_config=False):
    v, args, plan = fixture.verifier, fixture.args, fixture.plan
    plan.pop("content_sha256", None)
    plan["content_sha256"] = _digest(plan)
    payload = _canonical(plan)
    args.preflight_plan.write_bytes(payload)
    (args.output / "export.plan.json").write_bytes(payload)
    config = {
        "_comfy_omni": {
            "output_schema": v.OUTPUT_SCHEMA,
            "plan_content_sha256": plan["content_sha256"],
            "profile": v.PROFILE,
        },
        "quantization_config": {"quant_method": "int8"} if bad_config else None,
    }
    (args.output / "config.patch.json").write_bytes(_canonical(config))
    index = {
        "metadata": {"total_size": v.TARGET_PAYLOAD_BYTES},
        "weight_map": {name: fixture.shard for name in fixture.target_records},
    }
    (args.output / "model.safetensors.index.json").write_bytes(_canonical(index))
    manifest = {
        "schema": "comfy_omni.native_export.receipt/v1",
        "status": "COMMITTED",
        "component": "transformer",
        "profile": v.PROFILE,
        "output_schema": v.OUTPUT_SCHEMA,
        "plan_content_sha256": plan["content_sha256"],
        "source_files": plan["source_files"],
        "target": plan["target"],
        "runtime_quantization": plan["runtime_quantization"],
        "qkv_layout": plan["qkv_layout"],
        "tool": {
            "distribution": "comfy-omni",
            "version": "0.2.0a1",
            "source_commit": args.expected_commit,
            "wheel_sha256": args.expected_wheel_sha256,
        },
        "files": [],
    }
    for path in sorted(args.output.iterdir()):
        if path.name == "manifest.json":
            continue
        entry = {
            "name": path.name,
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if path.name == fixture.shard:
            entry["tensor_count"] = 5
        manifest["files"].append(entry)
    manifest["manifest_sha256"] = _digest(manifest)
    (args.output / "manifest.json").write_bytes(_canonical(manifest))
    common = {
        "candidate_commit": args.expected_commit,
        "wheel_sha256": args.expected_wheel_sha256,
        "source_sha256": v.SOURCE_SHA256,
        "plan_content_sha256": plan["content_sha256"],
        "plan_file_sha256": hashlib.sha256(payload).hexdigest(),
    }
    args.preflight_result.write_bytes(_canonical({**common, "status": "AUTHORIZED"}))
    args.result.write_bytes(
        _canonical(
            {
                **common,
                "status": "EXECUTED",
                "manifest_sha256": manifest["manifest_sha256"],
                "manifest_file_sha256": hashlib.sha256((args.output / "manifest.json").read_bytes()).hexdigest(),
            }
        )
    )


def _argv(args):
    return [part for key, value in vars(args).items() for part in ("--" + key.replace("_", "-"), str(value))]


def test_independent_verifier_covers_every_kind_and_records_numeric_limits(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    result = fixture.verifier._verify(fixture.args)
    assert result["status"] == "VERIFIED"
    assert (result["copy_raw_tensors"], result["qkv_reordered_tensors"], result["convrot_groups_checked"]) == (1, 2, 2)
    assert result["convrot_sample_rows"] == 6
    assert result["convrot_sample_elements"] == 48
    assert result["all_numeric_bitwise"] is False
    assert result["numerical_coverage"] == "deterministic-row-samples"
    assert result["sample_max_absolute_error"] == 0
    assert fixture.verifier.main(_argv(fixture.args)) == 0
    before = fixture.args.verification.read_bytes()
    assert fixture.verifier.main(_argv(fixture.args)) == 2
    assert fixture.args.verification.read_bytes() == before


@pytest.mark.parametrize("group_size", [4, 256])
def test_zeroed_small_weights_with_rehashed_receipts_are_rejected(tmp_path, monkeypatch, group_size):
    fixture = _fixture(tmp_path, monkeypatch, row_scale=2**-16, group_size=group_size)
    assert fixture.verifier._verify(fixture.args)["status"] == "VERIFIED"
    path = fixture.args.output / fixture.shard
    raw = bytearray(path.read_bytes())
    for name in ("blocks.0.attn.out_proj.weight", "blocks.0.attn.qkv_proj.weight"):
        record = fixture.target_records[name]
        start, end = fixture.target_offset + record["start"], fixture.target_offset + record["end"]
        values = [
            struct.unpack("<f", struct.pack("<I", value << 16))[0]
            for (value,) in struct.iter_unpack("<H", raw[start:end])
        ]
        assert 0 < max(abs(value) for value in values) < 0.05
        raw[start:end] = bytes(end - start)
    path.write_bytes(raw)
    _refresh(fixture)
    with pytest.raises(ValueError, match="numerical sample"):
        fixture.verifier._verify(fixture.args)


def test_numeric_duration_excludes_copy_work_and_estimate_is_explicit(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    clock = iter((5.0, 7.0, 11.0, 13.0))
    monkeypatch.setattr(fixture.verifier, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    result = fixture.verifier._verify(fixture.args)
    assert result["numeric_elapsed_seconds"] == 4.0
    assert result["convrot_total_elements"] == 80
    assert result["estimated_full_numeric_seconds"] == pytest.approx(result["numeric_elapsed_seconds"] * 80 / 48)
    assert result["numeric_estimate_is_rough"] is True


@pytest.mark.parametrize("group", [4, 256])
def test_hadamard_oracle_matches_independent_kronecker_basis_across_blocks(group):
    verifier = _verifier()
    h4 = ((1, 1, 1, -1), (1, 1, -1, 1), (1, -1, 1, 1), (-1, 1, 1, 1))
    levels = 1 if group == 4 else 4
    expected, values = [], [0.0] * (2 * group)
    for block, column in enumerate((0, group - 2)):
        values[block * group + column] = 1.0
        for row in range(group):
            coefficient = math.prod(h4[(row // 4**level) % 4][(column // 4**level) % 4] for level in range(levels))
            expected.append(coefficient / math.sqrt(group))
    assert verifier._regular_hadamard(values, group) == expected


@pytest.mark.parametrize(
    "fp32_bits,bf16_bits",
    [
        (0x3F807FFF, 0x3F80),
        (0x3F808000, 0x3F80),
        (0x3F808001, 0x3F81),
        (0x3F818000, 0x3F82),
        (0xBF808000, 0xBF80),
        (0xBF818000, 0xBF82),
        (0x00000000, 0x0000),
        (0x80000000, 0x8000),
        (0x00000001, 0x0000),
        (0x00008000, 0x0000),
        (0x00008001, 0x0001),
        (0x00018000, 0x0002),
        (0x80008000, 0x8000),
        (0x00010000, 0x0001),
    ],
)
def test_bf16_rounding_uses_nearest_even_and_preserves_zero_sign(fp32_bits, bf16_bits):
    verifier = _verifier()
    value = struct.unpack("<f", struct.pack("<I", fp32_bits))[0]
    assert verifier._bf16_rne_bits(value) == bf16_bits


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan, 2.0**128, (2.0 - 2**-23) * 2**127])
def test_nonfinite_or_overflow_oracle_values_fail_closed(value):
    with pytest.raises(ValueError):
        _verifier()._bf16_rne_bits(value)


def test_fp32_cancellation_uses_frozen_association_before_bf16():
    verifier = _verifier()
    # The first pair loses the small addend, and the second pair cancels
    # the large value. A binary64 algebraic result would incorrectly be 0.5.
    observed = verifier._regular_hadamard([2**24, 1, -(2**24), 0], 4)
    assert observed[0] == 0.0
    assert verifier._bf16_rne_bits(observed[0]) == 0
    assert verifier._fp32(2**-149) == 2**-149
    assert verifier._fp32(2**-150) == 0.0


def test_exact_oracle_matches_actual_cpu_backend_multiblock_nondyadic_scales(monkeypatch):
    pytest.importorskip("torch")
    from comfy_omni.conversion.numerics.serialization import torch_convrot_bf16_block

    monkeypatch.setenv("COMFY_OMNI_CONVROT_DEVICE", "cpu")
    verifier = _verifier()
    rows, columns, group = 5, 512, 256
    # Stored F32 values use decimal significands, cancellation, BF16 ties,
    # and the smallest positive F32 subnormal, without random input.
    scales = struct.pack("<5f", 0.001, 0.1, 1.234567, 1.00390625, 2**-149)
    factors = struct.unpack("<5f", scales)
    qvalues = [((row * 19 + column * 37) % 256) - 128 for row in range(rows) for column in range(columns)]
    for column in range(columns):
        qvalues[3 * columns + column] = 1  # Exact midpoint case after four stages.
    raw = struct.pack(f"<{len(qvalues)}b", *qvalues)
    actual = torch_convrot_bf16_block(raw, scales, rows=rows, columns=columns, group_size=group)
    expected = bytearray()
    for row, factor in enumerate(factors):
        decoded = verifier._regular_hadamard(
            [verifier._fp32(value * factor) for value in qvalues[row * columns : (row + 1) * columns]], group
        )
        expected.extend(b"".join(struct.pack("<H", verifier._bf16_rne_bits(value)) for value in decoded))
    assert actual == bytes(expected)


def test_failed_receipt_sync_leaves_no_success_and_redacts_paths(tmp_path, monkeypatch, capsys):
    fixture = _fixture(tmp_path, monkeypatch)

    def fail_sync(_):
        raise OSError(f"private path: {fixture.args.source}")

    monkeypatch.setattr(fixture.verifier.os, "fsync", fail_sync)
    assert fixture.verifier.main(_argv(fixture.args)) == 2
    assert not fixture.args.verification.exists()
    captured = capsys.readouterr()
    assert str(tmp_path) not in captured.out + captured.err
    assert json.loads(captured.err)["status"] == "VERIFICATION_FAILED"


@pytest.mark.parametrize(
    "change", ["int8", "config", "action", "qkv", "target", "template", "extra", "symlink", "directory", "traversal"]
)
def test_forged_self_consistent_receipts_do_not_override_authority(tmp_path, monkeypatch, change):
    fixture = _fixture(tmp_path, monkeypatch)
    if change == "int8":
        fixture.plan["runtime_quantization"] = {
            "required": True,
            "method": "int8",
            "ignored_layers": [],
            "checkpoint_int8_serialized": False,
        }
    elif change == "action":
        fixture.plan["actions"][0]["target_name"] = "forged"
    elif change == "qkv":
        fixture.plan["qkv_layout"]["target_layout"] = "runtime-qkv"
    elif change == "target":
        fixture.plan["target"]["schema_sha256"] = "0" * 64
    elif change == "template":
        fixture.plan["architecture_template"]["sha256"] = "0" * 64
    elif change == "traversal":
        fixture.plan["shards"][0]["name"] = "../escape.safetensors"
    _refresh(fixture, bad_config=change == "config")
    if change == "extra":
        (fixture.args.output / "unexpected").write_bytes(b"x")
    elif change == "symlink":
        (fixture.args.output / "unexpected").symlink_to(fixture.args.source)
    elif change == "directory":
        (fixture.args.output / "unexpected").mkdir()
    with pytest.raises(ValueError):
        fixture.verifier._verify(fixture.args)


@pytest.mark.parametrize(
    "name,reason",
    [
        ("adaln_basis", "passthrough"),
        ("token_refiner.blocks.1.attn.qkv_proj.weight", "QKV copy"),
        ("blocks.0.attn.out_proj.weight", "numerical sample"),
        ("blocks.0.attn.qkv_proj.weight", "numerical sample"),
    ],
)
def test_tampered_payload_rehashed_by_producer_fails_independent_math(tmp_path, monkeypatch, name, reason):
    fixture = _fixture(tmp_path, monkeypatch)
    path = fixture.args.output / fixture.shard
    raw = bytearray(path.read_bytes())
    start = fixture.target_offset + fixture.target_records[name]["start"]
    # Non-sampled row for copy-QKV, second 4-wide block for ConvRot.
    start += (
        30
        if "token_refiner" in name
        else fixture.target_records[name]["end"] - fixture.target_records[name]["start"] - 2
        if name == "adaln_basis"
        else 14
    )
    raw[start : start + 2] = _bf16([10000.0])
    path.write_bytes(raw)
    _refresh(fixture)
    with pytest.raises(ValueError, match=reason):
        fixture.verifier._verify(fixture.args)


@pytest.mark.parametrize("operation", ["rewrite", "replace", "extra"])
def test_changes_after_semantics_are_detected_on_exit(tmp_path, monkeypatch, operation):
    fixture = _fixture(tmp_path, monkeypatch)
    original = fixture.verifier._semantics

    def change(*args):
        result = original(*args)
        path = fixture.args.source
        if operation == "rewrite":
            raw = bytearray(path.read_bytes())
            raw[-1] ^= 1
            path.write_bytes(raw)
        elif operation == "replace":
            replacement = path.with_suffix(".replacement")
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
        else:
            (fixture.args.output / "extra").write_bytes(b"x")
        return result

    monkeypatch.setattr(fixture.verifier, "_semantics", change)
    assert fixture.verifier.main(_argv(fixture.args)) == 2
    assert not fixture.args.verification.exists()


def test_hashes_headers_and_payloads_share_descriptors(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    seen, final = {}, set()
    original, verify = fixture.verifier.HeldFile.read, fixture.verifier.HeldFile.verify

    def read(self, offset, length):
        seen.setdefault(self.path, set()).add(self.stream.fileno())
        return original(self, offset, length)

    def checked(self):
        final.add(self.path)
        return verify(self)

    monkeypatch.setattr(fixture.verifier.HeldFile, "read", read)
    monkeypatch.setattr(fixture.verifier.HeldFile, "verify", checked)
    fixture.verifier._verify(fixture.args)
    assert set(seen) == final
    assert all(len(fds) == 1 for fds in seen.values())


@pytest.mark.parametrize("bad", ["trailing", "gap", "overlap", "duplicate", "shape", "metadata", "length"])
def test_strict_safetensors_rejects_malformed_layout(tmp_path, bad):
    verifier = _verifier()
    path = tmp_path / "bad.safetensors"
    header = {
        "a": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
        "b": {"dtype": "BF16", "shape": [1], "data_offsets": [2, 4]},
    }
    payload = bytes(4)
    if bad == "trailing":
        payload += b"x"
    elif bad == "gap":
        header["b"]["data_offsets"] = [3, 5]
        payload += b"x"
    elif bad == "overlap":
        header["b"]["data_offsets"] = [1, 3]
    elif bad == "shape":
        header["b"]["shape"] = [True]
    elif bad == "metadata":
        header["__metadata__"] = {"bad": 1}
    raw = _canonical(header)
    if bad == "duplicate":
        raw = raw.replace(b'"b":', b'"a":')
    path.write_bytes(struct.pack("<Q", len(raw) if bad != "length" else 2**63) + raw + payload)
    with pytest.raises(ValueError), verifier._held(path, 10000) as held:
        verifier._safetensors(held)
