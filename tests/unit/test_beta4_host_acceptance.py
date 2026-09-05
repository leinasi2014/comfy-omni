"""Host-free policy/receipt tests; GPU execution belongs to explicit torchrun."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import struct
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "acceptance" / "beta4_host_runtime.py"
SPEC = importlib.util.spec_from_file_location("beta4_host_acceptance_under_test", SCRIPT)
acceptance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acceptance)

UUIDS = ["GPU-00000000-0000-0000-0000-000000000000", "GPU-11111111-1111-1111-1111-111111111111"]


def _cgroup(tmp_path, monkeypatch, *, version, limit):
    root = tmp_path / "cgroup"
    names = (
        ("memory.max", "memory.current", "memory.peak")
        if version == 2
        else ("memory.limit_in_bytes", "memory.usage_in_bytes", "memory.max_usage_in_bytes")
    )
    directory = root if version == 2 else root / "memory"
    directory.mkdir(parents=True)
    for name, value in zip(names, (limit, 1024**3, 2 * 1024**3), strict=True):
        (directory / name).write_text(str(value))

    def redirected(value):
        path = Path(value)
        return root / path.relative_to("/sys/fs/cgroup") if path.is_relative_to("/sys/fs/cgroup") else path

    monkeypatch.setattr(acceptance, "Path", redirected)
    monkeypatch.setattr(acceptance.resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=1024**2))


@pytest.mark.parametrize("version", [1, 2])
def test_tp2_with_eight_gib_container_reaches_gpu_stage_without_loading_a_model(tmp_path, monkeypatch, version):
    import comfy_omni

    _cgroup(tmp_path, monkeypatch, version=version, limit=8 * 1024**3)
    args = acceptance._parser().parse_args(_argv(tmp_path, tp=2))
    monkeypatch.setattr(
        acceptance,
        "installed_tool_identity",
        lambda: SimpleNamespace(source_commit=args.commit, wheel_sha256=args.wheel_sha256),
    )
    monkeypatch.setattr(
        acceptance.importlib.metadata,
        "distribution",
        lambda _: SimpleNamespace(locate_file=lambda _: Path(comfy_omni.__file__).parent),
    )
    monkeypatch.setattr(acceptance, "_fixture", lambda *_: SimpleNamespace(tiny_arch_config=lambda: {}))

    class ReachedGpuStage(Exception):
        pass

    @contextmanager
    def gpu_stage(*_):
        raise ReachedGpuStage
        yield  # pragma: no cover - the regression ends before any GPU work

    monkeypatch.setattr(acceptance, "_gpu_host", gpu_stage)
    with pytest.raises(ReachedGpuStage):
        acceptance._run(args, rank=0, local=0)


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("tp,gib", [(1, 2), (1, 4), (2, 4), (2, 8)])
def test_memory_observation_records_actual_limit_without_inventing_the_maximum(tmp_path, monkeypatch, version, tp, gib):
    _cgroup(tmp_path, monkeypatch, version=version, limit=gib * 1024**3)
    assert acceptance._memory(tp) == {
        "limit_bytes": gib * 1024**3,
        "max_rss_bytes": 1024**3,
        "current_bytes": 1024**3,
        "peak_bytes": 2 * 1024**3,
    }


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize(
    "tp,limit", [(1, "max"), (2, "max"), (1, 0), (2, -1), (1, 4 * 1024**3 + 1), (2, 8 * 1024**3 + 1)]
)
def test_memory_gate_rejects_unlimited_or_over_budget_containers(tmp_path, monkeypatch, version, tp, limit):
    _cgroup(tmp_path, monkeypatch, version=version, limit=limit)
    with pytest.raises(ValueError, match="enforced container memory limit"):
        acceptance._memory(tp)


@pytest.mark.parametrize("tp,limit,rss", [(1, 2 * 1024**3, 2 * 1024**3 + 1024), (2, 8 * 1024**3, 4 * 1024**3 + 1024)])
def test_memory_gate_keeps_the_individual_rank_rss_bounded(tmp_path, monkeypatch, tp, limit, rss):
    _cgroup(tmp_path, monkeypatch, version=2, limit=limit)
    monkeypatch.setattr(acceptance.resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=rss // 1024))
    with pytest.raises(ValueError, match="process RSS exceeds the per-rank acceptance limit"):
        acceptance._memory(tp)


def _argv(tmp_path, *, mode="tiny", tp=1):
    args = [
        "run",
        "--mode",
        mode,
        "--tp",
        str(tp),
        "--commit",
        "a" * 40,
        "--wheel-sha256",
        "b" * 64,
        "--harness-sha256",
        hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        "--image-id",
        "sha256:" + "c" * 64,
        "--device-uuids",
        *UUIDS[:tp],
        "--result-dir",
        str(tmp_path / "fresh-result"),
    ]
    if mode == "tiny":
        args.extend(["--fixture-file", str(tmp_path / "fixture.py"), "--fixture-sha256", "d" * 64])
    else:
        args.extend(
            [
                "--component",
                str(tmp_path / "component"),
                "--manifest-sha256",
                "e" * 64,
                "--manifest-file-sha256",
                "f" * 64,
            ]
        )
    return args


def _environ(tp=1, rank=0):
    return {
        "WORLD_SIZE": str(tp),
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "NVIDIA_VISIBLE_DEVICES": ",".join(str(i) for i in range(tp)),
    }


@pytest.mark.parametrize("mode,tp", [("tiny", 1), ("tiny", 2), ("real", 2)])
def test_only_fixed_external_single_node_lanes_are_accepted(tmp_path, mode, tp):
    args = acceptance._parser().parse_args(_argv(tmp_path, mode=mode, tp=tp))
    assert acceptance._validate_args(args, _environ(tp)) == (0, 0)
    if tp == 2:
        assert acceptance._validate_args(args, _environ(tp, rank=1)) == (1, 1)


@pytest.mark.parametrize(
    "case",
    [
        "real_tp1",
        "real_fixture",
        "real_missing_manifest",
        "tiny_component",
        "tiny_unpinned",
        "wrong_world",
        "wrong_local_rank",
        "implicit_devices",
        "gpu2",
        "duplicate_uuid",
        "short_uuid",
        "mutable_image",
        "short_commit",
        "short_wheel",
        "short_harness",
    ],
)
def test_invalid_lane_is_rejected_before_gpu_or_model_access(tmp_path, case):
    args = acceptance._parser().parse_args(_argv(tmp_path, mode="real" if case.startswith("real_") else "tiny", tp=2))
    env = _environ(2)
    if case == "real_tp1":
        args.tp, args.device_uuids, env = 1, UUIDS[:1], _environ()
    elif case == "real_fixture":
        args.fixture_file = tmp_path / "fixture.py"
    elif case == "real_missing_manifest":
        args.manifest_file_sha256 = None
    elif case == "tiny_component":
        args.component = tmp_path / "component"
    elif case == "tiny_unpinned":
        args.fixture_sha256 = None
    elif case == "wrong_world":
        env["WORLD_SIZE"] = "3"
    elif case == "wrong_local_rank":
        env["LOCAL_RANK"] = "1"
    elif case in {"implicit_devices", "gpu2"}:
        env["NVIDIA_VISIBLE_DEVICES"] = "all" if case == "implicit_devices" else "0,2"
    elif case == "duplicate_uuid":
        args.device_uuids = [UUIDS[0], UUIDS[0]]
    elif case == "short_uuid":
        args.device_uuids = ["GPU-123", UUIDS[1]]
    elif case == "mutable_image":
        args.image_id = "example:latest"
    else:
        setattr(
            args,
            {"short_commit": "commit", "short_wheel": "wheel_sha256", "short_harness": "harness_sha256"}[case],
            "abc",
        )
    with pytest.raises(ValueError):
        acceptance._validate_args(args, env)


def _output(rows, width, *, adjustment=0.0):
    values = [(i + 1) / 128 + adjustment for i in range((rows - 1) * width)] + [0.0] * width
    raw = struct.pack(f"<{len(values)}f", *values)
    return {
        "shape": [rows, width],
        "dtype": "torch.float32",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "fp32_hex": raw.hex(),
    }


def _rank(tp, rank, *, adjustment=0.0):
    ledger = [
        {
            "source": f"source-{i}",
            "target": f"target-{i}",
            "bytes": 2,
            "kind": "parameter" if i < 530 else "buffer",
            "numerical_forward_input": i < 532,
        }
        for i in range(534)
    ]
    inputs = {"synthetic_input": "independent test-only identity"}
    return {
        "status": "FORWARD_COMPLETED",
        "candidate": {"source_commit": "a" * 40, "wheel_sha256": "b" * 64},
        "image_id": "sha256:" + "c" * 64,
        "harness_sha256": "d" * 64,
        "host_commit": acceptance.HOST_COMMIT,
        "mode": "tiny",
        "fixture_sha256": "e" * 64,
        "tp": tp,
        "rank": rank,
        "binding": {"synthetic": True, "fixture_sha256": "e" * 64},
        "inputs": inputs,
        "input_sha256": acceptance._digest(inputs),
        "outputs": {"video": _output(4, 8, adjustment=adjustment), "audio": _output(2, 4, adjustment=adjustment)},
        "loading": {
            "source_slots": 534,
            "runtime_slots": 534,
            "parameter_bytes": 1060,
            "buffer_bytes": 8,
            "ledger": ledger,
        },
        "tiny_exact_slots": {
            "status": "EXACT_LOCAL_SLICES",
            "slots": 534,
            "qkv_slots": 52,
            "gate_up_slots": 52,
            "all_elements_checked": True,
            "local_state_sha256": str(rank) * 64,
        },
        "numerical_policy": dict(acceptance.COMPARISON_POLICY),
        "libraries": {"torch": "test-version"},
        "architecture": {"hidden_size": 32},
        "cuda_runtime": "test-version",
        "nccl_version": [2, 28, 9],
        "attention_backend": "actual-official-sdpa-cuda",
        "tf32": False,
    }


def _aggregate(tp, *, adjustment=0.0):
    result = acceptance._aggregate([_rank(tp, rank, adjustment=adjustment) for rank in range(tp)])
    result["schema"] = acceptance.SCHEMA
    result["receipt_sha256"] = acceptance._digest(result)
    return result


@pytest.mark.parametrize("adjustment", [0, 2**-20, 1.0])
def test_tp_comparison_reports_observation_without_claiming_numerical_pass(adjustment):
    result = acceptance.compare_receipts(_aggregate(1), _aggregate(2, adjustment=adjustment))
    assert result["status"] == "TP_NUMERICS_OBSERVED"
    assert result["numerical_policy"]["automatic_pass"] is False
    metric = result["metrics"]["video"]
    assert metric["bitwise_equal"] is (adjustment == 0)
    assert metric["max_abs"] == adjustment
    assert metric["rms"] == pytest.approx((3 / 4) ** 0.5 * adjustment)


def test_ulp_metric_is_exact_for_adjacent_fp32_values_and_signed_zero():
    left = _output(2, 1)
    raw = bytearray.fromhex(left["fp32_hex"])
    raw[:4] = struct.pack("<I", struct.unpack("<I", raw[:4])[0] + 1)
    right = {**left, "fp32_hex": raw.hex(), "sha256": hashlib.sha256(raw).hexdigest()}
    assert acceptance._float_metrics(left, right)["max_fp32_ulp"] == 1
    raw[:4] = bytes(4)
    left = {**left, "fp32_hex": raw.hex(), "sha256": hashlib.sha256(raw).hexdigest()}
    raw[:4] = struct.pack("<I", 0x80000000)
    right = {**left, "fp32_hex": raw.hex(), "sha256": hashlib.sha256(raw).hexdigest()}
    result = acceptance._float_metrics(left, right)
    assert result["max_fp32_ulp"] == 0 and not result["bitwise_equal"]


@pytest.mark.parametrize(
    "case",
    [
        "rank_duplicate",
        "missing_slot",
        "duplicate_slot",
        "byte_census",
        "false_basis_role",
        "loader_sampled",
        "rank_input",
        "replica_output",
        "nonfinite",
        "output_digest",
        "host_pin",
    ],
)
def test_aggregate_rejects_incomplete_or_inconsistent_rank_evidence(case):
    reports = [_rank(2, i) for i in range(2)]
    changed = reports[1]
    if case == "rank_duplicate":
        changed["rank"] = 0
    elif case == "missing_slot":
        changed["loading"]["ledger"].pop()
    elif case == "duplicate_slot":
        changed["loading"]["ledger"][-1] = copy.deepcopy(changed["loading"]["ledger"][0])
    elif case == "byte_census":
        changed["loading"]["parameter_bytes"] += 1
    elif case == "false_basis_role":
        changed["loading"]["ledger"][-1]["numerical_forward_input"] = True
    elif case == "loader_sampled":
        changed["tiny_exact_slots"]["all_elements_checked"] = False
    elif case == "rank_input":
        changed["inputs"]["synthetic_input"] = "other"
        changed["input_sha256"] = acceptance._digest(changed["inputs"])
    elif case == "replica_output":
        changed["outputs"]["video"] = _output(4, 8, adjustment=2**-20)
    elif case == "host_pin":
        changed["host_commit"] = "0" * 40
    elif case == "nonfinite":
        for report in reports:
            output = report["outputs"]["video"]
            raw = struct.pack("<f", float("nan")) + bytes.fromhex(output["fp32_hex"])[4:]
            output.update(fp32_hex=raw.hex(), sha256=hashlib.sha256(raw).hexdigest())
    else:
        for report in reports:
            report["outputs"]["video"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        acceptance._aggregate(reports)


@pytest.mark.parametrize(
    "field", ["candidate", "image_id", "fixture_sha256", "harness_sha256", "libraries", "inputs", "architecture"]
)
def test_tp_comparison_refuses_changed_candidate_or_inputs_even_with_resealed_aggregate(field):
    left, right = _aggregate(1), _aggregate(2)
    for report in right["ranks"]:
        if isinstance(report[field], dict):
            report[field]["changed"] = True
        else:
            report[field] = "changed"
        if field == "inputs":
            report["input_sha256"] = acceptance._digest(report["inputs"])
    right = acceptance._aggregate(right["ranks"])
    right["receipt_sha256"] = acceptance._digest(right)
    with pytest.raises(ValueError):
        acceptance.compare_receipts(left, right)


def test_top_level_aggregate_cannot_disagree_with_its_rank_receipts():
    left, right = _aggregate(1), _aggregate(2)
    right["candidate"] = {"forged": True}
    with pytest.raises(ValueError):
        acceptance.compare_receipts(left, right)


def test_fixture_is_sha_pinned_without_calling_its_cpu_host(tmp_path):
    path = Path(__file__).resolve().parents[1] / "host" / "beta4_host_fixture.py"
    payload = path.read_bytes()
    module = acceptance._fixture(path, hashlib.sha256(payload).hexdigest())
    assert len(module.tiny_inventory()) == 534
    assert module.tiny_arch_config()["attention_head_dim"] == 128
    with pytest.raises(ValueError, match="fixture source"):
        acceptance._fixture(path, "0" * 64)


def test_receipts_are_canonical_digest_bound_and_exclusive(tmp_path):
    path = tmp_path / "receipt.json"
    document = acceptance._write(path, _aggregate(1))
    assert acceptance._read_receipt(path) == document
    before = path.read_bytes()
    with pytest.raises((OSError, acceptance.fileops.FsopsError)):
        acceptance._write(path, {"status": "replacement"})
    assert path.read_bytes() == before
    path.chmod(0o644)
    path.write_bytes(before.replace(b'"tp":1', b'"tp":2', 1))
    with pytest.raises(ValueError, match="digest"):
        acceptance._read_receipt(path)


def test_runtime_failure_is_retained_without_private_paths_or_success(tmp_path, monkeypatch, capsys):
    (tmp_path / "fixture.py").write_text("# synthetic")
    for key, value in _environ().items():
        monkeypatch.setenv(key, value)

    def fail(*_):
        raise RuntimeError("sensitive /models/private-component/file.safetensors")

    monkeypatch.setattr(acceptance, "_run", fail)
    assert acceptance.main(_argv(tmp_path)) == 1
    root = tmp_path / "fresh-result"
    result = acceptance._read_receipt(root / "failed-rank-0.json")
    assert result["status"] == "FAILED" and result["error_type"] == "RuntimeError"
    assert not (root / "aggregate.json").exists()
    assert "private-component" not in capsys.readouterr().err
    before = {x.name: x.read_bytes() for x in root.iterdir() if x.is_file()}
    assert acceptance.main(_argv(tmp_path)) == 1
    assert {x.name: x.read_bytes() for x in root.iterdir() if x.is_file()} == before


def test_comparison_cli_writes_observation_and_never_overwrites(tmp_path):
    left, right, result = (tmp_path / name for name in ("tp1.json", "tp2.json", "comparison.json"))
    acceptance._write(left, _aggregate(1))
    acceptance._write(right, _aggregate(2, adjustment=2**-20))
    argv = ["compare", "--tp1-receipt", str(left), "--tp2-receipt", str(right), "--result", str(result)]
    assert acceptance.main(argv) == 0
    assert acceptance._read_receipt(result)["status"] == "TP_NUMERICS_OBSERVED"
    before = result.read_bytes()
    assert acceptance.main(argv) == 1 and result.read_bytes() == before
