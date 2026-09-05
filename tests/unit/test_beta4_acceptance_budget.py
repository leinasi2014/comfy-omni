"""Resource gates are exercised without mounting a real model or allocating its payload."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_beta4_dense_conversion import beta4_report
from test_native_export_transaction import _bound_plan, _tool, _write_fixture

from comfy_omni.artifacts.safetensors_writer import TensorPayload, _header
from comfy_omni.conversion.exporters.beta4 import build_beta4_dense_plan
from comfy_omni.conversion.exporters.execution import execute_native_export


def _harness():
    path = Path(__file__).resolve().parents[2] / "scripts/acceptance/beta4_dense_conversion.py"
    spec = importlib.util.spec_from_file_location("beta4_acceptance_budget", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_plan_disk_estimate_matches_writer_headers_without_tensor_allocation():
    harness = _harness()
    plan = build_beta4_dense_plan(beta4_report())
    estimate = harness._estimate_output(plan)
    actions = {a.target_name: a for a in plan.actions if a.target_name}
    exact = 0
    for shard in plan.shards:
        payloads = tuple(
            TensorPayload(name, actions[name].target_dtype, actions[name].shape, actions[name].target_bytes, lambda: ())
            for name in shard.tensor_names
        )
        header, _ = _header(payloads)
        exact += len(header) + shard.payload_bytes
    assert estimate["shard_file_bytes"] == exact
    assert estimate["payload_bytes"] == 40_222_925_872
    assert estimate["maximum_output_bytes"] < 45 * 1024**3
    assert estimate["maximum_output_bytes"] + 12 * 1024**3 < 67 * 1024**3


def test_disk_gate_rejects_inadequate_reserve_before_conversion(monkeypatch, tmp_path):
    harness = _harness()
    allocation = 40_222_925_872
    monkeypatch.setattr(
        harness.shutil, "disk_usage", lambda _: SimpleNamespace(free=allocation + harness.FREE_RESERVE - 1)
    )
    with pytest.raises(ValueError, match="12 GiB"):
        harness._disk_gate(tmp_path, allocation)
    monkeypatch.setattr(harness.shutil, "disk_usage", lambda _: SimpleNamespace(free=allocation + harness.FREE_RESERVE))
    assert harness._disk_gate(tmp_path, allocation) == allocation + harness.FREE_RESERVE


def test_publication_gate_rechecks_actual_staged_size_and_free_space(monkeypatch, tmp_path):
    harness = _harness()
    (tmp_path / "shard").write_bytes(b"example")
    estimate = {"maximum_output_bytes": len(b"example") + harness.MANIFEST_RESERVE}
    monkeypatch.setattr(
        harness.shutil, "disk_usage", lambda _: SimpleNamespace(free=harness.FREE_RESERVE + harness.MANIFEST_RESERVE)
    )
    harness._publication_gate(tmp_path, estimate)
    with pytest.raises(ValueError, match="allocation"):
        harness._publication_gate(tmp_path, {"maximum_output_bytes": 1})
    monkeypatch.setattr(harness.shutil, "disk_usage", lambda _: SimpleNamespace(free=harness.FREE_RESERVE))
    with pytest.raises(ValueError, match="12 GiB"):
        harness._publication_gate(tmp_path, estimate)


def test_failed_resource_callback_keeps_publication_absent_and_source_unchanged(tmp_path):
    source, output = tmp_path / "source.safetensors", tmp_path / "output"
    _write_fixture(source)
    original = source.read_bytes()
    observed = []

    def refuse(stage):
        assert (stage / "export.plan.json").is_file()
        assert not output.exists()
        observed.append(stage)
        raise ValueError("reserve exhausted before publication")

    with pytest.raises(ValueError, match="reserve exhausted"):
        execute_native_export(_bound_plan(source), output, tool=_tool(), before_publication=refuse)
    assert len(observed) == 1
    assert not output.exists()
    assert source.read_bytes() == original
