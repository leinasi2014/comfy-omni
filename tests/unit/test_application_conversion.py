from __future__ import annotations

from pathlib import Path

from comfy_omni.application import conversion


def test_convert_native_export_composes_explicit_plan_and_transaction(monkeypatch) -> None:
    sources = (Path("/models/source.safetensors"),)
    output = Path("/evidence/output")
    snapshot = Path("/contracts/source.json")
    plan = object()
    publication = object()
    tool = object()
    catalog = object()
    templates = object()
    observed: dict[str, object] = {}

    def fake_plan(given_sources, **kwargs):
        observed["plan"] = (given_sources, kwargs)
        return plan

    def fake_execute(given_plan, given_output, **kwargs):
        observed["execute"] = (given_plan, given_output, kwargs)
        return publication

    monkeypatch.setattr(conversion, "plan_native_export", fake_plan)
    monkeypatch.setattr(conversion, "execute_native_export", fake_execute, raising=False)

    result = conversion.convert_native_export(
        sources,
        output,
        tool=tool,
        component="transformer",
        source_profile="source-profile-v1",
        profile_name="target-profile-v1",
        max_rows=4096,
        max_shard_bytes=8 * 1024**3,
        catalog=catalog,
        templates=templates,
        source_contract_snapshot=snapshot,
    )

    assert result is publication
    assert observed == {
        "plan": (
            sources,
            {
                "catalog": catalog,
                "component": "transformer",
                "max_rows": 4096,
                "max_shard_bytes": 8 * 1024**3,
                "profile_name": "target-profile-v1",
                "source_profile": "source-profile-v1",
                "templates": templates,
            },
        ),
        "execute": (
            plan,
            output,
            {"source_contract_snapshot": snapshot, "tool": tool},
        ),
    }
