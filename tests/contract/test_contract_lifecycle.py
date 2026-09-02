from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from comfy_omni.artifacts import fileops
from comfy_omni.artifacts.snapshot_schema import load_snapshot
from comfy_omni.artifacts.snapshot_store import catalog_with_store
from comfy_omni.contracts import ARCHITECTURE_TEMPLATES, COMPILE_TIME_CATALOG, ContractError
from comfy_omni.conversion.contract_workflows.census import FileRecord, census_tensors
from comfy_omni.conversion.contract_workflows.drafting import build_draft
from comfy_omni.conversion.contract_workflows.matching import build_scan_report
from comfy_omni.conversion.contract_workflows.pinning import pin_draft
from comfy_omni.domain.checkpoints import TensorDescriptor

TEMPLATE_NAME = "h3-te-pruned24-convrot"


def _marker(group_size: int) -> bytes:
    return json.dumps(
        {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": group_size},
        separators=(",", ":"),
    ).encode()


def _exact_text_encoder_census(source: Path):
    template = ARCHITECTURE_TEMPLATES[TEMPLATE_NAME]
    descriptors: list[TensorDescriptor] = []
    marker_payloads: dict[str, bytes] = {}
    offset = 0
    for prefix, (shape, group_size) in sorted(template.convrot_table().items()):
        marker = _marker(group_size)
        descriptors.extend(
            [
                TensorDescriptor(f"{prefix}.weight", "I8", shape, (offset, offset + 1)),
                TensorDescriptor(f"{prefix}.weight_scale", "F32", (shape[0], 1), (offset + 1, offset + 2)),
                TensorDescriptor(
                    f"{prefix}.comfy_quant",
                    "U8",
                    (len(marker),),
                    (offset + 2, offset + 2 + len(marker)),
                ),
            ]
        )
        marker_payloads[f"{prefix}.comfy_quant"] = marker
        offset += len(marker) + 2
    for name, (dtype, shape) in template.non_quantized_inventory.items():
        descriptors.append(TensorDescriptor(name, dtype, shape, (offset, offset + 1)))
        offset += 1
    payload = source.read_bytes()
    record = FileRecord(str(source), len(payload), hashlib.sha256(payload).hexdigest())
    return census_tensors(descriptors, marker_payloads, files=(record,))


def _write_draft(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source.safetensors"
    source.write_bytes(b"bound source bytes")
    census = _exact_text_encoder_census(source)
    scan = build_scan_report(census)
    assert scan.matched_template == TEMPLATE_NAME
    draft = build_draft(census, scan.match, generator_identity={"operator": "generator-alice"})
    target = tmp_path / "draft.json"
    draft.write_to(target)
    return target, source


def test_draft_pin_load_and_explicit_catalog_are_immutable(tmp_path: Path) -> None:
    draft, _ = _write_draft(tmp_path)
    evidence = tmp_path / "review.md"
    evidence.write_text("reviewed exact template evidence", encoding="utf-8")
    store = tmp_path / "contracts"
    result = pin_draft(
        draft,
        name="external-te-reviewed-v1",
        reviewer="reviewer-bob",
        evidence_path=evidence,
        contract_dir=store,
        enforce_observed_schema=True,
    )
    loaded = load_snapshot(result.snapshot.path)
    assert loaded.manifest_sha256 == result.snapshot.manifest_sha256
    assert loaded.contract_block["schema_sha256"] is not None
    extended = catalog_with_store(COMPILE_TIME_CATALOG, store)
    assert len(COMPILE_TIME_CATALOG.records) == 3
    assert len(extended.records) == 4
    assert (
        extended.resolve("text_encoder", "external-te-reviewed-v1").snapshot_manifest_sha256 == loaded.manifest_sha256
    )
    with pytest.raises(ContractError, match="overwrite"):
        pin_draft(
            draft,
            name="external-te-reviewed-v1",
            reviewer="reviewer-bob",
            evidence_path=evidence,
            contract_dir=store,
            enforce_observed_schema=True,
        )


def test_pin_rejects_generator_as_reviewer_and_stale_source(tmp_path: Path) -> None:
    draft, source = _write_draft(tmp_path)
    evidence = tmp_path / "review.md"
    evidence.write_text("evidence", encoding="utf-8")
    with pytest.raises(ContractError, match="cannot be its reviewer"):
        pin_draft(
            draft,
            name="self-reviewed",
            reviewer="generator-alice",
            evidence_path=evidence,
            contract_dir=tmp_path / "self-store",
        )
    source.write_bytes(b"changed after drafting")
    with pytest.raises(ContractError, match="stale draft"):
        pin_draft(
            draft,
            name="stale-source",
            reviewer="reviewer-bob",
            evidence_path=evidence,
            contract_dir=tmp_path / "stale-store",
        )


def test_pin_rejects_canonical_template_digest_tamper(tmp_path: Path) -> None:
    draft, _ = _write_draft(tmp_path)
    document = json.loads(draft.read_bytes())
    document["provenance"]["template"]["digest"] = "0" * 64
    draft.chmod(stat.S_IWRITE | stat.S_IREAD)
    draft.write_bytes(fileops.canonical_json(document))
    evidence = tmp_path / "review.md"
    evidence.write_text("evidence", encoding="utf-8")
    with pytest.raises(ContractError, match="template drifted"):
        pin_draft(
            draft,
            name="tampered",
            reviewer="reviewer-bob",
            evidence_path=evidence,
            contract_dir=tmp_path / "tampered-store",
        )


def test_snapshot_tamper_fails_content_address_gate(tmp_path: Path) -> None:
    draft, _ = _write_draft(tmp_path)
    evidence = tmp_path / "review.md"
    evidence.write_text("evidence", encoding="utf-8")
    result = pin_draft(
        draft,
        name="snapshot-tamper",
        reviewer="reviewer-bob",
        evidence_path=evidence,
        contract_dir=tmp_path / "store",
    )
    target = result.snapshot.path
    document = json.loads(target.read_bytes())
    document["contract"]["name"] = "defaced"
    os.chmod(target, stat.S_IWRITE | stat.S_IREAD)
    target.write_bytes(fileops.canonical_json(document))
    with pytest.raises(ContractError, match="manifest digest"):
        load_snapshot(target)
