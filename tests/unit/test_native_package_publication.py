from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest

from comfy_omni.artifacts import fileops
from comfy_omni.conversion.packaging.materialization import materialize_package
from comfy_omni.conversion.packaging.models import ComponentFile, ComponentReceipt, NativePackagePlan
from comfy_omni.conversion.packaging.planning import (
    PACKAGE_COMPONENTS,
    PINNED_VLLM_OMNI_COMMIT,
    plan_native_package,
)
from comfy_omni.conversion.packaging.publication import PackagePublicationError, publish_package
from comfy_omni.domain.normalization import ToolIdentity


def _fixture(tmp_path: Path) -> tuple[NativePackagePlan, dict[str, bytes]]:
    tool = ToolIdentity("comfy-omni", "0.2.0a1", "a" * 40, "b" * 64)
    receipts: list[ComponentReceipt] = []
    payloads: dict[str, bytes] = {}
    for component in PACKAGE_COMPONENTS:
        source = tmp_path / "sources" / component
        source.mkdir(parents=True)
        payload = f"{component}:payload".encode()
        relative = "nested/artifact.bin"
        target = source / relative
        target.parent.mkdir()
        target.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        receipts.append(
            ComponentReceipt(
                component=component,
                source_dir=source.as_posix(),
                receipt_schema="test.component.receipt/v1",
                receipt_sha256=hashlib.sha256(f"{component}:receipt".encode()).hexdigest(),
                tool=tool,
                files=(ComponentFile(relative, len(payload), digest),),
            )
        )
        target_path = f"Ref2VA/{component}/{relative}"
        payloads[target_path] = payload
    return plan_native_package(tuple(receipts), vllm_omni_commit=PINNED_VLLM_OMNI_COMMIT), payloads


def _census(plan: NativePackagePlan) -> list[dict[str, object]]:
    return [{"path": item.target_path, "sha256": item.sha256, "size": item.size} for item in plan.files]


def test_publish_package_publishes_manifest_last_atomically(tmp_path: Path) -> None:
    from comfy_omni.conversion.packaging.publication import (
        PACKAGE_PUBLICATION_SCHEMA,
        publish_package,
    )

    plan, payloads = _fixture(tmp_path)
    output = tmp_path / "native-package"

    materialized = materialize_package(plan, output)
    published = publish_package(plan, materialized)

    assert published.schema == PACKAGE_PUBLICATION_SCHEMA
    assert published.plan_content_sha256 == plan.content_sha256
    assert not materialized.stage_dir.exists()
    assert output.is_dir()
    on_disk = {item.relative_to(output).as_posix(): item.read_bytes() for item in output.rglob("*") if item.is_file()}
    assert set(on_disk) == set(payloads) | {"h3-comfy-package.json"}
    for path, payload in payloads.items():
        assert on_disk[path] == payload
    assert published.file_count == len(plan.files)
    assert published.total_bytes == sum(item.size for item in plan.files)
    assert published.output_dir == output
    assert published.to_dict()["status"] == "PUBLISHED"

    manifest = fileops.parse_json_strict((output / "h3-comfy-package.json").read_bytes())
    assert manifest["schema"] == "h3-comfy-package/v3"
    assert manifest["plan_content_sha256"] == plan.content_sha256
    assert manifest["file_count"] == len(plan.files)
    assert manifest["total_bytes"] == sum(item.size for item in plan.files)
    assert manifest["files"] == _census(plan)
    assert manifest["routing"]["serving_entrypoint"] == "Ref2VA/"
    assert manifest["routing"]["resident_dit_count"] == 1
    assert manifest["routing"]["supported_tasks"] == ["ref2va", "t2va", "fl2va"]
    digest = hashlib.sha256(
        fileops.canonical_json({k: v for k, v in manifest.items() if k != "package_manifest_sha256"})
    ).hexdigest()
    assert manifest["package_manifest_sha256"] == digest
    assert published.manifest_sha256 == digest


def test_publish_package_refuses_a_plan_handle_digest_mismatch(tmp_path: Path) -> None:
    plan, _ = _fixture(tmp_path)
    output = tmp_path / "native-package"
    materialized = materialize_package(plan, output)

    mismatched = replace(plan, content_sha256="0" * 64)

    with pytest.raises(PackagePublicationError, match="plan digest") as failure:
        publish_package(mismatched, materialized)

    assert failure.value.evidence["stage"] == "plan-binding"
    assert not output.exists()
    assert materialized.stage_dir.exists()


def test_publish_package_rejects_post_materialization_staged_tampering(tmp_path: Path) -> None:
    plan, _ = _fixture(tmp_path)
    output = tmp_path / "native-package"
    materialized = materialize_package(plan, output)

    staged = materialized.stage_dir / "Ref2VA/transformer/nested/artifact.bin"
    original = staged.read_bytes()
    staged.chmod(0o600)
    staged.write_bytes(b"\x00" * len(original))

    with pytest.raises(PackagePublicationError, match="SHA256 differs") as failure:
        publish_package(plan, materialized)

    assert failure.value.evidence["stage"] == "file-verification"
    assert not output.exists()
    assert materialized.stage_dir.exists()


def test_publish_package_rejects_an_unexpected_staged_entry(tmp_path: Path) -> None:
    plan, _ = _fixture(tmp_path)
    output = tmp_path / "native-package"
    materialized = materialize_package(plan, output)

    (materialized.stage_dir / "unexpected.bin").write_bytes(b"unexpected")

    with pytest.raises(PackagePublicationError, match="file census differs") as failure:
        publish_package(plan, materialized)

    assert failure.value.evidence["stage"] == "staging-census"
    assert not output.exists()
    assert materialized.stage_dir.exists()


def test_publish_package_rejects_a_missing_staged_entry(tmp_path: Path) -> None:
    plan, _ = _fixture(tmp_path)
    output = tmp_path / "native-package"
    materialized = materialize_package(plan, output)

    staged = materialized.stage_dir / "Ref2VA/transformer/nested/artifact.bin"
    staged.chmod(0o600)
    staged.unlink()

    with pytest.raises(PackagePublicationError, match="file census differs") as failure:
        publish_package(plan, materialized)

    assert failure.value.evidence["stage"] == "staging-census"
    assert not output.exists()
    assert materialized.stage_dir.exists()


def test_publish_package_refuses_an_output_that_appeared_before_publication(tmp_path: Path) -> None:
    plan, _ = _fixture(tmp_path)
    output = tmp_path / "native-package"
    materialized = materialize_package(plan, output)

    output.mkdir()
    marker = output / "marker.txt"
    marker.write_bytes(b"keep-me")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        publish_package(plan, materialized)

    assert marker.read_bytes() == b"keep-me"
    assert materialized.stage_dir.exists()


def test_publish_package_rejects_a_replaced_staging_directory(tmp_path: Path) -> None:
    plan, payloads = _fixture(tmp_path)
    output = tmp_path / "native-package"
    materialized = materialize_package(plan, output)

    stage = materialized.stage_dir
    moved = stage.parent / "moved-away-stage"
    stage.rename(moved)
    stage.mkdir()
    for target_path, payload in payloads.items():
        path = stage.joinpath(*PurePosixPath(target_path).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    with pytest.raises(PackagePublicationError, match="identity changed") as failure:
        publish_package(plan, materialized)

    assert failure.value.evidence["stage"] == "staging"
    assert not output.exists()
    assert stage.exists()


def test_publish_package_emits_the_host_discovery_model_index(tmp_path: Path) -> None:
    plan, _ = _fixture(tmp_path)
    output = tmp_path / "native-package"

    materialized = materialize_package(plan, output)
    publish_package(plan, materialized)

    model_index = output / "model_index.json"
    assert model_index.is_file()
    expected_index = {
        "_class_name": "MiniMaxH3Pipeline",
        "_diffusers_version": "0.32.2",
        "_minimax_h3": {
            "partition": "ref2va",
            "sigma_shift_scales": {"audio": 3.0, "video": 12.0},
            "schema_version": 1,
            "task_aliases": {},
            "tasks": list(plan.supported_tasks),
        },
        "audio_vae": ["diffusers", "MiniMaxH3AudioVAE"],
        "processor": ["transformers", "Qwen3VLProcessor"],
        "scheduler": None,
        "text_encoder": ["transformers", "MiniMaxH3Qwen3VLHFEncoder"],
        "tokenizer": ["transformers", "Qwen2TokenizerFast"],
        "transformer": ["diffusers", "MiniMaxH3DiTModel"],
        "video_vae": ["diffusers", "MiniMaxH3VideoVAE"],
    }
    parsed = fileops.parse_json_strict(model_index.read_bytes())
    assert parsed == expected_index
    assert model_index.read_bytes() == fileops.canonical_json(parsed)

    manifest = fileops.parse_json_strict((output / "h3-comfy-package.json").read_bytes())
    assert manifest["model_index_sha256"] == hashlib.sha256(model_index.read_bytes()).hexdigest()
