from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from comfy_omni.artifacts import fileops
from comfy_omni.conversion.packaging.materialization import materialize_package
from comfy_omni.conversion.packaging.models import ComponentFile, ComponentReceipt, NativePackagePlan
from comfy_omni.conversion.packaging.planning import (
    PACKAGE_COMPONENTS,
    PINNED_VLLM_OMNI_COMMIT,
    plan_native_package,
)
from comfy_omni.conversion.packaging.publication import publish_package
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


def _published(tmp_path: Path) -> tuple[NativePackagePlan, Path]:
    plan, _ = _fixture(tmp_path)
    output = tmp_path / "native-package"
    materialized = materialize_package(plan, output)
    publish_package(plan, materialized)
    return plan, output


def test_validate_runtime_package_binds_the_published_contract(tmp_path: Path) -> None:
    from comfy_omni.integrations.vllm_omni.package_contract import RuntimePackageContractError, validate_runtime_package

    plan, _ = _fixture(tmp_path)
    output = tmp_path / "native-package"

    materialized = materialize_package(plan, output)
    publish_package(plan, materialized)

    contract = validate_runtime_package(output)

    assert not isinstance(contract, RuntimePackageContractError)
    assert contract.package_root == output.resolve()
    assert contract.plan_content_sha256 == plan.content_sha256

    manifest = fileops.parse_json_strict((output / "h3-comfy-package.json").read_bytes())
    manifest_digest = hashlib.sha256(
        fileops.canonical_json({key: value for key, value in manifest.items() if key != "package_manifest_sha256"})
    ).hexdigest()
    assert contract.manifest_sha256 == manifest_digest

    model_index = output / "model_index.json"
    assert contract.model_index_sha256 == hashlib.sha256(model_index.read_bytes()).hexdigest()

    assert contract.class_name == "MiniMaxH3Pipeline"
    assert contract.partition == "ref2va"
    assert contract.supported_tasks == tuple(plan.supported_tasks)
    assert contract.file_count == len(plan.files)
    assert contract.total_bytes == sum(item.size for item in plan.files)

    assert set(contract.component_paths) == set(PACKAGE_COMPONENTS)
    for component, directory in contract.component_paths.items():
        assert directory == (output / "Ref2VA" / component).resolve()

    assert contract.to_dict()["status"] == "RUNTIME_VERIFIED"


def test_validate_runtime_package_refuses_a_tampered_manifest(tmp_path: Path) -> None:
    from comfy_omni.integrations.vllm_omni.package_contract import (
        RuntimePackageContractError,
        validate_runtime_package,
    )

    _, output = _published(tmp_path)
    manifest_path = output / "h3-comfy-package.json"
    manifest_path.chmod(0o600)
    manifest = fileops.parse_json_strict(manifest_path.read_bytes())
    manifest["file_count"] = manifest["file_count"] + 1
    manifest_path.write_bytes(fileops.canonical_json(manifest))

    with pytest.raises(RuntimePackageContractError) as failure:
        validate_runtime_package(output)

    assert failure.value.evidence["stage"] == "manifest"


def test_validate_runtime_package_refuses_a_rewritten_payload(tmp_path: Path) -> None:
    from comfy_omni.integrations.vllm_omni.package_contract import (
        RuntimePackageContractError,
        validate_runtime_package,
    )

    _, output = _published(tmp_path)
    payload_path = output / "Ref2VA/transformer/nested/artifact.bin"
    payload_path.chmod(0o600)
    original = payload_path.read_bytes()
    payload_path.write_bytes(b"\x00" * len(original))

    with pytest.raises(RuntimePackageContractError) as failure:
        validate_runtime_package(output)

    assert failure.value.evidence["stage"] == "file-verification"


def test_validate_runtime_package_refuses_a_missing_component_file(tmp_path: Path) -> None:
    from comfy_omni.integrations.vllm_omni.package_contract import (
        RuntimePackageContractError,
        validate_runtime_package,
    )

    _, output = _published(tmp_path)
    payload_path = output / "Ref2VA/transformer/nested/artifact.bin"
    payload_path.chmod(0o600)
    payload_path.unlink()

    with pytest.raises(RuntimePackageContractError) as failure:
        validate_runtime_package(output)

    assert failure.value.evidence["stage"] == "tree-census"


def test_validate_runtime_package_refuses_an_unexpected_extra_file(tmp_path: Path) -> None:
    from comfy_omni.integrations.vllm_omni.package_contract import (
        RuntimePackageContractError,
        validate_runtime_package,
    )

    _, output = _published(tmp_path)
    (output / "unexpected.bin").write_bytes(b"unexpected")

    with pytest.raises(RuntimePackageContractError) as failure:
        validate_runtime_package(output)

    assert failure.value.evidence["stage"] == "tree-census"


def test_validate_runtime_package_refuses_a_missing_model_index(tmp_path: Path) -> None:
    from comfy_omni.integrations.vllm_omni.package_contract import (
        RuntimePackageContractError,
        validate_runtime_package,
    )

    _, output = _published(tmp_path)
    index_path = output / "model_index.json"
    index_path.chmod(0o600)
    index_path.unlink()

    with pytest.raises(RuntimePackageContractError) as failure:
        validate_runtime_package(output)

    assert failure.value.evidence["stage"] == "model-index"


def test_validate_runtime_package_refuses_a_tampered_class_name(tmp_path: Path) -> None:
    from comfy_omni.integrations.vllm_omni.package_contract import (
        RuntimePackageContractError,
        validate_runtime_package,
    )

    _, output = _published(tmp_path)
    index_path = output / "model_index.json"
    index_path.chmod(0o600)
    index = fileops.parse_json_strict(index_path.read_bytes())
    index["_class_name"] = "DifferentPipeline"
    index_path.write_bytes(fileops.canonical_json(index))

    with pytest.raises(RuntimePackageContractError) as failure:
        validate_runtime_package(output)

    assert failure.value.evidence["stage"] == "model-index"


def test_validate_runtime_package_refuses_task_routing_drift(tmp_path: Path) -> None:
    from comfy_omni.integrations.vllm_omni.package_contract import (
        RuntimePackageContractError,
        validate_runtime_package,
    )

    _, output = _published(tmp_path)
    index_path = output / "model_index.json"
    index_path.chmod(0o600)
    index = fileops.parse_json_strict(index_path.read_bytes())
    minimax = index["_minimax_h3"]
    minimax["tasks"] = ["ref2va"]
    new_index_bytes = fileops.canonical_json(index)
    index_path.write_bytes(new_index_bytes)
    index_sha256 = hashlib.sha256(new_index_bytes).hexdigest()

    manifest_path = output / "h3-comfy-package.json"
    manifest_path.chmod(0o600)
    manifest = fileops.parse_json_strict(manifest_path.read_bytes())
    manifest["model_index_sha256"] = index_sha256
    manifest_digest = hashlib.sha256(
        fileops.canonical_json({key: value for key, value in manifest.items() if key != "package_manifest_sha256"})
    ).hexdigest()
    manifest["package_manifest_sha256"] = manifest_digest
    manifest_path.write_bytes(fileops.canonical_json(manifest))

    with pytest.raises(RuntimePackageContractError) as failure:
        validate_runtime_package(output)

    assert failure.value.evidence["stage"] == "routing"


def test_validate_runtime_package_refuses_a_link_inside_the_tree(tmp_path: Path) -> None:
    from comfy_omni.integrations.vllm_omni.package_contract import (
        RuntimePackageContractError,
        validate_runtime_package,
    )

    _, output = _published(tmp_path)
    link = output / "Ref2VA/transformer/nested/link-target"
    try:
        os.symlink("artifact.bin", link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available on this platform")

    with pytest.raises(RuntimePackageContractError) as failure:
        validate_runtime_package(output)

    assert failure.value.evidence["stage"] == "tree-census"


def test_validate_runtime_package_refuses_a_non_package_path(tmp_path: Path) -> None:
    from comfy_omni.integrations.vllm_omni.package_contract import (
        RuntimePackageContractError,
        validate_runtime_package,
    )

    non_package = tmp_path / "not-a-package"
    non_package.mkdir()

    with pytest.raises(RuntimePackageContractError) as failure:
        validate_runtime_package(non_package)

    assert failure.value.evidence["stage"] == "model-index"
