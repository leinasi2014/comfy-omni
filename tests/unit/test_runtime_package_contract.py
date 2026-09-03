from __future__ import annotations

import hashlib
from pathlib import Path

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
    assert contract.supported_tasks == ["ref2va", "t2va", "fl2va"]
    assert contract.file_count == len(plan.files)
    assert contract.total_bytes == sum(item.size for item in plan.files)

    assert set(contract.component_paths) == set(PACKAGE_COMPONENTS)
    for component, directory in contract.component_paths.items():
        assert directory == (output / "Ref2VA" / component).resolve()

    assert contract.to_dict()["status"] == "RUNTIME_VERIFIED"
