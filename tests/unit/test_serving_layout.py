from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from comfy_omni.conversion.packaging.materialization import materialize_package
from comfy_omni.conversion.packaging.models import ComponentFile, ComponentReceipt, NativePackagePlan
from comfy_omni.conversion.packaging.planning import (
    PACKAGE_COMPONENTS,
    PINNED_VLLM_OMNI_COMMIT,
    plan_native_package,
)
from comfy_omni.conversion.packaging.publication import publish_package
from comfy_omni.domain.normalization import ToolIdentity
from comfy_omni.integrations.vllm_omni.package_contract import RuntimePackageContractError, validate_runtime_package


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


def test_prepare_serving_layout_creates_the_partition_view(tmp_path: Path) -> None:
    from comfy_omni.integrations.vllm_omni.serving import SERVING_PARTITION_NAME, prepare_serving_layout

    _, output = _published(tmp_path)
    work = tmp_path / "serving"

    view = prepare_serving_layout(output, work)

    assert view == work / SERVING_PARTITION_NAME
    assert view.is_symlink()
    assert view.resolve() == output.resolve()
    assert (view / "model_index.json").is_file()
    assert (work / ".comfy-omni-serving").is_file()

    contract = validate_runtime_package(view)
    assert contract.package_root == output.resolve()
    assert contract.partition == "ref2va"

    again = prepare_serving_layout(output, work)
    assert again == view


def test_prepare_serving_layout_propagates_an_invalid_package(tmp_path: Path) -> None:
    from comfy_omni.integrations.vllm_omni.serving import prepare_serving_layout

    _, output = _published(tmp_path)
    payload_path = output / "Ref2VA/transformer/nested/artifact.bin"
    original = payload_path.read_bytes()
    payload_path.write_bytes(b"\x00" * len(original))
    work = tmp_path / "serving"

    with pytest.raises(RuntimePackageContractError) as failure:
        prepare_serving_layout(output, work)

    assert failure.value.evidence["stage"] == "file-verification"
    assert not work.exists()


def test_prepare_serving_layout_refuses_a_divergent_work_dir(tmp_path: Path) -> None:
    from comfy_omni.integrations.vllm_omni.serving import ServingLayoutError, prepare_serving_layout

    _, output = _published(tmp_path)
    work = tmp_path / "serving"
    work.mkdir()
    (work / "stray.txt").write_bytes(b"stray")

    with pytest.raises(ServingLayoutError) as failure:
        prepare_serving_layout(output, work)

    assert failure.value.evidence["stage"] == "layout-binding"


def test_prepare_serving_layout_refuses_a_wrong_marker_content(tmp_path: Path) -> None:
    from comfy_omni.integrations.vllm_omni.serving import (
        SERVING_PARTITION_NAME,
        ServingLayoutError,
        prepare_serving_layout,
    )

    _, output = _published(tmp_path)
    work = tmp_path / "serving"
    work.mkdir()
    (work / ".comfy-omni-serving").write_bytes(b"wrong")
    os.symlink(output, work / SERVING_PARTITION_NAME, target_is_directory=True)

    with pytest.raises(ServingLayoutError) as failure:
        prepare_serving_layout(output, work)

    assert failure.value.evidence["stage"] == "layout-binding"


def test_clear_serving_layout_removes_only_its_own_layout(tmp_path: Path) -> None:
    from comfy_omni.integrations.vllm_omni.serving import (
        SERVING_PARTITION_NAME,
        ServingLayoutError,
        clear_serving_layout,
        prepare_serving_layout,
    )

    _, output = _published(tmp_path)
    work = tmp_path / "serving"
    prepare_serving_layout(output, work)

    clear_serving_layout(work)

    assert not (work / ".comfy-omni-serving").exists()
    assert not (work / SERVING_PARTITION_NAME).exists()
    assert not work.exists()

    other = tmp_path / "not-a-layout"
    other.mkdir()
    with pytest.raises(ServingLayoutError) as failure:
        clear_serving_layout(other)

    assert failure.value.evidence["stage"] == "layout-binding"


def test_describe_serving_command_returns_the_legacy_serve_shape(tmp_path: Path) -> None:
    from comfy_omni.integrations.vllm_omni.serving import describe_serving_command

    model_path = Path("serving/Ref2VA")
    command = describe_serving_command(model_path)

    assert "vllm serve" in command
    assert str(model_path) in command
    assert "--model-class-name MiniMaxH3Pipeline" in command

    with_args = describe_serving_command(model_path, host_args=("--host", "0.0.0.0", "--port", "8000"))

    assert "--host 0.0.0.0" in with_args
    assert "--port 8000" in with_args
