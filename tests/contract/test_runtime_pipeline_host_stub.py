from __future__ import annotations

import hashlib
import importlib
import sys
import types
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
from comfy_omni.integrations.vllm_omni.package_contract import RuntimePackageContractError


def _install_host_stubs(monkeypatch) -> types.ModuleType:
    host = types.ModuleType("vllm_omni")
    host.__path__ = []
    diffusion = types.ModuleType("vllm_omni.diffusion")
    diffusion.__path__ = []
    models = types.ModuleType("vllm_omni.diffusion.models")
    models.__path__ = []
    minimax = types.ModuleType("vllm_omni.diffusion.models.minimax_h3")
    minimax.__path__ = []
    leaf = types.ModuleType("vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3")

    class MiniMaxH3Pipeline:
        constructed: list[tuple[object, str]] = []

        def __init__(self, *, od_config, prefix: str = "") -> None:
            type(self).constructed.append((od_config, prefix))

    leaf.MiniMaxH3Pipeline = MiniMaxH3Pipeline
    leaf.get_minimax_h3_post_process_func = lambda: "official-post-process"

    host.diffusion = diffusion
    diffusion.models = models
    models.minimax_h3 = minimax
    minimax.pipeline_minimax_h3 = leaf

    for name, module in {
        "vllm_omni": host,
        "vllm_omni.diffusion": diffusion,
        "vllm_omni.diffusion.models": models,
        "vllm_omni.diffusion.models.minimax_h3": minimax,
        "vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3": leaf,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    return leaf


def _publish(tmp_path: Path) -> tuple[NativePackagePlan, Path]:
    tool = ToolIdentity("comfy-omni", "0.2.0a1", "a" * 40, "b" * 64)
    receipts: list[ComponentReceipt] = []
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
    plan = plan_native_package(tuple(receipts), vllm_omni_commit=PINNED_VLLM_OMNI_COMMIT)
    output = tmp_path / "native-package"
    materialized = materialize_package(plan, output)
    publish_package(plan, materialized)
    return plan, output


def test_pipeline_subclasses_the_official_pipeline_and_reexports_post_process(monkeypatch) -> None:
    leaf = _install_host_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules,
        "comfy_omni.integrations.vllm_omni.pipelines.runtime_pipeline",
        raising=False,
    )
    runtime_pipeline = importlib.import_module("comfy_omni.integrations.vllm_omni.pipelines.runtime_pipeline")

    assert issubclass(runtime_pipeline.H3ComfyMiniMaxH3Pipeline, leaf.MiniMaxH3Pipeline)
    assert runtime_pipeline.get_minimax_h3_post_process_func is leaf.get_minimax_h3_post_process_func


def test_pipeline_validates_the_package_before_super_init(monkeypatch, tmp_path: Path) -> None:
    leaf = _install_host_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules,
        "comfy_omni.integrations.vllm_omni.pipelines.runtime_pipeline",
        raising=False,
    )
    runtime_pipeline = importlib.import_module("comfy_omni.integrations.vllm_omni.pipelines.runtime_pipeline")

    plan, output = _publish(tmp_path)
    od_config = types.SimpleNamespace(model=str(output))

    pipeline = runtime_pipeline.H3ComfyMiniMaxH3Pipeline(od_config=od_config, prefix="p1")

    assert leaf.MiniMaxH3Pipeline.constructed == [(od_config, "p1")]
    assert pipeline.comfy_omni_package.plan_content_sha256 == plan.content_sha256


def test_pipeline_refuses_a_tampered_package_before_super_init(monkeypatch, tmp_path: Path) -> None:
    leaf = _install_host_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules,
        "comfy_omni.integrations.vllm_omni.pipelines.runtime_pipeline",
        raising=False,
    )
    runtime_pipeline = importlib.import_module("comfy_omni.integrations.vllm_omni.pipelines.runtime_pipeline")

    _, output = _publish(tmp_path)
    payload_path = output / "Ref2VA/transformer/nested/artifact.bin"
    payload_path.chmod(0o600)
    original = payload_path.read_bytes()
    payload_path.write_bytes(b"\x00" * len(original))
    od_config = types.SimpleNamespace(model=str(output))

    with pytest.raises(RuntimePackageContractError):
        runtime_pipeline.H3ComfyMiniMaxH3Pipeline(od_config=od_config, prefix="p1")

    assert leaf.MiniMaxH3Pipeline.constructed == []


def test_pipeline_resolves_the_real_package_root_through_a_serving_view(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    leaf = _install_host_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules,
        "comfy_omni.integrations.vllm_omni.pipelines.runtime_pipeline",
        raising=False,
    )
    runtime_pipeline = importlib.import_module("comfy_omni.integrations.vllm_omni.pipelines.runtime_pipeline")
    from comfy_omni.integrations.vllm_omni.serving import prepare_serving_layout

    _, output = _publish(tmp_path)
    view = prepare_serving_layout(output, tmp_path / "serving")
    od_config = types.SimpleNamespace(model=str(view))

    pipeline = runtime_pipeline.H3ComfyMiniMaxH3Pipeline(od_config=od_config, prefix="p1")

    assert leaf.MiniMaxH3Pipeline.constructed[-1][1] == "p1"
    assert pipeline.comfy_omni_package.package_root == output.resolve()


@pytest.mark.parametrize("named_sources", [False, True])
def test_explicit_raw_transformer_uses_existing_root_without_package_validation(monkeypatch, tmp_path, named_sources):
    """Selecting an original file must bypass the exported-package boundary."""
    _install_host_stubs(monkeypatch)
    module_name = "comfy_omni.integrations.vllm_omni.pipelines.runtime_pipeline"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    runtime = importlib.import_module(module_name)
    source = tmp_path / "original.safetensors"
    source.write_bytes(b"routing fixture, provider tested independently")
    component_root = tmp_path / "existing" / "Ref2VA"
    component_root.mkdir(parents=True)
    (component_root / "model_index.json").write_text('{"_minimax_h3":{"partition":"ref2va"}}')
    authenticated = object()
    calls = []
    provider_module = types.ModuleType("comfy_omni.runtime.h3.raw_beta4")

    class RawBeta4Binding:
        @staticmethod
        def establish(path):
            calls.append(path)
            return authenticated

    provider_module.RawBeta4Binding = RawBeta4Binding
    monkeypatch.setitem(sys.modules, provider_module.__name__, provider_module)
    implementation = types.ModuleType("comfy_omni.integrations.vllm_omni.pipelines.beta4_pipeline")

    class RawPipeline:
        def __init__(self, *, od_config, raw_binding, raw_selection="initial", prefix="", package=None):
            self.od_config = od_config
            self.raw_binding = raw_binding
            self.prefix = prefix
            self.registered = {}
            self.comfy_omni_active_h3_dit = self.comfy_omni_register_h3_dit(raw_selection, raw_binding)
            assert package is None

        def comfy_omni_register_h3_dit(self, selection, binding):
            self.registered[selection] = binding
            return selection

    implementation.H3Beta4Pipeline = RawPipeline
    monkeypatch.setitem(sys.modules, implementation.__name__, implementation)

    def forbidden_package(*args, **kwargs):
        raise AssertionError("original-file loading entered the exported-package boundary")

    monkeypatch.setattr(runtime, "_resolve_package_root", forbidden_package)
    settings = {"transformer_source": str(source)}
    if named_sources:
        settings = {
            "active": "a",
            "sources": {
                "a": {"path": str(source), "format": "h3-beta4-convrot"},
                "alias": {"path": str(source), "format": "h3-beta4-convrot"},
            },
        }
    config = types.SimpleNamespace(model=str(component_root), additional_config={"comfy_omni_h3": settings})
    result = runtime.H3ComfyMiniMaxH3Pipeline(od_config=config, prefix="original")
    assert isinstance(result, RawPipeline)
    assert result.raw_binding is authenticated
    assert result.od_config.model == str(component_root)
    assert result.prefix == "original"
    assert calls == [source]
    if named_sources:
        assert result.registered == {"a": authenticated, "alias": authenticated}
        assert result.comfy_omni_active_h3_dit == "a"
    assert sorted(item.name for item in component_root.iterdir()) == ["model_index.json"]
