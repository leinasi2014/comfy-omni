from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

from comfy_omni.conversion.packaging.materialization import materialize_package
from comfy_omni.conversion.packaging.models import ComponentFile, ComponentReceipt
from comfy_omni.conversion.packaging.planning import PACKAGE_COMPONENTS, PINNED_VLLM_OMNI_COMMIT, plan_native_package
from comfy_omni.domain.normalization import ToolIdentity


def _plan(tmp_path: Path):
    tool = ToolIdentity("comfy-omni", "0.2.0a1", "a" * 40, "b" * 64)
    receipts = []
    sources = {}
    for component in PACKAGE_COMPONENTS:
        directory = tmp_path / "sources" / component
        directory.mkdir(parents=True)
        payload = (component + ":immutable").encode()
        source = directory / "payload.bin"
        source.write_bytes(payload)
        source.chmod(0o444)
        sources[f"Ref2VA/{component}/payload.bin"] = source
        receipts.append(
            ComponentReceipt(
                component, str(directory), "test.component/v1", hashlib.sha256(component.encode()).hexdigest(), tool,
                (ComponentFile("payload.bin", len(payload), hashlib.sha256(payload).hexdigest()),),
            )
        )
    return plan_native_package(tuple(receipts), vllm_omni_commit=PINNED_VLLM_OMNI_COMMIT), sources


def _reuse(plan, output):
    # Characterize actual baseline copying; absence of a future keyword is not RED.
    options = {"reuse_immutable": True} if "reuse_immutable" in inspect.signature(materialize_package).parameters else {}
    return materialize_package(plan, output, **options)


def test_immutable_package_does_not_allocate_duplicate_payload_inodes(tmp_path):
    plan, sources = _plan(tmp_path)
    staged = _reuse(plan, tmp_path / "package")
    for relative, source in sources.items():
        target = staged.stage_dir / relative
        assert target.read_bytes() == source.read_bytes()
        assert target.stat().st_ino == source.stat().st_ino, "immutable package duplicated payload storage"
