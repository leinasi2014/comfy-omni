from __future__ import annotations

import errno
import hashlib
import importlib.util
import inspect
import os
import tempfile
from pathlib import Path

import pytest

from comfy_omni.artifacts import fileops, immutable_links
from comfy_omni.artifacts.immutable_links import reuse_file_pinned_exclusive
from comfy_omni.contracts.models import ContractError
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
                component,
                str(directory),
                "test.component/v1",
                hashlib.sha256(component.encode()).hexdigest(),
                tool,
                (ComponentFile("payload.bin", len(payload), hashlib.sha256(payload).hexdigest()),),
            )
        )
    return plan_native_package(tuple(receipts), vllm_omni_commit=PINNED_VLLM_OMNI_COMMIT), sources


def _reuse(plan, output):
    # Characterize actual baseline copying; absence of a future keyword is not RED.
    options = (
        {"reuse_immutable": True} if "reuse_immutable" in inspect.signature(materialize_package).parameters else {}
    )
    return materialize_package(plan, output, **options)


def test_immutable_package_does_not_allocate_duplicate_payload_inodes(tmp_path):
    plan, sources = _plan(tmp_path)
    staged = _reuse(plan, tmp_path / "package")
    for relative, source in sources.items():
        target = staged.stage_dir / relative
        assert target.read_bytes() == source.read_bytes()
        assert target.stat().st_ino == source.stat().st_ino, "immutable package duplicated payload storage"


def test_shared_package_publishes_and_accounts_for_zero_copied_bytes(tmp_path):
    from comfy_omni.conversion.packaging.publication import publish_package

    plan, sources = _plan(tmp_path)
    staged = materialize_package(plan, tmp_path / "package", reuse_immutable=True, max_copy_bytes=0)
    assert staged.to_dict()["storage"] == {
        "mode": "immutable-reuse/v1",
        "shared_bytes": sum(p.stat().st_size for p in sources.values()),
        "copied_bytes": 0,
    }
    published = publish_package(plan, staged)
    for relative, source in sources.items():
        target = published.output_dir / relative
        assert target.read_bytes() == source.read_bytes()
        assert target.samefile(source)
        assert target.stat().st_mode & 0o222 == 0


def test_default_copy_keeps_independent_inodes_and_legacy_serialization(tmp_path):
    plan, sources = _plan(tmp_path)
    staged = materialize_package(plan, tmp_path / "package")
    assert "storage" not in staged.to_dict()
    for relative, source in sources.items():
        assert not (staged.stage_dir / relative).samefile(source)


def test_writable_source_cannot_be_shared(tmp_path):
    plan, sources = _plan(tmp_path)
    next(iter(sources.values())).chmod(0o644)
    output = tmp_path / "package"
    with pytest.raises(ContractError, match="copy failed") as failure:
        materialize_package(plan, output, reuse_immutable=True)
    assert "writable sources" in failure.value.evidence["cause"]
    assert not output.exists()


def test_cross_filesystem_copy_uses_real_distinct_storage_and_budget(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"small immutable payload")
    source.chmod(0o444)
    with tempfile.TemporaryDirectory(prefix="comfy-reuse-", dir="/dev/shm") as parent:
        target = Path(parent) / "copied"
        assert source.stat().st_dev != target.parent.stat().st_dev
        digest, size, shared = reuse_file_pinned_exclusive(source, target, max_copy_bytes=source.stat().st_size)
        assert not shared and target.read_bytes() == source.read_bytes()
        assert digest == hashlib.sha256(target.read_bytes()).hexdigest() and size == target.stat().st_size
        denied = Path(parent) / "denied"
        with pytest.raises(fileops.FsopsIoError, match="allocation budget"):
            reuse_file_pinned_exclusive(source, denied, max_copy_bytes=size - 1)
        assert not denied.exists()


def test_copy_budget_checks_newly_opened_inode_before_destination_create(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_bytes(b"old")
    target = tmp_path / "target"
    real_open = fileops._open_pinned

    def replace_before_open(path):
        source.write_bytes(b"larger than the declared budget")
        return real_open(path)

    monkeypatch.setattr(fileops, "_open_pinned", replace_before_open)
    with pytest.raises(fileops.FsopsIoError, match="allocation budget"):
        fileops.copy_file_pinned_exclusive(source, target, max_bytes=3)
    assert not target.exists()


def _immutable_file(path, content=b"immutable"):
    path.write_bytes(content)
    path.chmod(0o444)
    return path


@pytest.mark.parametrize("after_link", [False, True])
def test_source_mutation_during_hash_or_link_refuses_success(tmp_path, monkeypatch, after_link):
    source = _immutable_file(tmp_path / "source")
    target = tmp_path / "target"

    def mutate():
        source.chmod(0o644)
        source.write_bytes(b"changed!!")
        source.chmod(0o444)

    if after_link:
        real_link = os.link

        def link_then_mutate(*args, **kwargs):
            real_link(*args, **kwargs)
            mutate()

        monkeypatch.setattr(immutable_links.os, "link", link_then_mutate)
    else:
        real_digest = immutable_links._digest

        def hash_then_mutate(descriptor):
            result = real_digest(descriptor)
            mutate()
            return result

        monkeypatch.setattr(immutable_links, "_digest", hash_then_mutate)
    with pytest.raises(fileops.FsopsModifiedError):
        reuse_file_pinned_exclusive(source, target)


def test_existing_target_is_not_overwritten(tmp_path):
    source = _immutable_file(tmp_path / "source")
    target = _immutable_file(tmp_path / "target", b"prior")
    with pytest.raises(fileops.FsopsExistsError):
        reuse_file_pinned_exclusive(source, target)
    assert target.read_bytes() == b"prior"


@pytest.mark.parametrize("budget", [-1, True, "0"])
def test_invalid_budget_is_rejected_before_sharing(tmp_path, budget):
    source = _immutable_file(tmp_path / "source")
    target = tmp_path / "target"
    with pytest.raises(fileops.FsopsIoError, match="allocation budget"):
        reuse_file_pinned_exclusive(source, target, max_copy_bytes=budget)
    assert not target.exists()
    assert source.stat().st_nlink == 1


def test_cross_mount_refusal_uses_budgeted_copy_fallback(tmp_path, monkeypatch):
    source = _immutable_file(tmp_path / "source")

    def cross_mount(*args, **kwargs):
        raise OSError(errno.EXDEV, "different bind mounts")

    monkeypatch.setattr(immutable_links.os, "link", cross_mount)
    with pytest.raises(fileops.FsopsIoError, match="allocation budget"):
        reuse_file_pinned_exclusive(source, tmp_path / "denied", max_copy_bytes=0)
    assert not (tmp_path / "denied").exists()
    target = tmp_path / "copied"
    _, size, shared = reuse_file_pinned_exclusive(source, target, max_copy_bytes=source.stat().st_size)
    assert not shared and size == target.stat().st_size
    assert not target.samefile(source) and target.read_bytes() == source.read_bytes()


def test_directory_swap_cannot_redirect_link_creation(tmp_path, monkeypatch):
    source = _immutable_file(tmp_path / "source")
    parent = tmp_path / "destination"
    parent.mkdir()
    moved = tmp_path / "original-destination"
    real_link = os.link

    def rename_then_link(*args, **kwargs):
        parent.rename(moved)
        parent.mkdir()
        real_link(*args, **kwargs)

    monkeypatch.setattr(immutable_links.os, "link", rename_then_link)
    with pytest.raises(fileops.FsopsModifiedError, match="directory path changed"):
        reuse_file_pinned_exclusive(source, parent / "target")
    assert list(parent.iterdir()) == []
    assert (moved / "target").samefile(source)


def test_copy_fallback_cannot_publish_after_target_parent_replacement(tmp_path, monkeypatch):
    source = _immutable_file(tmp_path / "source")
    parent = tmp_path / "destination"
    parent.mkdir()
    moved = tmp_path / "original-destination"
    real_open = os.open

    def cross_mount(*args, **kwargs):
        raise OSError(errno.EXDEV, "different bind mounts")

    def replace_before_create(path, flags, *args, **kwargs):
        if flags & os.O_CREAT and Path(path).name == "target":
            parent.rename(moved)
            parent.mkdir()
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(immutable_links.os, "link", cross_mount)
    monkeypatch.setattr(immutable_links.os, "open", replace_before_create)
    with pytest.raises(fileops.FsopsModifiedError, match="directory path changed"):
        reuse_file_pinned_exclusive(source, parent / "target", max_copy_bytes=source.stat().st_size)
    assert list(parent.iterdir()) == []


def test_replaced_temporary_target_cannot_be_accepted(tmp_path, monkeypatch):
    source = _immutable_file(tmp_path / "source")
    unrelated = _immutable_file(tmp_path / "unrelated", b"elsewhere")
    real_link = os.link

    def wrong_link(_source_name, destination, **kwargs):
        kwargs.pop("src_dir_fd")
        real_link(unrelated, destination, **kwargs)

    monkeypatch.setattr(immutable_links.os, "link", wrong_link)
    with pytest.raises(fileops.FsopsModifiedError):
        reuse_file_pinned_exclusive(source, tmp_path / "target")


@pytest.mark.parametrize("source_link", [False, True])
def test_linked_source_or_parent_is_refused(tmp_path, source_link):
    source = _immutable_file(tmp_path / "source")
    link = tmp_path / "linked"
    if source_link:
        link.symlink_to(source)
        source = link
        target = tmp_path / "target"
    else:
        real_parent = tmp_path / "real-parent"
        real_parent.mkdir()
        link.symlink_to(real_parent, target_is_directory=True)
        target = link / "target"
    with pytest.raises(fileops.FsopsError):
        reuse_file_pinned_exclusive(source, target)


def _harness(name):
    path = Path(__file__).resolve().parents[2] / "scripts" / "acceptance" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _accepted(tmp_path, monkeypatch):
    from comfy_omni.conversion.packaging.publication import publish_package

    assembly = _harness("native_package_assembly")
    verifier = _harness("verify_native_package_assembly")
    plan, sources = _plan(tmp_path)
    total = sum(path.stat().st_size for path in sources.values())
    monkeypatch.setattr(assembly, "TOTAL_BYTES", total)
    before = {relative: assembly._storage_identity(path) for relative, path in sources.items()}
    staged = materialize_package(plan, tmp_path / "package", reuse_immutable=True, max_copy_bytes=0)
    output = publish_package(plan, staged).output_dir
    root = tmp_path / "sources"
    evidence = assembly._storage_evidence(plan, root, output, before)
    result = {"storage": evidence, "total_bytes": total}
    verifier._verify_storage(result, output, root, set(sources))
    return verifier, result, output, root, sources


@pytest.mark.parametrize("change", ["copy", "chmod", "extra-link", "accounting", "wrong-root", "mapping"])
def test_independent_storage_verifier_rejects_changed_evidence_or_inodes(tmp_path, monkeypatch, change):
    verifier, result, output, root, sources = _accepted(tmp_path, monkeypatch)
    relative, source = next(iter(sources.items()))
    target = output / relative
    if change == "copy":
        payload = target.read_bytes()
        target.unlink()
        target.write_bytes(payload)
        target.chmod(0o444)
    elif change == "chmod":
        source.chmod(0o644)
    elif change == "extra-link":
        (tmp_path / "extra").hardlink_to(source)
    elif change == "accounting":
        result["storage"]["copied_bytes"] = 1
    elif change == "wrong-root":
        root = tmp_path
    else:
        result["storage"]["files"][0]["source_path"] = "../outside"
    with pytest.raises(RuntimeError, match="independent package verification failed"):
        verifier._verify_storage(result, output, root, set(sources))
