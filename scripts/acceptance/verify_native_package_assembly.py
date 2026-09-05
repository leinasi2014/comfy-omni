#!/usr/bin/env python3
"""Independently re-read a published native package assembly."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path, PurePosixPath

from comfy_omni.artifacts import fileops

MANIFEST_NAME = "h3-comfy-package.json"
MODEL_INDEX_NAME = "model_index.json"


def _fail(detail: str) -> None:
    raise RuntimeError(f"independent package verification failed: {detail}")


def _tree_files(root: Path) -> set[str]:
    result: set[str] = set()

    def visit(directory: Path, prefix: str) -> None:
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda item: item.name)
        for entry in entries:
            relative = f"{prefix}/{entry.name}" if prefix else entry.name
            path = Path(entry.path)
            if fileops.is_link(path):
                _fail(f"published package contains a link: {relative}")
            if entry.is_dir(follow_symlinks=False):
                visit(path, relative)
            elif entry.is_file(follow_symlinks=False):
                result.add(relative)
            else:
                _fail(f"published package contains a special entry: {relative}")

    visit(root, "")
    return result


def _verify_storage(result: dict, package: Path, source_root: Path | None, expected_files: set[str]) -> None:
    storage = result.get("storage")
    if storage is None:
        if source_root is not None:
            _fail("source storage authority supplied without an immutable reuse receipt")
        return
    if source_root is None or storage.get("mode") != "immutable-reuse/v1":
        _fail("immutable reuse requires an independently supplied source root")
    source_root = fileops.reject_linked_ancestors(source_root).resolve(strict=True)
    if source_root.as_posix() != storage["source_root"]:
        _fail("storage source root disagrees with command authority")
    if (storage["shared_bytes"], storage["copied_bytes"], storage["max_copy_bytes"]) != (result["total_bytes"], 0, 0):
        _fail("storage byte accounting drifted")
    records = storage["files"]
    if len(records) != len(expected_files) or {item["target_path"] for item in records} != expected_files:
        _fail("storage receipt file census drifted")
    increments = Counter((entry["before"]["dev"], entry["before"]["ino"]) for entry in records)
    for item in records:
        for key in ("source_path", "target_path"):
            relative = PurePosixPath(item[key])
            if relative.is_absolute() or ".." in relative.parts or "\\" in item[key]:
                _fail("storage path is not contained")
        if item["target_path"] != f"Ref2VA/{item['source_path']}":
            _fail("storage source and package destination mapping drifted")
        before, after = item["before"], item["after"]
        if before["mode"] & 0o222 or any(
            before[key] != after[key] for key in ("dev", "ino", "size", "mtime_ns", "mode", "uid", "gid")
        ):
            _fail("source immutable metadata drifted")
        if after["nlink"] != before["nlink"] + increments[(before["dev"], before["ino"])]:
            _fail("source link increment disagrees with planned file census")
        for path in (source_root / item["source_path"], package / item["target_path"]):
            status = fileops.reject_linked_ancestors(path).lstat()
            if any(getattr(status, f"st_{key}") != value for key, value in after.items()):
                _fail(f"storage inode or metadata changed: {item['target_path']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--verdict-out", type=Path, required=True)
    parser.add_argument("--components-root", type=Path, help="Independent source-root authority for immutable reuse")
    args = parser.parse_args()

    started = time.monotonic()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    if result.get("schema") != "comfy-omni.e3.package-assembly/v1" or result.get("status") != "ASSEMBLED_PUBLISHED":
        _fail("assembly result identity drifted")
    package = args.package.resolve(strict=True)
    manifest_path = package / MANIFEST_NAME
    manifest = fileops.parse_json_strict(manifest_path.read_bytes())

    expected_files = {item["path"] for item in manifest["files"]}
    _verify_storage(result, package, args.components_root, expected_files)
    observed_files = _tree_files(package)
    if observed_files != expected_files | {MANIFEST_NAME, MODEL_INDEX_NAME}:
        missing = sorted(expected_files - observed_files)
        unexpected = sorted(observed_files - expected_files - {MANIFEST_NAME, MODEL_INDEX_NAME})
        _fail(f"published census drifted: missing={missing} unexpected={unexpected}")

    total_bytes = 0
    for record in manifest["files"]:
        digest, size = fileops.sha256_file_pinned(package / record["path"])
        if (digest, size) != (record["sha256"], record["size"]):
            _fail(f"published file digest drifted: {record['path']}")
        total_bytes += size
    if manifest["file_count"] != len(manifest["files"]) or manifest["total_bytes"] != total_bytes:
        _fail("published manifest totals drifted")
    if (manifest["file_count"], manifest["total_bytes"]) != (result["file_count"], result["total_bytes"]):
        _fail("published totals disagree with the assembly result")
    if manifest["plan_content_sha256"] != result["plan_content_sha256"]:
        _fail("published plan digest disagrees with the assembly result")

    self_digest = hashlib.sha256(
        fileops.canonical_json({key: value for key, value in manifest.items() if key != "package_manifest_sha256"})
    ).hexdigest()
    if self_digest != manifest["package_manifest_sha256"] or self_digest != result["manifest_sha256"]:
        _fail("published manifest self-digest mismatch")

    model_index_bytes = (package / MODEL_INDEX_NAME).read_bytes()
    model_index = fileops.parse_json_strict(model_index_bytes)
    if model_index["_class_name"] != "MiniMaxH3Pipeline":
        _fail("published model_index class drifted")
    model_index_sha256 = hashlib.sha256(model_index_bytes).hexdigest()
    if model_index_sha256 != manifest["model_index_sha256"]:
        _fail("published model_index_sha256 disagrees with manifest")
    if model_index_bytes != fileops.canonical_json(model_index):
        _fail("published model_index is not canonical")
    _verify_storage(result, package, args.components_root, expected_files)

    verdict = {
        "schema": "comfy-omni.e3.package-assembly-verdict/v1",
        "status": "VERIFIED",
        "package_dir": package.as_posix(),
        "plan_content_sha256": manifest["plan_content_sha256"],
        "manifest_sha256": self_digest,
        "model_index_sha256": model_index_sha256,
        "file_count": len(manifest["files"]),
        "total_bytes": total_bytes,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    if "storage" in result:
        verdict["storage"] = {key: result["storage"][key] for key in ("mode", "shared_bytes", "copied_bytes")}
    args.verdict_out.parent.mkdir(parents=True, exist_ok=True)
    args.verdict_out.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(verdict, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
