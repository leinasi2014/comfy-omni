"""Independent standard-library verifier for native-export transaction evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import struct
from pathlib import Path
from typing import Any


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")


def _strict_json(path: Path) -> Any:
    return _strict_json_bytes(path.read_bytes())


def _strict_json_bytes(raw: bytes) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard constant {value!r}")

    return json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)


def _sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _require_regular(path: Path) -> None:
    status = path.lstat()
    assert stat.S_ISREG(status.st_mode) and not stat.S_ISLNK(status.st_mode), path


def _verify_shard(path: Path) -> None:
    raw = path.read_bytes()
    assert len(raw) >= 8
    (header_length,) = struct.unpack("<Q", raw[:8])
    assert 0 < header_length <= len(raw) - 8
    header = _strict_json_bytes(raw[8 : 8 + header_length])
    assert header == {
        "alpha": {"data_offsets": [0, 4], "dtype": "BF16", "shape": [2]},
        "beta": {"data_offsets": [4, 8], "dtype": "F32", "shape": [1]},
    }
    payload = raw[8 + header_length :]
    assert payload == b"\x01\x02\x03\x04\x05\x06\x07\x08"


def _verify(source: Path, output: Path, result_path: Path, expected_commit: str) -> dict[str, object]:
    assert output.is_dir() and not output.is_symlink()
    expected_names = {
        "config.patch.json",
        "export.plan.json",
        "manifest.json",
        "model-00001-of-00001.safetensors",
        "model.safetensors.index.json",
    }
    assert {item.name for item in output.iterdir()} == expected_names
    for item in output.iterdir():
        _require_regular(item)
    manifest = _strict_json(output / "manifest.json")
    assert manifest["status"] == "COMMITTED"
    claimed_manifest = manifest.pop("manifest_sha256")
    assert hashlib.sha256(_canonical(manifest)).hexdigest() == claimed_manifest
    records = {item["name"]: item for item in manifest["files"]}
    assert set(records) == expected_names - {"manifest.json"}
    for name, record in records.items():
        assert _sha256(output / name) == (record["sha256"], record["size"])
    source_sha256, source_size = _sha256(source)
    assert manifest["source_files"] == [{"path": str(source), "sha256": source_sha256, "size": source_size}]
    plan = _strict_json(output / "export.plan.json")
    claimed_plan = plan.pop("content_sha256")
    assert hashlib.sha256(_canonical(plan)).hexdigest() == claimed_plan
    assert claimed_plan == manifest["plan_content_sha256"]
    index = _strict_json(output / "model.safetensors.index.json")
    assert index == {
        "metadata": {"total_size": 8},
        "weight_map": {
            "alpha": "model-00001-of-00001.safetensors",
            "beta": "model-00001-of-00001.safetensors",
        },
    }
    config = _strict_json(output / "config.patch.json")
    assert config["_comfy_omni"]["plan_content_sha256"] == claimed_plan
    _verify_shard(output / "model-00001-of-00001.safetensors")
    result = _strict_json(result_path)
    assert result["status"] == "PASSED"
    assert result["manifest_sha256"] == claimed_manifest
    assert result["tool"]["source_commit"] == expected_commit
    assert result["source"] == {"path": str(source), "sha256": source_sha256, "size": source_size}
    return {
        "manifest_sha256": claimed_manifest,
        "plan_content_sha256": claimed_plan,
        "schema": "comfy_omni.acceptance.native-export-transaction-verification/v1",
        "source_sha256": source_sha256,
        "status": "VERIFIED",
        "tool_wheel_sha256": result["tool"]["wheel_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--verification", type=Path, required=True)
    arguments = parser.parse_args()
    verified = _verify(arguments.source, arguments.output, arguments.result, arguments.expected_commit)
    with arguments.verification.open("xb") as stream:
        stream.write(_canonical(verified))
    print(json.dumps(verified, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
