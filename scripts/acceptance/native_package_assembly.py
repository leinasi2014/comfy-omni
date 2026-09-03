#!/usr/bin/env python3
"""Assemble the fixed six-component native package in acceptance Docker."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

from comfy_omni.artifacts import fileops
from comfy_omni.artifacts.build_identity import installed_tool_identity
from comfy_omni.conversion.packaging.materialization import (
    PACKAGE_MATERIALIZATION_SCHEMA,
    materialize_package,
)
from comfy_omni.conversion.packaging.planning import (
    PACKAGE_COMPONENTS,
    PACKAGE_MANIFEST_NAME,
    PACKAGE_OUTPUT_SCHEMA,
    PACKAGE_PLAN_SCHEMA,
    PACKAGE_TASKS,
    PINNED_VLLM_OMNI_COMMIT,
    plan_native_package,
)
from comfy_omni.conversion.packaging.publication import PACKAGE_PUBLICATION_SCHEMA, publish_package
from comfy_omni.conversion.packaging.receipts import RECEIPT_SCHEMA, parse_component_receipt
from comfy_omni.conversion.packaging.verification import verify_package_sources

TOTAL_FILE_COUNT = 57
TOTAL_BYTES = 61_745_392_507

MODEL_INDEX_NAME = "model_index.json"

COMPONENT_CENSUS = {
    "audio_vae": (13, 605_286_955),
    "processor": (7, 11_498_352),
    "text_encoder": (2, 15_683_131_061),
    "tokenizer": (4, 11_492_078),
    "transformer": (14, 40_226_030_420),
    "video_vae": (17, 5_207_953_641),
}

SINGLE_PAYLOAD_SHA256 = {
    "audio_vae": (
        "minimax_h3_audio_vae_fp32.safetensors",
        "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
    ),
    "text_encoder": (
        "qwen3vl_32b_heretic_minimax_h3_nvfp4.strict.safetensors",
        "a166c7bbbe66a22065159e478335fee4a633c4a3e3bb34c8e8ac4cc91bf4996f",
    ),
    "video_vae": (
        "minimax_h3_video_vae_fp16.safetensors",
        "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522",
    ),
}

TRANSFORMER_CONFIG_SHA256 = {
    "config.patch.json": "3ec738237bcc7de59065ab538633c0494abe6775fa9c3d65f945dc28943d29c7",
    "export.plan.json": "07d8529e54a00bc6718f1bceda705b0119b30a43a44bbe731b58ffa2572eea9d",
    "manifest.json": "93b3fc29364dbca66570206ce5df72363d856630a17f58216f89b68ee50c7052",
    "model.safetensors.index.json": "4c790598b90ff246e185e5ff3900034cd8495a8303b48e9312b8a83f265684ca",
}

CONFIG_COMPONENT_FILES = {
    "processor": {
        "chat_template.json": (5499, "5c72a170d2a4a1a3bc5adad2e689ae28138a9700e5b8c96c0266331e86c0acce"),
        "merges.txt": (1671839, "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"),
        "preprocessor_config.json": (390, "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"),
        "tokenizer.json": (7032403, "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7"),
        "tokenizer_config.json": (11003, "a07e942ac874baa13758de8d1fbdb186683cc03416b5589e1b6671c6b3057c68"),
        "video_preprocessor_config.json": (385, "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"),
        "vocab.json": (2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
    },
    "tokenizer": {
        "merges.txt": (1671839, "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"),
        "tokenizer.json": (7032403, "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7"),
        "tokenizer_config.json": (11003, "a07e942ac874baa13758de8d1fbdb186683cc03416b5589e1b6671c6b3057c68"),
        "vocab.json": (2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
    },
}


def _fail(detail: str) -> None:
    raise RuntimeError(f"native package assembly acceptance failed: {detail}")


def _tool(expected_commit: str, expected_wheel_sha256: str):
    tool = installed_tool_identity()
    if tool.distribution != "comfy-omni" or tool.source_commit != expected_commit:
        _fail("installed wheel commit disagrees with command authority")
    if tool.wheel_sha256 != expected_wheel_sha256:
        _fail("installed wheel SHA256 disagrees with command authority")
    return tool


def _receipt(component: str, source: Path, tool):
    started = time.monotonic()
    receipt = parse_component_receipt(component, source, tool)
    if receipt.receipt_schema != RECEIPT_SCHEMA or receipt.component != component:
        _fail(f"receipt identity drifted for {component}")
    if receipt.source_dir != source.resolve(strict=True).as_posix():
        _fail(f"receipt source binding drifted for {component}")
    count, total = COMPONENT_CENSUS[component]
    if len(receipt.files) != count or sum(item.size for item in receipt.files) != total:
        _fail(f"receipt census drifted for {component}")
    by_path = {item.path: item for item in receipt.files}
    if component in SINGLE_PAYLOAD_SHA256:
        payload_name, payload_sha256 = SINGLE_PAYLOAD_SHA256[component]
        record = by_path.get(payload_name)
        if record is None or record.sha256 != payload_sha256:
            _fail(f"payload digest drifted for {component}")
    elif component in CONFIG_COMPONENT_FILES:
        for name, (size, sha256) in CONFIG_COMPONENT_FILES[component].items():
            record = by_path.get(name)
            if record is None or (record.size, record.sha256) != (size, sha256):
                _fail(f"official config digest drifted for {component}/{name}")
    elif component == "transformer":
        for name, sha256 in TRANSFORMER_CONFIG_SHA256.items():
            record = by_path.get(name)
            if record is None or record.sha256 != sha256:
                _fail(f"transformer export digest drifted for {name}")
    return receipt, {
        "component": component,
        "receipt_sha256": receipt.receipt_sha256,
        "file_count": len(receipt.files),
        "total_bytes": total,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--expect-wheel-sha256", required=True)
    parser.add_argument("--result-out", type=Path, required=True)
    args = parser.parse_args()

    root = args.components_root.resolve(strict=True)
    tool = _tool(args.expect_commit, args.expect_wheel_sha256)
    started = time.monotonic()

    parsed = [_receipt(component, root / component, tool) for component in PACKAGE_COMPONENTS]
    receipts = tuple(receipt for receipt, _ in parsed)
    receipt_evidence = [evidence for _, evidence in parsed]

    plan = plan_native_package(receipts, vllm_omni_commit=PINNED_VLLM_OMNI_COMMIT)
    if plan.schema != PACKAGE_PLAN_SCHEMA or plan.host_commit != PINNED_VLLM_OMNI_COMMIT:
        _fail("plan schema or host binding drifted")
    if plan.host_adapter != "vllm-omni" or plan.output_schema != PACKAGE_OUTPUT_SCHEMA:
        _fail("plan host adapter or output schema drifted")
    if plan.manifest_name != PACKAGE_MANIFEST_NAME or plan.serving_entrypoint != "Ref2VA/":
        _fail("plan manifest name or serving entrypoint drifted")
    if plan.resident_dit_count != 1 or tuple(plan.supported_tasks) != PACKAGE_TASKS:
        _fail("plan routing drifted")
    if len(plan.files) != TOTAL_FILE_COUNT or sum(item.size for item in plan.files) != TOTAL_BYTES:
        _fail("plan file census drifted")
    observed = hashlib.sha256(fileops.canonical_json(plan.to_dict(include_content_sha256=False))).hexdigest()
    if observed != plan.content_sha256:
        _fail("plan self-digest mismatch")

    verification = verify_package_sources(plan)
    if verification.to_dict()["status"] != "VERIFIED" or verification.file_count != TOTAL_FILE_COUNT:
        _fail("source verification drifted")
    if verification.total_bytes != TOTAL_BYTES or verification.plan_content_sha256 != plan.content_sha256:
        _fail("source verification totals drifted")

    materialization = materialize_package(plan, args.output)
    if materialization.schema != PACKAGE_MATERIALIZATION_SCHEMA:
        _fail("materialization schema drifted")
    if (materialization.file_count, materialization.total_bytes) != (TOTAL_FILE_COUNT, TOTAL_BYTES):
        _fail("materialization census drifted")
    if materialization.source_files_sha256 != verification.files_sha256:
        _fail("materialization source digest disagrees with verification")
    if (
        materialization.files_sha256
        != hashlib.sha256(
            fileops.canonical_json(
                [{"path": item.target_path, "sha256": item.sha256, "size": item.size} for item in plan.files]
            )
        ).hexdigest()
    ):
        _fail("staged census digest mismatch")
    if args.output.exists():
        _fail("output path became visible before publication")

    publication = publish_package(plan, materialization)
    if publication.schema != PACKAGE_PUBLICATION_SCHEMA or publication.file_count != TOTAL_FILE_COUNT:
        _fail("publication identity drifted")
    if publication.total_bytes != TOTAL_BYTES or publication.plan_content_sha256 != plan.content_sha256:
        _fail("publication totals drifted")
    manifest = fileops.parse_json_strict((args.output / PACKAGE_MANIFEST_NAME).read_bytes())
    if manifest["schema"] != PACKAGE_OUTPUT_SCHEMA or manifest["plan_content_sha256"] != plan.content_sha256:
        _fail("published manifest identity drifted")
    self_digest = hashlib.sha256(
        fileops.canonical_json({key: value for key, value in manifest.items() if key != "package_manifest_sha256"})
    ).hexdigest()
    if self_digest != manifest["package_manifest_sha256"] or self_digest != publication.manifest_sha256:
        _fail("published manifest self-digest mismatch")

    model_index_bytes = (args.output / MODEL_INDEX_NAME).read_bytes()
    model_index = fileops.parse_json_strict(model_index_bytes)
    if model_index["_class_name"] != "MiniMaxH3Pipeline":
        _fail("published model_index class drifted")
    if model_index_bytes != fileops.canonical_json(model_index):
        _fail("published model_index is not canonical")
    model_index_sha256 = hashlib.sha256(model_index_bytes).hexdigest()
    if model_index_sha256 != manifest["model_index_sha256"]:
        _fail("published model_index_sha256 disagrees with manifest")

    result = {
        "schema": "comfy-omni.e3.package-assembly/v1",
        "status": "ASSEMBLED_PUBLISHED",
        "tool": tool.to_dict(),
        "components": receipt_evidence,
        "plan_content_sha256": plan.content_sha256,
        "source_files_sha256": verification.files_sha256,
        "staged_files_sha256": materialization.files_sha256,
        "manifest_sha256": publication.manifest_sha256,
        "model_index_sha256": model_index_sha256,
        "file_count": TOTAL_FILE_COUNT,
        "total_bytes": TOTAL_BYTES,
        "output_dir": args.output.resolve(strict=True).as_posix(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    args.result_out.parent.mkdir(parents=True, exist_ok=True)
    args.result_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
