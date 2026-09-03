# ComfyOmni user guide

> Bring Comfy checkpoints to native Omni runtimes.

This guide describes what the current ComfyOmni pre-release (`0.2.0a1`, as recorded in
[`pyproject.toml`](../pyproject.toml)) ships today. It is scoped strictly to merged capabilities and to
the evidence recorded under `docs/`; it does not describe unfinished work. Every capability claim below
cites the document that evidences it.

<!-- guide-parity -->
> The English and Chinese user guides are a synchronized pair. They carry the **same** facts — names,
> paths, byte counts, and digests are identical, and only the prose language differs. When you change a
> fact in one file, update the other.

## What ComfyOmni is

ComfyOmni is an open-source bridge for inspecting, converting, packaging, and validating checkpoints
from the Comfy ecosystem for native Omni runtimes. Its tagline is "Bring Comfy checkpoints to native
Omni runtimes." The project keeps conversion offline and aims to produce immutable, verifiable runtime
packages instead of teaching inference workers to parse arbitrary Comfy checkpoints at startup. This
guide is the user-facing companion to the project
[README](../README.md); it focuses on the commands and Python APIs that exist in code today and on the
native package format those APIs produce.

ComfyOmni is an independent open-source project. It is not an official project of ComfyUI, Comfy.org,
vLLM, MiniMax, or their respective maintainers unless explicitly stated otherwise.

## Safety model

ComfyOmni treats checkpoint conversion and package assembly as offline, bounded, streaming, fail-closed
artifact operations.

- **Offline.** Inspection, normalization, conversion, and packaging do not contact a runtime inference
  host. Model work runs only inside the designated-server Docker container against read-only source
  mounts, per [`docs/development/docker-first.md`](../docs/development/docker-first.md).
- **Bounded.** Inspection reads safetensors headers only and never tensor payloads. Normalization
  streams at most 8 MiB per read. ConvRot producers cap the rows processed per intermediate block to the
  plan's `max_rows`. The native-package chain copies files in bounded 8 MiB chunks.
- **Streaming.** Large files are processed as bounded chunks through held, digest-bound descriptors
  rather than loaded into memory. The six-component 61,745,213,741-byte package is staged and published
  with a bounded memory envelope (the E3 run used a peak RSS of 52,367,360 bytes across that payload,
  per [`docs/evidence/native-package-assembly-b47d084.md`](../docs/evidence/native-package-assembly-b47d084.md)).
- **Fail-closed.** Every refusal aborts before publication or before weight loading. Sources are
  read-only: inspection and normalization never open a source checkpoint for writing, and server
  conversion mounts source trees read-only. Publication is staging-first and manifest-last: files are
  copied into a private staging tree, independently re-read and re-hashed, then released by a single
  same-parent atomic rename with `h3-comfy-package.json` written last as the only completion marker.
  Existing output paths are never overwritten; a manifest-less directory is by construction unfinished
  and is retained for diagnosis instead of recursive deletion.

## Installed capabilities today

| Capability | Status | Evidence |
|---|---|---|
| Header-only checkpoint inspection (`comfy-omni inspect`) | Shipped | [`docs/migration/checkpoint-inspection-e9cb011.md`](../docs/migration/checkpoint-inspection-e9cb011.md) |
| Digest-pinned text-encoder normalization (`comfy-omni normalize text-encoder`) | Shipped | [`docs/migration/text-encoder-normalization.md`](../docs/migration/text-encoder-normalization.md) |
| Immutable native-source contract workflow (`comfy-omni contract` scan / draft / pin / list) | Shipped | [`docs/migration/contract-workflows-e9cb011.md`](../docs/migration/contract-workflows-e9cb011.md) |
| ConvRot numerics and bounded conversion chain (inverse-ConvRot, QKV reorder, bounded payload producers, immutable transaction, manifest-last publication) | Shipped as an application-level chain; no `export-native` CLI | [`docs/migration/convrot-numerics-e9cb011.md`](../docs/migration/convrot-numerics-e9cb011.md) · [`docs/migration/convrot-payload-producers-e9cb011.md`](../docs/migration/convrot-payload-producers-e9cb011.md) · [`docs/migration/convrot-native-export-transaction-e9cb011.md`](../docs/migration/convrot-native-export-transaction-e9cb011.md) |
| Full Ref2VA native conversion (server-verified: 636.593 s; 932 actions, 532 output tensors, 10 shards; 40,225,668,192 tensor payload bytes) | Accepted real-model slice | [`docs/evidence/ref2va-full-conversion-25ceccdd5468.md`](../docs/evidence/ref2va-full-conversion-25ceccdd5468.md) |
| Immutable native-package chain (receipt → plan → verify → materialize → publish) | Shipped | [`docs/migration/component-receipt-parsing-e9cb011.md`](../docs/migration/component-receipt-parsing-e9cb011.md) · [`docs/migration/native-package-planning-e9cb011.md`](../docs/migration/native-package-planning-e9cb011.md) · [`docs/migration/native-package-source-verification-e9cb011.md`](../docs/migration/native-package-source-verification-e9cb011.md) · [`docs/migration/native-package-materialization-e9cb011.md`](../docs/migration/native-package-materialization-e9cb011.md) · [`docs/migration/native-package-publication-e9cb011.md`](../docs/migration/native-package-publication-e9cb011.md) |
| Six-component 61,745,213,741-byte native package, independently verified on `srv-00` | Verified | [`docs/evidence/native-package-assembly-b47d084.md`](../docs/evidence/native-package-assembly-b47d084.md) · [`docs/evidence/native-package-assembly-76e2ebb.md`](../docs/evidence/native-package-assembly-76e2ebb.md) |
| Single lazy idempotent vLLM-Omni bootstrap (`plugin:register`) | Shipped | [`docs/migration/vllm-omni-bootstrap-e9cb011.md`](../docs/migration/vllm-omni-bootstrap-e9cb011.md) |
| Fail-closed runtime package contract (`validate_runtime_package`) | Shipped | [`docs/migration/runtime-package-contract-e9cb011.md`](../docs/migration/runtime-package-contract-e9cb011.md) |

## CLI reference

The distribution exposes the `comfy-omni` CLI ([`pyproject.toml`](../pyproject.toml)
`[project.scripts]`). The commands below are the ones implemented in
[`src/comfy_omni/cli/`](../src/comfy_omni/cli/). These are container-internal commands; per the
Docker-first policy they are run inside the built image, never installed or executed on the host.

```text
comfy-omni --version
comfy-omni inspect CHECKPOINT.safetensors [--json]
comfy-omni normalize text-encoder SOURCE.safetensors DERIVED.safetensors [--json]
comfy-omni contract scan SOURCE.safetensors [--json]
comfy-omni contract draft SOURCE.safetensors -o DRAFT.json --generated-by OPERATOR [--json]
comfy-omni contract pin DRAFT.json --name PROFILE --reviewer REVIEWER \
  --evidence REVIEW.md [--contract-dir CONTRACTS] [--enforce-observed-schema] [--json]
comfy-omni contract list [--contract-dir CONTRACTS] [--json]
```

- `inspect` accepts one or more paths and emits header-only inspection. It rejects non-`.safetensors`
  files, enforces a 64 MiB header bound and 100,000 tensor bound, and refuses unindexed trailing bytes
  with exit code `2` and the stable `safetensors-unindexed-trailing-bytes` reason. It never reads tensor
  payloads and never imports Torch or vLLM.
- `normalize text-encoder` applies the single authorized normalization profile
  ([`docs/migration/text-encoder-normalization.md`](../docs/migration/text-encoder-normalization.md)).
  The source must exist and match the pinned byte count and SHA-256; the destination parent must already
  exist; the source and destination must differ; and neither the destination nor its sibling
  `DERIVED.safetensors.normalization.json` receipt may exist. It publishes the artifact and receipt
  through no-overwrite links.
- `contract scan` performs a read-only census and an exact three-level match. It returns exit code `0`
  only when exactly one L3 template matches, and `3` otherwise. It never materializes tensor payloads.
- `contract draft` writes one immutable pending-review draft binding source paths, sizes and SHA-256
  values, the census digest, template identity, and the installed generator identity. Draft files are
  created exclusively and never rewritten.
- `contract pin` reviews and publishes one immutable snapshot. It requires `--contract-dir` (or the
  legacy `H3_FORGE_CONTRACT_DIR` compatibility variable, read only by this CLI boundary), enforces
  generator/reviewer separation, and records the evidence file's byte digest rather than its path.
  `--enforce-observed-schema` freezes an otherwise census-only contract's observed schema.
- `contract list` lists compile-time and explicitly loaded snapshots. External contracts are visible only
  when a caller passes an explicit store.

Contract commands hash source files but do not import Torch or vLLM. General legacy conversion and
runtime commands remain unavailable.

## The native package format

A published native package is an immutable directory tree that conforms to the `h3-comfy-package/v3`
output schema (see [`src/comfy_omni/conversion/packaging/planning.py`](../src/comfy_omni/conversion/packaging/planning.py)).
It contains exactly six component directories placed canonically under `Ref2VA/`:

| Component | Placement | Typical contents |
|---|---|---|
| transformer | `Ref2VA/transformer/` | Native 10-shard DiT checkpoint (10 `model-00000N-of-00010.safetensors` files plus `model.safetensors.index.json`, `config.patch.json`, `export.plan.json`, `manifest.json`) |
| text_encoder | `Ref2VA/text_encoder/` | The strict digest-pinned normalization derivative of the text encoder |
| video_vae | `Ref2VA/video_vae/` | Video VAE weights |
| audio_vae | `Ref2VA/audio_vae/` | Audio VAE weights |
| tokenizer | `Ref2VA/tokenizer/` | Official tokenizer files, fetched at assembly time |
| processor | `Ref2VA/processor/` | Official processor files, fetched at assembly time |

The package root carries two generated files:

- **`model_index.json`** — the host-discovery index. It is written after the independent staged re-read
  and before the manifest. It carries `_class_name: "MiniMaxH3Pipeline"`, `_diffusers_version: "0.32.2"`,
  a `_minimax_h3` routing block (`partition: "ref2va"`, `sigma_shift_scales: {"audio": 3.0, "video":
  12.0}`, `schema_version: 1`, and the hybrid `tasks` list `ref2va|t2va|fl2va`), and the component
  classifier pairs (`transformer` → `("diffusers", "MiniMaxH3DiTModel")`, `text_encoder` →
  `("transformers", "MiniMaxH3Qwen3VLHFEncoder")`, `video_vae` → `("diffusers", "MiniMaxH3VideoVAE")`,
  `audio_vae` → `("diffusers", "MiniMaxH3AudioVAE")`, `tokenizer` → `("transformers",
  "Qwen2TokenizerFast")`, `processor` → `("transformers", "Qwen3VLProcessor")`, `scheduler` → `null`).
- **`h3-comfy-package.json`** — the package manifest. Its `package_manifest_sha256` field is the
  SHA-256 of the same document excluding exactly that field (the self-digest). It binds
  `schema` (`h3-comfy-package/v3`), `plan_content_sha256`, `tool`, `host` (`adapter: "vllm-omni"`,
  `commit`), the `components` array, `source_files_sha256`, `staged_files_sha256`,
  `model_index_sha256`, the `files` census (`path`/`sha256`/`size`), `file_count`, `total_bytes`, and a
  `routing` block (`manifest`, `serving_entrypoint: "Ref2VA/"`, `resident_dit_count: 1`, and
  `supported_tasks`).

The manifest is written last through an exclusive read-only create with fsync. A package directory
without its `h3-comfy-package.json` is by construction an unfinished publication.

The pinned host resolves a model directory to a pipeline class only through the root `model_index.json`
`_class_name` field (see
[`docs/migration/native-package-publication-e9cb011.md`](../docs/migration/native-package-publication-e9cb011.md)).

### What a consumer must verify before load

Before a runtime loads weights, a consumer must validate the package root with
`comfy_omni.integrations.vllm_omni.package_contract.validate_runtime_package`. The validator is
host-free and refuses fail-closed in this order: `package-binding` → `model-index` → `manifest` →
`routing` → `tree-census` → `file-verification` → `components`. It re-derives the manifest self-digest,
recomputes the `model_index_sha256` binding, confirms the model-index and manifest routing agree, censuses
the tree exactly (refusing links and special entries), re-hashes every declared file against the
manifest, and confirms all six components are present. Any refusal aborts the pipeline before weight
loading, per
[`docs/migration/runtime-package-contract-e9cb011.md`](../docs/migration/runtime-package-contract-e9cb011.md).

## Python API quick-start

The offline package chain lives in [`comfy_omni.conversion.packaging`](../src/comfy_omni/conversion/packaging/)
and is exposed from the package `__init__`. The canonical intent of each step:

```python
from pathlib import Path

from comfy_omni.artifacts.build_identity import installed_tool_identity
from comfy_omni.conversion.packaging import (
    parse_component_receipt,
    plan_native_package,
    verify_package_sources,
    materialize_package,
    publish_package,
)

tool = installed_tool_identity()                 # must report distribution "comfy-omni"

receipts = tuple(
    parse_component_receipt(component, source_dir, tool)
    for component, source_dir in [
        ("transformer", "/components/transformer"),
        ("text_encoder", "/components/text_encoder"),
        ("video_vae", "/components/video_vae"),
        ("audio_vae", "/components/audio_vae"),
        ("tokenizer", "/components/tokenizer"),
        ("processor", "/components/processor"),
    ]
)                                                # six receipts, exact tree, no writes

plan = plan_native_package(
    receipts,
    vllm_omni_commit="17285c2f55a41bf15772676121814d59a60ace35",
)                                                # AUTHORIZED_PLAN, canonical content SHA-256

verified = verify_package_sources(plan)          # re-hash every source tree, no writes
materialized = materialize_package(plan, Path("/out/native-package"))   # private staging only
publication = publish_package(plan, materialized)                       # manifest-last atomic publish
```

- `parse_component_receipt(component, source_dir, tool)` returns an immutable `ComponentReceipt` after a
  deterministic tree census and a both-before-and-after pinned hash of every file. It never writes.
- `plan_native_package(receipts, *, vllm_omni_commit)` returns one canonical `NativePackagePlan`. It
  requires exactly six component roles, one identical producer tool identity across all components, the
  fixed `vllm_omni` host commit `17285c2f55a41bf15772676121814d59a60ace35`, and canonical `Ref2VA/`
  component placement. It does not read or write files.
- `verify_package_sources(plan)` reconstructs the plan, re-censuses every source tree, and re-hashes
  every planned file against the plan.
- `materialize_package(plan, output_dir)` copies every planned file into a private sibling staging tree
  in bounded chunks, refuses existing or overlapping output paths, and returns a `STAGED_VERIFIED` handle.
  It publishes nothing.
- `publish_package(plan, materialization)` first re-validates the staging identity and tree, writes
  `model_index.json` and then the manifest last, and releases the tree with one same-parent `os.rename`.
  Existing output paths are never overwritten.

### Consumer-side validation

```python
from comfy_omni.integrations.vllm_omni.package_contract import validate_runtime_package

contract = validate_runtime_package("/data/models/comfy-omni/native-package")
print(contract.to_dict())  # status "RUNTIME_VERIFIED"
```

`validate_runtime_package(package_root, *, expected_class_name="MiniMaxH3Pipeline")` returns a frozen
`RuntimePackageContract` whose `to_dict()` reports `status: "RUNTIME_VERIFIED"`. It is the documented,
host-free entry point for a consumer before any weight load.

## The vLLM-Omni plugin

The distribution registers one `vllm_omni.general_plugins` entry point
([`pyproject.toml`](../pyproject.toml)) that resolves to `comfy_omni.plugin:register`, a thin shim over
[`comfy_omni.integrations.vllm_omni.bootstrap.register`](../src/comfy_omni/integrations/vllm_omni/bootstrap.py).

`register()` does the following, per
[`docs/migration/vllm-omni-bootstrap-e9cb011.md`](../docs/migration/vllm-omni-bootstrap-e9cb011.md):

- **Observes the host, never forces it.** Architecture registration runs only when `vllm_omni` is
  already resident in `sys.modules`, and the registry submodule is resolved from `sys.modules` first,
  falling back to a guarded import. A missing, or resident-but-partial, host defers silently — no
  exception, no latch — so a later `register()` call retries.
- **Registers declarative lazy strings.** It contributes the wire-compatible architecture keys
  `MiniMaxH3Pipeline` and `MiniMaxH3DensePipeline` with fully-qualified module/class names and
  `get_minimax_h3_post_process_func`. Importing `register()` imports no pipeline module; the host
  resolves them at model-load time.
- **Latches exactly once per process.** Registration is guarded by a thread-safe `NEW → REGISTERING →
  REGISTERED` state machine, resets to `NEW` on failure, and is safe under the host's every-process
  loading shape (process0, engine cores, workers).

It does **not** register any REST/API-server hook, route, model, or runtime service today; the
`_is_root_process` helper is a documented hook for future API-server-only wiring and arms nothing.

## Current limitations

The following are explicit, current limitations. They are not shipped and must not be assumed from the
capabilities above.

- **No native host load or generation is shipped yet.** The runtime package contract
  (`validate_runtime_package`) and the `H3ComfyMiniMaxH3Pipeline` subclass exist, but real host loading
  and minimal generation are still in flight (E4-S3, per
  [`docs/migration/runtime-package-contract-e9cb011.md`](../docs/migration/runtime-package-contract-e9cb011.md)).
- **No LoRA lifecycle.** LoRA conversion, preflight, activation, and deactivation are not delivered
  (issue #12).
- **No hot switching.** Full-DiT `A → B → A` hot switching in one host process is not delivered
  (issue #13).
- **No `export-native` CLI and no broad compatibility claim.** The complete Ref2VA conversion is
  accepted, but the legacy `export-native` command is intentionally not exposed, and the offline
  artifact operations are not a production runtime or a broad compatibility claim
  ([`docs/migration/convrot-native-export-plan-e9cb011.md`](../docs/migration/convrot-native-export-plan-e9cb011.md)).
- **External assets are not redistributed.** The tokenizer and processor component configs are external
  official assets fetched at assembly time, not redistributed by this repository
  ([`docs/evidence/native-package-assembly-b47d084.md`](../docs/evidence/native-package-assembly-b47d084.md)).
  Model weights are never committed to Git and never included in the wheel.

## License and provenance policy

ComfyOmni is licensed under the [Apache License 2.0](../LICENSE). Migrated modules are derivatives of the
Apache-2.0 `h3-forge` project and carry blob-exact provenance in `docs/migration/`: each migration record
names the legacy repository, the immutable source commit, the production source blob, the source license,
and the attribution. Third-party code, fixtures, and assets remain subject to their recorded attribution
and compatible license terms. Model payloads, generated packages, server evidence, and untracked legacy
files are never distributed by this repository.
