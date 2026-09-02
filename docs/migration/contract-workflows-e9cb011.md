# Contract workflow migration ledger

Status: frozen for issue #7  
Source: `h3-forge/main@e9cb011d00b028c149db3978de246c54f6e34acc`  
Target: ComfyOmni `0.2.0a1`

This ledger is the auditable boundary for migrating native-source contract discovery, drafting,
review, publication, loading, and explicit use. It is not permission to copy the old package as a
whole. Only the source blobs and behavior named below are in scope.

## Source inventory

| Legacy source | Git blob | Decision |
|---|---|---|
| `h3/contracts/model.py` | `7477343d99950e0682aa74fb045b587555a82fa1` | Reshape into pure ComfyOmni contract values. |
| `h3/contracts/templates.py` | `443a5cc9ca58891c3852079c8589fbe2f5af6484` | Preserve the four generated templates and pinned digests. |
| `h3/contracts/registry.py` | `edf6342233d86b112359abcd5e09e5c3b22028b3` | Preserve compile-time profiles; replace global activation with an explicit catalog value. |
| `h3/contracts/snapshots.py` | `b0a438df386278a1b0e7dcf783ca182748ec77ea` | Preserve strict validation and immutable publication; remove process-global state. |
| `contract_auto/census.py` | `41556958384d343b4a3a1baa9a4ff7e19020f0fb` | Preserve strict single/sharded/index-constrained census behavior. |
| `contract_auto/matcher.py` | `521bbd63051a2a617ddb2385751f9458bc024625` | Preserve evidence levels; L1 remains advisory and cannot authorize a source. |
| `contract_auto/generator.py` | `e57a4235ff6518b3d5148cf7df8616d3a3dd98a2` | Preserve observed/enforced separation and immutable draft evidence. |
| `contract_auto/pin.py` | `a24474938c0c1558a58ee2cd820f82b0b344c622` | Preserve the reviewer-controlled, fail-closed binding chain. |
| `fsops.py` | `ae40e46eef808f979ee085e806f2380e50b6c01d` | Reuse the strict JSON, pinned read, link rejection, hashing, and exclusive-write leaf. |
| `convrot.py` | `8b4b9eebacd8bdaf64b251d5635b0147e7d790db` | Retain marker/triplet validation only; retire Torch/runtime tensor operations from this slice. |
| `native_export.py` | `475cee5523be64e5b24a95e16c5de3f371cbdf67` | Reimplement only held-descriptor source enumeration and legacy-compatible schema hashing. |

All listed source files are Apache-2.0. Their origin remains visible in this ledger and in migrated
module headers. New code uses ComfyOmni names; persisted legacy schema identifiers remain stable as
described below.

The four template tables were mechanically exported by
`scripts/migration/export_h3_contract_templates.py`. The canonical package resource is 90,970 bytes
with SHA-256 `294d8cf5b790d7de42b91c385a72030dcbd318eddd44e6bd72d6f5b886c6125d`;
the loader separately re-derives and checks each legacy template digest.

## Behavior retained

1. Inputs are one `.safetensors` file, an explicit shard list, or a directory constrained by a
   strict `model.safetensors.index.json`.
2. Every JSON authority rejects duplicate keys and non-standard constants, and serialized evidence
   uses sorted compact UTF-8 JSON with one trailing newline.
3. L1 family routing and L2 classification are advisory. Only one exact L3 template match can
   authorize drafting.
4. ConvRot marker values are read before ordinary tensor payload access. Only markers that explicitly
   declare `format=int8_tensorwise` and `convrot=true` enter ConvRot discovery. Other valid declarations
   fail closed with a bounded format census; marker-free sources with quantization scales or `I8`
   tensors also fail closed.
5. Drafts bind source paths, sizes and SHA-256 values, the census digest, template name/version/digest,
   and the installed generator identity. Draft files are created exclusively and never rewritten.
6. Pinning re-reads canonical draft bytes, re-hashes every source through a held-descriptor protocol,
   rejects stale templates and sources, enforces generator/reviewer separation, and records the
   evidence file's byte digest rather than its path.
7. A human reviewer chooses whether an otherwise census-only contract freezes the observed schema.
   Existing enforced schema pins are inherited and cannot be weakened.
8. Snapshots are named by their manifest SHA-256, published with `O_EXCL`, flushed, re-opened, and
   verified before they become loadable.
9. Compile-time and external names cannot shadow each other. Duplicate external names fail closed.
10. The dense BF16 exception requires zero ConvRot groups and a complete exact tensor manifest.

## Compatibility decisions

The following identifiers are on-disk wire contracts and stay unchanged in this migration:

- `h3_forge.contract_auto.census/v1`
- `h3_forge.contract_auto.scan/v1`
- `h3_forge.contract_auto.draft/v1`
- `h3_forge.contract.snapshot/v1`
- `h3_forge.contract/v1`
- `h3_forge.contract.pin/v1`

Keeping these strings does not create a Python dependency on `h3_forge`. A future schema rename
requires an ADR, an explicit version transition, and compatibility fixtures.

`H3_FORGE_CONTRACT_DIR` remains a CLI compatibility input only. Core, contract, artifact, conversion,
and runtime modules never read it. The preferred ComfyOmni interface is the explicit `--contract-dir`
argument; a future `COMFY_OMNI_CONTRACT_DIR` alias requires a separate compatibility decision.

## Behavior retired

- The module-level `_ACTIVE_EXTERNAL` registry and activate/deactivate mutation API.
- Import-time or runtime discovery of contract stores from environment variables.
- Making external snapshots visible to the plugin or conversion runtime without an explicit catalog.
- Treating weak Oracle family guesses or advisory classifiers as authorization.
- Copying unrelated h3-forge Oracle, export, plugin, server, or runtime modules into this slice.
- Mutable registry files, overwrite-in-place publication, and path-only evidence references.

## Server-discovered boundary clarification

The pinned NVFP4 text encoder on `srv-00` contains 350 `nvfp4` markers plus one non-ConvRot
`int8_tensorwise` embedding marker. It is a runtime validation asset, not one of this slice's strict
ConvRot or marker-free BF16 source formats. The scanner therefore rejects it explicitly with
`unsupported-comfy-quant-storage`; this does not claim that the later runtime loader supports or
rejects NVFP4 execution. Runtime loading remains a separate GPU acceptance gate.

## Target ownership and dependency direction

```text
contracts  <--- artifacts
    ^              ^
    |              |
conversion/contract_workflows
             ^
             |
        application <--- cli
```

- `contracts/`: immutable values, generated template facts, compile-time profiles, and explicit
  `ContractCatalog`. It imports only the standard library.
- `artifacts/`: strict JSON, pinned file reads, safetensors source sets, snapshot validation and
  exclusive publication. It may depend on `contracts` but not on workflows or presentation layers.
- `conversion/contract_workflows/`: census, matching, drafting, and pin orchestration. It receives
  templates/catalogs explicitly.
- `application/`: use cases that load an external store into a new immutable catalog and pass it to
  one operation. No process-global activation exists.
- `cli/commands/contract.py`: argument and output translation only.

The plugin entry point remains outside this path and receives no environment-derived external
contracts.

## Acceptance map

| Risk | Required proof |
|---|---|
| Duplicate JSON keys or non-canonical rewrites | Strict parser and byte-identity tests. |
| Template/schema drift | Digest, version, census, and enforced-schema rejection tests. |
| Stale or swapped source | Size/SHA and held-descriptor TOCTOU tests. |
| Generator self-approval | Reviewer-separation test across every generator identity field. |
| Snapshot tamper or filename mismatch | Manifest and content-address filename tests. |
| Registry shadowing | Compile-time/external and external/external collision tests. |
| Hidden dependency or import cycle | Architecture import tests and installed-wheel smoke test. |
| Accidental Torch/vLLM dependency | CLI tests in a clean base installation with no optional extras. |

Unit and contract gates run in GitHub CI. Real checkpoint acceptance runs on the designated server;
local execution is not treated as release acceptance.
