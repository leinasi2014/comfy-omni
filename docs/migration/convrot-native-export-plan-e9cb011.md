# ConvRot native-export planning design

Status: implementation slice for issue #8

Source: `h3-forge/main@e9cb011d00b028c149db3978de246c54f6e34acc`

Target: ComfyOmni `0.2.0a1`

This document freezes the authorization and planning boundary for the first native H3 checkpoint
export route. It deliberately does not authorize tensor decoding, output writes, publication, or a
claim that a converted checkpoint runs correctly. Those require later Docker-only numerical,
transactional, verification, and GPU acceptance slices.

## Audited source inventory

| Legacy source | Git blob | Migrated responsibility |
|---|---|---|
| `qkv.py` | `43f3d9e1d243b3d5aebaa6281b2c9b383970abfd` | Pure grouped ↔ runtime-QKV row permutations. |
| `native_export.py` | `475cee5523be64e5b24a95e16c5de3f371cbdf67` | Characterized action planning, shard assignment, legacy output schema, and future transaction behavior. The monolith is not copied. |
| `h3/profiles.py` | `b85a8b1cbf4a882474c83ac0f6f25a6a7434cd3e` | H3 shapes, profile identifiers, QKV geometry, and runtime quantization policy. |
| `h3/contracts/registry.py` | `edf6342233d86b112359abcd5e09e5c3b22028b3` | Source storage/output-route compatibility and the 60 ignored-layer entries. |

All four source files are Apache-2.0. Migrated modules retain source annotations. Existing wire
identifiers remain unchanged: output schema `h3-comfy-int8-export/v2`, profile
`dense-bf16-online-int8`, source layout `runtime-qkv`, and target layout
`grouped-for-official-loader`.

## Why the planner is a separate boundary

The old exporter resolved contracts, parsed sources, decided tensor actions, imported Torch,
decoded weights, wrote shards, created manifests, and published a directory in one module. The
ComfyOmni split is:

```text
contracts/conversion.py        immutable output and layout policy
domain/qkv.py                  pure row permutations
conversion/exporters/models.py frozen plan values
conversion/exporters/planning.py exact authorization and action planning
application/conversion.py      explicit catalog/template/census orchestration
```

The planning import graph points only toward artifacts, contracts, contract workflows, exporter
values, and domain rules. It does not import Torch, vLLM, runtime, application, CLI, API, or plugin
code. The base wheel therefore remains importable without an ML stack.

## Authorization sequence

Planning fails before payload mutation unless all of these statements are proven:

1. The requested output profile exists and supports the component and observed storage kind.
2. The source contract carries an exact schema SHA-256. A census-only contract is insufficient.
3. The contract and template name/component agree.
4. Observed tensor count, ConvRot group count, reported schema digest, and independently recomputed
   schema digest all equal the contract.
5. Exact L3 template validation passes, including group prefixes, shapes, group sizes, scale census,
   and a full non-quantized inventory when the template provides one.
6. Every source file has a positive size and lowercase SHA-256 binding.
7. An external contract carries both immutable snapshot bytes and a valid manifest digest.
8. Every QKV action has exactly 21,504 rows under the pinned 56-group, one-head-per-group,
   128-head-dimension layout.
9. Target names are unique, the output tensor census is exact, and no target tensor exceeds the
   requested shard envelope.

The plan itself is canonical JSON with content SHA-256. It binds the source files, contract origin,
contract schema, optional snapshot, template digest, QKV permutation digest, resource envelope,
every source action, shard membership, output census, and runtime quantization policy.

## Action model

Every source tensor receives exactly one operation:

| Source role | Planned operation | Output |
|---|---|---|
| ConvRot marker | `omit-comfy-quant-marker` | none |
| ConvRot row scale | `omit-source-rowwise-scale` | none |
| ConvRot weight | `inverse-convrot-to-bf16` | dense BF16 |
| ConvRot block QKV weight | `inverse-convrot-to-bf16-runtime-qkv-to-grouped` | dense BF16 in official loader disk order |
| Dense token-refiner QKV weight | `copy-runtime-qkv-to-grouped` | original dtype in official loader disk order |
| Other tensor | `copy-raw` | byte-preserving copy |

This route means offline inverse ConvRot to dense BF16 followed by required runtime rowwise INT8.
It is not a payload-preserving conversion, not a lossless claim, not serialized native INT8, and
not direct ConvRot loading.

## Real-asset authority decision

The currently authorized first conversion source on `srv-00` is the MiniMax-H3 × Z-Image Ref2VA
hot-switch candidate:

- file SHA-256: `71b8085ac4221ee036708c230a007d617dccca1b0028b95bb4ee106cb2a385c5`;
- exact source schema: `cc7976f678e6d4a567e718aca56c1db4aa91adfa27108db84066cce3213edf9d`;
- 932 tensors and 200 ConvRot groups;
- template: `h3-transformer-50l-convrot`.

The primary 10Eros checkpoint is not authorized by that record. A Docker-only header comparison
found two additional BF16 tensors and 112 shared dtype differences: 102 F16→BF16 and 10 F32→BF16.
Its exact schema is `ae2456bc6ac904929a4b773f703f8a1baa99b6356b5a389994faf64a1a2d80f2`.
It needs an independent complete 934-tensor contract and human review; inheriting the Ref2VA
contract would silently weaken the source authority.

The comparison evidence is retained on `srv-00` at
`/home/hyl/comfy-omni-acceptance/convrot-conversion-1a15bec/primary-vs-ref2va-header-delta.json`.
It was produced in a network-disabled container with the model root mounted read-only. This is
header evidence only, not conversion or inference acceptance.

## Remaining issue #8 slices

1. Implement the lazy-Torch numerical backend and characterize inverse ConvRot and row-chunk limits.
2. Implement exclusive staging, streaming shard writing, source re-verification, independent output
   verification, and manifest-last atomic publication.
3. Restore the legacy `export-native` CLI only when the complete transaction exists; do not expose a
   command that sounds like an export but only produces a plan.
4. Run an authorized full Ref2VA conversion in a resource-bounded Docker container on `srv-00`, then
   bind numerical/output evidence to the source, converter image, and Git commit.
5. Keep the primary 10Eros path fail-closed until its independent contract is reviewed and pinned.

The completed Docker evidence for this planning slice is indexed in
[`docs/evidence/convrot-plan-ed08abbe2df5.md`](../evidence/convrot-plan-ed08abbe2df5.md).
