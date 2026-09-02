# ConvRot bounded payload-producer design

Status: accepted implementation slice for issue
[#8](https://github.com/leinasi2014/comfy-omni/issues/8)

Legacy authority: `h3-forge@e9cb011d00b028c149db3978de246c54f6e34acc`

Target: ComfyOmni `0.2.0a1`

This slice connects every operation in an already authorized native-export plan to the immutable
transaction. It adds bounded QKV row copying, inverse-ConvRot-to-BF16 production, their combined
form, exact marker/scale omission, and external contract-snapshot carry. It does not expose the
`export-native` CLI or claim that a complete Ref2VA checkpoint can load or generate in a native
runtime.

## Audited source and ownership

| Legacy source | Git blob | ComfyOmni ownership |
| --- | --- | --- |
| `src/h3_forge/qkv.py` | `43f3d9e1d243b3d5aebaa6281b2c9b383970abfd` | Pure grouped ↔ runtime-QKV row-index rules remain in `domain/qkv.py`; scheduling and held byte reads are separate. |
| `src/h3_forge/convrot.py` | `8b4b9eebacd8bdaf64b251d5635b0147e7d790db` | The accepted numerical transform remains in `conversion/numerics`; exact byte serialization is a lazy Torch adapter. |
| `src/h3_forge/native_export.py` | `475cee5523be64e5b24a95e16c5de3f371cbdf67` | Operation dispatch, triplet binding, payload scheduling, and snapshot carry are cleanly separated from the legacy monolith. |

All audited legacy sources are Apache-2.0. Derived numerical and QKV modules carry source
annotations in their module headers. No legacy module is imported at runtime.

## Module boundaries

| Module | Responsibility |
| --- | --- |
| `artifacts/sources.py` | Read only an authorized byte range through the already held source descriptor. |
| `conversion/exporters/payloads.py` | Recompute the QKV permutation, schedule bounded contiguous row runs, and call an injected numerical block backend. |
| `conversion/numerics/serialization.py` | Lazily obtain Torch, decode one exact I8/F32 block, run inverse ConvRot, and emit little-endian BF16 bytes. |
| `conversion/exporters/execution.py` | Bind each operation to exact source descriptors, discover complete ConvRot triplets, carry an authorized snapshot, and dispatch producers into the accepted transaction. |

Artifact I/O and row scheduling do not import Torch. Torch is imported only after a caller executes
an inverse-ConvRot action. The base wheel therefore remains independent of an ML runtime.

## Immutable plan v2

`comfy_omni.native_export.plan/v2` adds `group_size` to every `TensorAction`. It is populated from
the independently discovered marker/weight/scale triplet and is required for ConvRot operations.
Copy operations carry `null`. The executor rejects older or unknown plan schemas; pre-release v1
plans must be regenerated and are retained only as historical evidence.

Execution revalidates all of the following before publication:

1. the canonical plan content digest and source/action/shard census;
2. complete QKV permutation length, digest, and bijection, but only when a QKV operation exists;
3. rank, dtype, byte length, row geometry, group prefix, group size, and operation for every action;
4. exact marker/weight/scale triplets rediscovered through held descriptors, including marker
   payload and descriptor revalidation;
5. bounded source ranges and an exact byte count from every producer;
6. external snapshot schema, manifest self-digest, contract name, schema digest, file digest, and
   plan bindings before the snapshot is copied under its fixed artifact name;
7. the existing independent shard verification, final source rehash, no-overwrite publication,
   and manifest-last protocol.

Any mismatch fails while output is private or before staging begins. Marker and scale omission is
accepted only as part of an exact rediscovered ConvRot group; arbitrary tensors cannot opt into an
omit operation.

## Bounded operation behavior

| Operation | Producer behavior |
| --- | --- |
| `copy-raw` | Stream the complete authorized source span unchanged. |
| `copy-runtime-qkv-to-grouped` | Read complete rows in digest-bound grouped order using contiguous runs no larger than `max_rows`. |
| `inverse-convrot-to-bf16` | Read at most `max_rows` of I8 weight and F32 scale, inverse-transform in FP32, then emit BF16. |
| `inverse-convrot-to-bf16-runtime-qkv-to-grouped` | Apply the same bounded numerical producer in digest-bound grouped row order. |
| marker and scale omission | Emit no target only after exact triplet discovery and action binding. |

The implementation intentionally prioritizes a small, auditable memory envelope over large
sequential reads. Non-contiguous QKV order may create multiple held range reads, but never expands
the numerical intermediate beyond the plan's `max_rows` value.

## Acceptance boundary

GitHub Docker lanes cover pure permutation, plan-schema, operation dispatch, snapshot carry and
tamper rejection, producer byte-count rejection, architecture imports, and transaction regressions.
The designated-server acceptance additionally built and installed the exact wheel offline, then
used actual Torch for all three materializing producer forms. The independent standard-library
verifier compared exact BF16 and reordered bytes.

That bounded synthetic fixture proves the producer boundary, not a 932-tensor conversion. Full
Ref2VA conversion, CLI exposure, native runtime loading, generation, LoRA activation, and A → B → A
model switching remain separate issue slices. Exact acceptance evidence is recorded in
[`docs/evidence/convrot-payload-producers-1b2324ada243.md`](../evidence/convrot-payload-producers-1b2324ada243.md).
