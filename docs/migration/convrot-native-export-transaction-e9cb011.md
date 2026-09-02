# ConvRot native-export transaction design

Status: accepted historical copy-only transaction slice for issue #8; extended by the bounded
producer slice at `1b2324ada243`

Legacy authority: `h3-forge@e9cb011d00b028c149db3978de246c54f6e34acc`

This slice established the filesystem and receipt transaction used by ConvRot conversion. At this
historical candidate the executor supported only `copy-raw`; every numerical, QKV, marker, or scale
operation failed closed. The later bounded producer design preserves this transaction and adds
separately tested producers; this document remains the authority for transaction semantics.

## Audited source and module split

| Legacy source | Git blob | ComfyOmni ownership |
| --- | --- | --- |
| `src/h3_forge/native_export.py` | `475cee5523be64e5b24a95e16c5de3f371cbdf67` | Behavior was separated across held sources, generic safetensors writing, execution, and publication. |
| `src/h3_forge/fsops.py` | `ae40e46eef808f979ee085e806f2380e50b6c01d` | Existing zero-dependency file primitives provide pinned reads, canonical JSON, exclusive writes, and directory durability. |
| `src/h3_forge/converter/package_v6.py` | `d36b20a1feb2064082577e48dbc2d863190a9686` | Manifest-last and no-overwrite package behavior informed the new publication boundary. |

All audited legacy files are Apache-2.0. The old exporter monolith was not copied. Responsibility
is now explicit:

| Module | Responsibility |
| --- | --- |
| `artifacts/sources.py` | Hold source descriptors, parse and hash through those descriptors, stream bounded tensor ranges, then rehash the same descriptors before publication. |
| `artifacts/safetensors_writer.py` | Deterministic header construction, exclusive streaming writes, exact producer byte counts, durable flush, strict independent reopen, descriptor comparison, and SHA-256 comparison. |
| `conversion/exporters/execution.py` | Recompute plan content SHA-256 before using paths, bind source/action/shard coverage, produce copy-only shards and deterministic sidecars, then request publication. |
| `conversion/packaging/native_export.py` | Claim a fresh sibling staging directory and fresh output directory, hard-link verified artifacts without overwrite, rehash published files, and write `manifest.json` last. |

These modules import no Torch, vLLM, API, CLI, runtime, or integration surface. Base-wheel import
behavior is unchanged.

## Commit protocol

```text
plan content SHA-256
  -> source descriptors opened and hash-bound
  -> exact tensor/action/shard coverage
  -> exclusive private shard write
  -> independent strict descriptor + SHA-256 verification
  -> canonical index, plan, and config sidecars
  -> final same-descriptor source rehash
  -> fresh output directory claim
  -> exclusive hard-link publication and target rehash
  -> canonical manifest.json written last
```

Readers must treat `manifest.json` as the only completion marker. A private staging directory or
claimed output directory without that file is incomplete evidence, not a native checkpoint. The
implementation deliberately does not recursively delete ambiguous failure state: without a
conditional inode-unlink primitive, retaining a manifestless artifact is safer than deleting a
path another actor may have replaced.

## Fail-closed contracts

- The plan's canonical content digest, schema, output schema, profile, absolute source bindings,
  sizes, and lowercase SHA-256 values are validated before staging or source-path use.
- This slice accepts compile-time contract plans only. An external-snapshot plan is refused because
  snapshot-copy publication is not implemented yet.
- Source name, dtype, shape, payload span, operation, target semantics, shard sequence, membership,
  byte totals, and resource envelope must agree exactly.
- Output and manifest paths must not exist. Existing content is never replaced.
- Source header parsing, initial hashing, payload reads, and final hashing use the same held file
  descriptions. A path replacement, identity change, truncation, or content digest change fails.
- A producer that writes fewer or more bytes than authorized fails while still in private staging.
- Every staged shard is reopened through the strict safetensors reader and rehashed before any
  output directory is claimed. Every published artifact is rehashed again before the manifest.

## Receipt and current limits

The canonical `comfy_omni.native_export.receipt/v1` manifest binds the plan content SHA-256, source
files, output schema/profile/component, complete file identities, output tensor/byte census, and
installed wheel identity. Its `manifest_sha256` is the SHA-256 of canonical receipt JSON before the
self-digest field is added.

The accepted miniature transaction is recorded in
[`docs/evidence/convrot-transaction-1a8ce636aa06.md`](../evidence/convrot-transaction-1a8ce636aa06.md).
It proves the historical transaction and copy path only. Inverse ConvRot, QKV reorder, and external
snapshot copying are subsequently accepted in
[`docs/evidence/convrot-payload-producers-1b2324ada243.md`](../evidence/convrot-payload-producers-1b2324ada243.md).
Full Ref2VA conversion, CLI exposure, native runtime loading, LoRA, and A/B/A switching remain
separate acceptance boundaries.
