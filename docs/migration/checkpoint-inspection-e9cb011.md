# Checkpoint inspection migration record

## Source identity

| Field | Value |
|---|---|
| Legacy project | `h3-forge` |
| Source commit | `e9cb011d00b028c149db3978de246c54f6e34acc` |
| Production source | `src/h3_forge/inspection.py` |
| Production blob | `70bcb4b0ff46b5297e468cb91533dcb812a38a7f` |
| Characterization source | `tests/test_inspection.py` |
| Characterization blob | `16cbc6a5905e6c4dfc352f827daafd0bf66e7b44` |
| Last production-source change | `d37280b4b1592fcef2d8bab95ea40e74706edddd` |
| Source license | Apache-2.0 |
| Migration issue | [#3](https://github.com/leinasi2014/comfy-omni/issues/3) |

The source commit is the immutable migration authority. Untracked files and sibling plugin working
trees are not inputs to this slice. The original implementation contains no external verbatim,
heritage, or mechanically-copied marker; this migration is a structurally split derivative of the
Apache-2.0 legacy project and remains under ComfyOmni's Apache-2.0 distribution.

## Ownership split

| Legacy responsibility | New owner | Dependency boundary |
|---|---|---|
| Inspection values and component/quantization classification | `domain/checkpoints.py` | Pure standard-library domain; no I/O |
| Bounded JSON/header/descriptor validation | `artifacts/safetensors.py` | Filesystem or supplied stream; no payload reads |
| One-file inspection composition | `conversion/inspection/checkpoint.py` | Artifact reader + pure classifiers only |
| Recursive path expansion and multi-file orchestration | `application/inspection.py` | Shared use case; no rendering |
| `inspect PATH... [--json]` parsing and rendering | `cli/commands/inspect.py` | Thin CLI adapter; business rules forbidden |

The old `h3_forge.inspection` import path is not added as a compatibility shim. ComfyOmni has not
published that Python import, and the approved public identity change uses `comfy_omni`. Stable CLI
arguments, output fields, evidence strings, limits, dtype roster, and failure messages are preserved.

## Frozen behavior

- only `.safetensors` files are accepted;
- header length is bounded to 64 MiB and tensor count to 100,000;
- JSON rejects duplicate keys, non-finite/non-standard values, oversized integers, malformed UTF-8,
  and unpaired Unicode surrogates;
- metadata is a string-to-string object;
- dtype, rank, dimensions, shape/product and byte span must agree;
- tensor offsets form one exact contiguous index over the payload, with no gaps, overlaps, or
  unindexed bytes;
- the stream reader stops at the payload boundary and never reads tensor bytes;
- H3 component and quantization classification uses structured evidence and rejects contradiction;
- directory expansion is recursive and sorted; JSON/text representations retain the legacy shape.

Characterization uses synthetic, license-cleared headers only. The real-writer padding property is
covered by an equivalent space-padded JSON header without importing Torch or the safetensors writer.
No checkpoint or model payload is copied into the repository.

## Acceptance and rollback

Local/CI characterization proves deterministic parser and CLI behavior; it does not prove the
selected multi-gigabyte files or the vLLM-Omni runtime. Server acceptance must inspect every asset in
`docs/testing/model-baseline.v1.json`, bind observed size/SHA256, and confirm no payload/GPU work occurs.

Rollback is a normal revert of this migration slice. It removes the new command and modules without
changing artifacts, schemas, remote state, or the lightweight plugin entry. No legacy source is
deleted by this migration.
