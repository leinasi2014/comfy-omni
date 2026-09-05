# Fixed beta4 dense BF16 conversion

The explicit `build_beta4_dense_plan` entry authorizes only the fixed primary beta4 asset:
20,967,637,320 bytes, SHA256
`54d56b15c65923b54c9ca16b494dae641bfe9455cfcb1c19c49b1008e270bbc1`.
Its complete 934-descriptor schema is
`ae2456bc6ac904929a4b773f703f8a1baa99b6356b5a389994faf64a1a2d80f2`.
This authorization extends the supported conversion inputs; the unmodified legacy converter
did not authorize this asset. The existing 932-source and 535-dense contracts remain unchanged.

## Source and distribution disposition

The existing Apache-2.0 migration inputs are from `h3-forge` commit
`e9cb011d00b028c149db3978de246c54f6e34acc`:

| Source | Exact blob | Use |
| --- | --- | --- |
| `src/h3_forge/h3/contracts/templates.py` | `443a5cc9ca58891c3852079c8589fbe2f5af6484` | Already distributed architecture inventory; the new contract derives an independently pinned beta4 inventory. |
| `tests/fixtures/contract_auto/10eros_hybrid8_catalog.json` | `1969b8ac93bb46a1ca46af395542876ec6baa552` | Header-only beta3 comparison evidence; no new fixture copy or source paths are distributed. |
| `src/h3_forge/native_export.py` | `475cee5523be64e5b24a95e16c5de3f371cbdf67` | Existing streaming producer and manifest-last publication characterization. |
| `src/h3_forge/convrot.py` | `8b4b9eebacd8bdaf64b251d5635b0147e7d790db` | Existing inverse-ConvRot numerical transformation. |

The new owner is `comfy_omni.contracts.beta4`; execution stays in `conversion.exporters`.
New contract code and tests are distributed under this repository's Apache-2.0 license with the
above attribution. No new legacy Python module, model payload, or private evidence is copied.
The original 535 inventory is already hash-verified by `contracts.templates`. Removing the grid
and representing the observed 200 triplets must reproduce the independent beta4 source schema
pin; the dense target has its own schema pin. The derivation itself cannot authorize a drifted
inventory. Header observations are not fresh full-file identity verification.

The standalone verifier derives from this repository at
`0925862033b0a9fdf48935ce538f364bbc317e2d`,
`scripts/acceptance/verify_ref2va_full_conversion.py` blob
`1e9056553b969366426b3c7dc6ad30b61ff43fc9` (Apache-2.0). Its scalar oracle is retained,
with independent beta4 pins and held-file verification added. It imports no converter module.

## Conversion and output contract

The source contains 334 BF16 tensors and 200 exact INT8 ConvRot triplets. Each triplet has an I8
matrix, positive finite F32 row scales, and a supported U8 declaration with group size 256.
FP32 dequantization and normalized regular-Hadamard inversion produce BF16 rows. Source runtime
QKV rows are reordered for the official grouped loader; all other passthrough bytes remain exact.
The source is held read-only and reverified before publication.

The output is exactly 534 BF16 tensors, 40,222,925,872 payload bytes, with descriptor schema
`3684a0d21eebe12c27cbf2b54d0b8cef74bd9d2119d94a14a89d5f77ffd0ec4b`.
No `silu_t_emb_grid` is produced. The separate `beta4-dense-bf16` profile declares
`runtime_quantization.required=false`, `method=null`, and empty ignored layers. Its config patch
explicitly clears `quantization_config` to null. It does not relabel this output as the existing
`dense-bf16-online-int8` route or claim payload-preserving/lossless conversion.

Planning checks fixed file identity, every source descriptor, complete group geometry and the
independent target inventory. Execution reconstructs the entire new-profile plan from held
source descriptors, markers and observed hashes against the compiled beta4 authority. A caller
cannot change operations, identities, quantization policy or target schema and regain authority
by recomputing the plan self-digest. Publication retains the existing no-overwrite transaction.

## Acceptance boundary

`scripts/acceptance/beta4_dense_conversion.py` runs from an installed candidate wheel in offline
CPU Docker with at most 4 GiB memory. Its `plan` and `run` actions bind the exact candidate, wheel,
source and plan. The output uses a separate filesystem and a fresh directory. Preflight calculates
exact payload and shard bytes, bounds small documents, limits allocation to 45 GiB, and reserves
12 GiB of free space. A callback rechecks actual staging size and free space before publication.
The callback is optional for existing exporter callers and does not alter their serialization.
Sources and older artifacts are never deleted to make space.

The independent verifier checks complete file identities and descriptor coverage, every raw-copy
byte and both full QKV reorderings. It checks each of the 200 converted matrices with independently
computed first/middle/last rows, covering every column and each 256-column block; its receipt
reports numerical sampling and errors explicitly. Sampling is not an all-element numerical proof.
An optional full numerical audit would require another streaming pass and a measured CPU budget.

Synthetic gates establish contract rejection, plan reconstruction, budget enforcement and the
existing numerical/transaction operations. They do not substitute for the installed-wheel full
conversion and independent real-asset verifier. This artifact is an E3 transformer export, not a
servable six-component package. Base-only hybrid8 runtime, fixed NVFP4 text encoder execution,
full video/audio generation, LoRA activation and same-process DiT switching require their own
accepted behavior and evidence.
