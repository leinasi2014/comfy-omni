# Native package assembly (E4 revision) evidence — candidate 76e2ebb

Date: 2026-09-03 (UTC 01:02–01:21) · Host: `srv-00` · Issue:
[#11](https://github.com/leinasi2014/comfy-omni/issues/11) · Staged E4 READY:
<https://github.com/leinasi2014/comfy-omni/issues/11#issuecomment-5518522663>

## Candidate and environment

| Field | Value |
| --- | --- |
| ComfyOmni commit | `76e2ebba3ba7ccd6bfc7d6254e5aa3a90716b369` (PR #34 merge — host-discovery model index) |
| Source archive SHA-256 | `798b21aa83e28277334ab03e8ca3691680ac23adb7de21e031f1e5043dac0e3b` |
| Installed wheel SHA-256 | `ea124fd4f4a7a1c0fddf6a305e56102e4620597fe2ed92f4a51926b1464521d7` |
| Runtime image | `comfy-omni:e4-assembly-76e2ebb` = `sha256:f3ff2533ce52a08a8a9da85ee2b1deabc1e5ec80f060eccbccd65968e60df0d8` |
| Host | `srv-00` (Linux 7.0.12-cmp x86_64, Docker 29.1.3); offline containers, uid 65532 |
| Evidence root | `/home/hyl/comfy-omni-e3/run-76e2ebb-attempt1` |

## Result

Same fixed six components as the E3 run (unchanged receipts census; the receipt digests change only
through the bound tool identity, which now carries candidate `76e2ebb` and wheel `ea124fd4…`).
Chain: `installed_tool_identity → parse_component_receipt ×6 → plan_native_package →
verify_package_sources → materialize_package → publish_package` = `ASSEMBLED_PUBLISHED` in
1016.2 s.

- 28 component files, 61,745,213,741 bytes;
- plan content SHA-256 `b8d1efb5dc7b3254b03a31c053450667f366c4c5aaaa9000fbe21723374ac9a0`;
- package manifest SHA-256 `bc5884447f0ac0be76f16706799b8ad1fb17a479794ac96e675846cce37652de`;
- **host-discovery `model_index.json`** (the E4-S1 addition) SHA-256
  `a58019bf39e9be4476e8e555f78a04d9fb9e4a530bb5e09b08aa3f1a8ab52741`, bound into the manifest
  self-digest via `model_index_sha256`, canonical bytes with
  `_class_name: "MiniMaxH3Pipeline"` and the hybrid task routing `ref2va|t2va|fl2va`;
- published package at `srv-00:/data/models/comfy-omni/e4-output/native-package`
  (30 files = 28 components + `model_index.json` + `h3-comfy-package.json`); no staging leftovers;
  the E3 package at `/data/models/comfy-omni/e3-output/native-package` remains untouched.

Independent verifier (fresh offline container, no trust in the assembly process): full census
including both generated files, per-file pinned hashing against the manifest, manifest self-digest
recomputation, and model-index validation (strict parse, `_class_name`, canonical bytes, manifest
digest equality) — **`VERIFIED`** in 139.8 s.

## Scope limits

Package revision only — still no runtime claim. Native host load, minimal generation, LoRA, and hot
switching remain open under E4-S2/E4-S3 and issues #12/#13. Full evidence and
`EVIDENCE_SHA256SUMS` remain at `/home/hyl/comfy-omni-e3/run-76e2ebb-attempt1` on `srv-00`.
