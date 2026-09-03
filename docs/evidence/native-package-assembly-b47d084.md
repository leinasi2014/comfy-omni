# Native package assembly (E3) evidence — candidate b47d084

Date: 2026-09-03 (UTC 2026-09-02T23:29–23:52) · Host: `srv-00` · Issue:
[#9](https://github.com/leinasi2014/comfy-omni/issues/9) · Decision frozen at
<https://github.com/leinasi2014/comfy-omni/issues/9#issuecomment-5517264707>

## Candidate and environment

| Field | Value |
| --- | --- |
| ComfyOmni commit | `b47d084cc07a1f3c974527a13e7b76aede84ca2e` (PR #31 merge) |
| Source archive SHA-256 | `13d78feb0f8a678249f6f7d0c5b5c788ef5b192838bcc57e6975f073fa633cb4` |
| Installed wheel SHA-256 | `578665973c1000ccd8eb7c43ab49d158dca26cc4579f32a67a8c014ed8cde483` |
| Runtime image | `comfy-omni:e3-assembly-b47d084` = `sha256:89cd460e067a0f380e0224274a8870091ff0482ea521ef3278aea180cd8c3d27` |
| Host | `Linux srv-00 7.0.12-cmp x86_64`, Docker 29.1.3 |
| Build network | base images from the local `m.daocloud.io` cache; PyPI `build`/`twine` through the maintainer loopback-only tunnel |
| Evidence root | `/home/hyl/comfy-omni-e3/run-b47d084-attempt1` |

The wheel identity was extracted from the installed image itself via
`installed_tool_identity()` (PEP 610 direct-URL hash); the assembly runner refused to start until the
commit and wheel digest matched that authority.

## Six fixed components

| Component | Files | Bytes | Receipt SHA-256 | Source |
| --- | ---: | ---: | --- | --- |
| transformer | 14 | 40,226,030,420 | `ddd178422132de2eec0894710736678a115ae0c2daf9fcd939d852a2bdc24563` | Ref2VA full-conversion export (PR #25 evidence tree, hard-linked read-only) |
| text_encoder | 1 | 15,683,129,587 | `bf3c807bcc0f86fce5e20f6d9ca46db2b141a8d126cf4818582fb1dac1d778e3` | digest-pinned strict normalization derivative `a166c7bb…996f` |
| video_vae | 1 | 5,207,808,496 | `d5b1b26cd7708a40b0be558c4a60aa2dc7341e3dbd46379178dc9a66f3888729` | `7c1f1314…e522` |
| audio_vae | 1 | 605,254,808 | `e2a210115cbc0400e6238e40e3349ca73e6a5faface2d1c73538304fdcd99318` | `8e505d95…db48` |
| tokenizer | 4 | 11,492,078 | `593b39159687eff60f2e85d98b4019671ddfb31fe4ad852c283f3ff9f00459a0` | official `Ref2VA/tokenizer` download (this PR's baseline) |
| processor | 7 | 11,498,352 | `2ade06eb86224da4049c2feece1258e2369e6ffed9591b420ccad201c9c35f51` | official `Ref2VA/processor` download (this PR's baseline) |

Every component directory census was exactly the frozen list (no sidecars, locks, or extra files);
payload components were hard-linked read-only inside their own filesystems and presented to the
containers as read-only bind mounts under a unified `/components` root. No file in the existing
ComfyUI model tree or the conversion evidence tree was modified.

## Chain result

`installed_tool_identity → parse_component_receipt ×6 → plan_native_package →
verify_package_sources → materialize_package → publish_package`, executed offline
(`--network none`, dropped capabilities, `no-new-privileges`, uid/gid 65532):

- status `ASSEMBLED_PUBLISHED`; 28 files; 61,745,213,741 bytes total;
- plan content SHA-256 `bf514bbafb1e2ece0ee0b21c3fb379cb64c8a52f412181f0fca6528448276d20`;
- source-verification digest `8ce427e9180fac66b0ec36b4207f4ce894b8bf67fda9e33215eda287a1c6457a`;
- staged-census digest `ca54246f169468a3e0bb9eaa307d5b1dc83c5d38af51c44f485ad1713818b05d`;
- package manifest SHA-256 `e357764231e3a0d5e20154115a2153d2bbd932e462e08cb5c3ab2bd6f3ea21bd`;
- elapsed 1017.5 s; peak RSS 52,367,360 bytes (bounded memory across 61.7 GB of payload);
- the intended output path stayed absent until the same-parent atomic rename; no staging
  directory remains in the output parent.

## Independent verification

A fresh container from the same image re-read the published package from disk with no trust in the
assembly process: full tree census (links/special entries refused), per-file pinned hashing against
the manifest census, manifest self-digest recomputation, and equality with the recorded assembly
result. Verdict: `VERIFIED` — 28 files, 61,745,213,741 bytes, manifest SHA-256 and plan digest
matching, in 139.7 s.

## Container policy audit

All execution was Docker-only on `srv-00`; the orchestration itself ran as an unprivileged transient
systemd unit (uid 1000). Audit facts recorded in `evidence/container-policy.json`: assembly and
verifier containers used `network=none`, `user=65532:65532`, `cap_drop=ALL`,
`no-new-privileges=true`, read-only source mounts, no Docker socket, no GPU. The image build was the
only networked step (base images from the local registry cache, PyPI through the maintainer
loopback-only tunnel); no system proxy was configured. No containers remained after the run.

## Scope limits

This is E3 only: offline assembly, publication, and independent re-read of the fixed six-component
set. It is not a runtime claim — native host load, minimal generation, LoRA lifecycle, and
`A → B → A` hot switching (E4/E5) remain open, and the published package has not been presented to
the pinned vLLM-Omni host.

Full evidence (`EVIDENCE_SHA256SUMS`, census, logs, result and verdict JSON) remains at
`/home/hyl/comfy-omni-e3/run-b47d084-attempt1` on `srv-00`.
