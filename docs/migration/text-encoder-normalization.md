# Pinned text-encoder normalization design

## Decision and source audit

Issue [#6](https://github.com/leinasi2014/comfy-omni/issues/6) introduces one clean-room
ComfyOmni operation for a format exception discovered in the frozen model baseline. A search of the
latest consolidated legacy source, `h3-forge@e9cb011d00b028c149db3978de246c54f6e34acc`, found the
strict safetensors reader but no source-artifact normalization implementation to migrate. No legacy
code, third-party implementation, model bytes, or transport marker is copied into this change.

The operation is not a generic safetensors repair tool. It authorizes exactly this transformation:

| Identity | Bytes | SHA256 |
|---|---:|---|
| Source ModelScope file | 15,683,129,659 | `47babbb3e4b7e43c097351ca39cfb7f326d014ae53a584f8559dc8121abca94c` |
| Removed suffix | 72 | `8bbc743f1fdc67acb6b09c977485e7d8bed7ff073a12d70865e0e4b793ed8e75` |
| Strict derived file | 15,683,129,587 | `a166c7bbbe66a22065159e478335fee4a633c4a3e3bb34c8e8ac4cc91bf4996f` |

The derived digest was discovered on the designated validation host by hashing the exact bounded
prefix of the already E1-verified source. That read-only discovery froze the expected value; it did
not modify the source and is not itself acceptance of the implementation.

## Ownership and dependency direction

| Responsibility | Owner |
|---|---|
| Immutable identities, profile, receipt, stable normalization errors | `domain/normalization.py` |
| Installed source-commit and wheel identity | `artifacts/build_identity.py` plus the build hook in `setup.py` |
| Bounded streaming, hashing, strict reread, no-overwrite publication | `artifacts/normalization.py` |
| The one authorized ModelScope profile | `conversion/normalization/text_encoder.py` |
| Profile selection and installed-tool provenance | `application/normalization.py` |
| Argument parsing and receipt rendering | `cli/commands/normalize.py` |

This keeps the dependency direction `CLI -> application -> conversion -> artifacts/domain`.
Normalization does not import Torch, vLLM, vLLM-Omni, FastAPI, or runtime modules.

## Command and publication contract

The public command is:

```bash
comfy-omni normalize text-encoder SOURCE.safetensors DERIVED.safetensors --json
```

The source must exist and match the pinned byte count and SHA256. The destination parent must
already exist, the source and destination must differ, and neither the destination nor its sibling
`DERIVED.safetensors.normalization.json` receipt may exist. The implementation streams at most 8
MiB per read, copies only the authorized prefix to a distinct exclusive staging file, hashes the
source/prefix/suffix independently, and never opens the source for writing.

Before publication, the staged derivative must pass the same strict safetensors reader that rejects
the original. Publication uses no-overwrite filesystem links. The artifact becomes consumable only
when the sibling receipt exists as its commit marker; if the receipt cannot be published, the
operation removes the artifact link it owns and reports `normalization-publication-failed`.

The canonical `comfy-omni.normalization-receipt/v1` JSON binds:

- profile ID and version;
- source, removed suffix, and derived byte counts and SHA256 digests;
- strict reread result and tensor count;
- distribution version, clean source commit, and installed wheel SHA256.

Editable installs, dirty source builds, index installs without an archive digest, malformed build
identity, mismatched bytes, and pre-existing publications all fail closed. A release wheel receives
its source identity during build; installation from that wheel supplies the PEP 610 archive digest
used by the receipt.

## Verification, compatibility, and rollback

Synthetic tests cover success, immutable source bytes, all three digest mismatches, source-size
mismatch, invalid strict reread, existing artifact/receipt rejection, and simulated interruption
between artifact and receipt publication. Contract tests pin the production profile and dependency
direction. CI builds both sdist and wheel, then confirms the installed wheel can resolve its source
commit and archive SHA256.

The strict reader is unchanged and continues to reject any unindexed trailing bytes. The new CLI and
receipt schema are additive preview contracts. Rollback is a normal revert that removes the command,
profile, build hook, and derived-file capability; it never requires changing or deleting a source
checkpoint. Real-file E3 acceptance remains separate and must use the exact wheel and source
identities recorded in its receipt.
