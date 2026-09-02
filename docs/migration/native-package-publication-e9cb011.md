# Native package publication provenance

Status: migrated publication contract for issue
[#9](https://github.com/leinasi2014/comfy-omni/issues/9) and pull request
[#29](https://github.com/leinasi2014/comfy-omni/pull/29)

## Source authority

| Field | Value |
| --- | --- |
| Repository | local migration authority `h3-forge` |
| Commit | `e9cb011d00b028c149db3978de246c54f6e34acc` |
| Source path | `src/h3_forge/package_assembler.py` |
| Source blob | `e64558f1d3bb6e1ee6f714b70e783d9df907f9ce` |
| Source path | `src/h3_forge/fsops.py` |
| Source blob | `ae40e46eef808f979ee085e806f2380e50b6c01d` |
| License | Apache-2.0 |
| Attribution | h3-forge contributors |

Both sources were read from the latest accepted legacy `main` only. Its unrelated untracked files and
working tree were not modified.

## Retained behavior

This slice reimplements the manifest-last commit point of legacy package assembly:

- emit the package manifest as canonical JSON whose `package_manifest_sha256` is the SHA-256 of the
  same document excluding exactly that field;
- write the manifest last, through an exclusive read-only create with fsync, so a package directory
  without its manifest is by construction an unfinished publication;
- deliberately retain that manifest-less (or pre-rename) directory on failure for diagnosis instead
  of an unsafe recursive cleanup;
- refuse any overwrite of an existing package path.

The implementation lives in `comfy_omni.conversion.packaging.publication.publish_package` with the
`PackagePublication` value in `conversion.packaging.models`. It imports no Torch, vLLM, FastAPI,
plugin, or optional serving dependency.

## Deliberate divergences

- Legacy assembled component trees directly inside the final output root; this slice publishes with
  one same-parent `os.rename` of the private staging directory, after re-checking that the output is
  still absent, and fsyncs the parent directory afterwards.
- Legacy trusted its in-process assembly state; this slice first revalidates the staging identity
  `(st_dev, st_ino)`, then independently re-reads the staged tree from disk (census plus per-file
  pinned SHA-256 and size against the plan) before any manifest write.
- The manifest is the only generated file and carries the routing index (`Ref2VA/` serving entry,
  one resident DiT, `ref2va|t2va|fl2va`); the legacy per-partition `model_index.json` layout is not
  reproduced.
- Refusals raise the typed `PackagePublicationError` (with stage evidence) or `FileExistsError`
  instead of the legacy domain error.

## Characterization evidence

RED candidate `28eef7c` (single acceptance test, format-folded from the first push) failed only
because the publication module did not exist: GitHub Docker quality run
[`33683554019`](https://github.com/leinasi2014/comfy-omni/actions/runs/33683554019) reported
`ModuleNotFoundError: No module named 'comfy_omni.conversion.packaging.publication'` with
`1 failed, 220 passed` on both Python lanes and a passing package and installed-wheel smoke.
GREEN candidate `debdbfc` passed 227 tests on Python 3.10 and 3.13 plus the package and
installed-wheel smoke in Docker quality run
[`33686167540`](https://github.com/leinasi2014/comfy-omni/actions/runs/33686167540); documentation
contracts passed in run
[`33686167394`](https://github.com/leinasi2014/comfy-omni/actions/runs/33686167394).

Tests cover the manifest-last atomic happy path (self-digest, exact census, staging consumed,
`PUBLISHED` result) plus refusals for a plan/handle digest mismatch, post-materialization staged
tampering, an unexpected staged entry, a missing staged entry, an output that appeared before
publication (marker preserved), and a replaced staging directory (deterministic inode divergence by
moving the original away).

## Distribution disposition

The rewritten Apache-2.0-compatible implementation and synthetic tests are included in the
ComfyOmni source distribution and wheel. No legacy model payload, generated package, operational
script, server evidence, or untracked legacy file is distributed by this slice.

Receipt-directory parsing, real six-component assembly on `srv-00`, CLI exposure, native load,
minimal generation, LoRA lifecycle, and `A → B → A` hot switching remain subsequent slices.
