# Native package materialization provenance

Status: migrated staging contract for issue
[#9](https://github.com/leinasi2014/comfy-omni/issues/9) and pull request
[#28](https://github.com/leinasi2014/comfy-omni/pull/28)

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

This slice reimplements the write boundary of legacy package assembly as private, verified staging:

- copy every planned file in bounded 8 MiB chunks through a held, pinned source descriptor and
  digest the bytes during the pass;
- require a regular non-linked source, refuse linked ancestors, and compare the descriptor identity
  `(st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns)` across the copy;
- create each target exclusively (`O_EXCL`, mode `0o444`), fsync it before close, re-check the
  target path identity, and independently re-hash the closed target;
- re-verify the whole plan before copying, refuse an existing or overlapping output path, and stage
  into a private sibling directory under the output parent;
- compare every copied digest and size against the plan, then re-census the staged tree and reject
  links, special entries, missing, or unexpected files;
- return an immutable `STAGED_VERIFIED` result bound to the plan digest, source-verification digest,
  staging identity, file count, byte total, and census digest.

The implementation lives in `comfy_omni.artifacts.fileops.copy_file_pinned_exclusive` and
`comfy_omni.conversion.packaging.materialization.materialize_package`. It imports no Torch, vLLM,
FastAPI, plugin, or optional serving dependency.

## Deliberate divergences

- Legacy assembled directly inside the final output root; this slice stages privately and publishes
  nothing. Manifest-last atomic publication is a subsequent slice.
- Legacy required the source to sit on a read-only mount; this slice relies on plan re-verification
  plus pinned-descriptor identity and defers the read-only mount policy to the fixed server
  acceptance environment.
- Legacy also offered a hardlink copy mode; the exclusive bounded copy is the only mode here.
- Legacy materialized resolved snapshot trees; this slice copies exactly the planned file set.
- A failed materialization never creates the final output and deliberately retains the private
  staging tree for diagnosis instead of an unsafe recursive cleanup.
- Refusals raise the typed `FsopsError` family or `PackageMaterializationError` instead of the
  legacy domain error or a raw OS error.

## Characterization evidence

RED candidate `1471ba5b296e94892c951155782bf8157bd60ae8` failed only because the materialization
module did not exist: GitHub Docker quality run
[`33679298001`](https://github.com/leinasi2014/comfy-omni/actions/runs/33679298001) reported one
failure and 211 passes on both Python lanes with the package and installed-wheel smoke passing.
GREEN candidate `dd369fe5d241fa70ce6c5e536981d7fb0e62922d` passed 220 tests on Python 3.10 and 3.13
plus the package and installed-wheel smoke in Docker quality run
[`33680349335`](https://github.com/leinasi2014/comfy-omni/actions/runs/33680349335); documentation
contracts passed in run
[`33680349346`](https://github.com/leinasi2014/comfy-omni/actions/runs/33680349346). The handoff
head `6ffe1b42a3dc31a9f8bbf5b61e9e6652bb1348f7` re-passed both gates in runs
[`33680690390`](https://github.com/leinasi2014/comfy-omni/actions/runs/33680690390) and
[`33680690389`](https://github.com/leinasi2014/comfy-omni/actions/runs/33680690389).

Tests cover exact staging without publication, existing output refusal without staging, output and
source overlap, post-verification source drift, a linked source, copy interruption retaining the
private staging tree, an unexpected staging entry, an existing copy target, and a same-size source
rewrite.

## Distribution disposition

The rewritten Apache-2.0-compatible implementation and synthetic tests are included in the
ComfyOmni source distribution and wheel. No legacy model payload, generated package, operational
script, server evidence, or untracked legacy file is distributed by this slice.

Package manifest/model-index generation, an independent staged-output verifier, manifest-last
atomic publication, CLI exposure, real-model assembly, native load, generation, LoRA, and hot
switching remain subsequent slices.
