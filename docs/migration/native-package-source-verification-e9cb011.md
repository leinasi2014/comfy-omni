# Native package source-verification provenance

Status: migrated verification contract for issue
[#9](https://github.com/leinasi2014/comfy-omni/issues/9) and pull request
[#27](https://github.com/leinasi2014/comfy-omni/pull/27)

## Source authority

| Field | Value |
| --- | --- |
| Repository | local migration authority `h3-forge` |
| Commit | `e9cb011d00b028c149db3978de246c54f6e34acc` |
| Source path | `src/h3_forge/package_assembler.py` |
| Source blob | `e64558f1d3bb6e1ee6f714b70e783d9df907f9ce` |
| License | Apache-2.0 |
| Attribution | h3-forge contributors |

The source was read from the latest accepted legacy `main` only. Its unrelated untracked files and
working tree were not modified.

## Retained behavior

This slice reimplements the read-only boundary between an authorized package plan and later
materialization:

- reconstruct and compare the complete plan before reading component payloads;
- reject missing, extra, linked, special, or unreadable tree entries;
- stream-hash every planned regular file through the shared pinned-descriptor primitive;
- compare exact sizes and SHA-256 values and repeat the tree census after hashing;
- return an immutable result bound to the plan, producer, counts, byte total, and canonical census
  digest.

The implementation lives in `comfy_omni.conversion.packaging.verification`. It performs no writes,
does not load a model, and imports no Torch, vLLM, FastAPI, plugin, or optional serving dependency.

## Characterization evidence

RED candidate `3c5a4a3efa1f5bdf40d5e898ebafa4b41dcd56e4` failed only because the verification module did
not exist: GitHub Docker quality run
[`33678019449`](https://github.com/leinasi2014/comfy-omni/actions/runs/33678019449) reported one
failure and 204 passes on both Python lanes. GREEN candidate
`e9bbc32d83d79eac1989f64ee4ad621b045b5e93` passed 211 tests on Python 3.10 and 3.13 plus the
package and installed-wheel smoke in Docker quality run
[`33678567807`](https://github.com/leinasi2014/comfy-omni/actions/runs/33678567807); documentation
contracts passed in run
[`33678567901`](https://github.com/leinasi2014/comfy-omni/actions/runs/33678567901).

## Distribution disposition

The rewritten Apache-2.0-compatible implementation and synthetic tests are included in the
ComfyOmni source distribution and wheel. No legacy model payload, generated package, operational
script, server evidence, or untracked legacy file is distributed by this slice.

Package materialization, output verification, manifest-last atomic publication, CLI exposure,
real-model assembly, native load, generation, LoRA, and hot switching remain subsequent slices.
