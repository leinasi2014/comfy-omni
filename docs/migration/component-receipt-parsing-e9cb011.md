# Component receipt parsing provenance

Status: migrated receipt contract for issue
[#9](https://github.com/leinasi2014/comfy-omni/issues/9) and pull request
[#30](https://github.com/leinasi2014/comfy-omni/pull/30)

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

This slice reimplements the read-only boundary between real component directories and package
planning:

- census one source tree deterministically (sorted entries), refusing links, special entries,
  unreadable entries, and an empty census;
- hash every regular file through the shared pinned held-descriptor primitive (bounded memory,
  before/after identity checks) recording exact size and SHA-256;
- re-census and re-hash the tree after the first pass, refusing any structural or content drift;
- bind the result into one immutable `ComponentReceipt` whose `receipt_sha256` is the SHA-256 of the
  canonical JSON over component, schema, resolved POSIX source directory, tool identity, and the
  sorted file census.

The implementation lives in `comfy_omni.conversion.packaging.receipts.parse_component_receipt`. It
performs no writes, does not parse model formats (a safetensors file is just a censored regular
file), and imports no Torch, vLLM, FastAPI, plugin, or optional serving dependency.

## Deliberate divergences

- Legacy bound sources to converter manifests (`_validate_export`, `_materialize_tree` snapshot
  resolution); this slice binds plain directories to the slice-1 `ComponentReceipt` planning value.
- The post-hash recheck re-hashes every file (legacy `verify_h3_package` re-hashed only manifest
  entries); a same-size in-place rewrite between the two passes is therefore refused.
- Refusals raise the typed `ComponentReceiptError` with stage evidence instead of the legacy domain
  error.

## Characterization evidence

RED candidate `788306c` (single acceptance test; format-folded from the first push `55ff5e5`, whose
run `33687245144` failed only on test formatting) failed only because the receipts module did not
exist: GitHub Docker quality run
[`33687433419`](https://github.com/leinasi2014/comfy-omni/actions/runs/33687433419) reported
`ModuleNotFoundError: No module named 'comfy_omni.conversion.packaging.receipts'` with
`1 failed, 227 passed` on both Python lanes; documentation run
[`33687433271`](https://github.com/leinasi2014/comfy-omni/actions/runs/33687433271) passed. GREEN
candidate `745a1da` passed 233 tests on Python 3.10 and 3.13 plus the package and installed-wheel
smoke in Docker quality run
[`33688797338`](https://github.com/leinasi2014/comfy-omni/actions/runs/33688797338); documentation
contracts passed in run
[`33688797414`](https://github.com/leinasi2014/comfy-omni/actions/runs/33688797414).

Tests cover the deterministic multi-file happy path with a planner plus source-verifier round trip,
plus refusals for an unknown component, a missing source directory, a linked leaf in the tree, an
empty tree, and same-size drift between the hash pass and the recheck.

## Distribution disposition

The rewritten Apache-2.0-compatible implementation and synthetic tests are included in the ComfyOmni
source distribution and wheel. No legacy model payload, generated package, operational script,
server evidence, or untracked legacy file is distributed by this slice.

Tool-identity resolution for real invocations, CLI exposure, the fixed six-component real assembly
E3 on `srv-00`, native load, generation, LoRA lifecycle, and hot switching remain subsequent work.
