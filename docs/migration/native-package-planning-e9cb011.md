# Native package planning provenance

Status: migrated planning contract for issue
[#9](https://github.com/leinasi2014/comfy-omni/issues/9) and pull request
[#26](https://github.com/leinasi2014/comfy-omni/pull/26)

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

This slice characterizes and reimplements only the pure authorization boundary that precedes
materialization:

- exactly six required component roles;
- one identical converter commit/wheel identity across all components;
- fixed vLLM-Omni host identity;
- canonical `Ref2VA/` component placement and legacy-compatible package schema/manifest name;
- deterministic input-order-independent planning;
- fail-closed rejection of missing, duplicate, unknown, reused, malformed, traversing, or
  target-colliding authorities;
- canonical JSON content identity excluding the self-digest field.

The new implementation is split into immutable values in
`comfy_omni.conversion.packaging.models` and pure validation/planning in
`comfy_omni.conversion.packaging.planning`. It does not copy the 1,500-line legacy assembler or
import its runtime, LoRA, CLI, vLLM, Torch, VAE, or plugin dependencies.

## Characterization evidence

RED candidate `f1da3243f888f1a41cd8f1bad38431d11a662104` failed only because the new package-plan modules
did not exist: GitHub Docker quality run
[`33676906712`](https://github.com/leinasi2014/comfy-omni/actions/runs/33676906712) reported one
failure and 188 passes on both Python lanes. GREEN candidate
`f1f1fa0192e3df1a71263115ab2d59433867695b` passed the canonical happy path and the component,
producer, receipt, path, file-identity, and host refusal matrix in Docker quality run
[`33677384662`](https://github.com/leinasi2014/comfy-omni/actions/runs/33677384662); documentation
contracts passed in run
[`33677384784`](https://github.com/leinasi2014/comfy-omni/actions/runs/33677384784).

## Distribution disposition

The rewritten Apache-2.0-compatible implementation and tests are included in the ComfyOmni source
distribution and wheel. No legacy model payload, generated package, private evidence, operational
script, runtime implementation, or untracked legacy file is distributed by this slice.

Filesystem receipt parsing, source rehash, materialization, independent package verification,
manifest-last publication, CLI exposure, and real-model assembly remain explicit subsequent
slices; this planning evidence makes no claim about them.
