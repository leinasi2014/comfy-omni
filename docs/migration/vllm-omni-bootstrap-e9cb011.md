# vLLM-Omni bootstrap provenance

Status: migrated bootstrap contract for issue
[#10](https://github.com/leinasi2014/comfy-omni/issues/10) and pull request
[#33](https://github.com/leinasi2014/comfy-omni/pull/33)

This record describes the first bootstrap slice. The later
[H3 component API contract](h3-component-api.md) adds the deferred API phase
and resolves both architecture keys through the verified runtime dispatcher.

## Source authority

| Field | Value |
| --- | --- |
| Repository | local migration authority `h3-forge` |
| Commit | `e9cb011d00b028c149db3978de246c54f6e34acc` |
| Source path | `src/h3_forge/plugin.py` |
| Source blob | `304a776bf4daf1f7a28b1bc6192d320da30421fd` |
| License | Apache-2.0 |
| Attribution | h3-forge contributors |

The source was read from the latest accepted legacy `main` only. Its unrelated untracked files and
working tree were not modified.

## Retained behavior

This slice reimplements the registration lifecycle of the legacy general-plugin entry as one
explicit state-machine coordinator:

- the host is observed, never forced: architecture registration runs only when `vllm_omni` is
  already resident in `sys.modules`, and the registry submodule is resolved from `sys.modules`
  first, falling back to a guarded import;
- a missing or resident-but-partial host defers silently — no exception, no latch — so a later
  `register()` retries;
- contributions are declarative lazy strings — the wire-compatible architecture keys
  `MiniMaxH3Pipeline` and `MiniMaxH3DensePipeline` with module/class names and
  `get_minimax_h3_post_process_func` — and registering imports no pipeline module (the host resolves
  them at model-load time);
- registration latches exactly once per process with a thread-safe `NEW → REGISTERING →
  REGISTERED` machine, resets on failure, and is safe under the host's every-process loading shape
  (process0, engine cores, workers);
- a `_is_root_process` helper (multiprocessing `parent_process`) keeps the documented hook for
  future API-server-only wiring.

The implementation lives in `comfy_omni.integrations.vllm_omni.bootstrap`; `comfy_omni.plugin`
is a thin shim over it and stays the distribution's only `vllm_omni.general_plugins` entry point.
Both modules import only the standard library at module level — no Torch, vLLM, FastAPI, or model
data anywhere on the import path.

## Deliberate divergences

- The legacy entry carried three independent latch flags and recursive sub-plugin callbacks; the new
  coordinator is one explicit state machine (issue #10's contract).
- The legacy `after_import` meta-path hook for API-server REST wiring, the LoRA admission bridge,
  and the runtime pipeline classes are not migrated in this slice; the future class homes are
  registered as lazy strings under `comfy_omni.integrations.vllm_omni.pipelines.*`.
- The legacy `API_SERVER_MODULE` constant is not carried; nothing is armed yet.

## Characterization evidence

RED candidate `1cc6da6` (single acceptance test) failed only because the coordinator did not
exist: GitHub Docker quality run
[`33698445499`](https://github.com/leinasi2014/comfy-omni/actions/runs/33698445499) reported
`ImportError: cannot import name 'bootstrap' from 'comfy_omni.integrations.vllm_omni'` with
`1 failed, 233 passed` on both Python lanes (ruff clean); documentation run
[`33698445493`](https://github.com/leinasi2014/comfy-omni/actions/runs/33698445493) passed. GREEN
candidate `07572f5` passed 240 tests on Python 3.10 and 3.13 plus the package and installed-wheel
smoke in Docker quality run
[`33699596471`](https://github.com/leinasi2014/comfy-omni/actions/runs/33699596471); documentation
contracts passed in run
[`33699596500`](https://github.com/leinasi2014/comfy-omni/actions/runs/33699596500).

Tests cover resident-host registration with the exact declarative call arguments and idempotent
re-entry, absent-host deferral with retry after the host appears, resident-but-partial host
deferral, the no-heavy-import invariant around import and `register()`, eight-thread concurrent
registration producing exactly one registration, shim delegation, and the source-level confinement
of literal `vllm`/`torch`/`fastapi` imports to `integrations/vllm_omni/` (none exist today).

## Distribution disposition

The rewritten Apache-2.0-compatible implementation and tests are included in the ComfyOmni source
distribution and wheel. No legacy runtime code, model payload, operational script, server evidence,
or untracked legacy file is distributed by this slice.

The `after_import` API hook, REST routes, runtime pipeline implementations, designated-host package
load, minimal generation, LoRA lifecycle, and hot switching remain subsequent slices under
issues #10–#13.
