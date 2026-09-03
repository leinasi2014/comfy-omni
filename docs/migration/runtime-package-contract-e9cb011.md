# Runtime package contract provenance

Status: migrated a fail-closed runtime package contract for issue
[#11](https://github.com/leinasi2014/comfy-omni/issues/11) and pull request
[#36](https://github.com/leinasi2014/comfy-omni/pull/36)

## Source authority

| Field | Value |
| --- | --- |
| Repository | local migration authority `h3-forge` |
| Commit | `e9cb011d00b028c149db3978de246c54f6e34acc` |
| Source path | `h3/runtime_pipeline.py` |
| Source blob | `fa94f86da746ff9a11105584081464c1162d07b6` |
| License | Apache-2.0 |
| Attribution | h3-forge contributors |

The source was read from the latest accepted legacy `main` only. Its unrelated untracked files and
working tree were not modified.

## Retained behavior

This slice reimplements the fail-closed runtime package contract of the legacy runtime pipeline:

- the `H3ComfyMiniMaxH3Pipeline` subclass inherits the official pipeline shape: it validates the
  resolved package before `super().__init__` so no weight loading begins against an unverified
  package, and it re-exports `get_minimax_h3_post_process_func`;
- the `_converted_partition_path` local-package resolution, which maps the configured model path to
  the package root holding the index and manifest before any file is read;
- the fail-closed verification chain — package binding → model index → manifest → routing → tree
  census → file verification → components — so every refusal aborts the pipeline before weight load.

The validator lives in `comfy_omni.integrations.vllm_omni.package_contract` as a pure, host-free
module. It exposes the frozen `RuntimePackageContract` dataclass whose `to_dict` reports status
`RUNTIME_VERIFIED`. It imports no Torch, vLLM, FastAPI, plugin, or optional serving dependency.

## Deliberate divergences

- The legacy validator lived inside the pipeline module; this slice splits out a host-free pure
  validator that the pipeline subclass calls before `super().__init__`.
- The configured model path must be a package ROOT containing `model_index.json` plus
  `h3-comfy-package.json`; the legacy `_converted_partition_path` accepted a Ref2VA-named partition
  path, while the new layout's index lives at the package root.
- Curve-cache mechanics are not migrated: the DiT swap, per-worker latch, request locks, and
  schedule replay from the legacy runtime pipeline are deliberately excluded.

## Characterization evidence

E4-S2 RED candidate `cd6ff7e` (single acceptance test) failed only because the validator module did
not exist: GitHub Docker quality run
[`33704632351`](https://github.com/leinasi2014/comfy-omni/actions/runs/33704632351) reported
`ModuleNotFoundError: No module named 'comfy_omni.integrations.vllm_omni.package_contract'` with
`1 failed, 240 passed` on both Python lanes; documentation contracts passed in run
[`33704632253`](https://github.com/leinasi2014/comfy-omni/actions/runs/33704632253).

GREEN candidate `d30336c` passed 254 tests on Python 3.10 and 3.13 plus the package and
installed-wheel smoke in Docker quality run
[`33708501596`](https://github.com/leinasi2014/comfy-omni/actions/runs/33708501596);
documentation contracts passed in run
[`33708501606`](https://github.com/leinasi2014/comfy-omni/actions/runs/33708501606). Earlier
same-slice iterations (44324f3, 28a79b1, 4b193df, a785214, acc384e) were folded by amend into the
same GREEN candidate for ruff format joins, an isort aliased-import split, an unused import, a
host-stub stale-module fix via `importlib.import_module`, and a tuple-vs-list assertion type
correction.

Tests cover the fail-closed acceptance path (`RuntimePackageContract` validated to
`RUNTIME_VERIFIED`) plus nine refusal scenarios in `tests/unit/test_runtime_package_contract.py`,
and three host-stub tests in `tests/contract/test_runtime_pipeline_host_stub.py`.

## Distribution disposition

The rewritten Apache-2.0-compatible implementation and synthetic tests are included in the ComfyOmni
source distribution and wheel. No legacy runtime code, model payload, generated package, operational
script, server evidence, or untracked legacy file is distributed by this slice.

Real host load (E4-S3), LoRA lifecycle (#12), and `A → B → A` hot switching (#13) remain subsequent
slices.
