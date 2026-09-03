# Serving layout provenance

Status: migrated a fused single-partition serving layout for issue
[#39](https://github.com/leinasi2014/comfy-omni/issues/39) and pull request
[#40](https://github.com/leinasi2014/comfy-omni/pull/40)

## Source authority

| Field | Value |
| --- | --- |
| Repository | local migration authority `h3-forge` |
| Commit | `e9cb011d00b028c149db3978de246c54f6e34acc` |
| Source path | `deploy/206/serve_010_native.sh` |
| Source blob | `8d266bbb07777c3808c9bb4c100261b7fbaddb2c` |
| License | Apache-2.0 |
| Attribution | h3-forge contributors |

The source was read from the latest accepted legacy `main` only. Its unrelated untracked files and
working tree were not modified.

## Retained behavior

This slice keeps the operational serving shape of the legacy one-shot native package serve:

- serve the package's `Ref2VA` directory with `vllm serve <path>/Ref2VA` and pass a
  `--model-class-name` selector, so the pinned host resolves the `ref2va` partition by the served
  path's basename and its `model_index.json`;
- a single fused partition view for a self-contained published package.

The implementation lives in `comfy_omni.integrations.vllm_omni.serving` as a pure, host-free
module. It imports no Torch, vLLM, FastAPI, plugin, or optional serving dependency.

## Deliberate divergences

- The legacy script served a `Ref2VA` directory that already sat under a package path the operator
  had prepared; the frozen ComfyOmni package format keeps `model_index.json` and
  `h3-comfy-package.json` at the package ROOT with the `Ref2VA/` component tree below it, so an
  operator cannot point `vllm serve` at a bare partition directory the pin expects. This slice adds
  a directory **symlink view** `<work_dir>/Ref2VA -> <package_root>` plus a layout marker
  (`.comfy-omni-serving`, content `comfy-omni.serving-layout/v1`), leaving the published package
  tree untouched; the layout is a view, not a copy.
- `comfy_omni.integrations.vllm_omni.package_contract.validate_runtime_package` now accepts a
  single top-level symlink (the serving view) as the package root — it resolves the view before the
  fail-closed tree census — while still refusing linked ancestors and any link inside the tree. This
  is the minimal change that lets `validate_runtime_package(work_dir/"Ref2VA")` pass through the
  view, proving the layout is servable shape.
- `describe_serving_command` reports `--model-class-name MiniMaxH3Pipeline`, the literal key our
  bootstrap registers for the fused single-partition architecture; the legacy script used
  `MiniMaxH3DensePipeline` for its dense-native package.
- Refusals raise the typed `ServingLayoutError` (with stage evidence) or propagate the fail-closed
  `RuntimePackageContractError` instead of the legacy script's container/exit-code behavior.

## Characterization evidence

RED candidate `f5ab153` (single acceptance test) failed only because the serving module did not
exist: GitHub Docker quality run
[`33715218795`](https://github.com/leinasi2014/comfy-omni/actions/runs/33715218795) reported
`ModuleNotFoundError: No module named 'comfy_omni.integrations.vllm_omni.serving'` with
`6 failed, 254 passed` on both Python lanes after `140 files already formatted` / `All checks passed!`
(ruff clean); documentation contracts passed in run
[`33715218800`](https://github.com/leinasi2014/comfy-omni/actions/runs/33715218800).

GREEN candidate `1c5fc95` passed `254 + 6 = TODO` tests on Python 3.10 and 3.13 plus the package and
installed-wheel smoke in Docker quality run
[`TODO`](https://github.com/leinasi2014/comfy-omni/actions/runs/TODO); documentation contracts
passed in run [`TODO`](https://github.com/leinasi2014/comfy-omni/actions/runs/TODO).

Tests cover the servable happy path (marker + `Ref2VA` symlink view, model index visible through the
view, `validate_runtime_package` passing through the view, idempotent re-call), the invalid-package
refusal propagating `RuntimePackageContractError`, divergent and wrong-marker `work_dir` refusals,
`clear_serving_layout` removing its own layout and refusing a non-layout directory, and the legacy
serve command string.

## Distribution disposition

The rewritten Apache-2.0-compatible implementation and synthetic tests are included in the ComfyOmni
source distribution and wheel. No legacy model payload, generated package, operational script, server
evidence, or untracked legacy file is distributed by this slice.

Real host load (E4-S3 rerun of serving `<layout>/Ref2VA` to a ready orchestrator), a `comfy-omni
serve-layout` CLI, LoRA lifecycle, and `A → B → A` hot switching remain subsequent slices.
