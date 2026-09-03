# LoRA compatibility oracle provenance

Status: LoRA compatibility oracle slice for issue
[#12](https://github.com/leinasi2014/comfy-omni/issues/12) and pull request
[#41](https://github.com/leinasi2014/comfy-omni/pull/41)

## Source authority

| Field | Value |
| --- | --- |
| Repository | local migration authority `h3-forge` |
| Commit | `e9cb011d00b028c149db3978de246c54f6e34acc` |
| License | Apache-2.0 |
| Attribution | h3-forge contributors |

The six legacy sources below live under `src/h3_forge/lora_hotswap/` at that commit and were
wholesale migrated under the maintainer-approved B-mode; each original path maps to a new home under
`src/comfy_omni/conversion/lora/`.

| Legacy source path | Source blob | New home |
| --- | --- | --- |
| `lora.py` | `35083005d8bbc4b187b95455d7980e8a92c55f18` | `src/comfy_omni/conversion/lora/normalize.py` |
| `bake_plan.py` | `767ffd216c1a755fe60d04082c908627f46b311b` | `src/comfy_omni/conversion/lora/bake_plan.py` |
| `bake_audit.py` | `25b1664d206e5319a12455c352a01fb3ac9d5869` | `src/comfy_omni/conversion/lora/bake_audit.py` |
| `comfy_lora_bake.py` | `323794f758692eb5549d285dec69c7e3ff6591a9` | `src/comfy_omni/conversion/lora/comfy_bake.py` |
| `comfy_oracle.py` | `edb52dc3d30b1b1dbe2c393f7aa5dd439ef33cde` | `src/comfy_omni/conversion/lora/comfy_oracle.py` |
| `security.py` | `9c55b71391899c2b5a0620cb86b13c51571c270` | `src/comfy_omni/conversion/lora/security.py` |

All sources were read from the latest accepted legacy `main` only. Its unrelated untracked files and
working tree were not modified.

## Retained behavior

This slice wholesales the legacy six-module LoRA compatibility oracle under the maintainer-approved
B-mode: the source is byte-preserving apart from the added provenance headers, the import retargeting
to the library equivalents (`comfy_omni.artifacts.safetensors.read_safetensors_header_stream`,
`comfy_omni.domain.qkv`, and `comfy_omni.domain.checkpoints.TensorDescriptor`), and the mechanical
ruff-format wrapping. No semantic rewrite was performed.

The two-gate oracle structure is retained:

- `comfy_oracle.py` is the fail-closed pinned-Comfy reference-fold micro-oracle over exactly five
  pinned tensors (`MICRO_INDICES=(1,2,3,4,252)`); its only two decision outcomes are `INCOMPLETE`
  (all five equal) and `REFERENCE_FOLD_PARITY_MISMATCH` (any unequal). `promotion_capable` stays
  `False`, `actual_vllm_loader` stays `NOT_RUN`, and full 259-operation coverage stays `NOT_RUN`.
- `bake_audit.py` is the residual-survival diagnostic that stops before writing a checkpoint and
  reports `DIAGNOSTIC_PASS` / `DIAGNOSTIC_PASS_UNDEPLOYABLE` / `IDEAL_RESIDUAL_NOT_PRESERVED` /
  `PARTITION_DIAGNOSTIC_PASS` / `REJECT_BF16_BAKE`; it is an ideal-FP32 fidelity diagnostic, not the
  production gate (the production gate is the Comfy reference-fold byte consistency required by
  `COMFY_BAKED_NATIVE_PRODUCT_GATE`).

Torch stays function-body lazy: importing any migrated module loads no Torch, vLLM, FastAPI, plugin,
or optional serving dependency. Per B-mode, real-GPU characterization is the later acceptance run on
the designated host, not part of the unit-test matrix; the migrated modules are not unit-tested here.

## Deliberate divergences

- The NEW `conversion/oracle/` contract and preflight (`contract.py`, `preflight.py`) are introduced
  in this slice: the LoRA-compatibility schema is `comfy_omni.lora-compatibility/v1`, with a stable
  fail-closed reason-code registry including `BASE_REPRESENTATION_UNBINDABLE` (and
  `ORACLE_BASE_CONTRACT_NOT_BINDING`) for int8-convrot bases that cannot present the official BF16
  13-shard census, and the pinned-candidate identity (SHA256 + size) is injected from
  `docs/testing/model-baseline.v1.json`.
- Coordinator layering correction: `preflight_candidate` takes an injected
  `runtime_contract_resolver` callable instead of importing `validate_runtime_package` from the
  integrations layer. The conversion layer must not import the integrations layer; the frozen
  contract was amended accordingly. The unit test injects `validate_runtime_package` only at call
  time.
- The unit-test matrix covers only the NEW `conversion/oracle/` code; the migrated legacy modules are
  characterized on a later GPU run (B-mode).

## Characterization evidence

RED candidate `62db872` (single acceptance test) failed because the preflight module did not exist:
Docker quality run
[`33714691370`](https://github.com/leinasi2014/comfy-omni/actions/runs/33714691370) reported the
missing-module failure for `comfy_omni.conversion.oracle.preflight` (and
`comfy_omni.conversion.oracle.contract`), while the documentation run
[`33714691344`](https://github.com/leinasi2014/comfy-omni/actions/runs/33714691344) passed.

The intermediate GREEN iteration `12d5089` ran green everywhere except that Docker quality run
[`33715018956`](https://github.com/leinasi2014/comfy-omni/actions/runs/33715018956) failed on ruff
formatting of the migrated modules.

Final GREEN candidate `17c022f` passed 262 tests on both Python lanes plus the package and
installed-wheel smoke in Docker quality run
[`33716440848`](https://github.com/leinasi2014/comfy-omni/actions/runs/33716440848); documentation
contracts passed in run
[`33716440850`](https://github.com/leinasi2014/comfy-omni/actions/runs/33716440850). The
format/lint/layering corrections were folded into the same GREEN commit by amend.

## Distribution disposition

The migrated modules, the NEW `conversion/oracle/` contract and preflight, and the new-code unit
matrix are included in the ComfyOmni source distribution and wheel. No LoRA weights, NVIDIA or
proprietary components, or server evidence is distributed by this slice.

Offline fold on the `SUPPORTED` branch, the off -> on -> off activation lifecycle, runtime
hot-swap/control plane, CLI, and actual baked-checkpoint publication remain subsequent slices;
normalization stops at `CONVERTIBLE` and is never treated as `SUPPORTED` or zero-overhead dynamic
PEFT.
