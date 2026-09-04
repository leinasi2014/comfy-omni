# Dense hybrid8 runtime route: migration record (from h3-forge e9cb011)

Status: PLANNED - first recorded slice of the dense hybrid8 runtime port.

## Purpose

Serve the MiniMax-H3 ref2va / hybrid8 dense form (`h3-transformer-50l-hybrid8-bf16-plain`,
per the ComfyOmni contract templates census) natively in the pinned vLLM-Omni host. Today the
host only constructs the "official old form" transformer (`time_embed_dim=2688` online MLP time
embedder); the hybrid8 form instead carries per-block narrow `[96768, 8]` AdaLN projections fed
by a shared top-level `adaln_t_table` (`[1025, 8]`) plus `silu_t_emb_grid` (`[1025, 2688]`) and
no `time_embedder` parameters.

## Provenance (required by HANDOFF before a legacy slice enters a PR)

- Legacy repository: `h3-forge` at commit `e9cb011d00b028c149db3978de246c54f6e34acc`.
- License: Apache-2.0 (repository LICENSE at that commit).
- Legacy implementation blob to migrate:
  `src/h3_forge/h3/dense_pipeline.py`, blob `6ddd34c49d532d56f568ec0010e925dbd86e5a2a`
  (verbatim at e9cb011; worktree file matches the blob so the working copy is clean).
- Legacy reference PRs: `prs/008-servable-hybrid8-hostload.md` (hybrid8 host-load, fail-closed
  535-tensor census) and `prs/009-servable-hybrid8-forward.md` (denoise-loop forward, contract
  mirror of the official `MiniMaxH3DiTModel.forward`).
- Public distribution disposition: the legacy code is already Apache-2.0; the migration keeps
  the same license and records the source blob above in this document and in the module
  docstrings of the ported code.

## Root cause evidence (srv-00, not committed)

- Package: `/data/models/comfy-omni/e4v33-output/native-package`
  (manifest SHA `6d8f2b3b...`; transformer `config.patch.json` profile
  `dense-bf16-online-int8`; `export.plan.json` `qkv_layout` =
  `{num_query_groups: 56, heads_per_group: 1, head_dim: 128, row_count: 21504,
  target_layout: grouped-for-official-loader}`; `architecture_template` =
  `h3-transformer-50l-convrot`; runtime-ignored AdaLN/condition layers).
- Run `run-e4s3-attempt12` (candidate `e6c5473`, environment + digests in
  `run-e4s3-attempt12/evidence/environment.txt`) failed at weight load:
  `minimax_h3_transformer.load_weights -> linear.weight_loader_v2 ->
  _ColumnvLLMParameter.load_column_parallel_weight -> AssertionError`.
- Instrumented re-run `run-e4s3-attempt13` (same image + debug probe only) printed the exact
  mismatch for both TP ranks:
  `LOADFAIL name=blocks.14.adaln_proj.linear.weight param_shape=(48384, 2688)
  loaded_shape=(96768, 8) dtype=torch.float16 args=()`.
  - checkpoint: `[96768, 8]` (hybrid8 narrow AdaLN, in = 8-dim code);
  - host model: `[48384, 2688]` (old form, in = `time_embed_dim` 2688; 48384 = 96768 / TP2).
- No Xid / NVRM error in dmesg; GPUs idle and healthy after the run. This is a form/architecture
  contract mismatch, not a hardware or memory failure.
- Legacy note: the same hybrid8 forward was verified on srv-206 to reach the denoise loop and
  fail only on activation VRAM (`~62.42 GiB` allocated vs `63.53 GiB` card), with the recorded
  minimal repair path being token-dim chunking of qkv/MLP plus TP slicing (see pr/009).

## Why the host chose the old form

`config.patch.json` in the native package carries only `quantization_config` (int8 + ignored
layers). The host falls back to `TransformerConfig()` defaults, and the vllm-omni
`MiniMaxH3DiTModel` constructs the official old-form AdaLN/time embedder. The hybrid8 signature
(top-level `adaln_t_table`/`adaln_basis`/`adaln_mean`/`silu_t_emb_grid` and per-block
`[96768, 8]` AdaLN) is not detected by the host; the dense route must switch on this signature
before constructing the transformer tree, exactly like the legacy
`DenseHybridDiT`/`discover_hybrid8_dit_form` route.

## Port design (matches docs/post-merge-refactoring-plan.md section 8.3)

```text
comfy_omni/runtime/h3/hybrid8/
  contracts.py   # hybrid8 signature + pinned 535-tensor manifest (from templates.v1.json census)
  geometry.py    # derive hidden/heads/head_dim/rot/ffn/patch/text from pinned census; fail-closed
  qkv.py         # grouped <-> qkv row-index algorithms (already exist in comfy_omni/domain/qkv.py)
  modules.py     # _Hybrid8RMSNorm/_Attention/_MLP/_AdalnProj/_DiTBlock/_FinalLayer (torch + vllm)
  loader.py      # census digest validation + assembly (1:1 params/buffers, reject unknown/dtype/shape)
  forward.py     # hybrid8_conditioning (t -> table rows floor+lerp), forward contract mirroring
  text_encoder.py# pruned Qwen3-VL 24-layer encoder route (deferred unless required by M4 slice)

comfy_omni/integrations/vllm_omni/pipelines/dense_pipeline.py
  # MiniMaxH3DensePipeline adapter (bootstrap.py already declares this contribution)
```

Pure contract/geometry/QKV code must not import vLLM or torch; only the final pipeline adapter
may inherit host classes (dependency direction preserved:
`core -> domain/contracts/artifacts -> conversion/runtime -> application -> CLI/API/integrations`).

## Slice acceptance

- Ported hybrid8 forward mirrors the official `MiniMaxH3DiTModel.forward` kwargs and math
  (port and adapt the legacy `test_dense_hybrid8_forward.py` contract tests).
- Assembly is fail-closed: census digest equals manifest-derived digest, 1:1 name coverage,
  unknown/duplicate/dtype/shape deviations abort construction.
- Runtime performs no Comfy parsing, dequantization, inverse rotation, or package mutation in
  the denoising path (hybrid8 is already dense BF16 in the package; the route only consumes it).
- Server verification stays on srv-00 (Docker-only) after a wheel + host/preflight image rebuild.
