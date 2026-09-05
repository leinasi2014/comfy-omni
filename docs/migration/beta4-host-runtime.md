# Beta4 host runtime source and contract

This bounded adapter consumes the fixed 534-tensor `beta4-dense-bf16` export
described in [the source attribution](source-attribution.md). This existing adapter requires exported tensors;
the current plan removes that prerequisite from general runtime loading.
Runtime acceptance decisions and pending work remain in the issue; this document
records source ownership and the behavior boundary.

## Source attribution and distribution

The mathematical reference is `h3-forge` at
`e9cb011d00b028c149db3978de246c54f6e34acc`, licensed Apache-2.0:

| Source | Exact Git blob | Use |
| --- | --- | --- |
| `src/h3_forge/h3/dense_pipeline.py` | `6ddd34c49d532d56f568ec0010e925dbd86e5a2a` | FP32 table interpolation, rank-eight AdaLN without SiLU, and final modulation precision. |
| `src/h3_forge/h3/contracts/templates.py` | `443a5cc9ca58891c3852079c8589fbe2f5af6484` | Already distributed geometry; independent synthetic 534-slot characterization fixture. |
| `src/h3_forge/h3/runtime_pipeline.py` | `fa94f86da746ff9a11105584081464c1162d07b6` | Existing cache construction isolation retained through one common scoped lock. |

The execution host is `vllm-omni` at
`17285c2f55a41bf15772676121814d59a60ace35`, also licensed Apache-2.0:

| Source beneath `vllm_omni/diffusion` | Exact Git blob | Use |
| --- | --- | --- |
| `models/minimax_h3/minimax_h3_transformer.py` | `91e03c865b22ffaaa5dbb3bf3ceeaf804a5564c8` | Inherited official DiT, attention, MLP, RoPE, TP loaders, and the short derived final-layer forward. |
| `models/minimax_h3/pipeline_minimax_h3.py` | `62d10488ce0a82e9d3f72b3d0b23550565ff4db9` | Inherited pipeline with a scoped transformer factory substitution. |

The new integration owner is
`comfy_omni.integrations.vllm_omni.pipelines.beta4_pipeline`.
The host-free component identity owner is `comfy_omni.runtime.h3.beta4_binding`.
Derived local formulas and tests are distributed under this repository's
Apache-2.0 license, preserving h3-forge and vLLM-Omni contributor attribution.
Official host modules are imported from the pinned external installation;
their full implementations are not copied into this distribution. No checkpoint
payloads, machine-local evidence or legacy forward implementation are distributed.

## Loading and numerical boundary

The validator binds the fixed source identity, source and target schemas, every
descriptor, all shard hashes, complete export plan and dense execution policy.
Unknown, incomplete, modified or competing cache/beta4 bindings fail before
the dedicated constructor. Existing unrelated native package routing remains
unchanged.

The model requires all 534 source slots exactly once. The official loader
consumes 530 parameters and four persistent buffers, retaining its grouped QKV
and gate/up TP mapping. The table and AdaLN parameters execute in FP32; main
block modulation returns BF16 and final modulation remains FP32. The table is
the direct conditioning input. Basis and mean are retained persistent geometry
state, explicitly marked as not read by the forward calculation. No missing
`silu_t_emb_grid` is invented. Dense execution requires `quant_config=None`.

The common construction lock restores substituted host classes on exceptions.
A worker permits one completed model and pipeline. The dedicated constructor
refuses CPU/layerwise offload and HSDP; same-process model switching and runtime
LoRA lifecycle are outside this behavior boundary.

CPU characterization uses actual official linears and Gloo. Its fixture adapts
only the unsupported CPU attention selector/dispatch to the official unchanged
pure-Torch SDPA method. This does not establish production CUDA kernel parity.
Three local formulas and complete source-to-rank loading use exact comparisons.
Whole-forward TP1/TP2 characterization reports differences without a numerical
PASS or a promise of bitwise equivalence to the legacy attention implementation.

The installed-wheel harness separates synthetic tiny geometry from the fixed
real component. GPU TP1/TP2 characterization and real-component loading require
explicitly scheduled device, wheel, image and artifact identities. A successful
single-component forward does not establish a complete text-encoder/VAE/media
workflow or support for other ComfyUI models and nodes.
