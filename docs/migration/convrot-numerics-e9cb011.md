# ConvRot numerical backend design

Status: implementation slice for issue #8

Source: `h3-forge/main@e9cb011d00b028c149db3978de246c54f6e34acc`

Audited source blob: `convrot.py@8b4b9eebacd8bdaf64b251d5635b0147e7d790db` (Apache-2.0)

This slice implements only the bounded inverse-ConvRot mathematics required by the already
authorized `dense-bf16-online-int8` plan. It does not read safetensors payloads, reorder QKV rows,
write output tensors, construct shards, publish a directory, or expose `export-native`.

## Mathematical contract

For each signed INT8 source row and its positive finite FP32 row scale:

1. dequantize the row in FP32;
2. reshape the input width into independent regular-Hadamard groups;
3. multiply every group by the transpose of the normalized regular Hadamard matrix;
4. place the result into a dense BF16 destination tensor;
5. process at most `max_rows` source rows in one FP32 intermediate block.

The base sign matrix is:

```text
 1  1  1 -1
 1  1 -1  1
 1 -1  1  1
-1  1  1  1
```

Larger matrices use repeated Kronecker products and normalization by `sqrt(group_size)`. The matrix
is symmetric and orthonormal, so the same transform is its own inverse. Registered H3 checkpoints
currently require group sizes 64 and 256. The implementation deliberately accepts only powers of
four from 4 through 256; expanding that resource boundary requires review.

## Independent implementations

`conversion/numerics/reference.py` is a standard-library, bounded oracle. It supports at most
1,048,576 input elements and exists for small fixtures, orthogonality checks, involution checks, and
cross-backend evidence. It is not a model converter.

`conversion/numerics/torch_backend.py` is the production numerical adapter. Torch is loaded only
inside the explicit backend call through `importlib`; importing ComfyOmni or the numerics package
does not import Torch. The adapter:

- requires rank-2 `torch.int8` weights and `[rows, 1]` `torch.float32` scales;
- requires weights and scales on one device;
- rejects empty/misaligned shapes and non-finite or non-positive scales;
- runs under `torch.inference_mode()`;
- constructs one FP32 Hadamard matrix per conversion call, not once per row block;
- limits `max_rows` to 1..4096 and returns a dense BF16 tensor.

The base wheel intentionally declares no Torch dependency. Runtime-specific Torch/CUDA versions are
owned by the conversion image so pip cannot silently replace the runtime's ABI-compatible stack.

## Docker boundary

The repository `numerics-runtime` target uses the same `vllm/vllm-openai:v0.27.0` release boundary
as the current adjacent vLLM-Omni integration checkout. `COMFY_OMNI_NUMERICS_BASE_IMAGE` may point to
a reviewed mirror of that tag on a constrained server. Server evidence records the resolved image
ID; a release candidate must replace the tag with an immutable digest.

GitHub's regular Python 3.10/3.13 Docker lanes validate the independent reference oracle, resource
partitioning, lazy-import failure, and architecture boundaries without downloading Torch. The
production Torch adapter must additionally pass CPU and CUDA comparisons against the reference
oracle in the designated-server `numerics-runtime` container.

The bounded CPU/CUDA acceptance for candidate `b0b757a696d9` is recorded in
[`docs/evidence/convrot-numerics-b0b757a696d9.md`](../evidence/convrot-numerics-b0b757a696d9.md).

## Scope boundary

This record preserves the retained inverse-ConvRot numerical behavior only. It does not describe an
export, package, or runtime setup path. See the [post-merge refactoring plan](../post-merge-refactoring-plan.md).
