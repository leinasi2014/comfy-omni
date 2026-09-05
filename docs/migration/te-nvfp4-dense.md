# Fixed H3 text encoder normalization contract

This offline component accepts one pinned strict text encoder and its unchanged
configuration. It does not provide general ComfyUI model or workflow compatibility.
It does not establish the checkpoint's historical writer identity.

The source file is 15,683,129,587 bytes, SHA256
`a166c7bbbe66a22065159e478335fee4a633c4a3e3bb34c8e8ac4cc91bf4996f`.
The configuration is 1,474 bytes, SHA256
`d2dd0c60d01b9e195d9447c52da61c7302d28828524914c044d9c6e1b81d0427`.
The complete source schema contains 1,954 tensors. The output contains 902 BF16
tensors and 51,506,191,840 payload bytes. All 1,052 auxiliary descriptors are
accounted for: 350 NVFP4 groups, one INT8 embedding and 551 BF16 tensors.

## Numerical reference and attribution

The consumer reference is comfy-kitchen commit
`b678fdf63378409676aa5596721445d33794d0ea`. Its eager quantization implementation
has Git blob `1df1514e38216b0deeb1977075a187fdda5886ad`, SHA256
`71ad880e9aadf4e9e8f144a3a1ba7a5e2c836df727fc90e059a4431bab94ceb8`.
Its float utilities have blob `29077a7b5375a596eab64bab449bfc2e842beb9d`,
SHA256 `de03e340ded7d21f136d0f8c662b976588c2464194dba1c13168f091a02fc216`.

The producer consumes high nibble first and reverses the consumer's blocked
scale layout. The global scale, block scale and scale product are materialized
as BF16 before the final BF16 product. INT8 embedding multiplication is FP32
followed by BF16 conversion. FP32 multiplication with a single final cast is a
different NVFP4 interpretation and is not interchangeable.

The derived numerical implementation is distributed under the upstream Apache
2.0 and applicable torchao BSD 3-Clause terms. The full upstream LICENSE and
NOTICE are in `third_party/comfy-kitchen/`; `THIRD_PARTY_NOTICES.md` identifies
the derived files. No ComfyUI GPL implementation is copied or executed here.
The acceptance scalar oracle is independently expressed using the standard
library and does not import the producer or Torch.

## Publication and independent verification

Planning and execution reauthorize the fixed input from retained file
descriptors. Execution decodes at most 128 matrix rows per stripe, writes to an
exclusive sibling staging directory, rereads output artifacts, then checks
source and configuration again before publishing the manifest. Source files
are never opened for writing. Existing output paths are rejected.

The acceptance harness requires Docker, read-only source mounts, no network or
GPU, a memory limit of at most 4 GiB, a 50 GiB output cap, 60 GiB initial free
space and a 12 GiB reserve. Its declared image identity must also be bound to
external Docker inspection evidence. Run it with an installed candidate wheel;
source-tree tests do not establish installed-wheel acceptance.

The independent verifier checks all source and target descriptors, all 551
passthrough tensors byte for byte, and the first, middle and last complete rows
of every NVFP4 matrix. It also checks distributed embedding rows using exact
BF16 bits. Sampling does not prove every converted value. It rehashes held
files and binds the producer receipt, plan, wheel and source identities.

Output names use `model.language_model.*` and `model.visual.*` for the pinned
H3 host loader. The 902 input tensors project to 752 logical host parameters;
the 50 QKV and 50 gate/up groups must each supply every shard exactly once.
This metadata projection does not prove a successful host load, prompt encode,
inference result or support for another checkpoint.
