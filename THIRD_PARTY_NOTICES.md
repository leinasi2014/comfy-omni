# Third-party notices

ComfyOmni is distributed under the root Apache-2.0 LICENSE. Existing migrated
h3-forge code retains its per-file source and attribution notices.

The bounded text-encoder decoder in
`src/comfy_omni/conversion/numerics/te_nvfp4.py` adapts numerical and scale-layout
behavior from [Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen)
commit `b678fdf63378409676aa5596721445d33794d0ea`:

- `comfy_kitchen/backends/eager/quantization.py`, Git blob
  `1df1514e38216b0deeb1977075a187fdda5886ad`.
- `comfy_kitchen/float_utils.py`, Git blob
  `29077a7b5375a596eab64bab449bfc2e842beb9d`.

Copyright (c) 2025 Comfy Org. All rights reserved. Licensed under Apache-2.0.
The upstream files also identify portions derived from PyTorch AO, copyright
Meta Platforms, Inc. and affiliates, under BSD-3-Clause. The complete upstream
[LICENSE](third_party/comfy-kitchen/LICENSE) and
[NOTICE](third_party/comfy-kitchen/NOTICE), including that BSD text, are retained
unmodified and included as license files in wheels and source distributions.

The independent scalar verifier uses a separate standard-library expression of
the characterized format and rounding contract; it imports neither this backend
nor any upstream numerical implementation. No GPL ComfyUI or writer code is
copied or executed by the decoder or verifier. Reference identities are recorded in
[the source attribution](docs/migration/source-attribution.md).
