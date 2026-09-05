# Source attribution

This record contains attribution for retained source code. It specifies no model preparation or
runtime procedure. Product behavior is defined by the [H3 refactoring plan](../post-merge-refactoring-plan.md).

- H3 numerical and file-handling references: Apache-2.0 `h3-forge` commit
  `e9cb011d00b028c149db3978de246c54f6e34acc`, attributed to h3-forge contributors.
  Relevant blobs include `native_export.py` `475cee5523be64e5b24a95e16c5de3f371cbdf67`
  and `convrot.py` `8b4b9eebacd8bdaf64b251d5635b0147e7d790db`.
  Exact per-module use is recorded in retained source headers.
- Beta4 source SHA256: `54d56b15c65923b54c9ca16b494dae641bfe9455cfcb1c19c49b1008e270bbc1`;
  descriptor schema: `ae2456bc6ac904929a4b773f703f8a1baa99b6356b5a389994faf64a1a2d80f2`.
- Text-encoder numerical reference: `comfy-kitchen` commit
  `b678fdf63378409676aa5596721445d33794d0ea`, eager quantization blob
  `1df1514e38216b0deeb1977075a187fdda5886ad`, float-utility blob
  `29077a7b5375a596eab64bab449bfc2e842beb9d`. Its license and distribution boundaries remain
  in [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md); ComfyOmni code is Apache-2.0.

No model payload or generated package is distributed by this repository.
