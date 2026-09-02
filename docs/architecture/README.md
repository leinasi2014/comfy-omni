# ComfyOmni architecture diagrams

The target diagram is a design contract, not an implementation-complete claim. Its status cards
separate the M0 capabilities already proved from the conversion, runtime, LoRA, hot-swap, and GPU
work that remains.

- [Interactive target architecture](comfy-omni-target.html)
- [Version-controlled Archify specification](comfy-omni-target.architecture.json)
- [Static light preview](comfy-omni-target.visual-check.2048x1320.light.png)
- [Static dark preview](comfy-omni-target.visual-check.2048x1320.dark.png)
- [Visual-check receipt](comfy-omni-target.visual-check.json)

Evidence inputs are the latest consolidated legacy source
`h3-forge@e9cb011d00b028c149db3978de246c54f6e34acc`, the target ownership rules in
[`../post-merge-refactoring-plan.md`](../post-merge-refactoring-plan.md), and the ComfyOmni source
revision pinned inside the diagram specification. The primary dependency direction is:

`public edges -> application -> offline conversion -> validation -> publication -> immutable package`

Runtime loading is downstream of the immutable package. Host-specific behavior remains behind
`integrations/vllm_omni`, and only publication may commit a final package directory.

Archify delivery validation passed all 9 showcase checks with 0 errors and 0 warnings. Automated
containment/readability checks passed at 1440x900, 1600x1000, 1920x1080, and 2048x1320 in light and
dark capture modes; the generated screenshots were also visually reviewed for legibility, route
clarity, boundary placement, and status-card truthfulness.
