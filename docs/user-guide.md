# ComfyOmni user guide

> A vLLM-Omni plugin for using existing ComfyUI H3 assets.

<!-- guide-parity -->
> The English and Chinese user guides are synchronized. They state the same product facts and links; only the prose language differs.

## What it is

ComfyOmni is a single `vllm_omni.general_plugins` plugin. Its first release targets loading, unloading and switching the H3 model and component files already present in a ComfyUI installation. It uses RAM/VRAM and does not require an offline conversion run, a new BF16 model copy or a complete package build. LoRA composition, tools and node workflows follow later and do not block the first release.

It is an independent Apache-2.0 project and is not an official ComfyUI, vLLM, or MiniMax project.

## Current use

The plugin entry point is `comfy_omni.plugin:register`. It registers lazily when vLLM-Omni is already loaded; it does not start a server, download assets, create a model copy, or load weights during import. See the [bootstrap record](migration/vllm-omni-bootstrap-e9cb011.md).

Current code provides one plugin integration, the component-directory path used to describe existing H3 assets, and the audited legacy H3 v3 curve-cache recipe route. The legacy route is the verified compatibility path for its documented input layout; it does not make conversion or packaging part of ordinary serving. Direct loading of existing H3 raw quantized A-format weights is a separate, unverified path. See the [H3 cache-runtime record](migration/h3-cache-runtime-e9cb011.md).

The target uses model and component files already present in the configured ComfyUI installation. Keep them read-only during deployment validation. Loading and switching manage RAM/VRAM within one control-service instance and its existing workers, reusing unchanged components. Worker reconstruction is a reported recovery or fallback path and does not count as normal hot-loading acceptance.

## First-release scope

The following work is the intended H3-first product, but is not yet a user-facing command or supported deployment contract:

| Area | Status |
|---|---|
| H1: Direct loading of existing H3 original files | Implementation and acceptance remain incomplete |
| H2: RAM/VRAM residency, component reuse and A → B → A in existing workers | Not yet accepted |
| H3: Real-host loading, generation, switching and delivery | Legacy compatibility has evidence; the new direct-loading path is not yet accepted |

Do not infer an HTTP route, CLI command, environment variable, model format, or switching workflow from this table. The normal workflow remains limited to the capabilities that have been integrated and verified for the selected host.

## Later extensions

Complete in-memory LoRA composition, H3 tools and typed node workflows are deferred. They are not first-release acceptance requirements, and this guide does not claim that arbitrary ComfyUI workflows or third-party nodes are supported.

## Boundaries

The repository does not redistribute model weights, LoRA payloads, generated packages or server evidence. Retained code ownership and licensing are recorded in [source attribution](migration/source-attribution.md); they are not model-preparation instructions.

For development and host validation rules, see [Docker-first development](development/docker-first.md) and the [model-validation baseline](testing/model-validation-baseline.md).
