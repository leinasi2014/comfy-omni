# ComfyOmni

> ComfyUI model assets and composable generation capabilities for vLLM-Omni.

**English** · [简体中文](README.zh-CN.md)

<!-- README_SYNC: overview -->
## Overview

ComfyOmni is a vLLM-Omni plugin. Its first release targets reuse of existing ComfyUI/H3 model and
component files, with loading, unloading and switching in RAM/VRAM. LoRA composition, tools, node
workflows and other model families follow later and do not block this release. A new converted
checkpoint or assembled model directory is not a normal startup or model-switch prerequisite.

This is an independent project, not an official project of ComfyUI, vLLM or MiniMax.

<!-- README_SYNC: naming -->
## Naming

| Role | Name/path |
|---|---|
| Product | `ComfyOmni` |
| Repository and distribution | `comfy-omni` |
| Python package | `comfy_omni` |
| Source | `src/comfy_omni/` |
| CLI | `comfy-omni` |
| Host entry point group | `vllm_omni.general_plugins` |

<!-- README_SYNC: status -->
## Current capability

The migration is incomplete. Single plugin registration, component catalog/API and the audited
legacy H3 curve-cache compatibility path exist. That existing path has real-host parity evidence;
it remains the fixed working baseline. A beta4 DiT forward test does not establish complete
original-quantized-H3 loading or generation.

Direct loading of the existing original H3 quantized assets and runtime model switching still need
implementation and real-host acceptance. LoRA lifecycle, tools and nodes are deferred work.
The [user guide](docs/user-guide.md) separates current capabilities from targets.
Live progress and evidence belong in [Issue #4](https://github.com/leinasi2014/comfy-omni/issues/4).

<!-- README_SYNC: goals -->
## Design goals

- Reference existing ComfyUI component files and reuse one fixed model environment.
- Manage residency, request isolation, loading and recovery in RAM/VRAM.
- Support each format explicitly, using native quantization or bounded load-time memory adaptation.
- Preserve proven legacy behavior while delivering verified H3 slices.

In-memory LoRA composition, tools and node workflows are later extensions, not first-release gates.

<!-- README_SYNC: architecture -->
## Architecture

Application services coordinate source bindings, component loaders and model sessions.
The vLLM-Omni adapter connects those services to the pinned host. Existing exported
packages remain usable as one source kind; runtime no longer targets mandatory package creation.

See the [runtime architecture](docs/architecture/README.md) and
[H3-first refactoring plan](docs/post-merge-refactoring-plan.md). These describe the target;
they do not claim the refactor is already implemented.

<!-- README_SYNC: milestones -->
## Delivery sequence

| ID | User-visible result | Acceptance |
|---|---|---|
| H1 | Load existing component sources | Fixed H3 loading regression, preserved legacy behavior, no new full model copy |
| H2 | Reuse resident components and switch models | One control instance and existing workers perform A→B→A; identities, cache invalidation, recovery and resource release are verified |
| H3 | Validate and deliver on the real host | Fixed-asset loading, generation and normal worker switching pass; affected software gates pass and the candidate reaches main |

This is a dependency sequence, not a live status table. The
[fixed model validation baseline](docs/testing/model-validation-baseline.md) defines assets and
verification boundaries. Ordinary CI never downloads model payloads.
Worker reconstruction is a reported recovery or fallback path, not successful normal hot loading.
LoRA composition, tools and node workflows are outside this first-release sequence.

<!-- README_SYNC: layout -->
## Repository layout

```text
src/comfy_omni/    plugin, contracts, loaders, runtime and application code
tests/            unit, contract, integration, packaging and host checks
docs/             current design, usage, testing and source attribution
scripts/          container execution and repository checks
.worktrees/       ignored development isolation within this plugin
```

<!-- README_SYNC: development -->
## Development

`DOCKER_FIRST_POLICY: v1`

Read [AGENTS.md](AGENTS.md), [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Docker execution policy](docs/development/docker-first.md). Project Python, tests and model
execution run inside Docker; hosts edit files and operate Git, Docker and SSH.

```bash
./scripts/docker.sh docs 3.13
./scripts/docker.sh quality 3.10
./scripts/docker.sh quality 3.13
./scripts/docker.sh package 3.12
```

PowerShell uses `scripts/docker.ps1`. Real runtime regressions reuse existing read-only model
mounts across branches. Use small fixtures for numerical and failure tests; reuse valid evidence
when the relevant code and inputs have not changed. Do not create new model copies for code tests.

<!-- README_SYNC: contributing -->
## Contributing

Use short-lived branches and focused pull requests against protected `main`. Keep English and
Chinese READMEs synchronized. See [CONTRIBUTING.md](CONTRIBUTING.md).

<!-- README_SYNC: license -->
## License

[Apache License 2.0](LICENSE). Migrated code and assets retain their recorded attribution and
applicable license terms.
