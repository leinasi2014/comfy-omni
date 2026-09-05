# ComfyOmni

> Bring Comfy checkpoints to native Omni runtimes.

**English** · [简体中文](README.zh-CN.md)

<!-- README_SYNC: overview -->
## Overview

ComfyOmni is an open-source bridge for inspecting, converting, packaging, and validating checkpoints
from the Comfy ecosystem for native Omni runtimes. It keeps conversion offline and aims to produce
immutable, verifiable runtime packages instead of teaching inference workers to parse arbitrary
Comfy checkpoints at startup.

The project is being rebuilt from the consolidated `h3-forge` codebase under one distribution, one
Python package, one CLI, and one runtime-plugin entry. The first planned and validated integration is
the pinned vLLM-Omni host; additional runtimes must provide their own adapters and acceptance
evidence.

ComfyOmni is an independent open-source project. It is not an official project of ComfyUI, Comfy.org,
vLLM, MiniMax, or their respective maintainers unless explicitly stated otherwise.

<!-- README_SYNC: naming -->
## Naming and source layout

| Role | Name/path |
|---|---|
| Product and documentation title | `ComfyOmni` |
| GitHub repository and clone directory | `comfy-omni` |
| PyPI distribution | `comfy-omni` |
| CLI command | `comfy-omni` |
| Python import package | `comfy_omni` |
| Importable source path | `src/comfy_omni/` |

A clone therefore has the path `comfy-omni/src/comfy_omni`. The repository name is not repeated
inside the repository: `src/` separates importable code from project files, while `comfy_omni` uses
an underscore because Python import packages should be valid identifiers. GitHub may compact a
single-child directory chain and display `src/comfy_omni` as one row.

This follows the Python Packaging User Guide's
[src-layout guidance](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)
and its distinction between
[distribution and import packages](https://packaging.python.org/en/latest/discussions/distribution-package-vs-import-package/).

<!-- README_SYNC: status -->
## Project status

**Pre-alpha migration.** The repository includes lightweight package/plugin registration,
strict checkpoint inspection, pinned text-encoder normalization, immutable source contracts,
bounded offline ConvRot export, and native package assembly and verification. The H3 integration
also contains a [curve-cache compatibility adapter](docs/migration/h3-cache-runtime-e9cb011.md)
for the explicitly audited `h3-forge@e9cb011` v3 recipe, alongside the existing native v3 layout.
The adapter preserves legacy artifacts and wire identifiers and separates artifact producer
identity from the executing ComfyOmni wheel.

Capabilities require their own representative host acceptance. Current acceptance and release
decisions live in [the runtime issue](https://github.com/leinasi2014/comfy-omni/issues/11) and
[the project epic](https://github.com/leinasi2014/comfy-omni/issues/4). HTTP API migration, additional
runtime profiles, LoRA lifecycle and complete DiT switching are separate work; this candidate
does not claim broad runtime compatibility or completion of the preview.

<!-- README_SYNC: goals -->
## Design goals

- Inspect checkpoint structure without loading model payloads into GPU memory.
- Convert unsupported representations offline through explicit, versioned profiles.
- Preserve tensor bytes when the target runtime supports the same native representation.
- Package outputs immutably with hashes, provenance, schema validation, and fail-closed publication.
- Keep conversion, dequantization, and mapping work out of the inference hot path.
- Isolate runtime-specific code behind integrations such as `integrations/vllm_omni`.
- Provide one consistent Python API, CLI, error model, test baseline, and release process.
- Prove LoRA compatibility before runtime mutation and fail closed for unsupported base/adapter pairs.
- Demonstrate full-DiT A-to-B-to-A hot switching in one host process with correct model identity,
  cache invalidation, failure recovery, and resource reclamation.

<!-- README_SYNC: architecture -->
## Architecture

```text
CLI / HTTP API / runtime integrations
                 |
                 v
            application
             /        \
     conversion      runtime
             \        /
          artifacts / contracts / domain
                       |
                       v
                      core
```

The intended source layout lives under [`src/comfy_omni`](src/comfy_omni). The
[validated target architecture](docs/architecture/README.md) shows the dependency and publication
boundaries while distinguishing proven M0 capability from planned work. Internal modules must
follow the dependency direction above; the public facade is for external consumers and must not be
used as an internal dependency shortcut.

<!-- README_SYNC: milestones -->
## High-level milestones

| ID | Milestone | Outcome | Status |
|---|---|---|---|
| M0 | Repository foundation and public audit | Independent repository, bilingual documentation, consumer inventory, license/history audit, and frozen baseline | In progress |
| M1 | Trustworthy release gates | Full pytest collection, Ruff, reproducible sdist/wheel builds, package-resource checks, and clean-install smoke tests | Planned |
| M2 | Atomic ComfyOmni migration | Distribution, import package, CLI, plugin target, and authority documents move together without unapproved wire changes | Planned |
| M3 | Single bootstrap and dependency direction | One lazy, idempotent plugin bootstrap; no plugin recursion, public-facade back edges, or import cycles | Planned |
| M4 | Conversion modularization | Inspection, contracts, mapping, exporters, LoRA conversion, packaging, and publication have explicit owners | Planned |
| M5 | Runtime modularization | Runtime services are separated from offline conversion and vLLM-Omni host subclasses | Planned |
| M6 | Runtime acceptance | Pinned vLLM-Omni adapter and digest-bound model suite pass package, load, request, LoRA preflight/lifecycle, full-DiT A-to-B-to-A hot-swap, parity, and fail-closed gates | Planned |
| M7 | Open-source preview | License-cleared `0.2.0a1` or `0.2.0b1` release with synchronized docs and reproducible artifacts | Planned |

Detailed sequencing and exit criteria are maintained in the
[post-merge refactoring and open-source plan](docs/post-merge-refactoring-plan.md).
The external test assets and evidence rules are frozen in the
[model validation baseline](docs/testing/model-validation-baseline.md); model payloads are never
stored in this repository or downloaded by ordinary CI.

<!-- README_SYNC: layout -->
## Repository layout

```text
.
├── AGENTS.md                 # Architecture, coding, testing, and Git rules
├── CONTRIBUTING.md           # Contribution and pull-request workflow
├── Dockerfile                # Development, quality, package, and CLI image targets
├── pyproject.toml             # Distribution metadata, CLI, plugin entry, and tool configuration
├── README.md                 # English project overview
├── README.zh-CN.md           # Simplified Chinese project overview
├── docs/                     # Design, ADRs, and public evidence indexes
├── scripts/                  # Repository checks
├── src/comfy_omni/           # New modular Python package
└── tests/                    # Unit, contract, integration, packaging, and host lanes
```

Legacy repositories and local evidence remain outside this independent Git root. They are not part
of the public ComfyOmni repository and may enter it only through an audited migration.

<!-- README_SYNC: development -->
## Development

`DOCKER_FIRST_POLICY: v1`

Before making changes, read:

1. [`AGENTS.md`](AGENTS.md)
2. [`CONTRIBUTING.md`](CONTRIBUTING.md)
3. [`docs/post-merge-refactoring-plan.md`](docs/post-merge-refactoring-plan.md)
4. [`docs/development/docker-first.md`](docs/development/docker-first.md)

Docker is the authoritative execution boundary. Do not install ComfyOmni, Python dependencies, or
model/runtime dependencies on the host. The host is limited to editing, Git/GitHub, Docker/Compose,
SSH/SCP, and diagnostics required to operate the container boundary. Run repository gates and the
current CLI through the provided wrapper:

```bash
./scripts/docker.sh docs 3.13
./scripts/docker.sh quality 3.10
./scripts/docker.sh quality 3.13
./scripts/docker.sh package 3.12
./scripts/docker.sh cli 3.13 --help
```

PowerShell users use `scripts/docker.ps1` with the same actions. Model/checkpoint commands use the
built image with explicit read-only input mounts and a separate bounded output/evidence mount, as
defined by the Docker-first policy. Missing local Docker is an unavailable local gate, not
permission to fall back to host Python; trusted CI and designated-server Docker evidence remain
required.

After the required mounts are declared, these are commands *inside* the container, never host-shell
installation or execution instructions:

```text
comfy-omni inspect CHECKPOINT.safetensors --json
comfy-omni normalize text-encoder SOURCE.safetensors DERIVED.safetensors --json
comfy-omni contract scan SOURCE.safetensors --json
comfy-omni contract draft SOURCE.safetensors --generated-by OPERATOR -o DRAFT.json
comfy-omni contract pin DRAFT.json --name PROFILE --reviewer REVIEWER \
  --evidence REVIEW.md --contract-dir CONTRACTS
comfy-omni contract list --contract-dir CONTRACTS --json
```

This exposes project identity, strict header-only inspection, the
[one exact normalization profile](docs/migration/text-encoder-normalization.md), and the
[audited immutable contract workflow](docs/migration/contract-workflows-e9cb011.md). Contract
commands hash source files but do not materialize tensor payloads or import Torch/vLLM. External
contracts are visible only when a caller passes a store explicitly; the legacy environment variable
is read only by the CLI compatibility boundary. General legacy conversion/runtime commands remain
unavailable. Fast deterministic repository checks run in Docker locally and in CI; GPU and runtime
acceptance runs only in Docker on the designated server against digest-bound assets. A deferred,
missing, or differently targeted check is not a pass.

<!-- README_SYNC: contributing -->
## Contributing

Use a short-lived branch, Conventional Commits, and a focused pull request. Update both README files
when public-facing information changes. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the complete
workflow and quality checklist.

<!-- README_SYNC: license -->
## License

ComfyOmni is licensed under the [Apache License 2.0](LICENSE). Migrated third-party code, fixtures,
and assets remain subject to their recorded attribution and compatible license terms.
