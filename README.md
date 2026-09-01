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

**Early refactoring / repository-foundation stage.** The repository currently contains the approved
architecture skeleton, development rules, and refactoring plan. It does not yet provide an
installable `comfy-omni` release or a functional CLI. Do not use current scaffolding as a production
runtime or compatibility claim.

The existing H3 implementation remains in the legacy workspace while code origin, licensing,
contracts, tests, and module ownership are audited before migration.

<!-- README_SYNC: goals -->
## Design goals

- Inspect checkpoint structure without loading model payloads into GPU memory.
- Convert unsupported representations offline through explicit, versioned profiles.
- Preserve tensor bytes when the target runtime supports the same native representation.
- Package outputs immutably with hashes, provenance, schema validation, and fail-closed publication.
- Keep conversion, dequantization, and mapping work out of the inference hot path.
- Isolate runtime-specific code behind integrations such as `integrations/vllm_omni`.
- Provide one consistent Python API, CLI, error model, test baseline, and release process.

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

The intended source layout lives under [`src/comfy_omni`](src/comfy_omni). Internal modules must
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
| M6 | Runtime acceptance | Pinned vLLM-Omni adapter passes package, load, request, parity, and fail-closed acceptance gates | Planned |
| M7 | Open-source preview | License-cleared `0.2.0a1` or `0.2.0b1` release with synchronized docs and reproducible artifacts | Planned |

Detailed sequencing and exit criteria are maintained in the
[post-merge refactoring and open-source plan](docs/post-merge-refactoring-plan.md).

<!-- README_SYNC: layout -->
## Repository layout

```text
.
├── AGENTS.md                 # Architecture, coding, testing, and Git rules
├── CONTRIBUTING.md           # Contribution and pull-request workflow
├── README.md                 # English project overview
├── README.zh-CN.md           # Simplified Chinese project overview
├── docs/                     # Design, ADRs, and public evidence indexes
├── scripts/                  # Repository checks
├── src/comfy_omni/           # New modular Python package skeleton
└── tests/                    # Unit, contract, integration, packaging, and host lanes
```

Legacy repositories and local evidence remain outside this independent Git root. They are not part
of the public ComfyOmni repository and may enter it only through an audited migration.

<!-- README_SYNC: development -->
## Development

Before making changes, read:

1. [`AGENTS.md`](AGENTS.md)
2. [`CONTRIBUTING.md`](CONTRIBUTING.md)
3. [`docs/post-merge-refactoring-plan.md`](docs/post-merge-refactoring-plan.md)

The repository is not installable yet. During the foundation milestone, the available documentation
check is:

```bash
python scripts/check_readme_sync.py
```

Language, lint, test, package, and host gates will become mandatory as their corresponding milestone
lands. Missing required tooling is a failure, not a fabricated pass.

<!-- README_SYNC: contributing -->
## Contributing

Use a short-lived branch, Conventional Commits, and a focused pull request. Update both README files
when public-facing information changes. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the complete
workflow and quality checklist.

<!-- README_SYNC: license -->
## License

ComfyOmni is licensed under the [Apache License 2.0](LICENSE). Migrated third-party code, fixtures,
and assets remain subject to their recorded attribution and compatible license terms.
