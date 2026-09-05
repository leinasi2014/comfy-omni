# ComfyOmni contributor instructions

This file governs the independent ComfyOmni repository. Sibling legacy repositories and server
evidence are migration inputs, not part of this Git root unless a reviewed change imports them.

## Delivery

- Live tasks: [GitHub Issues](https://github.com/leinasi2014/comfy-omni/issues). Do not keep changing
  status, priority, dependencies, WIP, or acceptance decisions in repository Markdown.
- Integration target: protected `main` through a short-lived branch and pull request.
- Working agreement: [docs/development/delivery.md](docs/development/delivery.md).
- Branch, commit, review, and handoff rules: [CONTRIBUTING.md](CONTRIBUTING.md).
- Trusted checks are Docker-only:
  `./scripts/docker.sh docs 3.13`, `quality 3.10`, `quality 3.13`, and `package 3.12`.
- Definition of Done: the accepted behavior is on `main`, affected automated and representative
  server checks pass, documentation/contracts are current, and limitations are explicit.
- Executable behavior follows a frozen contract and RED -> GREEN -> REFACTOR. Preserve the failing
  RED observation and bind later evidence to the exact candidate.

## Parallel worktree location

- Create parallel Git worktrees only in this plugin's `.worktrees/<task>` directory. Do not create
  sibling development checkouts under `plugins/` or in the system temporary directory.
- Keep `.worktrees/` excluded from Git and Docker build contexts. Read-only investigation does not
  need a new worktree.
- The coordinator owns creation, relocation and cleanup. Pause affected writers before moving a
  worktree, preserve branches and uncommitted files, verify Git identity and status afterwards, and
  update active scripts and agent assignments to the new path.

## Required reading

- All work: [docs/development/docker-first.md](docs/development/docker-first.md).
- Architecture or migration: [docs/post-merge-refactoring-plan.md](docs/post-merge-refactoring-plan.md).
- Checkpoint, conversion, LoRA, runtime, or server work:
  [docs/testing/model-validation-baseline.md](docs/testing/model-validation-baseline.md).
- Read the nearest nested `AGENTS.md` in any legacy source repository before inspecting it.

## Non-negotiable boundaries

`DOCKER_FIRST_POLICY: v1`

- Docker is the execution boundary for project Python, tests, lint, builds, packaging, downloads,
  conversion, and inference. Missing Docker is an unavailable gate, never a host-Python fallback.
- Hosts may edit/read files and use Git/GitHub, Docker/Compose, SSH/SCP, and read-only diagnostics.
- Preserve the dependency direction `core -> domain/contracts/artifacts -> conversion/runtime ->
  application -> CLI/API/integrations`; lower layers never import higher layers or `public`.
- Keep the single distribution `comfy-omni`, package `comfy_omni`, CLI `comfy-omni`, and one
  `vllm_omni.general_plugins` entry point. Legacy wire identifiers remain stable until a reviewed,
  versioned compatibility migration authorizes change.
- Conversion is offline, bounded, streaming, and fail-closed. Sources are read-only; publication is
  staging-first, independently verified, and manifest-last. Never overwrite an existing artifact.
- Importing the package or plugin must not load Torch, FastAPI, vLLM, models, or checkpoint data.
- Migrated or derived code requires exact source commit/blob, license, attribution, ownership,
  characterization evidence, and distribution disposition before implementation enters a PR.
- Preserve dirty/untracked user files and sibling repositories. Never commit models, local evidence,
  caches, wheels, secrets, private infrastructure, or generated media.

## Handoff

Report files changed, exact Docker/server evidence, unavailable checks with reasons, public-contract
impact, provenance/licensing, deliberately untouched user work, and remaining risk. Do not claim a
branch, download, mock, or restarted-process observation as completion.
