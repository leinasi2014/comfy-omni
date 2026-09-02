# Contributing to ComfyOmni

Thank you for helping build ComfyOmni. The project is in an architecture-first migration stage, so
small, reviewable changes and explicit compatibility decisions matter more than rapid code movement.

## Required reading

- [`AGENTS.md`](AGENTS.md)
- [`docs/post-merge-refactoring-plan.md`](docs/post-merge-refactoring-plan.md)
- [`docs/development/docker-first.md`](docs/development/docker-first.md)
- The nearest nested `AGENTS.md` for any legacy source being inspected

## Docker-first development

`DOCKER_FIRST_POLICY: v1`

All project execution is containerized by default. Do not install project dependencies into the
host Python, a host virtual environment, a user site, or the host operating system. The host may use
Git/GitHub, Docker/Compose, SSH/SCP, editors, and read-only diagnostics needed to operate Docker.
Use the repository wrappers for checks:

```bash
./scripts/docker.sh docs 3.13
./scripts/docker.sh quality 3.10
./scripts/docker.sh quality 3.13
./scripts/docker.sh package 3.12
```

PowerShell users invoke the same actions through `scripts/docker.ps1`. A missing Docker daemon is an
unavailable gate, not permission to run `pip`, pytest, Ruff, builds, conversion, or models on the
host. Any necessary exception must satisfy the documented exception protocol before it runs.

## Branch workflow

1. Start from an up-to-date `main`.
2. Create a short-lived branch:
   - `feat/<topic>`
   - `fix/<topic>`
   - `refactor/<topic>`
   - `docs/<topic>`
   - `test/<topic>`
   - `build/<topic>`
   - `chore/<topic>`
3. Make focused commits using Conventional Commits, for example:
   - `docs(readme): add bilingual roadmap`
   - `refactor(contracts): separate snapshot persistence`
   - `fix(packaging): include mapping pack resources`
4. Push the branch and open a pull request against `main`.
5. Resolve review findings and required checks before merge.

Do not develop ordinary features directly on `main`. Do not force-push or rewrite shared `main`.

## Pull-request scope

A pull request should have one primary goal. Separate these when practical:

- pure file/directory movement;
- dependency inversion;
- behavior changes;
- public contract or schema changes;
- mechanical formatting;
- third-party code migration.

The PR description must include:

- problem and intended outcome;
- files/modules changed;
- architecture and dependency impact;
- public-contract compatibility impact;
- tests and exact results;
- migration and rollback plan;
- documentation and license/attribution impact.

## Bilingual documentation

`README.md` and `README.zh-CN.md` are synchronized public documents. If either changes public-facing
information, update the other in the same commit or pull request and run:

```bash
./scripts/docker.sh docs 3.13
```

The two documents may use natural wording in each language, but their section keys, milestone IDs,
status, features, commands, compatibility statements, and links must agree.

## Migrating legacy code

Do not bulk-copy the old plugin packages. For each bounded migration slice:

1. Record source repository, path, commit, license, and attribution.
2. Add characterization tests around the existing behavior.
3. Move one responsibility into its target module.
4. Keep wire formats, schemas, paths, error codes, and safety behavior unchanged unless the PR is an
   approved contract migration.
5. Remove the old implementation only after the new owner passes equivalent tests.

Code marked `verbatim`, `mechanically copied`, or `heritage` requires an explicit distribution
decision before it enters the public repository.

## Quality expectations

- Preserve offline conversion and fail-closed publication.
- Keep the plugin import lightweight and free of model I/O.
- Do not introduce new import cycles or public-facade back edges.
- Add tests before fixing bugs.
- Do not commit secrets, private infrastructure, model weights, build outputs, or local evidence.
- Keep model mounts read-only and write outputs/evidence only to separate bounded container mounts.
- Run every check available for the affected milestone and report any unavailable gate honestly.

## Security reports

Do not open a public issue containing credentials, private infrastructure, exploitable path details,
or sensitive checkpoint information. A dedicated security policy and private reporting channel will
be added before the first preview release; until then, do not publish sensitive details.
