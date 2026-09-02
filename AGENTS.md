# ComfyOmni contributor instructions

This file governs the independent ComfyOmni repository rooted at the directory containing this
file. Legacy plugin repositories and local evidence are siblings outside this Git root; they are
migration inputs, not part of the new package unless a reviewed migration explicitly imports them.

## Delivery

- Live tasks: [GitHub Issues](https://github.com/leinasi2014/comfy-omni/issues).
- Integration target: protected `main`; deliver changes through a short-lived branch and pull request.
- Trusted checks: the commands under “Tests and quality gates”; runtime acceptance runs only on its
  declared server against an identified candidate.
- Definition of Done: behavior is present on `main`, affected automated and representative server
  checks pass, public contracts and documentation are current, and limitations are explicit.
- Architecture contract and TDD: freeze the boundary in the live issue; executable behavior normally
  follows RED → GREEN → REFACTOR. A maintainer-approved deferred test must stay open and cannot be
  represented as a pass.

## Project identity

- Project: `ComfyOmni`
- GitHub repository: `comfy-omni`
- PyPI distribution: `comfy-omni`
- Python package: `comfy_omni`
- CLI: `comfy-omni`
- Tagline: `Bring Comfy checkpoints to native Omni runtimes.`
- New source root: `src/comfy_omni/`
- Design authority during the refactor: `docs/post-merge-refactoring-plan.md`
- Validation-model authority: `docs/testing/model-baseline.v1.json` with the interpretation in
  `docs/testing/model-validation-baseline.md`

The tagline describes product direction. A runtime is supported only after its own adapter and
real-host acceptance pass. The first planned adapter is the `UPSTREAM.toml`-pinned vLLM-Omni
integration inherited from `h3-forge`.

## Before changing code

1. Read `docs/post-merge-refactoring-plan.md` and the nearest applicable `AGENTS.md`. For checkpoint,
   conversion, LoRA, runtime, or host work, also read `docs/testing/model-validation-baseline.md`.
2. Classify the change as `core`, `domain`, `artifact I/O`, `contract`, `conversion`, `runtime`,
   `application`, `integration`, `API`, `CLI`, `validation`, packaging, or documentation.
3. List affected public contracts: Python imports, CLI, entry points, HTTP paths, environment
   variables, runtime architecture keys, JSON schemas, artifact schemas, and error codes.
4. For migrated code, record its source path, commit, license/attribution disposition, new owner,
   characterization tests, and rollback/removal plan before copying implementation.
5. Keep directory moves, dependency inversion, behavior changes, and contract changes in separate
   commits or PRs whenever they can be reviewed independently.

Do not edit or delete sibling repositories, archives, worktrees, model artifacts, evidence, build
outputs, or dirty/untracked files unless the task explicitly names them.

## Required architecture

The intended dependency direction is:

```text
CLI / API / runtime integrations
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

Rules:

- `core` depends only on the Python standard library.
- `domain` depends only on `core`; it does not perform I/O or import Torch, FastAPI, vLLM, or
  vLLM-Omni.
- `contracts` may depend on `core` and `domain`; it does not read files or environment variables.
- `artifacts` owns filesystem access, strict JSON, hashing, safetensors metadata I/O, provenance,
  staging, and atomic publication.
- `conversion` is offline and may depend on `core`, `domain`, `contracts`, and `artifacts`. It does
  not depend on `runtime`, API, CLI, or a host integration.
- `runtime` does not import conversion, FastAPI, argparse, or CLI modules.
- `application` orchestrates use cases shared by CLI, API, and integrations. Lower layers never
  import it.
- `integrations/vllm_omni` is the only owner of direct `vllm` and `vllm_omni` imports.
- `api` maps HTTP schemas and errors; `cli` parses and renders. Neither contains business rules.
- `validation` owns compatibility status, preflight, parity, and release evidence.
- Internal modules do not import `comfy_omni.public`; the public facade is for external consumers.
- Delayed imports may preserve a process boundary, but may not conceal a forbidden reverse
  dependency or cycle.

New responsibilities go into the directory that owns them. Do not add new responsibilities to a
large legacy module merely because it is nearby.

## Plugin and host-integration invariants

- The eventual `comfy_omni.plugin:register` entry must be lightweight, idempotent, thread-safe,
  re-entry-safe, and free of checkpoint/model I/O.
- Importing the package or plugin must not import FastAPI, Torch, a vLLM pipeline, or model code.
- Contributions use lazy dotted references and are resolved only in the correct host process and
  bootstrap phase.
- Duplicate profile, route, runtime architecture, or patch ownership fails closed unless an
  explicit replacement contract identifies the previous owner.
- Profile registration, runtime registration, import-hook arming, host patching, and API mounting
  are separate observable stages. A retry repeats only incomplete stages.
- Any host monkeypatch must live in a reviewed patch registry with an owner, pinned target shape,
  version guard, process scope, idempotency test, failure behavior, and removal condition.
- Runtime workers never scan checkpoint directories, mount HTTP routes, or perform offline
  conversion.

## Safety and artifact rules

- Checkpoint conversion is offline, bounded, and streaming; never mutate a source checkpoint.
- Write to a distinct staging directory and publish only after structural, digest, and contract
  verification.
- Reject source/output collisions, path traversal, symlink/reparse/junction escapes, duplicate JSON
  keys, unknown required schema fields, and overwrite of an existing publication.
- Preserve fail-closed behavior. Do not introduce silent fallback, best-effort format guessing, or
  implicit quantization/dequantization.
- Domain and planning steps should be pure or read-only. A single publication owner commits final
  output.
- Secrets, tokens, private endpoints, internal hostnames, full sensitive paths, checkpoint content,
  and unredacted requests must not enter logs, fixtures, docs, or Git history.

## Public contracts and naming

The following identities are fixed for the new project: `ComfyOmni`, `comfy-omni`, `comfy_omni`,
and the `comfy-omni` command. Do not rename stable wire contracts merely for branding.

Legacy HTTP routes, `H3_FORGE_*` variables, runtime architecture keys, error schemas such as
`h3_forge.error/v1`, and artifact schema identifiers remain compatibility contracts until an ADR,
consumer inventory, versioned migration, and regression tests authorize a change. The
`vllm_omni.general_plugins` entry-point key is also a consumer-facing selector and must follow the
Phase 0 decision in the refactoring plan.

Public Python API is limited to explicitly documented exports. Do not expose private validators,
mutable registries, FastAPI routers, host subclasses, or test seams through `comfy_omni.public`.

## Code standards

- Support Python `>=3.10,<3.14` until project metadata changes through a reviewed contract update.
- Ruff formatter is the formatting authority. Ruff lint owns `E`, `F`, `I`, `UP`, and `B` rules.
- Do not mix unrelated repository-wide formatting with a functional change.
- New production Python files should remain at or below 600 lines. Existing files above the legacy
  ceiling are only allowed to shrink.
- New or materially changed functions should remain at or below 80 lines and cyclomatic complexity
  15. Split responsibilities instead of suppressing the limit.
- New public APIs, application services, protocols, and dataclasses require complete parameter and
  return annotations.
- Parse boundary JSON into typed models. Do not propagate unconstrained `dict[str, Any]` into
  domain or runtime code.
- Prefer frozen dataclasses and immutable collections for value objects, plans, contracts, and
  receipts.
- Each top-level module documents its responsibility, allowed dependencies, and forbidden
  dependencies. Comments explain why, not what the next line says.
- Production code uses module loggers. Only CLI rendering may use `print`.
- Do not swallow exceptions. Boundary translation preserves the cause and maps it to a stable error
  kind/code. Never use `except: pass` or an untested broad fallback.
- Use `pathlib.Path` and the shared artifact primitives for filesystem work.
- Package version has one source in distribution metadata and is read with `importlib.metadata`.

## Tests and quality gates

Every bug fix starts with a failing regression test. Wire, schema, path, and publication changes
cover positive cases plus malformed, missing, duplicate, tampered, collision, and interrupted cases.

As the corresponding tooling lands, the authoritative local checks are:

```bash
python -m ruff format --check src tests scripts deploy
python -m ruff check src tests scripts deploy
python -m pytest -q --strict-markers
python scripts/check_release.py
```

`check_release.py` must build in an empty temporary directory, run metadata checks, rebuild a wheel
from the sdist, install wheels into clean environments, and validate CLI, imports, entry points,
licenses, and package resources. It must not consume or delete old repository `dist/` or `build/`
contents.

Missing Ruff, pytest, build, twine, or a required test dependency is a failed gate, not
`NOT_CONFIGURED`. GPU and pinned-host acceptance run in their declared environments and remain
bound to the exact source commit and artifact digests.

Ordinary CI validates the model-baseline contract but never downloads model payloads. A host run
must verify exact byte size and SHA256 before inspection or load. Asset presence, a single-model
generation, a LoRA rejection, or two loads separated by a process restart must not be reported as
full runtime, LoRA-activation, or hot-swap acceptance.

Before handing off a change, report:

- files changed and any user-owned files deliberately left untouched;
- checks run, exact results, and checks not run with a reason;
- public-contract impact and compatibility decision;
- remaining migration risk or follow-up work.

## Git and open-source hygiene

- Preserve dirty and untracked user files. Never use destructive reset/checkout to clean a tree.
- Do not commit caches, wheels, build directories, model weights, local evidence dumps, private
  runbooks, or generated media.
- New third-party or mechanically derived code requires source commit, license, attribution, and
  distribution disposition before it enters the public candidate.
- Keep the existing private remotes and histories intact. Publish through an audited public mirror
  or export repository; never repoint a private remote and push unreviewed history directly.
- A PR should have one primary structural goal. If it changes directory layout, behavior, wire
  contracts, and artifact schema together, split it unless an approved design proves they are
  inseparable.

### Git workflow

- `main` is the protected integration branch. After the repository bootstrap commit, normal
  development goes through a short-lived branch and pull request; do not push feature work directly
  to `main`.
- Branch names use `feat/`, `fix/`, `refactor/`, `docs/`, `test/`, `build/`, or `chore/` followed by a
  short kebab-case description.
- Commit subjects follow Conventional Commits: `type(scope): imperative summary`. Keep commits
  atomic and do not mix generated artifacts or unrelated formatting into them.
- Rebase or merge the latest `main` before final review according to repository policy. Never
  rewrite shared `main`, delete a remote branch owned by someone else, or force-push without an
  explicit recovery reason.
- Every pull request states the problem, scope, public-contract impact, tests, migration/rollback,
  and documentation impact. Complete the repository PR checklist before merge.
- Required checks must pass. A skipped check needs a written, technically specific reason and may
  not be represented as a pass.

### Bilingual README synchronization

`README.md` (English) and `README.zh-CN.md` (Simplified Chinese) are equal public entry points. Any
change to positioning, status, features, architecture, milestones, installation, commands,
compatibility, contribution links, or license wording must update both files in the same commit or
pull request. Keep their `README_SYNC` section keys and milestone IDs in the same order, and run:

```bash
python scripts/check_readme_sync.py
```

Pure grammar fixes may differ in wording, but neither language may advertise a capability, status,
or schedule that the other omits.
