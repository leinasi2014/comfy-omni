# ComfyOmni handoff

This is a stable orientation document, not a task board or a runtime-evidence
record. Git history preserves prior handoffs. Current status, priority,
ownership, dependencies, acceptance decisions, and blockers live only in
[GitHub Issues](https://github.com/leinasi2014/comfy-omni/issues) and the
associated pull requests.

## Project identity

- Project: `ComfyOmni`
- Repository: <https://github.com/leinasi2014/comfy-omni>
- Distribution: `comfy-omni`
- Python package: `comfy_omni`
- CLI: `comfy-omni`
- Tagline: *Bring Comfy checkpoints to native Omni runtimes.*

## Non-negotiable constraints

- The integration target is protected `main`; all changes use a short-lived
  branch, pull request, required Docker checks, squash merge, and `main`
  read-back.
- Docker is the only execution boundary for project Python, tests, builds,
  downloads, checkpoint processing, conversion, and inference. A missing local
  Docker daemon makes that local gate unavailable; it never authorizes a host
  Python fallback.
- Sources are read-only. Conversion and package publication are bounded,
  staging-first, independently verified, manifest-last, and never overwrite an
  existing artifact.
- Do not commit model files, caches, wheels, generated media, server evidence,
  private infrastructure, credentials, or proxy configuration.
- The package and plugin import paths stay lightweight: no Torch, FastAPI,
  vLLM, model, or checkpoint load during import.
- Preserve the dependency direction:
  `core -> domain/contracts/artifacts -> conversion/runtime -> application -> CLI/API/integrations`.

The authoritative details are in [AGENTS.md](AGENTS.md),
[CONTRIBUTING.md](CONTRIBUTING.md),
[Docker-first policy](docs/development/docker-first.md), and the
[delivery agreement](docs/development/delivery.md).

## Legacy migration authority

The only legacy implementation input is `h3-forge` at commit
`e9cb011d00b028c149db3978de246c54f6e34acc` under Apache-2.0. Before a legacy
slice enters a pull request, record its exact source path/blob, license,
attribution, retained behavior, characterization evidence, owner, and public
distribution disposition. Do not treat newer, uncommitted, or other-branch
legacy code as authority.

## Start or resume a slice

1. Read the repository instructions and fetch the protected target.
2. Check Issues and open PRs; finish an accepted candidate before starting new
   work.
3. Freeze one small contract in its Issue: outcome, non-goals, scenario,
   ownership, failure semantics, compatibility, limits, and acceptance evidence.
4. Follow `RED -> GREEN -> REFACTOR`, using the repository Docker gates and the
   applicable representative server boundary.
5. Record candidate-bound evidence in the PR, merge only after the required
   checks, then reread `main` and its push checks.

For model, runtime, LoRA, or server work, start with
[the model validation baseline](docs/testing/model-validation-baseline.md).
That document and its machine-readable JSON define the test assets; model
presence, a download, a branch, or a CI-only result never proves runtime
compatibility.
