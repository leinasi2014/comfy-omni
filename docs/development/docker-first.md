# Docker-first execution policy

`DOCKER_FIRST_POLICY: v1`

Docker is the default and authoritative execution boundary for ComfyOmni. This applies to
development commands, tests, linting, packaging, checkpoint inspection and conversion, model
downloads, generation, CI, and designated-server acceptance. Installing project dependencies on a
developer or server host is forbidden merely because a container image is not ready.

## Host allowlist

The host may perform only the work needed to create and control the container boundary:

- edit or inspect repository files;
- use Git and GitHub tooling for version control, review, and task state;
- use Docker Engine, BuildKit, and Docker Compose to build, inspect, start, stop, and remove bounded
  project containers, images, networks, and volumes;
- use SSH/SCP to transfer repository-owned orchestration scripts or artifacts and invoke Docker on
  a designated server;
- perform read-only operating-system, filesystem, storage, network, Docker-daemon, GPU-driver, or
  `nvidia-smi` diagnostics required to make container execution possible.

The host must not run project Python, `pip`, `uv`, Conda, pytest, Ruff, Twine, Torch, vLLM,
conversion code, model parsers, model downloads, or inference. Do not use `apt`, `dnf`, `yum`,
`brew`, `choco`, a system Python, a host virtual environment, or a user-site install for project
dependencies. Existing unrelated host environments are not ComfyOmni execution targets.

An exception is valid only when the operation cannot technically execute in a container and is
required to establish or repair the container boundary. Record the exact command, reason, target,
expected writes, rollback, and result in the live Issue or PR before execution. Convenience,
performance speculation, a missing image, or a failing container is not an exception. An exception
never authorizes installing or running ComfyOmni itself on the host.

## Repository images and commands

The root `Dockerfile` owns five reviewable targets:

| Target | Purpose | Network while running |
|---|---|---|
| `documentation` | README and Docker-policy contracts | none after image build |
| `quality` | Ruff, pytest, documentation, and policy gates | none after image build |
| `package-check` | sdist/wheel build, metadata checks, clean wheel install, CLI smoke | none after image build |
| `runtime` | minimal non-root `comfy-omni` CLI image | none by default |
| `numerics-runtime` | lazy-Torch conversion math on the vLLM integration image | none by default |

Image builds may use the network to obtain declared base images and dependencies. Running a built
quality, package, conversion, or runtime image is offline by default. A download-specific container
may have network access only for its declared source and writes only into its dedicated download
volume or staging directory.

Use the repository wrappers; they use host Git only to bind the image to the exact commit and dirty
state, then delegate project execution to Docker:

```bash
./scripts/docker.sh docs 3.13
./scripts/docker.sh quality 3.10
./scripts/docker.sh quality 3.13
./scripts/docker.sh package 3.12
./scripts/docker.sh cli 3.13 --help
./scripts/docker.sh numerics 3.13
```

```powershell
.\scripts\docker.ps1 docs 3.13
.\scripts\docker.ps1 quality 3.10
.\scripts\docker.ps1 quality 3.13
.\scripts\docker.ps1 package 3.12
.\scripts\docker.ps1 cli 3.13 --help
.\scripts\docker.ps1 numerics 3.13
```

A numerics image defaults to the same `vllm/vllm-openai:v0.27.0` release boundary currently pinned
by the adjacent vLLM-Omni integration checkout. A constrained registry may set
`COMFY_OMNI_NUMERICS_BASE_IMAGE` to a reviewed mirror of that exact tag. Its resolved image ID must
be recorded in acceptance evidence, and a release candidate must replace the tag with an immutable
digest.

A missing local Docker daemon makes the local gate unavailable; it does not authorize a host-Python
fallback. The same targets must then pass in trusted CI and, where required, on the designated
Docker server.

The default Python base registry is `docker.io/library`. A network-constrained environment may set
`COMFY_OMNI_PYTHON_REGISTRY` to a reviewed pull-through mirror prefix for a build without changing
the host daemon configuration. Record the mirror, resolved base image ID/digest, and reason in the
live evidence. A mirror does not make a mutable tag an acceptable release identity.

## Container safety contract

- Run application containers as a numeric non-root user, with all Linux capabilities dropped,
  `no-new-privileges`, a read-only root filesystem, a bounded temporary filesystem, and no network
  unless the specific lane requires it.
- Never mount the Docker socket, host root, home directory, SSH directory, package-manager caches,
  or unrelated model/data trees into a workload container. Do not use `--privileged`, host PID, or
  host IPC by default.
- Mount checkpoint sources and validation models read-only. Conversion output and evidence use a
  separate, explicitly named, bounded read-write path or volume. Source and output paths cannot
  overlap.
- GPU containers receive only the devices required by the acceptance case. Host driver checks are
  diagnostic evidence, not runtime acceptance.
- Secrets enter only through an approved runtime secret mechanism, never image layers, build args,
  repository files, logs, or evidence bundles.
- Pin production/runtime base images and integration dependencies before claiming a release
  candidate. Record image ID and content digest in server evidence.

## Explicit immutable storage administration

The opt-in package `reuse_immutable` mode is a metadata write operation, separate from ordinary
conversion and inference. Hardlinks may require a shared writable mount for the named related
component roots and a distinct private output subtree. A live Issue/PR must first bind that
specific operation, roots, copy budget, verification and rollback. Source payloads must have no
write permission and may only be opened read-only; no source chmod, replacement or payload write
is permitted. Use directory-descriptor operations and exclusive planned staging names. Record
the intentional link-count/ctime changes and actual shared/copied bytes. Cross-mount fallback
must fit its explicit copy allocation. The non-root, capability, network, evidence and container
execution requirements above still apply. Normal source/model mounts remain read-only.

See [immutable package reuse](../migration/immutable-package-reuse.md) for the API and invariant.

## CI and server acceptance

GitHub Actions may check out source, inspect Git state, and invoke Docker on the runner. It must not
set up Python or execute project tools on the runner. Docker targets are the only Python quality and
packaging authority.

Server acceptance uses a candidate image built from the exact source commit. Model byte size and
SHA256 are verified inside a container before inspection or load. Model roots are mounted
read-only; only a dedicated evidence directory is writable. Each receipt records the source commit,
dirty state, wheel SHA256, image ID/digest, Docker and GPU runtime versions, sanitized container
configuration, model/LoRA/VAE SHA256 values, commands, exit codes, timestamps, and resource
observations. LoRA lifecycle and full-DiT A → B → A switching must occur in the same declared host
process when that is the acceptance contract.

Temporary containers are removed after a run. Persistent evidence and deliberately cached model
assets are retained only in their declared directories or volumes; cleanup resolves and verifies
the exact target before removal.
