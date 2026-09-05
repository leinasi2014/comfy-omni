# ComfyOmni delivery agreement

This is the stable working agreement for the ComfyOmni refactor. It defines how a bounded change
moves from an observed need to protected `main`. It is not a roadmap or task-status document.

## Live authority

[GitHub Issues](https://github.com/leinasi2014/comfy-omni/issues) is the single authority for live
status, priority, dependencies, ownership, WIP, scope changes, and acceptance decisions. Pull
requests, commits, checks, and evidence records preserve candidate history. Stable documentation
may link to an Issue but must not mirror its changing lifecycle.

Before pulling new work, check for a proven candidate waiting for review or integration and finish
it first. Do not create another tracker, issue mirror, agent-status stream, or repository backlog.

## Work in progress

Parallel delivery is deliberately enabled (maintainer decision, 2026-09-03) for disjoint work
lanes: concurrent executable slices may run when they own disjoint mutable surfaces — for example
one repository code slice, one server-preparation lane, and one research/contract lane. Total
executable WIP is capped at three slices. Design, implementation, review, CI, server acceptance,
and integration each count toward that cap. One writer owns each mutable surface; shared files
(including shared test fixtures, `HANDOFF.md`, and cross-cutting documentation) are integrated
serially by the coordinator, and merges into `main` stay serialized through the normal gates.
Every slice keeps its own frozen contract, observed RED, GREEN, full Docker gates, squash merge,
and `main` read-back — parallel capacity never weakens the acceptance or integration target. An
urgent repair that unblocks an active slice stays under the same Issue.

## Work-item contract

Every executable Issue freezes the smallest useful contract before production code changes:

- beneficiary and observable operational/user outcome;
- explicit non-goals;
- one scenario: `start state -> action -> important boundary -> observable result`;
- boundary and state owner;
- failure and rollback semantics;
- public/wire/artifact compatibility decision;
- resource and security limits;
- exact acceptance evidence, integration target, and first material action.

Implementation notes may guide work, but acceptance describes behavior. Architecture diagrams and
plans support a current boundary; they do not replace an executable slice.

The H3 runtime target reuses one existing set of read-only ComfyUI/H3 assets. Startup and hot
loading manage RAM/VRAM and must not depend on generating new disk conversions or complete model
packages. Record current implementation gaps without presenting this target as already delivered.
The first-release sequence is H1 existing-source loading, H2 RAM/VRAM residency, component reuse
and switching in existing workers, then H3 real-host validation and delivery. Complete LoRA
lifecycle, tools and node workflows are deferred and do not block this release.

## Definition of Ready

A slice is ready only when:

- its predecessor or accepted dependency is present on `main` and required main push checks passed;
- the Issue contains the contract and one independently demonstrable acceptance example;
- the accepted base revision and affected Docker gates are known;
- executable behavior has a specific RED path; a non-automatable live boundary has a repeatable
  observation and the nearest useful future automation instead;
- legacy migration input is pinned by repository, commit/blob, path, license, attribution, and
  distribution disposition;
- acceptance that needs model files identifies the existing authorized read-only assets and their
  retained verification records; container bases and integration versions resolve to an immutable digest
  or exact source identity. Reuse the declared environment rather than prefetching or copying assets
  for each code candidate. A genuinely missing asset blocks its dependent check, not independent
  code review or small-fixture development;
- model sources will be mounted read-only and outputs/evidence have separate bounded writable roots;
- existing user changes and the current integration state have been read back.

Asset availability is readiness evidence, not feature acceptance. A cached image or downloaded
model never proves conversion, loading, generation, LoRA, or hot-swap behavior. Acquiring a missing
asset is a separate declared action; neither readiness nor a documentation edit authorizes new
model downloads, conversion outputs or package copies.

## Delivery loop

Executable work uses `RED -> GREEN -> REFACTOR`:

1. **Contract:** freeze the behavior, ownership, failure semantics, compatibility, limits, and
   non-goals in the Issue.
2. **RED:** add the smallest acceptance/regression check and run it on the accepted base or
   predecessor candidate. It must fail for the intended missing behavior, not import/setup noise.
   Record candidate, Docker command/run, expected reason, and observed failure in the PR.
3. **GREEN:** make the smallest causal production change. Run RED first, then the affected focused
   checks. Do not expand the frozen contract to make the test pass.
4. **REFACTOR:** improve names or structure only while focused and representative checks stay green.
5. **Full gate:** run documentation, Python 3.10/3.13 quality, package/clean-install, and every
   applicable real server boundary. A skipped or unavailable gate has a precise reason and is never
   represented as a pass.

Documentation-only work preserves a repeatable before/after observation and adds automation only
when it prevents a concrete regression. New uncertain behavior may use a bounded spike, but the
spike is discarded or explicitly classified before the executable contract enters this loop.

## Evidence and real boundaries

Evidence proves only the surface it exercised. Bind it to the source commit/archive, dirty state,
container base digest, candidate image/wheel digest, relevant configuration, fixture/model/LoRA
SHA256, process identity, and state/output root whenever those dimensions can drift.

Separate initial full-content verification from subsequent bounded identity checks. Repeated code
validation reuses the same model files and prior source digests, checks for file/header/configuration
drift, and reports which observations are fresh. Do not label a lightweight check as a new full hash,
and do not invalidate unchanged source or numerical evidence merely because unrelated code changed.
Re-run the affected behavior, preserving both the earlier failure and later result. Rebuilding a
software wheel is not a reason to regenerate model payloads or assemble another model package.

GPU and stateful acceptance additionally verifies the artifact actually loaded by the declared
workers. First-release A -> B -> A acceptance keeps one control-service instance and its existing
workers, verifies component reuse and records both control and worker identities. Worker
reconstruction is reported recovery or fallback, not normal hot-loading proof. Model presence is
not load proof. Failed observations are retained, and a corrected stateful attempt runs in a clean
or deliberately reset target. LoRA, tools and nodes have separate future acceptance contracts.

Docker isolation, server mounts, exception handling, and host allowlists are owned by
[`docker-first.md`](docker-first.md). Model identities and the scope of acceptance evidence are owned by
[`../testing/model-validation-baseline.md`](../testing/model-validation-baseline.md).

## Git, review, and integration

Use the branch and Conventional Commit rules in [`CONTRIBUTING.md`](../../CONTRIBUTING.md). A pull
request has one primary structural/behavioral goal and records the Issue, contract, RED/GREEN/
REFACTOR evidence, exact gates, public impact, provenance, rollback, and limitations.

After all required checks and reviews pass, squash-merge through protected `main`. Read back the
merge commit from the remote and require the main push documentation and quality workflows to pass.
If merge resolution, target drift, generated files, or configuration could change the behavior,
rerun the affected representative boundary against the integrated candidate.

## Definition of Done

A slice is done only when:

- the stated observable behavior works and failures remain fail-closed;
- affected focused, full Docker, and representative server checks pass on the identified candidate;
- the accepted change is present on protected `main` and post-merge checks pass;
- public/wire/artifact compatibility and migration provenance are resolved;
- required operator, user, architecture, testing, and license documentation is current;
- known limitations and remaining work are explicit in the live Issue;
- temporary containers and outputs are handled under the Docker policy, while declared evidence and
  caches are retained or removed only through verified bounded targets.

A branch, PR, patch, plan, download, unit-only mock, or unbound model claim is not Done.
