## Outcome and acceptance

<!-- Link the live Issue. Who benefits, what observable behavior changes, and what one scenario proves it? -->

## Non-goals

<!-- State what this slice deliberately does not claim or change. -->

## Frozen contract

<!-- Boundary/state owner, failure/rollback semantics, compatibility, security/resource limits. -->

## Scope

<!-- Primary goal, modules/files changed, and dependency direction. Keep one main goal per PR. -->

## Contract impact

<!-- Python/CLI/plugin/HTTP/environment/runtime/artifact/error contracts. -->

- [ ] No public-contract change
- [ ] Public-contract change is documented, versioned, and migration-tested

## TDD evidence

### RED

<!-- Accepted base/candidate, exact Docker run or observation, intended failure, and result. -->

### GREEN

<!-- Smallest causal change and focused result, with exact candidate/run identity. -->

### REFACTOR

<!-- Structural cleanup, or state none; show that focused behavior stayed green. -->

## Verification

<!-- Exact Docker/CI/server commands or run URLs, counts, digests, identities, and unavailable gates. -->

- [ ] Relevant focused and full tests pass
- [ ] Ruff/format checks pass when Python changes
- [ ] Packaging/clean-install checks pass when packaging changes
- [ ] Representative server evidence is attached when a real boundary changes
- [ ] Evidence is bound to the exact source, image/package, configuration, and assets

## Documentation and provenance

- [ ] `README.md` and `README.zh-CN.md` changed together, or neither needed an update
- [ ] Docker documentation contracts pass
- [ ] No project command/dependency ran or installed on a host; approved exceptions are documented
- [ ] Architecture, operator, testing, and license documentation is current
- [ ] Migrated code/assets record source commit/blob, license, attribution, and disposition

## Migration and rollback

<!-- Migration order, compatibility shim, state mutation, rollback, and remaining limitations. -->

## Integration read-back

<!-- Complete after merge: protected-main commit and main push check results. -->
