# Test layout

All lanes execute through the repository Docker targets described in
[`docs/development/docker-first.md`](../docs/development/docker-first.md). A host Python environment
is not a test lane.

- `unit/`: standard-library or isolated dependency tests; no network, host, or GPU.
- `contract/`: schemas, manifests, mapping packs, receipts, and compatibility fixtures.
- `integration/`: CLI, API, plugin host stubs, subprocesses, and cross-module flows.
- `packaging/`: sdist/wheel contents, clean installs, entry points, and package resources.
- `host/`: pinned real-host or frozen-host acceptance tests.
- `fixtures/`: small, license-cleared, non-sensitive test inputs.

Tests are added with the implementation slices they characterize. The first contract test freezes
the external validation-model manifest without downloading or redistributing model payloads. Empty
lane markers keep the remaining reviewed test taxonomy visible without pretending those lanes have
already been implemented.
