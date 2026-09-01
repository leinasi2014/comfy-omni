# Test layout

- `unit/`: standard-library or isolated dependency tests; no network, host, or GPU.
- `contract/`: schemas, manifests, mapping packs, receipts, and compatibility fixtures.
- `integration/`: CLI, API, plugin host stubs, subprocesses, and cross-module flows.
- `packaging/`: sdist/wheel contents, clean installs, entry points, and package resources.
- `host/`: pinned real-host or frozen-host acceptance tests.
- `fixtures/`: small, license-cleared, non-sensitive test inputs.

Test files will be added with the implementation slices they characterize. Empty lane markers keep
the reviewed test taxonomy visible without pretending that tests already exist.
