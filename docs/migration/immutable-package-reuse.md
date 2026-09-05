# Reuse immutable package components

Package materialization copies files by default. This creates independent payload storage and
preserves the existing API and serialized result. For a trusted immutable store on POSIX,
callers may explicitly use:

```python
staging = materialize_package(
    plan, output, reuse_immutable=True, max_copy_bytes=copy_budget,
)
publication = publish_package(plan, staging)
```

Same-mount files share hardlinks. Cross-filesystem or cross-bind-mount files use a directory-pinned
bounded copy, provided their actual opened size fits the remaining `max_copy_bytes` budget.
Set the budget to zero to forbid any payload allocation. Other link errors fail rather than
silently allocate a copy. Invalid budgets and writable sources are rejected.

Reuse changes inode link counts and ctime. It never opens a source payload for writing, changes
source permissions, replaces a source path, or changes its owner/mtime. Payloads must have no
write permission. The store owner must retain that invariant: create a replacement file when
changing a component; do not chmod and edit a shared inode in place. Unlinking one package
does not remove other links. Use default copying to materialize independent storage again.

The helper holds no-follow source and directory descriptors, verifies complete hashes and
identities before/after link creation, and creates only exclusive planned staging entries.
A directory rename cannot redirect an operation to its replacement. Staging and publication
continue to verify their exact file census and all content digests. Failed stages remain
unpublished for diagnosis; no success receipt is issued.

Opt-in materialization results add `storage.mode`, `shared_bytes` and `copied_bytes`. Package
manifest bytes and the default materialization serialization are unchanged. The copy budget
does not include small directories/documents or guarantee free disk space; an operator must
also reserve filesystem capacity for those and for unrelated activity.

Docker normally mounts model sources read-only. Hardlinks across separate bind mounts may be
unavailable even on one underlying filesystem. Explicit reuse is a storage-administration
operation and may use a shared writable mount containing only the named related component
roots and a separate private output subtree. Its exact mounts and budget must be recorded in
the live acceptance contract. The payloads remain read-only; ordinary conversion and runtime
model mounts retain the default policy. No elevated capabilities or host-side Python is needed.

This helper is new ComfyOmni code. Existing copy/materialization/publication code retains its
Apache-2.0 h3-forge provenance; this change does not incorporate another external implementation.
Tests cover duplicate allocation on the prior implementation, shared publication, unchanged
default copying, real cross-filesystem copy, mount fallback budgets, writable inputs, late
mutations, existing outputs and source/destination directory replacement.

The installed-wheel acceptance runner `scripts/acceptance/native_package_assembly.py` accepts
`--reuse-immutable` for its fixed six-component corpus, with a zero copy budget. It records
source metadata before linking, exact link increments, target/source inode equality and byte
accounting. The separate `verify_native_package_assembly.py` invocation requires its own
`--components-root` authority for such a receipt, checks those identities before and after
re-hashing every published file, and rejects equal-byte copies, extra links and changed modes.
This storage acceptance alone does not establish model inference or conversion correctness.
