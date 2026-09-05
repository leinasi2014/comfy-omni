# Checkpoint LoRA preflight

`conversion.oracle.checkpoint_preflight.preflight_checkpoint_candidate` inspects a
digest-pinned raw checkpoint and LoRA before a native runtime package exists.
The original package-based `preflight_candidate` API keeps its existing behavior.
The acceptance boundary is tracked in [issue #12](https://github.com/leinasi2014/comfy-omni/issues/12).

Each input is opened as a regular file through the existing `SafeTensorSources`
reader. Complete file hashes, strict safetensors structure and final descriptor
checks use the held input descriptors. Matrix payloads are not decoded; reads
beyond the header are limited to bounded quantization declarations and scalar
alpha values. Importing this boundary does not load Torch or the runtime host.

The receipt records both expected and actual file identities, tensor and dtype
census, descriptor schema hashes, adapter tensor names/shapes, pair rank, alpha
and scale declarations, known target mapping, and ConvRot representation. An
unknown or missing target does not erase observations about the adapter pair
itself. Malformed inputs and changed identities raise `CheckpointInputError`;
they do not produce a successful inspection receipt.

A completed inspection returns `UNSUPPORTED` with a stable reason and the observed
evidence. Its scope is `checkpoint-only`, `promotion_capable` is false, and both
the numerical fold and runtime activation are `NOT_RUN`. Shape agreement alone
does not prove a numerical mapping or authorize mutation. In particular,
`BASE_REPRESENTATION_UNBINDABLE` describes the lack of a proved route for the
observed representation, not universal mathematical incompatibility. A later
supported route still requires the project's numerical and lifecycle acceptance.

`scripts/acceptance/lora_checkpoint_preflight.py` exercises the installed wheel
against the fixed primary checkpoint and either fixed LoRA. The caller supplies
read-only input mounts, candidate source/wheel identities, the container image
digest, scale, and a new output path. The script refuses an existing receipt,
verifies installed tool identity, and writes an exclusive receipt with a content
digest. Container identity is explicitly recorded as a caller declaration and
must also be captured by the external Docker orchestrator.

The implementation is new code under this repository's Apache-2.0 license and
reuses the already migrated strict reader, census and ConvRot contracts. It
copies no additional legacy modules and distributes no model payloads or private
validation records.
