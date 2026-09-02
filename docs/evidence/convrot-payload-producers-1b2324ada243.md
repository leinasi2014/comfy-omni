# ConvRot bounded payload-producer acceptance

Status: accepted bounded producer slice for issue
[#8](https://github.com/leinasi2014/comfy-omni/issues/8) and pull request
[#24](https://github.com/leinasi2014/comfy-omni/pull/24)

Candidate: `1b2324ada2432402a6071c9909abb77303107846`

This evidence proves that the exact installed wheel executes bounded raw-QKV, inverse-ConvRot, and
combined inverse-ConvRot/QKV producers; omits only exact marker/scale triplets; and publishes an
independently verified immutable artifact. Unit and contract tests additionally prove external
snapshot carry and tamper rejection. It does not prove full Ref2VA conversion, a native model load,
generation, LoRA behavior, or A → B → A hot switching.

## Agile trace and Docker quality gates

The READY contract was frozen on issue #8 before implementation. Corrected RED candidate
`23a952ba9166706afbaeda9f94561ba2ed2c7df6` produced 11 intended failures and 171 passes on each
Python lane in Docker run `33664020474`. No product code was added before that failing contract.

The exact accepted candidate passed:

- quality run [`33666330054`](https://github.com/leinasi2014/comfy-omni/actions/runs/33666330054):
  Python 3.10 and 3.13 each passed Ruff, architecture and delivery policies, and 182 tests; the
  package and installed-wheel smoke also passed;
- documentation run
  [`33666330086`](https://github.com/leinasi2014/comfy-omni/actions/runs/33666330086): all
  documentation contracts passed.

The local workstation had no Docker executable, so no host-Python fallback was used.

## Candidate identities

| Item | Immutable identity |
| --- | --- |
| Git source archive | SHA-256 `ca8fc2b89991d561b8eb3e155d71dad719b5ce67efa762cda965d45f0dee7733` |
| Torch runtime and offline wheel-builder image | `sha256:9d563b81c9393080a4e9c51fa807eec50c090870f1a10e7ca0123143eeb482fd` |
| Independent verifier image | `sha256:104312a3f865c6a09f818165d2083a97d82eb5a5a8cc2eb669cc22c72439475d` |
| Candidate wheel | SHA-256 `bd552a3a88c12f245e7d22513368a2a27b3853a22235a92b406ec1e34ad6383f` |
| Acceptance harness | SHA-256 `4636be9529adad34429e5d172e1f30845210d243de99c83d0df96d4fca079871` |
| Independent verifier | SHA-256 `7c118e17ab82a03bd76cd68fd9921e5c602d36bb8c8a17fd5d9ee373cf40d6c3` |
| Tracked acceptance Dockerfile | SHA-256 `161b1b09017d25726afec7caed305a7c693018fe569822cf7d4bcda4f8a00ef8` |
| Server orchestration | SHA-256 `174d2edeb535f39bb8220553f79abb7db44d73faf8c7c53b08d0114fb41fd5b7` |

The server used Docker client/server 29.1.3 with `overlayfs`. The exact source archive was mounted
read-only into the cached Torch image, its wheel was built with no network, and that wheel was
installed offline into a dedicated tree. Runtime calls used only that installed tree. The tool
identity was recovered from embedded source metadata and installed PEP 610 metadata, rather than
accepted from untrusted command arguments.

## Isolation audit

Wheel build, offline installation, fixture preparation, actual-Torch execution, and independent
verification each used:

- network `none`, read-only root filesystem, user `1000:1000`;
- all capabilities dropped, `no-new-privileges`, no privileged mode and no host devices;
- explicit memory, CPU, PID, and temporary-filesystem limits;
- read-only source, acceptance code, installed wheel, and verification inputs;
- exactly one purpose-specific writable output mount per stage.

No Docker socket, model tree, home directory, SSH directory, or GPU was exposed. A separate
network-disabled audit container parsed every saved `docker inspect` record and returned
`VERIFIED`. The final residue check found no acceptance containers.

## Verified numerical result

The synthetic source contains one six-row ConvRot QKV triplet, one three-row ordinary ConvRot
triplet, and one six-row dense QKV tensor. The plan uses group size 4, a two-row maximum block, and
three output tensors totaling 96 payload bytes. Actual Torch `2.13.0+cu130` executed the conversion;
an independent standard-library verifier compared exact target descriptors, row order, BF16 bytes,
file digests, plan digest, manifest self-digest, source identity, commit, and wheel identity.

| Result | Immutable value |
| --- | --- |
| Source fixture SHA-256 | `5d2315c749a75ea924fc0d9a8a59eca8f59fd75ebcab5030a3cdf137ee78b0f1` |
| QKV permutation SHA-256 | `726dbd3cccf6bf671cae40865d84221cdc29f1a117387f3d6984eb329d6d7ffd` |
| Plan content SHA-256 | `f90782a06c73b246719671cace3fb4e25869e25585f80e8abb0334b1cb20bc94` |
| Manifest self-digest | `68d6826b278f50cda34e809f087676b60186c451c2c9163d84e676bc95bd0e0d` |
| Output shard SHA-256 | `f6d9a6bb002a8bda9d542672699a17bec80b7c2e531779f28aebc85d524ab7ef` |
| Independent result | `VERIFIED`, 3 tensors |

Published files were:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `config.patch.json` | 264 | `16add9ed7254d6c5abc3383efe1faf77522cc7e7c49e3d40407904ccd61cd2f4` |
| `export.plan.json` | 3,485 | `a64bade3e0d10bde7799d19a8606dae2672b17b7e52c44438794d3e5189a0528` |
| `manifest.json` | 1,357 | `10477833b7103db4df2da52229d0c4f157c0349f66bb08d57ba261d06f7371df` |
| `model-00001-of-00001.safetensors` | 368 | `f6d9a6bb002a8bda9d542672699a17bec80b7c2e531779f28aebc85d524ab7ef` |
| `model.safetensors.index.json` | 251 | `c68b303651707add0ea876b52fb1b87b8a66e0438810d3c85edff2e149d8ec29` |

The complete evidence tree contains 367 files and 5,322,853 bytes. Its `SHA256SUMS` file was
successfully rechecked and has SHA-256
`2b40c89ae984f316627893192761a5de4e90aca675951b6bd7e255786095d450`.

## Preserved RED and harness failures

No failed attempt was rewritten as acceptance:

1. Formatting-invalid RED run `33663865224` was discarded as a process sample; corrected RED run
   `33664020474` is the development-contract evidence.
2. The first server attempt was interrupted during image preparation and produced no workload
   result.
3. The second server attempt built and executed successfully but its independent verifier expected
   a noncanonical `commit` key instead of the stable `source_commit` tool-identity key. That harness
   mismatch remained failed. Candidate `1b2324ada243...` fixed the acceptance harness to derive and
   verify installed-wheel provenance, then passed in a fresh evidence root.

## Evidence retention

The accepted private evidence set is retained on `srv-00` under
`/home/hyl/comfy-omni-acceptance/convrot-payload-producers-1b2324ada243-attempt3`. It includes the
exact archive, extracted source, wheel, installed tree, harnesses, orchestration, environment,
container policies, result, independent verification, logs, and fully verified `SHA256SUMS`.
Earlier attempts remain in separate sibling roots.
