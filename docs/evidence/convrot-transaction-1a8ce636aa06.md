# ConvRot native-export transaction acceptance

Status: accepted copy-only transaction slice for issue
[#8](https://github.com/leinasi2014/comfy-omni/issues/8) and pull request
[#23](https://github.com/leinasi2014/comfy-omni/pull/23)

Candidate: `1a8ce636aa06e7c60b0166afc1479881db0a9d28`

This evidence proves held-descriptor source binding, deterministic copy-only safetensors writing,
independent shard verification, no-overwrite publication, and a manifest-last commit marker. It
does not prove ConvRot materialization, QKV reorder, a full model conversion, native model loading,
generation, LoRA, or hot-switch behavior.

## Docker quality gates

GitHub Actions passed all Docker jobs for the exact candidate:

- quality run `33661389372`: Python 3.10 and 3.13 each passed Ruff, architecture and delivery
  policies, and 172 tests; package/installed-wheel smoke also passed;
- documentation run `33661389291`: documentation contracts passed.

The local workstation had no Docker executable, so no host-Python fallback was used.

## Candidate identities

| Item | Immutable identity |
| --- | --- |
| Git source archive | SHA-256 `d307da4118414c0a0c272cbd9c67694d3e42a124dbb95a115b64db134a150519` |
| Offline wheel builder image | `sha256:9d563b81c9393080a4e9c51fa807eec50c090870f1a10e7ca0123143eeb482fd` |
| Runtime base image | `sha256:104312a3f865c6a09f818165d2083a97d82eb5a5a8cc2eb669cc22c72439475d` |
| Candidate runtime image | `sha256:8fb6fdc84e2b0ed7149a3cbf8e3050cf6719e211a4f49e11d47ee76a9223cfef` |
| Candidate wheel | SHA-256 `29df7085f126a7c9612ef9b406db3e0eb43d0ac6ef6877df45f1bbe8ac5d223a` |
| Acceptance harness | SHA-256 `9d02f0920a47d5cdc80abb7526ef101a572aa4359b0c35b3fabfed9764648f4d` |
| Independent verifier | SHA-256 `5943ea99d704cabab6671d9c47a7e285f1afd1ea20528ef3e3fb8f995d3cc347` |
| Acceptance Dockerfile | SHA-256 `46a6029b245986cb309d139992116a365f395c4e9b905e256218e570722c54ff` |
| Docker orchestration | SHA-256 `befef4038951ab406b7b26207440bb15ebee0f23078528ca4b4eaf87988a1ebf` |

The server used Docker client/server 29.1.3 with `overlayfs`. The wheel was built offline from the
read-only candidate archive using the already accepted numerical image, then installed offline
over the already cached minimal runtime image. The receipt independently recovered the same wheel
SHA-256 from installed PEP 610 metadata and bound source commit `1a8ce636aa06...`.

## Isolation and mount policy

The wheel build, fixture preparation, transaction, and independent verification containers all
used network `none`, a read-only root filesystem, all capabilities dropped,
`no-new-privileges`, numeric user `1000:1000`, bounded memory/CPU/process limits, and a bounded
temporary filesystem. In particular:

- candidate source was read-only and wheel output was the only writable mount during wheel build;
- the fixture creator could write only its dedicated fixture directory;
- the transaction saw `/source` read-only and could write only `/evidence`;
- the independent verifier saw both `/source` and `/evidence` read-only and could write only its
  separate `/verify` directory.

No Docker socket, model tree, home directory, SSH directory, GPU, or network was exposed to a
workload container. All temporary containers were removed; only the candidate image and declared
evidence were retained.

## Verified result

The source fixture was 131 bytes with SHA-256
`ef49fb7587d62744ac1bffd289123fcc0084a567c697cdaf33e876e1c22ee05c`. The transaction produced
two tensors in one shard plus an index, immutable plan, runtime config patch, and last-written
manifest:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `config.patch.json` | 264 | `21a68dd59624a90a1682bee1ef7fab6d98ecd3d8680ea26071bb4d2cbd7d3791` |
| `export.plan.json` | 1,878 | `f669fd61142c862b0d0c92bc7f475f5ef1e3901b1074705ee51b2251c4e7d50d` |
| `manifest.json` | 1,356 | `ca7e23e268b65c6830268d80528abe5ac30b7054e8bec2cf543dcc059ae686f1` |
| `model-00001-of-00001.safetensors` | 136 | `84ad8377f05ffd7204b4bb59286c1022b8ec0d909afc0eb587f46b2a224e60f2` |
| `model.safetensors.index.json` | 130 | `a13ab55e2b0088d9b40f1fe56fb8fefe2916beb87835d50737251e98db895db8` |

The plan content SHA-256 was
`213c1dc35dd82118ec63b4307b0b7b615072d4927385cd5c7abf5fed64ff9e0c`; the manifest self-digest
was `ecfb6adefca1aa6a600447ab75a29100721c7b17147eb0ab1e64ebf2c48d6177`. The independent
standard-library verifier recomputed the source, plan, every listed output, safetensors header and
payload, manifest self-digest, tool commit, and wheel digest, then emitted status `VERIFIED`.

## Preserved readiness and harness failures

Three non-product failures were not overwritten or presented as acceptance:

1. A Docker Hub prefetch for `python:3.13-slim-bookworm` timed out while resolving over IPv6. No
   daemon, DNS, proxy, or host configuration was changed. The accepted route used existing cached
   immutable images and remained offline.
2. The first wheel-build attempt used the cached quality image. It had `build` but could not import
   `setuptools.build_meta` without build isolation, so it failed before producing a wheel, image,
   or transaction output. Its separate evidence directory was retained.
3. The first post-acceptance evidence-index refresh included its own temporary
   `SHA256SUMS.next` path. The index alone was regenerated with temporary names excluded; the
   candidate output and all evidence files were unchanged, and the final full checksum audit
   passed.

## Evidence retention

The accepted private evidence set is retained on `srv-00` under
`/home/hyl/comfy-omni-acceptance/convrot-transaction-1a8ce636aa06-attempt2`. It contains the exact
archive, source tree, wheel, harnesses, orchestration, image/build logs, container-policy inspect
records, result, independent verification, and a fully verified `SHA256SUMS`. The first wheel-build
failure and Docker Hub prefetch failure remain in separate sibling evidence directories.
