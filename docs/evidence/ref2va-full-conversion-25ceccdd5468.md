# Complete Ref2VA native-export acceptance

Status: accepted real-model conversion slice for issue
[#8](https://github.com/leinasi2014/comfy-omni/issues/8) and pull request
[#25](https://github.com/leinasi2014/comfy-omni/pull/25)

Candidate: `25ceccdd54680a5e32ea1574974c138d39d08bd6`

This evidence proves that the exact installed ComfyOmni wheel planned, converted, atomically
published, and independently verified the complete pinned Ref2VA checkpoint as a native 10-shard
package. It does not prove multi-component package assembly, CLI exposure, a vLLM-Omni load,
generation, LoRA behavior, or A -> B -> A hot switching.

## Agile trace and Docker quality gates

The READY contract was frozen in issue #8 before implementation. Initial RED candidate
`2c29b070240a42b3b6f5b55237571aeb8d488c11` failed for the intended missing application
composition with one failure and 182 passes in Docker quality run
[`33668209466`](https://github.com/leinasi2014/comfy-omni/actions/runs/33668209466).
Subsequent focused RED candidates required a fast ConvRot backend, explicit CPU/CUDA selection,
and buffer-backed BF16 serialization before the full model was attempted.

The exact accepted candidate passed:

- quality run [`33673841850`](https://github.com/leinasi2014/comfy-omni/actions/runs/33673841850):
  Python 3.10 and 3.13 each passed Ruff, architecture and delivery policies, and 188 tests; the
  package and installed-wheel smoke also passed;
- documentation run
  [`33673841843`](https://github.com/leinasi2014/comfy-omni/actions/runs/33673841843): all
  documentation contracts passed;
- a bounded CUDA smoke using the same runtime image: dense-oracle agreement for group sizes 4, 16,
  64, and 256, plus a representative 4,096 x 7,168 group-256 transform in 0.103 seconds.

The local workstation had no Docker executable, so no host-Python fallback was used.

## Immutable inputs and execution identities

| Item | Immutable identity |
| --- | --- |
| Source checkpoint | `minimax_h3_ref2va_pruned_zs05_int8_convrot.safetensors` |
| Source bytes | `20,970,379,680` |
| Source SHA-256 | `71b8085ac4221ee036708c230a007d617dccca1b0028b95bb4ee106cb2a385c5` |
| Source contract | `minimax-h3-dasiwa-ref2va-hybrid-int8-convrot-v1` |
| Source-contract schema | `cc7976f678e6d4a567e718aca56c1db4aa91adfa27108db84066cce3213edf9d` |
| Candidate archive SHA-256 | `bc9805a0d22afce82b880a1b6407aa5c612732dbdad06cf1b5223a4322b61c77` |
| Candidate wheel SHA-256 | `862a070adc841862397cbadb888c22149ed0787a4530d60c69ee3f6fe400e5bc` |
| Torch runtime image | `sha256:9d563b81c9393080a4e9c51fa807eec50c090870f1a10e7ca0123143eeb482fd` |
| Independent verifier image | `sha256:104312a3f865c6a09f818165d2083a97d82eb5a5a8cc2eb669cc22c72439475d` |
| Torch / CUDA | `2.13.0+cu130` / `13.0` |
| Selected GPU | device `0`, NVIDIA CMP 170HX, capability 8.0, `68,212,293,632` bytes |

The source checkpoint was mounted read-only. The source archive was mounted read-only into the
cached Torch image, the wheel was built without network access, and that exact wheel was installed
offline into a dedicated tree. Runtime code loaded only the installed wheel tree.

## Authorized plan and conversion result

The preflight took 96.503 seconds and reproduced plan content SHA-256
`754be195eba3f8d3eb0f65a24de20ce0110cf0fe80a50fa0bef4221da9452ab4`. The plan contained 932
actions, 532 output tensors, 10 shards, and `40,225,668,192` tensor payload bytes:

| Operation | Count |
| --- | ---: |
| Raw copy | 330 |
| Runtime-QKV to grouped copy | 2 |
| Inverse ConvRot to BF16 | 150 |
| Inverse ConvRot plus QKV reorder | 50 |
| Exact Comfy quantization marker omission | 200 |
| Exact source row-scale omission | 200 |

The complete conversion took 636.593 seconds with peak RSS `1,249,308,672` bytes. It published the
manifest last with self-digest
`6def8c29c3faa88b9365f939dfc1dd2610f36f23bea95d1c43f1d1815d46dfc9`; the manifest file SHA-256
is `93b3fc29364dbca66570206ce5df72363d856630a17f58216f89b68ee50c7052`.

## Independent verification

A separate standard-library verifier image reopened the 20.97 GB source and every published file.
It independently checked exact source identity, plan and manifest self-digests, output census,
file sizes and SHA-256 values, safetensors spans, all 532 target descriptors, all 330 raw-copy
payloads, both QKV reorder payloads, and three rows from each of the 200 ConvRot outputs. The 600
sampled rows covered 4,838,400 elements; the largest absolute BF16 comparison error was
`0.006017439300194383`, within the verifier's combined relative/absolute tolerance.

| Published file | Bytes | SHA-256 |
| --- | ---: | --- |
| `config.patch.json` | 2,076 | `3ec738237bcc7de59065ab538633c0494abe6775fa9c3d65f945dc28943d29c7` |
| `export.plan.json` | 267,341 | `07d8529e54a00bc6718f1bceda705b0119b30a43a44bbe731b58ffa2572eea9d` |
| `manifest.json` | 3,018 | `93b3fc29364dbca66570206ce5df72363d856630a17f58216f89b68ee50c7052` |
| `model-00001-of-00010.safetensors` | 4,173,108,120 | `1a7f494e788fd8148a8abe36c43866bef3220708b0daa965099d1b04ba1f5f75` |
| `model-00002-of-00010.safetensors` | 4,170,622,616 | `4ecc82a0f540f595acdd65539e0e895b02971e1fda839cca8f43813d94098f2b` |
| `model-00003-of-00010.safetensors` | 4,095,316,816 | `8cd8af8b91532ca77135a81762b710d2cb3933a2041850608d573299d77b2a74` |
| `model-00004-of-00010.safetensors` | 4,093,552,288 | `fb3c7fc4e26ab82a4488515d39d09bd0c6b6e0b03cf7718368a3348b50503550` |
| `model-00005-of-00010.safetensors` | 4,170,622,608 | `6947d295a1783fa229f4223c494ee57331dcb3ce2efee6b83d95afaf7593d120` |
| `model-00006-of-00010.safetensors` | 4,095,316,824 | `247e2ef9c1da8bd3b94a84bc9c4900aa061c516944a44aeae3a2384612d2ef4c` |
| `model-00007-of-00010.safetensors` | 4,093,552,280 | `1de95881c8a521911a64839d9dc97dd2445b5836268a0744c19c9b5d7b189af9` |
| `model-00008-of-00010.safetensors` | 4,170,622,616 | `8e5399dc284735f8ad37b4215ec181273ec63f4ba274fecf7093fde25d55c7b9` |
| `model-00009-of-00010.safetensors` | 4,095,316,792 | `1b5e73be72f69437b404bff401b50504d75a04a44a41f76b1a576462ee4f0520` |
| `model-00010-of-00010.safetensors` | 3,067,692,144 | `548ed3ef562fae850ec431402950b49d24d3693113dbdbe1f097771e58208745` |
| `model.safetensors.index.json` | 34,881 | `4c790598b90ff246e185e5ff3900034cd8495a8303b48e9312b8a83f265684ca` |

## Isolation and publication audit

An independent, network-disabled policy-audit container parsed the saved Docker inspection records
for wheel build, offline install, preflight, GPU conversion, and verification. All five used user
`1000:1000`, read-only root filesystems, dropped all capabilities, enabled `no-new-privileges`, and
had explicit CPU, memory, PID, and temporary-filesystem limits. Only the conversion container
received a GPU, and it received device `0` only. Mount destinations and read/write modes were exact;
no Docker socket or broad host directory was exposed.

The output census showed link count two for the plan, config, index, and ten shards: atomic
publication used hard links from the private staging directory instead of duplicating the 40 GB
payload. `manifest.json` had link count one and was published last. The final residue check found
zero acceptance containers.

Key evidence digests are:

| Evidence | SHA-256 |
| --- | --- |
| Preflight result | `6fae8061522a4fef0db6de04ea76e9630be2cc7386ba4e33e6f3eef21947e651` |
| Conversion result | `0433411cb0f6b23022b6dc84171a13cc142bd8d411a8234d8916d2f51cfdecf2` |
| Independent verification result | `bfb7c17be10bcac81e48de082cdc3905c985cb678b8bbd1e9e8f4ede58db146e` |
| Output census | `04a037da7102edc4d4d35951c554763c7b0cfab4de6ea00c893702cdd42a7626` |
| Policy-audit checksum set | `9b88069dd2497b558dbe4ffacf83f6d730e99bed32aa35aafcd225faef7ad66d` |
| GPU-identity checksum set | `e2c24978cd0cbd3b37b38e5cd029bdbe7e2c82aa77867bbc52e0c8a267dd1826` |

## Preserved failed attempts and retention

Attempt 1 rejected an orchestration filename mismatch before producing an output. Attempt 2 used
the correct source identity but exposed the dense CPU Hadamard and byte-by-byte storage paths; it
was stopped without publishing an output. Those observations drove the focused performance RED
tests and are retained rather than represented as acceptance.

The accepted private evidence and output are retained on `srv-00` under
`/home/hyl/comfy-omni-acceptance/ref2va-full-conversion-25ceccdd5468-attempt3`. Model payloads are
not committed to Git or included in the wheel.
