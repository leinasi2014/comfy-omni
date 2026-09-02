# ConvRot native-export plan acceptance

Status: passed

Implementation commit: `ed08abbe2df50f95502b21b14214255c08a8e508`

Issue/PR: [#8](https://github.com/leinasi2014/comfy-omni/issues/8) /
[#19](https://github.com/leinasi2014/comfy-omni/pull/19)

This receipt covers only exact source authorization and payload-free plan generation. It is not
evidence for inverse-ConvRot numerical correctness, output writing, checkpoint loading, generation,
LoRA activation, or full-DiT hot switching.

## Docker quality gates

The local workstation had no Docker executable. The local gate was reported unavailable and no
host Python, pytest, Ruff, pip, Torch, conversion code, or model parser was used as a fallback.

GitHub Actions run `33648299571` passed all Docker jobs for the implementation commit:

- Python 3.10 quality: Ruff format, Ruff lint, and 130 tests passed;
- Python 3.13 quality: Ruff format, Ruff lint, and 130 tests passed;
- package and clean installed-wheel smoke passed;
- documentation contracts passed in run `33648299378`.

The `srv-00` quality target also passed from source archive SHA-256
`1b65d4819a5aa3585d54beadb78e73fd54c7fd2f7be231e5ee7f8f5bcaf724d9`.

| Image | Image ID |
|---|---|
| `comfy-omni:convrot-plan-quality-ed08abbe2df5` | `sha256:ec798f83ccdaa11fdc4f9fa3e3b50bedad8a7b09e71d3b6421d1952dd4fea591` |
| `comfy-omni:convrot-plan-runtime-ed08abbe2df5` | `sha256:104312a3f865c6a09f818165d2083a97d82eb5a5a8cc2eb669cc22c72439475d` |

The server used Docker client/server 29.1.3, `overlayfs`, and Docker root `/data/docker`. The reviewed
DaoCloud pull-through prefix `m.daocloud.io/docker.io/library` was used because direct Docker Hub
access was unavailable; no daemon configuration was changed.

## Container boundary

The plan and its independent verification ran with:

- network `none` and no Docker socket;
- read-only root filesystem;
- `/tmp` tmpfs limited to 64 MiB with `noexec,nosuid`;
- all Linux capabilities dropped and `no-new-privileges` enabled;
- 128 PID, 1 GiB memory/swap, and 4 CPU limits;
- numeric UID/GID 1000:1000;
- the bounded model root mounted read-only;
- no model output mount and no model payload writes.

No workload container remained after verification.

## Source and result

The authorized source was
`diffusion_models/minimax_h3_ref2va_pruned_zs05_int8_convrot.safetensors`:

- size: 20,970,379,680 bytes;
- file SHA-256: `71b8085ac4221ee036708c230a007d617dccca1b0028b95bb4ee106cb2a385c5`;
- source schema SHA-256: `cc7976f678e6d4a567e718aca56c1db4aa91adfa27108db84066cce3213edf9d`;
- source contract: `minimax-h3-dasiwa-ref2va-hybrid-int8-convrot-v1`;
- architecture template: `h3-transformer-50l-convrot`.

The canonical plan has content SHA-256
`2a9a7895ab1bae68a2d7ffa19d95b9dcbc301cf2cb182ad8c37f8fb2d8a41fda`. It contains 932 source
actions, 532 target tensors, 10 planned shards, and 40,225,668,192 target payload bytes.

| Operation | Count |
|---|---:|
| raw copy | 330 |
| dense runtime-QKV to grouped order | 2 |
| inverse ConvRot to BF16 | 150 |
| inverse ConvRot to BF16 plus QKV grouped order | 50 |
| omitted Comfy quant marker | 200 |
| omitted source row scale | 200 |

The retained server evidence directory is
`/home/hyl/comfy-omni-acceptance/convrot-plan-ed08abbe2df5`.

| Evidence file | SHA-256 |
|---|---|
| `native-export-plan.json` | `6bcc8c35e6d51138a813450bc99fef16075d275bca9f9f7386091f274d639f3c` |
| `verification.json` | `3aa4b485f406e18307db438ec9c6641d1280dc39a4cebf1a707bd0d7b758e3d2` |
| `verification-attempts.txt` | `1f3735f2360ea952dec27ca28292ab4352db7ddc239936719d9e2f5ed03edae7` |
| `environment.txt` | `66c7bf9b6298e5f00010570eada22d2b34c7b1ae0d73fae7aac9301b74cbb2d3` |
| `container-policy.txt` | `a9c6490f5bf84c4f19f3c7baa29f9fda9cc3229af5bbeb37c1f4d5caa952a5ec` |

## Corrected acceptance assertion

The first independent verification attempt rejected the generated plan because the orchestration
script compared its source binding with an unverified expansion of the previously abbreviated
`71b808…` digest. The plan itself already contained the correct digest and was retained unchanged.
The second attempt used the complete digest already pinned in
`docs/testing/model-baseline.v1.json`, re-derived the plan's canonical content digest, and passed all
operation, contract, template, source, semantics, and resource assertions. The failed assertion and
correction are retained in `verification-attempts.txt`; they were not erased from the evidence trail.
