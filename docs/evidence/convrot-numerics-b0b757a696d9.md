# ConvRot numerical backend acceptance

Status: accepted bounded numerical slice for issue
[#8](https://github.com/leinasi2014/comfy-omni/issues/8) and pull request
[#20](https://github.com/leinasi2014/comfy-omni/pull/20)

Candidate: `b0b757a696d9b62a3f30aca8a1124bd3c8b01b88`

This evidence proves the regular-Hadamard oracle and the Torch inverse-ConvRot adapter on bounded
fixtures. It does not prove safetensors payload reading, transactional output writing, a full
checkpoint conversion, native runtime loading, generation parity, LoRA activation, or hot-swap.

## Docker quality gates

GitHub Actions passed all Docker jobs for the exact candidate:

- run `33651339711`: Python 3.10 quality, Python 3.13 quality, and package/installed-wheel smoke;
- run `33651339651`: documentation contracts.

The quality target reported 156 passing tests. The local workstation had no Docker executable, so
no host-Python fallback was used.

## Candidate and runtime identity

The designated server built the `numerics-runtime` target from a Git archive whose SHA256 was
`86f027e45c9a5e4ba07586697af6e8b58a1e1d7079043119e702433ae59378b6`.

| Item | Immutable identity |
| --- | --- |
| vLLM 0.27.0 base | `sha256:07ea4e292adf3a26b05ac97114b28849cf4551a26beb1fbe7decd3842d752ed7` |
| ComfyOmni numerical image | `sha256:9d563b81c9393080a4e9c51fa807eec50c090870f1a10e7ca0123143eeb482fd` |
| Acceptance harness | `3547fe4b510c3b50aea854559f77264f114171bcaab4e14857fae79b6e10268d` |
| Independent verifier | `792fb931e33e6647d131746e70814f507d8b933ce699fdeef42d009c2c60dcfb` |

The server used Docker client/server 29.1.3 with `overlayfs`. The container supplied Torch
2.13.0+cu130 and CUDA 13.0. The public release must continue to reference the base by digest rather
than relying on its mutable tag.

## Isolation policy

Both CPU and CUDA acceptance containers used:

- network `none`, a read-only root filesystem, and no Docker socket;
- all capabilities dropped and `no-new-privileges` enabled;
- a non-root `1000:1000` identity;
- a 128 MiB `noexec,nosuid` temporary filesystem;
- limits of 4 GiB memory/swap, four CPUs, and 128 processes;
- only GPU device 0 for the CUDA case.

No candidate container remained after verification.

## Numerical result

The CPU and NVIDIA CMP 170HX CUDA paths exercised group sizes 4, 16, 64, and 256. For every case:

- the generated matrix exactly matched the independent standard-library oracle;
- the measured orthogonality error was zero for the fixture;
- FP32 output exactly matched the oracle;
- BF16 output exactly matched the oracle;
- `max_rows=1` and `max_rows=2` produced identical tensors;
- zero and NaN row scales failed closed.

Both devices produced the same output digest:
`635369a74dec76b7bc9333c0c1de740f03d972d5e3fac8eb3f48498e2aa7663c`.
The CUDA run reported compute capability 8.0 and 9,582,080 peak allocated bytes.

## Corrected harness attempts

Two failed acceptance attempts were retained rather than overwritten:

1. The first harness attempted to create a byte digest from a nested Python list after the CPU
   calculations. It failed before emitting `cpu.json`.
2. The corrected digest harness completed CPU acceptance, then passed an explicit `torch.device`
   to a memory-statistics API that rejected that argument in this runtime. It failed before the
   CUDA numerical assertions.

The final harness flattened the byte view and selected the current CUDA device before calling the
no-argument memory-statistics API. Neither correction changed repository source, the candidate Git
archive, the base image, or the candidate image. Failed evidence remains beside the accepted record
under `failed-digest-harness` and `failed-cuda-device-harness` suffixes.

## Evidence retention

The full private evidence set, including the two failed observations, is retained on the designated
server under `/home/hyl/comfy-omni-acceptance/convrot-numerics-b0b757a696d9*`. The accepted set
contains individually verified hashes for the harnesses, CPU/CUDA JSON, independent verification,
environment, and container policy.
