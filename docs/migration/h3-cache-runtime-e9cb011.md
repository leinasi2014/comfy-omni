# H3 curve-cache runtime compatibility

This adapter retains the proven legacy Ref2VA model recipe while moving its
execution into the independent `comfy_omni` distribution. Its acceptance contract
is tracked in [Issue #11](https://github.com/leinasi2014/comfy-omni/issues/11).
This document describes the implementation contract and source attribution;
runtime acceptance decisions remain in the issue and pull request.

## Artifact and executor identities

Two existing formats use the name `h3-comfy-package/v3`. ComfyOmni's offline
producer writes nested `routing` plus an export-plan identity. The audited
legacy producer writes top-level routing, two model indexes and a converter
identity. The runtime selects their mutually exclusive layouts explicitly.
It preserves the existing ComfyOmni validator and validates the legacy layout
separately, including its VAE exports, file census, quantization policy, folded
LoRA and curve-cache relationship. Mixed layouts and unknown versions fail.

The legacy artifact remains byte-for-byte unchanged. Its producer identity is
the audited `h3-forge` wheel at the commit below. The executing ComfyOmni wheel
has its own source and wheel identities; it does not impersonate the producer
or require `h3_forge` to be installed. Legacy artifacts have no export-plan
digest, so the runtime represents that field as absent rather than inventing one.

All package modules remain lightweight at import. Full validation of a legacy
curve-cache package additionally uses CPU Torch to reproduce the exact FP32
schedule. Native ComfyOmni package validation retains its existing dependencies.
Validation is read-only and occurs before model construction, not in denoising.

## Runtime behavior

The registered H3 entrypoint selects the cache pipeline only from a validated
cache binding. The adapter constructs the official DiT with cache-only time
and AdaLN modules, then restores all temporarily replaced host attributes.
The four-step Turbo schedule uses five API sigma points with video/audio
shifts 12/3. The legacy wire identifiers and all five conditioning modes stay
unchanged. Host schedule plans are checked by their FP32 bits before inference.

Each worker accepts one constructed model and pipeline. Requests serialize
through a lock; overlapping cache scopes fail, and exceptions clear the active
scope. Cache lookup returns views of the resident BF16 payload without changing
the original math or adding GPU synchronization to each step. A cache failure
does not fall back to an online AdaLN computation or another model path.

This adapter accepts acceleration `off`. Other acceleration profiles remain
unsupported until independently migrated and verified. The current slice does
not establish dense hybrid8, NVFP4, runtime LoRA lifecycle or full-DiT switching
support; those retain their own acceptance contracts under the project epic.

## Reproducible comparison

`scripts/run_h3_reference.py` runs the same ordinary-scene reference case in two
separate environments, first with the legacy plugin and then with ComfyOmni.
The model, quantization, prompt, reference image, seed, dimensions, frame rate,
duration and schedule are supplied identically. Video and audio retain their
original output precision. The comparison rejects changed input identities,
modified output files or two results from the same plugin, and reports numerical
differences separately from the required visual review.

Use the immutable validated host image digest with `deploy/Dockerfile.h3` and
a wheel built from the exact candidate. That image removes the old plugin,
installs only the ComfyOmni wheel and verifies the installed wheel identity.
The source checkout left inside a host image is not proof of the executing
host version: validate installed module bytes against the pinned host source.

Containers mount models and inputs read-only and write outputs to a separate
directory. The pinned Triton runtime compiles and loads shared libraries from
its temporary cache, so its bounded `/tmp` tmpfs requires `exec`; `noexec`
prevents the first request from loading a compiled kernel. Numeric container
users receive `USER` and `LOGNAME` because this host's Torch cache initialization
calls `getpass.getuser()` even when `/etc/passwd` has no entry for that UID.

## Source attribution and distribution

All migrated implementation and characterization sources below are from
`h3-forge@e9cb011d00b028c149db3978de246c54f6e34acc`, licensed Apache-2.0.
The h3-forge contributors retain attribution in derived module headers.
The adapter, contract constants and derived tests are distributed under the
same license. Model payloads, generated media, local server evidence and
third-party dynamic VAE code are external assets and are not distributed here.

| Legacy source | Exact blob | New owner |
|---|---|---|
| `src/h3_forge/h3/h3_schedule.py` | `961a03c4df3a3e9f3ce5b209bb48561320fe17a9` | `runtime/h3/schedule.py` |
| `src/h3_forge/h3/api_contract.py` | `dda7c5806cd8c94d022b9c9bfb16a31d15f69b0b` | `runtime/h3/requests.py` |
| `src/h3_forge/h3/runtime_pipeline.py` | `fa94f86da746ff9a11105584081464c1162d07b6` | integration cache pipeline |
| `src/h3_forge/package_assembler.py` | `e64558f1d3bb6e1ee6f714b70e783d9df907f9ce` | legacy package and cache verification |
| `src/h3_forge/vae_export.py` | `531a63b91354a38214db5a07ce72815427e1d6d5` | `runtime/h3/legacy_vae.py` |
| `src/h3_forge/h3/profiles.py` | `b85a8b1cbf4a882474c83ac0f6f25a6a7434cd3e` | `runtime/h3/legacy_profiles.py` |
| `src/h3_forge/lora_hotswap/curve_adaln_cache.py` | `25e12cf6ec4299b79de988b38edc2ec718f9ccad` | raw cache verification |
| `tests/test_package_assembler.py` | `bd5c8ad67ca613c4e9a89511939dcc59145eb4b9` | legacy package regressions |
| `tests/test_vae_export.py` | `2c7eccfecc3f9b93e481883f72fb7f4a27310dd1` | legacy VAE regressions |
| `tests/lora_hotswap/test_curve_adaln_cache.py` | `93fbb4d95702b8fe70ffcfc9ebdfdfef9b290dc7` | raw cache regressions |
