"""Deterministic audio decode is scoped to the original-file H3 route."""

from __future__ import annotations

import hashlib
import logging
import os

import pytest
from beta4_host_fixture import actual_cpu_host
from test_h3_raw_residency import pytestmark as pytestmark
from test_h3_raw_residency import raw_runtime as raw_runtime


def _cudnn_flags(torch):
    cudnn = torch.backends.cudnn
    return cudnn.enabled, cudnn.allow_tf32, cudnn.benchmark, cudnn.deterministic


def _sha256(torch, tensor):
    payload = tensor.detach().contiguous().cpu().view(torch.uint8).numpy()
    return hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("raises", [False, True])
def test_raw_audio_decode_scopes_deterministic_cudnn_flags_and_restores_them(raises):
    with actual_cpu_host() as host:
        torch = host.torch
        from comfy_omni.integrations.vllm_omni.pipelines import beta4_pipeline as adapter

        observed = []

        class RecordingAudioVAE:
            def decode_latent(self, latent):
                observed.append(_cudnn_flags(torch))
                if raises:
                    raise RuntimeError("decode failed")
                return latent

        audio_vae_type = adapter._raw_audio_vae_type(RecordingAudioVAE)
        audio_vae = audio_vae_type()
        outer = (not raises, not raises, True, False)
        cudnn = torch.backends.cudnn
        with cudnn.flags(enabled=outer[0], allow_tf32=outer[1], benchmark=outer[2], deterministic=outer[3]):
            if raises:
                with pytest.raises(RuntimeError, match="decode failed"):
                    audio_vae.decode_latent("latent")
            else:
                assert audio_vae.decode_latent("latent") == "latent"
            assert _cudnn_flags(torch) == outer

        assert observed == [(outer[0], outer[1], False, True)]


def test_pipeline_construction_only_substitutes_audio_vae_for_raw_sources():
    with actual_cpu_host():
        from comfy_omni.integrations.vllm_omni.pipelines import beta4_pipeline as adapter

        original_audio_vae = adapter.pipeline.MiniMaxH3AudioVAE
        package_replacements = adapter._pipeline_construction_replacements(raw_binding=None)
        raw_replacements = adapter._pipeline_construction_replacements(raw_binding=object())

        assert "MiniMaxH3AudioVAE" not in package_replacements
        raw_audio_vae = raw_replacements["MiniMaxH3AudioVAE"]
        assert issubclass(raw_audio_vae, original_audio_vae)
        assert raw_audio_vae._comfy_omni_raw_deterministic_decode is True


@pytest.mark.parametrize("trace", [False, True])
def test_raw_audio_trace_is_explicit_and_records_input_and_output_hashes(monkeypatch, caplog, trace):
    with actual_cpu_host() as host:
        torch = host.torch
        from comfy_omni.integrations.vllm_omni.pipelines import beta4_pipeline as adapter

        latent = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
        waveform = torch.tensor([0.25, -0.5], dtype=torch.float32)

        class RecordingAudioVAE:
            def decode_latent(self, value):
                assert value is latent
                return waveform

        if trace:
            monkeypatch.setenv("COMFY_OMNI_H3_AUDIO_TRACE", "1")
        else:
            monkeypatch.delenv("COMFY_OMNI_H3_AUDIO_TRACE", raising=False)
        caplog.set_level(logging.INFO, logger=adapter.__name__)
        audio_vae = adapter._raw_audio_vae_type(RecordingAudioVAE)()

        assert audio_vae.decode_latent(latent) is waveform
        messages = [record.getMessage() for record in caplog.records if record.name == adapter.__name__]
        if trace:
            assert len(messages) == 1
            assert f"worker_pid={os.getpid()}" in messages[0]
            assert "latent_shape=(1, 2)" in messages[0]
            assert "latent_dtype=torch.bfloat16" in messages[0]
            assert f"latent_sha256={_sha256(torch, latent)}" in messages[0]
            assert f"waveform_sha256={_sha256(torch, waveform)}" in messages[0]
        else:
            assert messages == []


def test_raw_pipeline_constructs_the_scoped_audio_decoder(raw_runtime):
    fixture = raw_runtime
    audio_vae = fixture.pipeline.audio_vae

    assert type(audio_vae)._comfy_omni_raw_deterministic_decode is True
    assert type(audio_vae).__mro__[1].__name__ == "TinyVAE"
    assert fixture.pipeline._raw_beta4_binding is fixture.bindings["a"]
