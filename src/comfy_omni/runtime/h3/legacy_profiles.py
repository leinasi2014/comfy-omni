"""Audited legacy VAE verifier constants (Apache-2.0).

Derived from h3-forge e9cb011 h3/profiles.py lines 348-458,
blob b85a8b1cbf4a882474c83ac0f6f25a6a7434cd3e. No conversion runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

VAE_EXPORT_SCHEMA = "h3-comfy-vae-export/v2"
VAE_FAILURE_SCHEMA = "h3-comfy-vae-export-failure/v2"
VAE_MANIFEST_NAME = "h3-comfy-vae-export.json"
PINNED_VLLM_OMNI_COMMIT = "17285c2f55a41bf15772676121814d59a60ace35"
PINNED_TEMPLATE_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
PROFILE_VIDEO_FP16 = "minimax-h3-comfy-video-vae-fp16-v1"
PROFILE_AUDIO_FP32_WEIGHTNORM = "minimax-h3-comfy-audio-vae-fp32-weightnorm-v1"


@dataclass(frozen=True)
class VaeProfileContract:
    profile: str
    component: str
    metadata_namespace: str
    source_tensor_count: int
    source_dtype: str
    source_schema_sha256: str
    metadata_static_sha256: str
    output_tensor_count: int
    output_schema_sha256: str
    stats_length: int
    template_weight_path: str
    template_config_static_sha256: str
    template_static_files: Mapping[str, str]
    weight_norm_prefixes: frozenset[str] = frozenset()


def _audio_weight_norm_prefixes() -> frozenset[str]:
    prefixes = {"decoder.conv_pre", "decoder.conv_post", "encoder.block.0", "encoder.block.7"}
    prefixes.update(f"decoder.ups.{index}.0" for index in range(7))
    prefixes.update(
        f"decoder.resblocks.{block}.convs{side}.{layer}" for block in range(21) for side in (1, 2) for layer in range(3)
    )
    prefixes.update(
        f"encoder.block.{block}.block.{inner}.block.{layer}"
        for block in range(1, 6)
        for inner in range(3)
        for layer in (1, 3)
    )
    prefixes.update(f"encoder.block.{block}.block.4" for block in range(1, 6))
    if len(prefixes) != 172:
        raise AssertionError("internal audio VAE weight-norm census is invalid")
    return frozenset(prefixes)


_VIDEO_TEMPLATE_FILES = {
    "attention.py": "c9db8465c57f0bfb40c0194227be6e34b2c1ff7c5b5d63abca9964eed6283767",
    "base_module.py": "0e7ddf5086179a306298693ac461ffcbc27ccb28dbc786b1a05060de7a3357c0",
    "conv.py": "b3adee35f27e5d372543242aa86ce5e0138f13b794b6c81f919001e3e2346723",
    "flash.py": "c1918463d303a16f278a670bb4f33d3f8d6f7653b0d473313ff5835699b86d04",
    "func.py": "0612c26a65f095699cc7acd51937cd153584fa5a0476956e46a4679aa575c22d",
    "klvae.py": "d05a22b4918ad303f147d6b0d0ae9038bd1af1f810a5553b04f3542ab9bcff96",
    "minimax_h3_video_vae.py": "77fdea69a48485b5434f65d08096657b5d78b78b7cdf24e1241038f237720138",
    "norm.py": "258b120ca8d9a16366d7b811fe93eebf98fe1564db0995e8b7747229c651919a",
    "normalize.py": "8db651acb9eb551c906a2850bbca8e9d851f00d07b30ccf66ceb6057da687a13",
    "parallel.py": "7193522beafd65bc3dbe40b9843e360ab8da3a0bb211e9bfb67bb0d5d0e66919",
    "source/config.json": "66c68f541e6578ce613ce7a0fc985eb59097038829e49f7535e6d08e6d95ab12",
    "utils.py": "309e45e9d9e33cd5516a10eac2d207eb9f940079a6d9bbe0ee80aaaf7eb42dcd",
    "vae_cnn.py": "3949b701e6893bd2ea070cd5b18ed8135a7c21d884290df449b1997212d65b11",
    "vae_module.py": "9558fb302e10a619f198e12e04701c68988e897d0b627f958f33d27fd2a30d94",
    "vae_processor.py": "32870580a1802d9163104a4d68170e35a4772106867181e1bd343dff11f87d2e",
    "vae_vit.py": "a372c28441b56ec2dbc6041dbd6ce2766c64c0e61b2c69a16594a97a11da3f58",
}

_AUDIO_TEMPLATE_FILES = {
    "config.yaml": "259f675c9f71eedc24ca4f23965cd32a1b3878fc894f56664c40328d99831a5e",
    "dac_activations.py": "74ca8bb9e8039f1ff362ca4219eee5240a7df9502f3dbdc3161d4e86fbd6dc04",
    "dac_alias_free_act.py": "db8761e66c0eaf9fce2dcccb59162d52b22ca0de9f699cf7d21ec7cd507e86fd",
    "dac_alias_free_filter.py": "a369523cd7f14f4f299d8e2e47f3c473da76ed1910e6e9134fd7a9fce02537e8",
    "dac_alias_free_resample.py": "fe35055893833c563e42adc5e5882cbe7628c82d48d33c300fcf0a9576997910",
    "dac_attn_proj.py": "0e04556fa5fd38d4a71e62fcf4d3242fd9a2464fd72ed62baef2ec0980d910c6",
    "dac_audio_vae.py": "ec04939602d4710d039e83f558a7572547669652f7fa967e3de0a3b6c48cfd48",
    "dac_bigvgan.py": "f5df4e6f633e74da479549ef7a6d2684964e4b65ac42c6765be42f9c7fcb7bd2",
    "dac_utils.py": "74ef0ba2c2a11ca35a7c0cc166fe88e564de01230000a0bcc8429103312ad9f0",
    "metadata.json": "755d0529d43b2b5c83590f6f44ca659bc68e6a21b01d5669c93e8b2965749bff",
    "minimax_h3_audio_vae.py": "63bc0dab6def69480cfd77ea820f9a9a84fc4bd6e56868e4f45ec1e1e200578f",
}

VAE_PROFILES: dict[str, VaeProfileContract] = {
    PROFILE_VIDEO_FP16: VaeProfileContract(
        profile=PROFILE_VIDEO_FP16,
        component="video_vae",
        metadata_namespace="minimax_h3_video_vae",
        source_tensor_count=562,
        source_dtype="F16",
        source_schema_sha256="5dda984db89024ab37c5994d8dd83f35ca0acc73f199bf755a67bcc1154565ed",
        metadata_static_sha256="7043a3769a111705ce0dbae7d82fa2a34aca0468169cf5a337e2d6a2b18958c7",
        output_tensor_count=560,
        output_schema_sha256="b7ffdf4fe1a0b15429f0ef50b7d485f608a752477290bfcb6502b8cc91400bdf",
        stats_length=24,
        template_weight_path="source/model.safetensors",
        template_config_static_sha256="a830d408a3ca545f253978e8c1b0a2bc1d67fa5bb8d40293a0abe5a516252682",
        template_static_files=_VIDEO_TEMPLATE_FILES,
    ),
    PROFILE_AUDIO_FP32_WEIGHTNORM: VaeProfileContract(
        profile=PROFILE_AUDIO_FP32_WEIGHTNORM,
        component="audio_vae",
        metadata_namespace="minimax_h3_audio_vae",
        source_tensor_count=917,
        source_dtype="F32",
        source_schema_sha256="2bd372977cb35d59c056b24ccc0ee286523d50a63d0c141dd28a7ab47ca48c32",
        metadata_static_sha256="1237700244cc1b1f086aef966fd3aac0684982aa0ae404c56b8428fb9e09074c",
        output_tensor_count=1087,
        output_schema_sha256="da4ff84a9cf635ef249a812adb7df47b1008f4cc0910ce5958bcc8ebca105abb",
        stats_length=32,
        template_weight_path="model.safetensors",
        template_config_static_sha256="192baa523cdd6abbf36002723706d879405dfaf92aaaaa67583e4384196f34ba",
        template_static_files=_AUDIO_TEMPLATE_FILES,
        weight_norm_prefixes=_audio_weight_norm_prefixes(),
    ),
}
