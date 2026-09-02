from __future__ import annotations

from comfy_omni.contracts import ARCHITECTURE_TEMPLATES, COMPILE_TIME_CATALOG, template_digest

EXPECTED = {
    "h3-te-pruned24-convrot": (168, 448, "ca3196815cec871f606ac14f8cd50e995674008b64c835867dd06423a3889a8e"),
    "h3-transformer-50l-convrot": (200, 0, "1bc9c6241c0b1e7c6b95f494b09b4bc8aee8ae07e59804522ebcc1ab657361d1"),
    "h3-transformer-50l-convrot-adaln64": (
        250,
        285,
        "41430e728c2ef641f2b1f5ee3db796fd9ec37a2c237f9bf9f7bc1c11f5833a25",
    ),
    "h3-transformer-50l-hybrid8-bf16-plain": (
        0,
        535,
        "9e6124e9d90c121198637ae42f6384cea4e72983a721fbc208304eac89ca94fb",
    ),
}


def test_audited_templates_keep_exact_counts_and_digests() -> None:
    observed = {
        name: (len(template.convrot_table()), len(template.non_quantized_inventory), template_digest(template))
        for name, template in ARCHITECTURE_TEMPLATES.items()
    }
    assert observed == EXPECTED


def test_compile_time_catalog_is_immutable_and_bound_to_templates() -> None:
    assert len(COMPILE_TIME_CATALOG.records) == 3
    for record in COMPILE_TIME_CATALOG.records.values():
        assert record.template_name in ARCHITECTURE_TEMPLATES
        assert record.contract.convrot_group_count == len(ARCHITECTURE_TEMPLATES[record.template_name].convrot_table())
    try:
        COMPILE_TIME_CATALOG.records["mutated"] = next(iter(COMPILE_TIME_CATALOG.records.values()))  # type: ignore[index]
    except TypeError:
        pass
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("catalog mapping must be immutable")
