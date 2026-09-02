from __future__ import annotations

import json
import sys

from comfy_omni.cli import main


def test_contract_list_uses_base_install_without_optional_runtimes(capsys, monkeypatch) -> None:
    monkeypatch.delenv("H3_FORGE_CONTRACT_DIR", raising=False)
    before = set(sys.modules)
    assert main(["contract", "list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 3
    imported = set(sys.modules) - before
    assert not any(name == "torch" or name.startswith("torch.") for name in imported)
    assert not any(name == "vllm" or name.startswith("vllm.") for name in imported)
