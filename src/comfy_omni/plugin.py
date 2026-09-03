"""Lightweight vLLM-Omni general-plugin entry point.

This shim delegates registration to the bootstrap coordinator. Importing this module is
intentionally lightweight: it loads no runtime architectures, host runtimes, model code, or HTTP
frameworks. Registration happens only when a resident host is present and only when ``register()``
is invoked.
"""

from __future__ import annotations


def register() -> None:
    from comfy_omni.integrations.vllm_omni import bootstrap

    bootstrap.register()


__all__ = ["register"]
