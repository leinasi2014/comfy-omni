"""Pinned TP control transport: rank-zero dictionaries are not rank consensus.

Only tiny in-memory worker state is used. The native WorkerProc and executor
own RPC status aggregation; this test adds no plugin collective implementation.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import multiprocessing
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("vllm_omni") is None,
    reason="requires the actual pinned host image",
)


def _host_transport():
    os.environ.setdefault("USER", "h3-tp-status-test")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/h3-tp-status-inductor")
    import vllm_omni.diffusion.executor.multiproc_executor as executor_module
    import vllm_omni.diffusion.worker.diffusion_worker as worker_module

    for module, expected in (
        (worker_module, "25bca55e1dd2f3f6ff106c5912a4e286ca93bc8b"),
        (executor_module, "ff92539625a55a0ca76083765380bfc0f9705d88"),
    ):
        source = Path(module.__file__).read_bytes()
        assert hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source).hexdigest() == expected
    return worker_module, executor_module.MultiprocDiffusionExecutor


def _envelope(result, other_result=True, *, failed=False):
    from vllm_omni.diffusion.ipc import DIFFUSION_RPC_RESULT_ENVELOPE

    return {
        "type": DIFFUSION_RPC_RESULT_ENVELOPE,
        "method": "comfy_omni_check_h3_dit_status",
        "result": result,
        "rank_statuses": [
            {"rank": 0, "ok": True, "bool_result": result if type(result) is bool else None},
            {
                "rank": 1,
                "ok": not failed,
                "bool_result": other_result if type(other_result) is bool else None,
                "error_type": "RuntimeError" if failed else None,
                "error": "rank-one-mutation-failed" if failed else None,
            },
        ],
    }


def test_pinned_executor_aggregates_boolean_results_and_rank_errors():
    _, executor = _host_transport()
    unwrap = executor._unwrap_rpc_result_envelope
    assert unwrap(_envelope(True, True)) is True
    assert unwrap(_envelope(True, False)) is False
    assert unwrap(_envelope(False, True)) is False
    with pytest.raises(RuntimeError, match="rank 1.*rank-one-mutation-failed"):
        unwrap(_envelope(True, failed=True))
    # This is the exact information loss that the independent predicate closes.
    rank_zero = {"active_identity": "registered:a"}
    assert unwrap(_envelope(rank_zero, {"active_identity": "registered:b"})) == rank_zero


def test_controller_rejects_rank_one_mismatch_hidden_by_rank_zero_status():
    from test_h3_residency_control import _InlineClientBoundary, _omni

    from comfy_omni.integrations.vllm_omni.residency_control import (
        H3ResidencyControlError,
        H3ResidencyCoordinator,
    )

    _, executor = _host_transport()

    async def scenario():
        client = _InlineClientBoundary(0)
        native_call = client._engine.collective_rpc
        checks = []

        def rank_one_mismatch(method, timeout=None, args=(), kwargs=None, unique_reply_rank=None):
            if method == "comfy_omni_check_h3_dit_status":
                checks.append((args, kwargs))
                return [executor._unwrap_rpc_result_envelope(_envelope(True, False))]
            return native_call(method, timeout, args, kwargs, unique_reply_rank)

        client._engine.collective_rpc = rank_one_mismatch
        omni = _omni(client)
        try:
            with pytest.raises(H3ResidencyControlError):
                await H3ResidencyCoordinator(omni, stage_id=0).switch("b", transaction_id="tx-tp-mismatch")
            assert checks, "the controller never verified the hidden TP rank"
            assert await omni.is_paused() is True
            assert "comfy_omni_finalize_h3_dit" not in client.calls
        finally:
            client._executor.shutdown(wait=True)

    asyncio.run(scenario())


def _tiny_worker(worker_module):
    import torch

    from comfy_omni.integrations.vllm_omni.residency import H3ResidencyWorkerExtension
    from comfy_omni.runtime.hotel import H3TensorDescriptor, PreparedH3DitSelection

    def descriptor(selection):
        return PreparedH3DitSelection(
            selection=selection,
            identity=f"registered:{selection}",
            execution_profile="cpu-tp-status-17285",
            tensors=(H3TensorDescriptor(name="weight", shape=(1,), dtype="torch.float32"),),
            logical_bytes=4,
        )

    class TinyDiT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.beta4_ready = True

        def load_weights(self, weights):
            raise AssertionError("metadata RPC must not open or load weights")

        def post_load_weights(self):
            raise AssertionError("metadata RPC must not finalize weights")

    pipeline = SimpleNamespace(
        transformer=TinyDiT(),
        comfy_omni_active_h3_dit=descriptor("a"),
        comfy_omni_prepare_h3_dit=descriptor,
    )

    class EmptyWorker:
        pass

    wrapper = object.__new__(worker_module.WorkerWrapperBase)
    wrapper.base_worker_class = EmptyWorker
    wrapper.worker_extension_cls = H3ResidencyWorkerExtension
    wrapper.custom_pipeline_args = None
    worker = object.__new__(wrapper._prepare_worker_class())
    worker.model_runner = SimpleNamespace(
        pipeline=pipeline, state_cache={}, input_batch=None, cache_backend=None, offload_backend=None
    )
    worker._step_lora_state = {}
    wrapper.worker = worker
    return wrapper, pipeline


def _tp_worker(rank, directory):
    import torch

    worker_module, executor = _host_transport()
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=(Path(directory) / "gloo-init").as_uri(),
        timeout=timedelta(seconds=25),
    )
    try:
        wrapper, pipeline = _tiny_worker(worker_module)
        proc = object.__new__(worker_module.WorkerProc)
        proc.gpu_id = rank
        proc.result_mq = object() if rank == 0 else None
        proc.worker = wrapper

        def rpc(method, args=()):
            response, should_reply = proc._execute_rpc(
                {
                    "method": method,
                    "args": args,
                    "kwargs": {},
                    "output_rank": 0,
                    "exec_all_ranks": True,
                    "collect_rank_status": True,
                }
            )
            assert should_reply is (rank == 0)
            return response

        expected = {
            "phase": "idle",
            "active_selection": "a",
            "active_identity": "registered:a",
            "execution_profile": "cpu-tp-status-17285",
            "transaction_id": None,
            "target_selection": None,
            "target_identity": None,
            "poison_reason": None,
            "weight_residency": "loaded",
        }
        response = rpc("comfy_omni_check_h3_dit_status", (expected,))
        if rank == 0:
            assert executor._handle_rpc_response(response) is True
        if rank == 1:
            wrapper.worker._comfy_omni_h3_residency.active = pipeline.comfy_omni_prepare_h3_dit("b")
        response = rpc("comfy_omni_h3_residency_status")
        if rank == 0:
            assert executor._handle_rpc_response(response)["active_identity"] == "registered:a"
        response = rpc("comfy_omni_check_h3_dit_status", (expected,))
        if rank == 0:
            assert executor._handle_rpc_response(response) is False
        if rank == 1:
            wrapper.worker.model_runner.state_cache["busy"] = object()
        response = rpc("comfy_omni_prepare_h3_dit", ("tx-failure", "b"))
        if rank == 0:
            with pytest.raises(RuntimeError, match="rank 1.*not quiescent"):
                executor._handle_rpc_response(response)
        # Both ranks must reach this barrier after the mutation error. A plugin
        # status collective in the successful rank would break this ordering.
        torch.distributed.barrier()
        (Path(directory) / f"rank-{rank}.passed").write_text("complete", encoding="utf-8")
    finally:
        torch.distributed.destroy_process_group()


def test_actual_gloo_worker_rpc_rejects_mismatch_and_finishes_rank_failure(tmp_path):
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_tp_worker, args=(rank, str(tmp_path))) for rank in range(2)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=60)
            assert process.exitcode == 0, f"TP status worker failed or deadlocked: {process.exitcode}"
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    assert [path.read_text(encoding="utf-8") for path in sorted(tmp_path.glob("rank-*.passed"))] == [
        "complete",
        "complete",
    ]
