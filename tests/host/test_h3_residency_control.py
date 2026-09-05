from __future__ import annotations

import asyncio
import inspect
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

os.environ.setdefault("USER", "h3-residency-host-test")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/h3-residency-inductor")
pytest.importorskip("vllm_omni", reason="the actual pinned host image is required")

from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.inline_stage_diffusion_client import InlineStageDiffusionClient
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.entrypoints.async_omni import AsyncOmni

from comfy_omni.integrations.vllm_omni.residency_control import (
    H3ResidencyControlError,
    H3ResidencyCoordinator,
)


class _Scheduler:
    def __init__(self, busy: bool = False) -> None:
        self.busy = busy

    def has_requests(self) -> bool:
        return self.busy


class _InlineEngineBoundary:
    def __init__(self, busy: bool = False, malformed_prepare: bool = False) -> None:
        self._cv = threading.Condition()
        self.scheduler = _Scheduler(busy)
        self._out_streams = {"request": object()} if busy else {}
        self.abort_queue: queue.Queue[str] = queue.Queue()
        self._closed = False
        self.active = "a"
        self.phase = "idle"
        self.transaction_id: str | None = None
        self.target: str | None = None
        self.calls: list[str] = []
        self.malformed_prepare = malformed_prepare
        self.fail_commit = False
        self.hang_prepare_seconds = 0.0
        self.weight_residency = "loaded"

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None, unique_reply_rank=None):
        self.calls.append(method)
        if method == "comfy_omni_check_h3_dit_status":
            expected = args[0]
            return [all(self._status().get(key) == value for key, value in expected.items())]
        if method == "comfy_omni_prepare_h3_dit":
            if self.hang_prepare_seconds:
                time.sleep(self.hang_prepare_seconds)
            if self.malformed_prepare:
                return [True]
            self.transaction_id, self.target = args[0], args[1]
            self.phase = "prepared"
        elif method == "comfy_omni_commit_h3_dit":
            if self.fail_commit:
                raise RuntimeError("rank transaction failed")
            self.phase = "committed"
            self.weight_residency = "loaded"
        elif method == "comfy_omni_finalize_h3_dit":
            self.active = self.target or self.active
            self.phase = "idle"
            self.transaction_id = None
            self.target = None
        elif method == "comfy_omni_rollback_h3_dit":
            self.phase = "rolled_back"
            self.transaction_id = None
            self.target = None
        elif method == "comfy_omni_unload_h3_dit":
            self.weight_residency = "released" if args[0] == "release" else "cpu"
        elif method == "comfy_omni_load_h3_dit":
            self.weight_residency = "loaded"
        elif method != "comfy_omni_h3_residency_status":
            raise AssertionError(f"unexpected method: {method}")
        return [self._status()]

    def _status(self):
        return {
            "phase": self.phase,
            "active_selection": self.active,
            "active_identity": f"registered:{self.active}",
            "execution_profile": "test-h3-pinned-17285",
            "transaction_id": self.transaction_id,
            "target_selection": self.target,
            "target_identity": f"registered:{self.target}" if self.target else None,
            "cpu_cache_budget_bytes": 0,
            "poison_reason": None,
            "cpu_cached_bytes": 0,
            "cpu_cached_identity": None,
            "weight_residency": self.weight_residency,
            "cpu_weight_bytes": 12 if self.weight_residency in {"loaded", "cpu"} else 0,
            "device_weight_bytes": 0,
            "resident_weight_bytes": 12 if self.weight_residency in {"loaded", "cpu"} else 0,
            "cuda_memory_allocated_bytes": 0,
            "cuda_memory_reserved_bytes": 0,
            "worker_pid": os.getpid(),
            "pipeline_id": id(self),
            "transformer_id": id(self.scheduler),
            "shared_object_ids": {"text_encoder": id(self.abort_queue)},
        }


class _InlineClientBoundary(InlineStageDiffusionClient):
    def __init__(self, replica_id: int, *, busy: bool = False, malformed_prepare: bool = False) -> None:
        self.stage_id = 0
        self.replica_id = replica_id
        self.final_output = True
        self._engine = _InlineEngineBoundary(busy, malformed_prepare)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._tasks: dict[str, object] = {"request": object()} if busy else {}

    @property
    def active(self):
        return self._engine.active

    @property
    def phase(self):
        return self._engine.phase

    @property
    def calls(self):
        return self._engine.calls

    @property
    def fail_commit(self):
        return self._engine.fail_commit

    @fail_commit.setter
    def fail_commit(self, value):
        self._engine.fail_commit = value


def _omni(*clients: _InlineClientBoundary):
    pool = StagePool(0, list(clients))
    omni = object.__new__(AsyncOmni)
    omni._pause_cond = asyncio.Condition()
    omni._paused = False
    omni.request_states = {}
    omni.engine = SimpleNamespace(stage_pools=[pool])
    return omni


def test_pinned_host_boundaries_used_by_the_coordinator_are_present() -> None:
    pause = inspect.signature(AsyncOmni.pause_generation).parameters
    pool_rpc = inspect.signature(StagePool.collective_rpc).parameters
    inline_rpc = inspect.signature(InlineStageDiffusionClient.collective_rpc_async).parameters

    assert {"wait_for_inflight_requests", "clear_cache"} <= set(pause)
    assert {"replica_id", "method", "timeout", "args", "kwargs"} <= set(pool_rpc)
    assert {"method", "timeout", "args", "kwargs"} <= set(inline_rpc)
    assert "self._processes = processes" in inspect.getsource(MultiprocDiffusionExecutor._init_executor)


def test_switch_uses_every_live_replica_and_resumes_only_after_finalize() -> None:
    async def scenario():
        clients = (_InlineClientBoundary(0), _InlineClientBoundary(1))
        omni = _omni(*clients)
        coordinator = H3ResidencyCoordinator(omni, stage_id=0, drain_timeout_seconds=0.1, rpc_timeout_seconds=0.1)

        result = await coordinator.switch("b", transaction_id="tx-b")

        assert result["active_selection"] == "b"
        assert result["replica_ids"] == [0, 1]
        assert await omni.is_paused() is False
        for client in clients:
            assert client.active == "b"
            mutations = [
                method
                for method in client.calls
                if method not in {"comfy_omni_h3_residency_status", "comfy_omni_check_h3_dit_status"}
            ]
            assert mutations == [
                "comfy_omni_prepare_h3_dit",
                "comfy_omni_commit_h3_dit",
                "comfy_omni_finalize_h3_dit",
            ]

    asyncio.run(scenario())


def test_pause_does_not_count_as_drain_and_waits_for_real_scheduler_idle() -> None:
    async def scenario():
        client = _InlineClientBoundary(0, busy=True)
        omni = _omni(client)
        coordinator = H3ResidencyCoordinator(omni, stage_id=0, drain_timeout_seconds=0.2, rpc_timeout_seconds=0.1)

        async def finish_request():
            await asyncio.sleep(0.01)
            client._tasks.clear()
            with client._engine._cv:
                client._engine.scheduler.busy = False
                client._engine._out_streams.clear()

        clearing = asyncio.create_task(finish_request())
        await coordinator.switch("b", transaction_id="tx-b")
        await clearing
        assert "comfy_omni_prepare_h3_dit" in client.calls

    asyncio.run(scenario())


def test_truthy_or_partial_nested_reply_is_rejected_and_all_routes_are_rolled_back() -> None:
    async def scenario():
        clients = (_InlineClientBoundary(0), _InlineClientBoundary(1, malformed_prepare=True))
        omni = _omni(*clients)
        coordinator = H3ResidencyCoordinator(omni, stage_id=0, drain_timeout_seconds=0.1, rpc_timeout_seconds=0.1)

        with pytest.raises(H3ResidencyControlError, match="rank status"):
            await coordinator.switch("b", transaction_id="tx-b")

        assert await omni.is_paused() is True
        assert all("comfy_omni_rollback_h3_dit" in client.calls for client in clients)

    asyncio.run(scenario())


def test_rank_failure_rolls_back_every_prepared_replica_and_keeps_admission_paused() -> None:
    async def scenario():
        clients = (_InlineClientBoundary(0), _InlineClientBoundary(1))
        clients[1].fail_commit = True
        omni = _omni(*clients)
        coordinator = H3ResidencyCoordinator(omni, stage_id=0, drain_timeout_seconds=0.1, rpc_timeout_seconds=0.1)

        with pytest.raises(H3ResidencyControlError, match="commit"):
            await coordinator.switch("b", transaction_id="tx-b")

        assert await omni.is_paused() is True
        assert all(client.phase == "rolled_back" for client in clients)
        assert all("comfy_omni_rollback_h3_dit" in client.calls for client in clients)

    asyncio.run(scenario())


def test_unknown_scheduler_boundary_and_drain_timeout_both_fail_closed() -> None:
    async def scenario():
        unknown_client = _InlineClientBoundary(0)
        del unknown_client._engine
        unknown_omni = _omni(unknown_client)
        unknown = H3ResidencyCoordinator(unknown_omni, stage_id=0, drain_timeout_seconds=0.1)
        with pytest.raises(H3ResidencyControlError, match="scheduler idle cannot be proved"):
            await unknown.switch("b", transaction_id="tx-unknown")
        assert await unknown_omni.is_paused() is True

        busy_client = _InlineClientBoundary(0, busy=True)
        busy_omni = _omni(busy_client)
        timeout = H3ResidencyCoordinator(busy_omni, stage_id=0, drain_timeout_seconds=0.01)
        with pytest.raises(H3ResidencyControlError, match="drain timed out"):
            await timeout.switch("b", transaction_id="tx-timeout")
        assert await busy_omni.is_paused() is True
        assert busy_client.calls == []

    asyncio.run(scenario())


def test_rpc_timeout_is_unknown_state_and_requires_explicit_recovery() -> None:
    async def scenario():
        client = _InlineClientBoundary(0)
        client._engine.hang_prepare_seconds = 0.2
        omni = _omni(client)
        coordinator = H3ResidencyCoordinator(omni, stage_id=0, drain_timeout_seconds=0.1, rpc_timeout_seconds=0.01)

        with pytest.raises(H3ResidencyControlError, match="prepare.*timed out"):
            await coordinator.switch("b", transaction_id="tx-timeout")

        assert await omni.is_paused() is True
        client._executor.shutdown(wait=True)

    asyncio.run(scenario())


def test_explicit_unload_keeps_pause_until_load_succeeds_on_every_replica() -> None:
    async def scenario():
        clients = (_InlineClientBoundary(0), _InlineClientBoundary(1))
        omni = _omni(*clients)
        coordinator = H3ResidencyCoordinator(omni, stage_id=0, drain_timeout_seconds=0.1, rpc_timeout_seconds=0.1)

        unloaded = await coordinator.unload(mode="release")
        assert unloaded["weight_residency"] == "released"
        assert await omni.is_paused() is True
        assert all(client._engine.weight_residency == "released" for client in clients)

        status = await coordinator.status()
        assert status["weight_residency"] == "released"
        assert status["worker_pids"] == [os.getpid(), os.getpid()]
        assert status["worker_pid_scope"] == "reporting-rank"
        loaded = await coordinator.load()
        assert loaded["weight_residency"] == "loaded"
        assert await omni.is_paused() is False
        assert all(client._engine.weight_residency == "loaded" for client in clients)

    asyncio.run(scenario())


def test_cancelled_caller_does_not_abandon_an_inflight_switch_transaction() -> None:
    async def scenario():
        client = _InlineClientBoundary(0)
        client._engine.hang_prepare_seconds = 0.05
        omni = _omni(client)
        coordinator = H3ResidencyCoordinator(omni, stage_id=0, drain_timeout_seconds=0.1, rpc_timeout_seconds=0.5)

        request = asyncio.create_task(coordinator.switch("b", transaction_id="tx-cancelled-caller"))
        while "comfy_omni_prepare_h3_dit" not in client.calls:
            await asyncio.sleep(0)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request

        while coordinator._owned_tasks:
            await asyncio.sleep(0.01)
        assert client.active == "b"
        assert client.phase == "idle"
        assert await omni.is_paused() is False

    asyncio.run(scenario())


def test_switch_loads_target_directly_from_released_residency() -> None:
    async def scenario():
        client = _InlineClientBoundary(0)
        omni = _omni(client)
        coordinator = H3ResidencyCoordinator(omni, stage_id=0, drain_timeout_seconds=0.1, rpc_timeout_seconds=0.1)

        await coordinator.unload(mode="release")
        result = await coordinator.switch("b", transaction_id="tx-released-to-b")

        assert result["active_selection"] == "b"
        assert client._engine.weight_residency == "loaded"
        assert await omni.is_paused() is False

    asyncio.run(scenario())


def test_status_reads_all_rank_pids_from_inline_engine_parent_inventory() -> None:
    async def scenario():
        client = _InlineClientBoundary(0)
        client._engine.executor = SimpleNamespace(_processes=[SimpleNamespace(pid=31001), SimpleNamespace(pid=31002)])
        omni = _omni(client)
        coordinator = H3ResidencyCoordinator(omni, stage_id=0, drain_timeout_seconds=0.1, rpc_timeout_seconds=0.1)

        before = await coordinator.status()
        await coordinator.switch("b", transaction_id="tx-pid-inventory")
        after = await coordinator.status()

        assert before["worker_pids"] == after["worker_pids"] == [31001, 31002]
        assert before["worker_pids_by_replica"] == {"0": [31001, 31002]}
        assert before["worker_pid_scope"] == "parent-owned-all-ranks"

    asyncio.run(scenario())
