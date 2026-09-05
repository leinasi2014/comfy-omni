"""Root-process transaction coordinator for live H3 DiT residency.

The coordinator follows the pinned vLLM-Omni 17285 control path directly:
``AsyncOmni`` gates new admission, its ``AsyncOmniEngine`` owns logical
``StagePool`` objects, and each pool routes an RPC to one physical diffusion
client.  A diffusion client returns the singleton list produced by
``DiffusionEngine.collective_rpc``.  On the multiprocess executor that one
reply is emitted only after ``WorkerProc`` gathers the status of every rank.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any


class H3ResidencyControlError(RuntimeError):
    """A fail-closed root coordinator error."""

    def __init__(self, message: str, *, phase: str, rollback_errors: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.phase = phase
        self.rollback_errors = rollback_errors


@dataclass(frozen=True, slots=True)
class _Route:
    pool: Any
    replica_id: int
    client: Any


class H3ResidencyCoordinator:
    """Coordinate one H3 DiT selection across every live stage replica."""

    def __init__(
        self,
        async_omni: Any,
        *,
        stage_id: int,
        drain_timeout_seconds: float = 300.0,
        rpc_timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 0.01,
    ) -> None:
        if not isinstance(stage_id, int) or isinstance(stage_id, bool) or stage_id < 0:
            raise ValueError("stage_id must be a non-negative integer")
        if drain_timeout_seconds <= 0:
            raise ValueError("drain_timeout_seconds must be positive")
        if rpc_timeout_seconds <= 0:
            raise ValueError("rpc_timeout_seconds must be positive")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self.async_omni = async_omni
        self.stage_id = stage_id
        self.drain_timeout_seconds = float(drain_timeout_seconds)
        self.rpc_timeout_seconds = float(rpc_timeout_seconds)
        self.poll_interval_seconds = float(poll_interval_seconds)
        self._lock = asyncio.Lock()
        self._unload_pause_owned = False
        self._owned_tasks: set[asyncio.Task[dict[str, object]]] = set()

    async def status(self) -> dict[str, object]:
        """Return one strict, uniform status across every live stage replica."""
        return await self._run_owned(self._status())

    async def _status(self) -> dict[str, object]:
        async with self._lock:
            routes = self._snapshot_routes()
            statuses = await self._run_status(routes)
            return self._summarize_status(routes, statuses)

    async def unload(
        self,
        *,
        mode: str = "release",
        cpu_budget_bytes: int = 0,
    ) -> dict[str, object]:
        """Explicitly unload DiT weights and retain the admission pause."""
        if mode not in {"release", "cpu"}:
            raise ValueError("unload mode must be 'release' or 'cpu'")
        if not isinstance(cpu_budget_bytes, int) or isinstance(cpu_budget_bytes, bool) or cpu_budget_bytes < 0:
            raise ValueError("cpu_budget_bytes must be a non-negative integer")
        return await self._run_owned(self._unload(mode=mode, cpu_budget_bytes=cpu_budget_bytes))

    async def _unload(self, *, mode: str, cpu_budget_bytes: int) -> dict[str, object]:
        async with self._lock:
            was_paused = await self._is_paused()
            await self._pause()
            routes = self._snapshot_routes()
            await self._wait_for_scheduler_idle(routes)
            baseline = await self._run_status(routes)
            self._validate_baseline(routes, baseline, require_uniform_weight_residency=True)
            if any(status.get("weight_residency") != "loaded" for status in baseline):
                raise H3ResidencyControlError("H3 weights are not uniformly loaded before unload", phase="unload")
            expected = "released" if mode == "release" else "cpu"
            try:
                statuses = await self._run_memory_phase(
                    routes,
                    method="comfy_omni_unload_h3_dit",
                    args=(mode,),
                    kwargs={"cpu_budget_bytes": cpu_budget_bytes},
                    expected_residency=expected,
                )
            except Exception as error:
                raise H3ResidencyControlError(
                    f"H3 residency unload failed: {error}; generation remains paused",
                    phase="unload",
                ) from error
            self._unload_pause_owned = not was_paused
            return self._summarize_status(routes, statuses)

    async def load(self, *, resume_on_success: bool = True) -> dict[str, object]:
        """Reload active DiT weights on every route, then optionally resume admission."""
        if not isinstance(resume_on_success, bool):
            raise ValueError("resume_on_success must be a boolean")
        return await self._run_owned(self._load(resume_on_success=resume_on_success))

    async def _load(self, *, resume_on_success: bool) -> dict[str, object]:
        async with self._lock:
            await self._pause()
            routes = self._snapshot_routes()
            await self._wait_for_scheduler_idle(routes)
            baseline = await self._run_status(routes)
            self._validate_baseline(routes, baseline, require_uniform_weight_residency=False)
            try:
                statuses = await self._run_memory_phase(
                    routes,
                    method="comfy_omni_load_h3_dit",
                    args=(),
                    expected_residency="loaded",
                )
            except Exception as error:
                raise H3ResidencyControlError(
                    f"H3 residency load failed: {error}; generation remains paused",
                    phase="load",
                ) from error
            result = self._summarize_status(routes, statuses)
            if resume_on_success:
                await self.async_omni.resume_generation()
                self._unload_pause_owned = False
                result["resumed"] = True
            else:
                result["resumed"] = False
            return result

    async def switch(
        self,
        selection: str,
        *,
        transaction_id: str | None = None,
        cpu_cache_budget_bytes: int = 0,
    ) -> dict[str, object]:
        """Pause admission, drain, and run prepare/commit/finalize on all routes.

        Any uncertain or failed phase attempts rollback on every snapshotted
        route and deliberately leaves admission paused.  A successful switch
        resumes only if this coordinator acquired the pause.
        """
        if not isinstance(selection, str) or not selection:
            raise ValueError("selection must be a non-empty registered ID")
        if transaction_id is None:
            transaction_id = uuid.uuid4().hex
        if not isinstance(transaction_id, str) or not transaction_id:
            raise ValueError("transaction_id must be a non-empty string")
        if not isinstance(cpu_cache_budget_bytes, int) or isinstance(cpu_cache_budget_bytes, bool):
            raise ValueError("cpu_cache_budget_bytes must be an integer")
        if cpu_cache_budget_bytes < 0:
            raise ValueError("cpu_cache_budget_bytes must not be negative")
        return await self._run_owned(
            self._switch(
                selection,
                transaction_id=transaction_id,
                cpu_cache_budget_bytes=cpu_cache_budget_bytes,
            )
        )

    async def _switch(
        self,
        selection: str,
        *,
        transaction_id: str,
        cpu_cache_budget_bytes: int,
    ) -> dict[str, object]:
        async with self._lock:
            was_paused = await self._is_paused()
            await self._pause()
            routes = self._snapshot_routes()
            await self._wait_for_scheduler_idle(routes)
            baseline = await self._run_status(routes)
            self._validate_baseline(routes, baseline, require_uniform_weight_residency=False)

            phase = "prepare"
            try:
                await self._run_phase(
                    routes,
                    method="comfy_omni_prepare_h3_dit",
                    args=(transaction_id, selection),
                    kwargs={"cpu_cache_budget_bytes": cpu_cache_budget_bytes},
                    expected_phase="prepared",
                    transaction_id=transaction_id,
                    selection=selection,
                )
                phase = "commit"
                await self._run_phase(
                    routes,
                    method="comfy_omni_commit_h3_dit",
                    args=(transaction_id,),
                    expected_phase="committed",
                    transaction_id=transaction_id,
                    selection=selection,
                    expected_weight_residency="loaded",
                )
                phase = "finalize"
                finalized = await self._run_phase(
                    routes,
                    method="comfy_omni_finalize_h3_dit",
                    args=(transaction_id,),
                    expected_phase="idle",
                    transaction_id=None,
                    selection=selection,
                    require_active_selection=True,
                    expected_weight_residency="loaded",
                )
            except Exception as error:
                rollback_errors = await self._rollback_all(routes, transaction_id)
                detail = f"H3 residency {phase} failed: {error}"
                if rollback_errors:
                    detail += "; rollback incomplete: " + "; ".join(rollback_errors)
                raise H3ResidencyControlError(
                    detail,
                    phase=phase,
                    rollback_errors=rollback_errors,
                ) from error

            identities = {str(status["active_identity"]) for status in finalized}
            profiles = {str(status["execution_profile"]) for status in finalized}
            if len(identities) != 1 or len(profiles) != 1:
                rollback_errors = await self._rollback_all(routes, transaction_id)
                raise H3ResidencyControlError(
                    "H3 residency finalize produced inconsistent active identity or execution profile",
                    phase="finalize",
                    rollback_errors=rollback_errors,
                )

            resume = not was_paused or self._unload_pause_owned
            if resume:
                await self.async_omni.resume_generation()
                self._unload_pause_owned = False
            return {
                "transaction_id": transaction_id,
                "stage_id": self.stage_id,
                "replica_ids": [route.replica_id for route in routes],
                "active_selection": selection,
                "active_identity": identities.pop(),
                "execution_profile": profiles.pop(),
                "resumed": resume,
            }

    async def resume_after_recovery(self, expected_active_selection: str) -> dict[str, object]:
        """Resume only after every route reports one healthy, uniform active selection."""
        if not isinstance(expected_active_selection, str) or not expected_active_selection:
            raise ValueError("expected_active_selection must be a non-empty string")
        return await self._run_owned(self._resume_after_recovery(expected_active_selection))

    async def _resume_after_recovery(self, expected_active_selection: str) -> dict[str, object]:
        async with self._lock:
            if not await self._is_paused():
                raise H3ResidencyControlError("generation is not paused", phase="recovery")
            routes = self._snapshot_routes()
            await self._wait_for_scheduler_idle(routes)
            statuses = await self._run_status(routes)
            for route, status in zip(routes, statuses, strict=True):
                if status.get("phase") not in {"idle", "rolled_back"}:
                    raise H3ResidencyControlError(
                        f"route {route.replica_id} is not recoverable: phase={status.get('phase')!r}",
                        phase="recovery",
                    )
                if status.get("transaction_id") is not None or status.get("poison_reason") is not None:
                    raise H3ResidencyControlError(
                        f"route {route.replica_id} has unresolved transaction or poison state",
                        phase="recovery",
                    )
                if status.get("weight_residency") != "loaded":
                    raise H3ResidencyControlError(
                        f"route {route.replica_id} has no loaded H3 weights",
                        phase="recovery",
                    )
                if status.get("active_selection") != expected_active_selection:
                    raise H3ResidencyControlError(
                        f"route {route.replica_id} active selection differs during recovery",
                        phase="recovery",
                    )
            identities = {str(status["active_identity"]) for status in statuses}
            profiles = {str(status["execution_profile"]) for status in statuses}
            if len(identities) != 1 or len(profiles) != 1:
                raise H3ResidencyControlError(
                    "recovery status is inconsistent across replicas",
                    phase="recovery",
                )
            await self.async_omni.resume_generation()
            return {
                "stage_id": self.stage_id,
                "replica_ids": [route.replica_id for route in routes],
                "active_selection": expected_active_selection,
                "active_identity": identities.pop(),
                "execution_profile": profiles.pop(),
                "resumed": True,
            }

    async def _run_owned(
        self,
        operation: Coroutine[Any, Any, dict[str, object]],
    ) -> dict[str, object]:
        """Keep the serialized operation alive if its awaiting caller is cancelled."""
        task = asyncio.create_task(operation)
        self._owned_tasks.add(task)
        task.add_done_callback(self._owned_task_done)
        return await asyncio.shield(task)

    def _owned_task_done(self, task: asyncio.Task[dict[str, object]]) -> None:
        self._owned_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    async def _is_paused(self) -> bool:
        method = getattr(self.async_omni, "is_paused", None)
        if not callable(method):
            raise H3ResidencyControlError("AsyncOmni has no is_paused control", phase="pause")
        result = await method()
        if not isinstance(result, bool):
            raise H3ResidencyControlError("AsyncOmni returned an unknown pause state", phase="pause")
        return result

    async def _pause(self) -> None:
        method = getattr(self.async_omni, "pause_generation", None)
        if not callable(method):
            raise H3ResidencyControlError("AsyncOmni has no pause_generation control", phase="pause")
        await method(wait_for_inflight_requests=True, clear_cache=False)
        if not await self._is_paused():
            raise H3ResidencyControlError("AsyncOmni did not enter paused state", phase="pause")

    def _snapshot_routes(self) -> tuple[_Route, ...]:
        engine = getattr(self.async_omni, "engine", None)
        pools = getattr(engine, "stage_pools", None)
        if not isinstance(pools, list):
            raise H3ResidencyControlError("AsyncOmniEngine stage_pools are unavailable", phase="discovery")
        matches = [pool for pool in pools if getattr(pool, "stage_id", None) == self.stage_id]
        if len(matches) != 1:
            raise H3ResidencyControlError(
                f"expected one StagePool for stage {self.stage_id}, found {len(matches)}",
                phase="discovery",
            )
        pool = matches[0]
        if getattr(pool, "stage_type", None) != "diffusion":
            raise H3ResidencyControlError(f"stage {self.stage_id} is not a diffusion StagePool", phase="discovery")
        live_replica_ids = getattr(pool, "live_replica_ids", None)
        if not callable(live_replica_ids):
            raise H3ResidencyControlError("StagePool has no live replica inventory", phase="discovery")
        replica_ids = live_replica_ids()
        if (
            not isinstance(replica_ids, list)
            or not replica_ids
            or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in replica_ids)
            or len(replica_ids) != len(set(replica_ids))
        ):
            raise H3ResidencyControlError("StagePool live replica inventory is empty or invalid", phase="discovery")
        clients = getattr(pool, "clients", None)
        if not isinstance(clients, list):
            raise H3ResidencyControlError("StagePool clients inventory is unavailable", phase="discovery")

        routes: list[_Route] = []
        for replica_id in replica_ids:
            if replica_id >= len(clients) or clients[replica_id] is None:
                raise H3ResidencyControlError(
                    f"StagePool live replica {replica_id} has no attached client",
                    phase="discovery",
                )
            client = clients[replica_id]
            if getattr(client, "stage_type", None) != "diffusion":
                raise H3ResidencyControlError(
                    f"StagePool replica {replica_id} is not a diffusion client",
                    phase="discovery",
                )
            routes.append(_Route(pool=pool, replica_id=replica_id, client=client))
        return tuple(routes)

    def _assert_routes_unchanged(self, routes: tuple[_Route, ...]) -> None:
        pool = routes[0].pool
        current_ids = pool.live_replica_ids()
        expected_ids = [route.replica_id for route in routes]
        if current_ids != expected_ids:
            raise H3ResidencyControlError(
                f"StagePool membership changed during transaction: {current_ids!r} != {expected_ids!r}",
                phase="membership",
            )
        for route in routes:
            if pool.clients[route.replica_id] is not route.client:
                raise H3ResidencyControlError(
                    f"StagePool replica {route.replica_id} client changed during transaction",
                    phase="membership",
                )

    async def _wait_for_scheduler_idle(self, routes: tuple[_Route, ...]) -> None:
        request_states = getattr(self.async_omni, "request_states", None)
        if not isinstance(request_states, dict):
            raise H3ResidencyControlError("AsyncOmni request inventory is unavailable", phase="drain")
        for route in routes:
            engine = getattr(route.client, "_engine", None)
            tasks = getattr(route.client, "_tasks", None)
            if engine is None or not isinstance(tasks, dict):
                raise H3ResidencyControlError(
                    f"scheduler idle cannot be proved for stage {self.stage_id} replica {route.replica_id}",
                    phase="drain",
                )
            condition = getattr(engine, "_cv", None)
            scheduler = getattr(engine, "scheduler", None)
            if condition is None or not callable(getattr(scheduler, "has_requests", None)):
                raise H3ResidencyControlError(
                    f"scheduler idle cannot be proved for stage {self.stage_id} replica {route.replica_id}",
                    phase="drain",
                )

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.drain_timeout_seconds
        while True:
            self._assert_routes_unchanged(routes)
            busy = bool(request_states)
            for route in routes:
                engine = route.client._engine
                tasks = route.client._tasks
                condition = engine._cv
                with condition:
                    if getattr(engine, "_closed", False):
                        raise H3ResidencyControlError(
                            f"diffusion engine for replica {route.replica_id} is closed",
                            phase="drain",
                        )
                    out_streams = getattr(engine, "_out_streams", None)
                    abort_queue = getattr(engine, "abort_queue", None)
                    if not isinstance(out_streams, dict) or not callable(getattr(abort_queue, "empty", None)):
                        raise H3ResidencyControlError(
                            f"scheduler idle cannot be proved for stage {self.stage_id} replica {route.replica_id}",
                            phase="drain",
                        )
                    busy = busy or bool(tasks) or bool(engine.scheduler.has_requests())
                    busy = busy or bool(out_streams) or not abort_queue.empty()
            if not busy:
                return
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise H3ResidencyControlError(
                    f"diffusion scheduler drain timed out after {self.drain_timeout_seconds:g} seconds",
                    phase="drain",
                )
            await asyncio.sleep(min(self.poll_interval_seconds, remaining))

    async def _run_phase(
        self,
        routes: tuple[_Route, ...],
        *,
        method: str,
        args: tuple[object, ...],
        expected_phase: str,
        transaction_id: str | None,
        selection: str,
        kwargs: dict[str, object] | None = None,
        require_active_selection: bool = False,
        expected_weight_residency: str | None = None,
    ) -> list[dict[str, object]]:
        for route in routes:
            self._assert_routes_unchanged(routes)
            result = await self._call_route(route, method, args=args, kwargs=kwargs)
            self._validate_rank_status(result, route, method)
        statuses = await self._run_status(routes)
        for route, status in zip(routes, statuses, strict=True):
            if status.get("phase") != expected_phase:
                raise H3ResidencyControlError(
                    f"{method} returned phase {status.get('phase')!r} for replica {route.replica_id}",
                    phase=method,
                )
            if status.get("transaction_id") != transaction_id:
                raise H3ResidencyControlError(
                    f"{method} returned an unexpected transaction for replica {route.replica_id}",
                    phase=method,
                )
            selected_field = "active_selection" if require_active_selection else "target_selection"
            if status.get(selected_field) != selection:
                raise H3ResidencyControlError(
                    f"{method} returned an unexpected {selected_field} for replica {route.replica_id}",
                    phase=method,
                )
            if expected_weight_residency is not None and status.get("weight_residency") != expected_weight_residency:
                raise H3ResidencyControlError(
                    f"{method} returned unexpected weight residency for replica {route.replica_id}",
                    phase=method,
                )
        return statuses

    async def _run_status(self, routes: tuple[_Route, ...]) -> list[dict[str, object]]:
        statuses: list[dict[str, object]] = []
        for route in routes:
            self._assert_routes_unchanged(routes)
            result = await self._call_route(route, "comfy_omni_h3_residency_status", args=())
            status = self._validate_rank_status(result, route, "comfy_omni_h3_residency_status")
            expected = {key: status[key] for key in self._critical_status_keys()}
            consensus = await self._call_route(
                route,
                "comfy_omni_check_h3_dit_status",
                args=(expected,),
            )
            self._validate_bool_consensus(consensus, route)
            statuses.append(status)
        return statuses

    @staticmethod
    def _validate_baseline(
        routes: tuple[_Route, ...],
        statuses: list[dict[str, object]],
        *,
        require_uniform_weight_residency: bool,
    ) -> None:
        for route, status in zip(routes, statuses, strict=True):
            if status.get("phase") not in {"idle", "rolled_back"} or status.get("transaction_id") is not None:
                raise H3ResidencyControlError(
                    f"replica {route.replica_id} has an unresolved residency transaction",
                    phase="baseline",
                )
        identities = {str(status["active_identity"]) for status in statuses}
        profiles = {str(status["execution_profile"]) for status in statuses}
        if len(identities) != 1 or len(profiles) != 1:
            raise H3ResidencyControlError(
                "active H3 identity or execution profile is inconsistent across replicas",
                phase="baseline",
            )
        residencies = {str(status["weight_residency"]) for status in statuses}
        if require_uniform_weight_residency and len(residencies) != 1:
            raise H3ResidencyControlError(
                "weight residency is inconsistent across replicas",
                phase="baseline",
            )

    async def _run_memory_phase(
        self,
        routes: tuple[_Route, ...],
        *,
        method: str,
        args: tuple[object, ...],
        expected_residency: str,
        kwargs: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        for route in routes:
            self._assert_routes_unchanged(routes)
            result = await self._call_route(route, method, args=args, kwargs=kwargs)
            self._validate_rank_status(result, route, method)
        statuses = await self._run_status(routes)
        for route, status in zip(routes, statuses, strict=True):
            if status.get("weight_residency") != expected_residency:
                raise H3ResidencyControlError(
                    f"{method} returned weight residency {status.get('weight_residency')!r} "
                    f"for replica {route.replica_id}",
                    phase=method,
                )
        return statuses

    def _summarize_status(
        self,
        routes: tuple[_Route, ...],
        statuses: list[dict[str, object]],
    ) -> dict[str, object]:
        fields = ("active_selection", "active_identity", "execution_profile", "weight_residency")
        values = {field: {str(status[field]) for status in statuses} for field in fields}
        inconsistent = [field for field, observed in values.items() if len(observed) != 1]
        if inconsistent:
            raise H3ResidencyControlError(
                f"H3 status is inconsistent across replicas: {inconsistent}",
                phase="status",
            )
        worker_pid_inventories = [
            self._route_worker_pids(route, status) for route, status in zip(routes, statuses, strict=True)
        ]
        worker_pid_scopes = {scope for _, scope in worker_pid_inventories}
        worker_pids_by_replica = {
            str(route.replica_id): pids for route, (pids, _) in zip(routes, worker_pid_inventories, strict=True)
        }
        return {
            "stage_id": routes[0].pool.stage_id,
            "replica_ids": [route.replica_id for route in routes],
            **{field: observed.pop() for field, observed in values.items()},
            "cpu_weight_bytes": sum(int(status["cpu_weight_bytes"]) for status in statuses),
            "device_weight_bytes": sum(int(status["device_weight_bytes"]) for status in statuses),
            "resident_weight_bytes": sum(int(status["resident_weight_bytes"]) for status in statuses),
            "cuda_memory_allocated_bytes": sum(int(status["cuda_memory_allocated_bytes"]) for status in statuses),
            "cuda_memory_reserved_bytes": sum(int(status["cuda_memory_reserved_bytes"]) for status in statuses),
            "weight_bytes_scope": "reporting-rank-per-replica",
            "worker_pids": [pid for pids, _ in worker_pid_inventories for pid in pids],
            "worker_pids_by_replica": worker_pids_by_replica,
            "worker_pid_scope": worker_pid_scopes.pop() if len(worker_pid_scopes) == 1 else "mixed",
            "routes": statuses,
        }

    @staticmethod
    def _route_worker_pids(route: _Route, status: dict[str, object]) -> tuple[list[int], str]:
        """Read the inline engine's parent-owned process inventory when present."""
        engine = getattr(route.client, "_engine", None)
        executor = getattr(engine, "executor", None)
        processes = getattr(executor, "_processes", None)
        if isinstance(processes, list) and processes:
            pids = [getattr(process, "pid", None) for process in processes]
            if any(not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0 for pid in pids):
                raise H3ResidencyControlError(
                    f"invalid parent-owned worker process inventory for replica {route.replica_id}",
                    phase="status",
                )
            return pids, "parent-owned-all-ranks"
        return [int(status["worker_pid"])], "reporting-rank"

    async def _call_route(
        self,
        route: _Route,
        method: str,
        *,
        args: tuple[object, ...],
        kwargs: dict[str, object] | None = None,
    ) -> object:
        call = route.pool.collective_rpc(
            replica_id=route.replica_id,
            method=method,
            timeout=self.rpc_timeout_seconds,
            args=args,
            kwargs=kwargs,
        )
        cancellation_margin = min(1.0, max(0.05, self.rpc_timeout_seconds * 0.1))
        try:
            return await asyncio.wait_for(call, timeout=self.rpc_timeout_seconds + cancellation_margin)
        except TimeoutError as error:
            raise H3ResidencyControlError(
                f"{method} timed out for stage {self.stage_id} replica {route.replica_id}",
                phase=method,
            ) from error

    @staticmethod
    def _validate_rank_status(
        result: object,
        route: _Route,
        method: str,
    ) -> dict[str, object]:
        if not isinstance(result, list) or len(result) != 1:
            raise H3ResidencyControlError(
                f"{method} returned an unknown or partial rank status for replica {route.replica_id}",
                phase=method,
            )
        status = result[0]
        if not isinstance(status, dict):
            raise H3ResidencyControlError(
                f"{method} returned a non-dict rank status for replica {route.replica_id}",
                phase=method,
            )
        if status.get("supported") is False or status.get("todo"):
            raise H3ResidencyControlError(
                f"{method} is unsupported for replica {route.replica_id}: {status.get('error')}",
                phase=method,
            )
        required = {
            "phase",
            "active_selection",
            "active_identity",
            "execution_profile",
            "transaction_id",
            "target_selection",
            "target_identity",
            "poison_reason",
            "weight_residency",
            "cpu_weight_bytes",
            "device_weight_bytes",
            "resident_weight_bytes",
            "cuda_memory_allocated_bytes",
            "cuda_memory_reserved_bytes",
            "worker_pid",
            "pipeline_id",
            "transformer_id",
            "shared_object_ids",
        }
        if not required <= status.keys():
            missing = sorted(required - status.keys())
            raise H3ResidencyControlError(
                f"{method} rank status is incomplete for replica {route.replica_id}: missing={missing}",
                phase=method,
            )
        if status.get("poison_reason") is not None or status.get("phase") == "poisoned":
            raise H3ResidencyControlError(
                f"{method} reported a poisoned worker for replica {route.replica_id}",
                phase=method,
            )
        if status["resident_weight_bytes"] != status["cpu_weight_bytes"] + status["device_weight_bytes"]:
            raise H3ResidencyControlError(
                f"{method} returned inconsistent resident weight bytes for replica {route.replica_id}",
                phase=method,
            )
        if (
            not isinstance(status["worker_pid"], int)
            or isinstance(status["worker_pid"], bool)
            or status["worker_pid"] <= 0
            or not isinstance(status["pipeline_id"], int)
            or not isinstance(status["transformer_id"], int)
            or not isinstance(status["shared_object_ids"], dict)
        ):
            raise H3ResidencyControlError(
                f"{method} returned invalid worker identity observations for replica {route.replica_id}",
                phase=method,
            )
        return status

    @staticmethod
    def _critical_status_keys() -> tuple[str, ...]:
        return (
            "phase",
            "active_selection",
            "active_identity",
            "execution_profile",
            "transaction_id",
            "target_selection",
            "target_identity",
            "poison_reason",
            "weight_residency",
        )

    @staticmethod
    def _validate_bool_consensus(result: object, route: _Route) -> None:
        if not isinstance(result, list) or len(result) != 1 or result[0] is not True:
            raise H3ResidencyControlError(
                f"H3 rank status differs within replica {route.replica_id}",
                phase="rank_consensus",
            )

    async def _rollback_all(self, routes: tuple[_Route, ...], transaction_id: str) -> tuple[str, ...]:
        errors: list[str] = []
        for route in routes:
            clients = getattr(route.pool, "clients", None)
            if (
                not isinstance(clients, list)
                or route.replica_id >= len(clients)
                or clients[route.replica_id] is not route.client
            ):
                errors.append(f"replica {route.replica_id}: client changed before rollback")
                continue
            try:
                result = await self._call_route(
                    route,
                    "comfy_omni_rollback_h3_dit",
                    args=(transaction_id,),
                )
                status = self._validate_rank_status(result, route, "comfy_omni_rollback_h3_dit")
                if status.get("phase") != "rolled_back" or status.get("transaction_id") is not None:
                    raise H3ResidencyControlError(
                        f"rollback returned an unexpected state for replica {route.replica_id}",
                        phase="rollback",
                    )
            except Exception as error:
                errors.append(f"replica {route.replica_id}: {error}")
        try:
            statuses = await self._run_status(routes)
            for route, status in zip(routes, statuses, strict=True):
                if status.get("phase") not in {"idle", "rolled_back"} or status.get("transaction_id") is not None:
                    errors.append(f"replica {route.replica_id}: rollback left unresolved state")
        except Exception as error:
            errors.append(f"rollback rank consensus: {error}")
        return tuple(errors)


__all__ = ["H3ResidencyControlError", "H3ResidencyCoordinator"]
