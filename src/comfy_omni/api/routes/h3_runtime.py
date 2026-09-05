"""Runtime HTTP operations; the host injects its live coordinator factory."""

API_PREFIX = "/v1/comfy-omni/h3/runtime"


def build_router(coordinator_factory, control_error):
    """Create routes lazily, without importing a host or scanning model files."""
    from fastapi import APIRouter, HTTPException, Query, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, ConfigDict, Field

    router = APIRouter(prefix=API_PREFIX, tags=["ComfyOmni H3 runtime"])

    class StageRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        stage_id: int = Field(default=0, ge=0)

    class SwitchRequest(StageRequest):
        selection: str = Field(min_length=1, max_length=128)
        cpu_cache_budget_bytes: int = Field(default=0, ge=0, strict=True)

    class UnloadRequest(StageRequest):
        mode: str = Field(default="release", pattern="^(release|cpu)$")
        cpu_budget_bytes: int = Field(default=0, ge=0, strict=True)

    class RecoveryRequest(StageRequest):
        expected_active_selection: str = Field(min_length=1, max_length=128)

    def coordinator(request, stage_id):
        state = request.app.state
        engine = getattr(state, "diffusion_engine", None) or getattr(state, "engine_client", None)
        if engine is None:
            raise HTTPException(status_code=503, detail="H3 runtime engine is not initialized")
        controls = getattr(state, "comfy_omni_h3_controls", None)
        if controls is None:
            controls = {}
            state.comfy_omni_h3_controls = controls
        cached = controls.get(stage_id)
        if cached is None or cached.async_omni is not engine:
            cached = coordinator_factory(engine, stage_id=stage_id, rpc_timeout_seconds=600)
            controls[stage_id] = cached
        return cached

    async def run(operation):
        try:
            return await operation
        except control_error as error:
            return JSONResponse(
                status_code=409,
                content={"error": str(error), "phase": error.phase, "rollback_errors": error.rollback_errors},
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @router.get("")
    async def status(request: Request, stage_id: int = Query(default=0, ge=0)):
        return await run(coordinator(request, stage_id).status())

    @router.post("/switch")
    async def switch(request: Request, body: SwitchRequest):
        return await run(
            coordinator(request, body.stage_id).switch(
                body.selection, cpu_cache_budget_bytes=body.cpu_cache_budget_bytes
            )
        )

    @router.post("/unload")
    async def unload(request: Request, body: UnloadRequest):
        return await run(
            coordinator(request, body.stage_id).unload(mode=body.mode, cpu_budget_bytes=body.cpu_budget_bytes)
        )

    @router.post("/load")
    async def load(request: Request, body: StageRequest):
        return await run(coordinator(request, body.stage_id).load())

    @router.post("/resume")
    async def resume(request: Request, body: RecoveryRequest):
        return await run(coordinator(request, body.stage_id).resume_after_recovery(body.expected_active_selection))

    return router
