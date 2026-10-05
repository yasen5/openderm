"""FastAPI routes for the RX-axis service."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...config import GantryConfig
from ..cubemars import CubeMarsError
from ..security import bearer_token_matches
from .service import RxAxisService
from .types import RxAxisServerConfig, RxAxisServerError

if TYPE_CHECKING:
    from fastapi import FastAPI

    from ..api_models import RxMoveToRequest, RxVelocityRequest


def build_app(
    config: GantryConfig | None = None,
    server_config: RxAxisServerConfig | None = None,
    service: RxAxisService | None = None,
) -> "FastAPI":
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse

    # Keep the HTTP/Pydantic stack optional for client-only installations.
    # FastAPI resolves postponed endpoint annotations against module globals.
    from .. import api_models

    globals().update(
        {
            "RxMoveToRequest": api_models.RxMoveToRequest,
            "RxVelocityRequest": api_models.RxVelocityRequest,
        }
    )

    resolved_server_config = server_config or RxAxisServerConfig()
    resolved_config = (config or GantryConfig.from_env()).with_overrides(
        axis=resolved_server_config.axis
    )
    resolved_service = service or RxAxisService(resolved_config, resolved_server_config)
    app = FastAPI(title="OpenDerm RX-Axis Server", version="0.1.0")
    app.state.rx_axis = resolved_service

    @app.middleware("http")
    async def authenticate_control_request(http_request, call_next):
        token = resolved_server_config.control_token
        if token is not None and not bearer_token_matches(
            http_request.headers.get("Authorization"), token
        ):
            return JSONResponse(
                status_code=401,
                content={"detail": "invalid_or_missing_control_token"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(http_request)

    @app.on_event("startup")
    async def startup() -> None:
        await resolved_service.start()

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await resolved_service.stop()

    @app.get("/healthz", response_model=None)
    async def healthz():
        payload = resolved_service.health_payload()
        if not payload["ok"]:
            return JSONResponse(status_code=503, content=payload)
        return payload

    @app.get("/state", response_model=None)
    async def state():
        if not resolved_service.is_homed:
            raise HTTPException(status_code=409, detail="axis_not_homed")
        payload = resolved_service.state_payload()
        if payload["stale"]:
            return JSONResponse(status_code=503, content=payload)
        return payload

    @app.get("/limit-switches")
    async def limit_switches() -> dict[str, Any]:
        try:
            return resolved_service.limit_switch_payload()
        except (RxAxisServerError, CubeMarsError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/poll", response_model=None)
    async def poll():
        if not resolved_service.is_homed:
            raise HTTPException(status_code=409, detail="axis_not_homed")
        try:
            await resolved_service.poll_once()
        except (RxAxisServerError, CubeMarsError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        payload = resolved_service.state_payload()
        if payload["stale"]:
            return JSONResponse(status_code=503, content=payload)
        return payload

    @app.post("/move-to")
    async def move_to(body: RxMoveToRequest) -> dict[str, Any]:
        try:
            return await resolved_service.move_to(
                body.position_rad,
                body.speed_rad_s,
                body.accel_rad_s2,
            )
        except (RxAxisServerError, CubeMarsError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/velocity")
    async def velocity(body: RxVelocityRequest) -> dict[str, Any]:
        try:
            return await resolved_service.set_velocity(body.velocity_rad_s)
        except (RxAxisServerError, CubeMarsError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/home")
    async def home() -> dict[str, Any]:
        try:
            return (await resolved_service.home_axis()).to_payload()
        except (RxAxisServerError, CubeMarsError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/stop")
    async def stop() -> dict[str, Any]:
        return await resolved_service.stop_motor()

    @app.post("/clear-errors")
    async def clear_errors() -> dict[str, Any]:
        return await resolved_service.clear_errors()

    return app
