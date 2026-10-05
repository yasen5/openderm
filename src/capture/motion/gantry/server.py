from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, replace
import json
import socket
from typing import TYPE_CHECKING, Any
from urllib import error, request

from ...config import GantryConfig
from ..security import (
    MotionSecurityError,
    bearer_token_matches,
    control_token_from_env,
    require_secure_bind,
)
from .manager import MotionError, MotionManager
from .moonraker import MoonrakerError
from .moonraker_coordinator import MoonrakerCoordinatorClient
from .state import StateSnapshot, StateStore, project_position

if TYPE_CHECKING:
    from fastapi import FastAPI, WebSocket

    from ..api_models import (
        GantryHomeRequest,
        GantryMoveRequest,
        GantryStopRequest,
        GantryStreamStartRequest,
        GantryStreamTargetRequest,
    )


def x_only_gantry_config(config: GantryConfig) -> GantryConfig:
    """Return the fixed hardware configuration for the Klipper gantry service."""
    return replace(
        config,
        axis="x",
        travel_min_mm=0.0,
        travel_max_mm=800.0,
    )


class GantryServerError(RuntimeError):
    """Raised when the gantry server or its client encounters an error."""


# Conservative Y capture window used by tools that require an additional
# rectangular bound. Contour scanning uses the full-pose collision envelope.
Y_SOFT_MIN_MM = 50.0
Y_SOFT_MAX_MM = 570.0


class GantrySoftLimitError(GantryServerError):
    """Raised when a Pico-controlled axis rejects a target at a soft limit."""


@dataclass(frozen=True)
class GantryServerConfig:
    host: str = "127.0.0.1"
    port: int = 8090
    control_token: str | None = None


class GantryCoordinatorService:
    def __init__(self, config: GantryConfig) -> None:
        # The OpenDerm Klipper instance owns X only.
        self.config = x_only_gantry_config(config)
        self.state_store = StateStore()
        self.moonraker = MoonrakerCoordinatorClient(self.config.moonraker_ws_url, self.state_store)
        self.motion = MotionManager(self.config, self.state_store, self.moonraker)
        self._moonraker_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._moonraker_task = asyncio.create_task(self.moonraker.run(), name="moonraker-ws-client")
        await self.motion.start()

    async def stop(self) -> None:
        await self.motion.stop()
        await self.moonraker.close()
        if self._moonraker_task is not None:
            self._moonraker_task.cancel()
            try:
                await self._moonraker_task
            except asyncio.CancelledError:
                pass
        self._moonraker_task = None

    def health_payload(self) -> dict[str, Any]:
        snapshot = self.state_store.get()
        moonraker_task_running = (
            self._moonraker_task is not None and not self._moonraker_task.done()
        )
        issues: list[str] = []
        if not moonraker_task_running:
            issues.append("moonraker_loop_not_running")
        if not self.moonraker.connected:
            issues.append("moonraker_disconnected")
        if not self.motion.worker_running:
            issues.append("motion_worker_not_running")
        if snapshot.stale:
            issues.append("state_stale")
        if snapshot.webhooks_state != "ready":
            issues.append("klipper_not_ready")
        if snapshot.fault is not None:
            issues.append("klipper_fault")
        if snapshot.emergency_latched:
            issues.append("emergency_stop_latched")
        return {
            "ok": not issues,
            "issues": issues,
            "moonraker_connected": self.moonraker.connected,
            "moonraker_loop_running": moonraker_task_running,
            "motion_worker_running": self.motion.worker_running,
            "state_stale": snapshot.stale,
            "klipper_state": snapshot.webhooks_state,
            "fault": snapshot.fault,
            "emergency_latched": snapshot.emergency_latched,
        }


def build_app(
    config: GantryConfig | None = None,
    server_config: GantryServerConfig | None = None,
) -> "FastAPI":
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse

    # Keep the HTTP/Pydantic stack optional for users of GantryServerClient.
    # FastAPI resolves postponed endpoint annotations against module globals,
    # so publish the lazily imported model classes before declaring routes.
    from .. import api_models

    globals().update(
        {
            "WebSocket": WebSocket,
            "GantryHomeRequest": api_models.GantryHomeRequest,
            "GantryMoveRequest": api_models.GantryMoveRequest,
            "GantryStopRequest": api_models.GantryStopRequest,
            "GantryStreamStartRequest": api_models.GantryStreamStartRequest,
            "GantryStreamTargetRequest": api_models.GantryStreamTargetRequest,
        }
    )

    resolved_config = config or GantryConfig.from_env()
    resolved_server_config = server_config or GantryServerConfig(
        control_token=control_token_from_env()
    )
    service = GantryCoordinatorService(resolved_config)
    app = FastAPI(title="OpenDerm Gantry Coordinator", version="0.2.0")
    app.state.coordinator = service

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
        await service.start()

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await service.stop()

    @app.get("/healthz", response_model=None)
    async def healthz():
        payload = service.health_payload()
        if not payload["ok"]:
            return JSONResponse(status_code=503, content=payload)
        return payload

    @app.get("/position", response_model=None)
    async def position():
        snapshot = service.state_store.get()
        payload = position_payload(snapshot)
        if snapshot.stale:
            return JSONResponse(status_code=503, content=payload)
        return payload

    @app.get("/state", response_model=None)
    async def state():
        snapshot = service.state_store.get()
        payload = state_payload(snapshot)
        if snapshot.stale:
            return JSONResponse(status_code=503, content=payload)
        return payload

    @app.get("/move/{move_id}")
    async def move_status(move_id: str) -> dict[str, Any]:
        record = service.motion.get_record(move_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown_move_id")
        return record.to_dict()

    @app.post("/move")
    async def move(body: GantryMoveRequest) -> dict[str, Any]:
        if service.motion.is_streaming:
            raise HTTPException(status_code=409, detail="stream_active")
        try:
            record = await service.motion.enqueue_move(
                target={"x": body.x},
                feed_mm_s=body.feed_mm_s,
                tolerance_mm=body.tolerance_mm,
                commander_id=body.commander_id,
            )
        except MotionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if body.blocking:
            await record.waiter.wait()
            return service.motion.get_record(record.move_id).to_dict()
        return {"move_id": record.move_id, "status": "queued"}

    @app.post("/home")
    async def home(body: GantryHomeRequest | None = None) -> dict[str, Any]:
        if service.motion.is_streaming:
            raise HTTPException(status_code=409, detail="stream_active")
        payload = body or GantryHomeRequest()
        try:
            record = await service.motion.enqueue_home(payload.axes)
        except MotionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if payload.blocking:
            await record.waiter.wait()
            return service.motion.get_record(record.move_id).to_dict()
        return {"move_id": record.move_id, "status": "queued"}

    @app.post("/stop")
    async def stop(body: GantryStopRequest | None = None) -> dict[str, Any]:
        payload = body or GantryStopRequest()
        if payload.mode == "soft":
            try:
                return await service.motion.request_soft_stop()
            except MotionError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
        try:
            return await service.motion.request_emergency_stop()
        except MotionError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/stream/start")
    async def stream_start(
        body: GantryStreamStartRequest | None = None,
    ) -> dict[str, Any]:
        payload = body or GantryStreamStartRequest()
        try:
            return await service.motion.start_stream(
                feed_mm_s=payload.resolved_feed_mm_s,
                tick_s=payload.tick_s,
                min_step_mm=payload.min_step_mm,
            )
        except MotionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/stream/target")
    async def stream_target(body: GantryStreamTargetRequest) -> dict[str, Any]:
        try:
            resolved = await service.motion.update_stream_target({"x": body.x})
        except MotionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "target": resolved}

    @app.post("/stream/stop")
    async def stream_stop() -> dict[str, Any]:
        return await service.motion.stop_stream()

    @app.post("/reset")
    async def reset() -> dict[str, Any]:
        record = await service.motion.enqueue_control("reset")
        await record.waiter.wait()
        return service.motion.get_record(record.move_id).to_dict()

    @app.post("/enable")
    async def enable() -> dict[str, Any]:
        try:
            record = await service.motion.enqueue_control("enable")
        except MotionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        await record.waiter.wait()
        return record.to_dict() | (service.motion.jobs[record.move_id].result or {})

    @app.post("/disable")
    async def disable() -> dict[str, Any]:
        snapshot = service.state_store.get()
        if snapshot.is_moving:
            raise HTTPException(status_code=409, detail="disable_conflict")
        try:
            record = await service.motion.enqueue_control("disable")
        except MotionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        await record.waiter.wait()
        if record.status == "failed":
            raise HTTPException(status_code=400, detail=record.error or "disable_failed")
        return record.to_dict() | (service.motion.jobs[record.move_id].result or {})

    @app.websocket("/stream")
    async def stream(websocket: WebSocket) -> None:
        await websocket.accept()
        subscription = service.state_store.subscribe()
        receive_task = asyncio.create_task(
            _receive_stream_messages(
                websocket,
                service.motion,
                WebSocketDisconnect,
            )
        )
        try:
            while True:
                snapshot = await subscription.get()
                await websocket.send_json(state_payload(snapshot))
        except WebSocketDisconnect:
            pass
        finally:
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass
            service.state_store.unsubscribe(subscription)

    return app


def position_payload(snapshot: StateSnapshot) -> dict[str, Any]:
    return {
        "position": round_position(project_position(snapshot)),
        "velocity": round(snapshot.velocity_mm_s, 6),
        "moving": snapshot.is_moving,
        "timestamp": round(snapshot.timestamp, 6),
        "stale": snapshot.stale,
    }


def state_payload(snapshot: StateSnapshot) -> dict[str, Any]:
    return {
        "position": round_position(project_position(snapshot)),
        "velocity": round(snapshot.velocity_mm_s, 6),
        "moving": snapshot.is_moving,
        "timestamp": round(snapshot.timestamp, 6),
        "stale": snapshot.stale,
        "last_commanded_target": None
        if snapshot.last_commanded_target is None
        else round_position(snapshot.last_commanded_target),
        "homed_axes": snapshot.homed_axes,
        "enabled": snapshot.enabled,
        "active_move_id": snapshot.active_move_id,
        "queue_depth": snapshot.queue_depth,
        "fault": snapshot.fault,
        "stop_requested": snapshot.stop_requested,
        "streaming": snapshot.streaming,
        "emergency_latched": snapshot.emergency_latched,
        "webhooks_state": snapshot.webhooks_state,
        "webhooks_message": snapshot.webhooks_message,
    }


def round_position(position: dict[str, float]) -> dict[str, float]:
    return {axis: round(value, 6) for axis, value in position.items()}


async def _receive_stream_messages(
    websocket: "WebSocket",
    motion: MotionManager,
    disconnect_error: type[Exception],
) -> None:
    while True:
        try:
            incoming = await websocket.receive_text()
        except disconnect_error:
            return
        message = json.loads(incoming)
        if message.get("type") == "ping" and message.get("commander_id"):
            motion.register_commander_ping(str(message["commander_id"]))


class GantryServerClient:
    def __init__(
        self,
        base_url: str,
        axis: str = "x",
        timeout_s: float = 5.0,
        control_token: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.axis = axis.lower()
        if self.axis != "x":
            raise ValueError("The OpenDerm Klipper gantry client controls X only.")
        self.timeout_s = timeout_s
        self.control_token = (
            control_token if control_token is not None else control_token_from_env()
        )

    def status(self) -> dict[str, Any]:
        return self._request_json("/state")

    def move_to(
        self,
        position_mm: float,
        feed_mm_min: float | None = None,
        tolerance_mm: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {self.axis: position_mm, "blocking": True}
        if feed_mm_min is not None:
            payload["feed_mm_s"] = feed_mm_min / 60.0
        if tolerance_mm is not None:
            payload["tolerance_mm"] = tolerance_mm
        return self._request_json("/move", method="POST", payload=payload)

    def home(self) -> dict[str, Any]:
        return self._request_json(
            "/home", method="POST", payload={"axes": [self.axis], "blocking": True}
        )

    def stop(self, mode: str = "emergency") -> dict[str, Any]:
        return self._request_json("/stop", method="POST", payload={"mode": mode})

    def enable(self) -> dict[str, Any]:
        return self._request_json("/enable", method="POST", payload={})

    def disable(self) -> dict[str, Any]:
        return self._request_json("/disable", method="POST", payload={})

    def move_by(self, delta_mm: float, feed_mm_min: float | None = None) -> dict[str, Any]:
        status = self.status()
        current_position = float(status["position"][self.axis])
        return self.move_to(current_position + delta_mm, feed_mm_min=feed_mm_min)

    def move_xyz(
        self,
        target_mm: dict[str, float],
        feed_mm_min: float | None = None,
        tolerance_mm: float | None = None,
        blocking: bool = True,
    ) -> dict[str, Any]:
        """Submit an X target using the mapping accepted by scan dispatch code.

        Y/Z targets are rejected because those axes are owned by the Pico.
        """
        unsupported = set(target_mm) - {"x"}
        if unsupported:
            axes = ", ".join(sorted(unsupported))
            raise GantryServerError(
                f"The OpenDerm Klipper gantry client controls X only; rejected: {axes}."
            )
        payload: dict[str, Any] = {axis: float(v) for axis, v in target_mm.items()}
        payload["blocking"] = blocking
        if feed_mm_min is not None:
            payload["feed_mm_s"] = feed_mm_min / 60.0
        if tolerance_mm is not None:
            payload["tolerance_mm"] = tolerance_mm
        return self._request_json("/move", method="POST", payload=payload)

    # --- Streaming mode (continuous setpoint chasing; see MotionManager) ---

    def stream_start(
        self,
        *,
        feed_mm_min: float | None = None,
        tick_s: float = 0.05,
        min_step_mm: float = 0.01,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"tick_s": tick_s, "min_step_mm": min_step_mm}
        if feed_mm_min is not None:
            payload["feed_mm_s"] = feed_mm_min / 60.0
        return self._request_json("/stream/start", method="POST", payload=payload)

    def stream_to(self, position_mm: float) -> dict[str, Any]:
        """Update the live streaming setpoint for X."""
        return self._request_json("/stream/target", method="POST", payload={self.axis: position_mm})

    def stream_stop(self) -> dict[str, Any]:
        return self._request_json("/stream/stop", method="POST", payload={})

    def _request_json(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"} if data is not None else {}
        if self.control_token is not None:
            headers["Authorization"] = f"Bearer {self.control_token}"
        req = request.Request(
            f"{self.base_url}{path}",
            method=method,
            data=data,
            headers=headers,
        )
        try:
            with request.urlopen(req, timeout=self.timeout_s) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8")
            raise GantryServerError(f"Gantry server request failed: {exc.code} {detail}") from exc
        except error.URLError as exc:
            raise GantryServerError(f"Gantry server request failed: {exc}") from exc
        except TimeoutError as exc:
            raise GantryServerError("Gantry server request timed out.") from exc
        except socket.timeout as exc:
            raise GantryServerError("Gantry server request timed out.") from exc
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise GantryServerError(f"Invalid JSON from gantry server: {body!r}") from exc


def serve(config: GantryServerConfig, gantry_config: GantryConfig | None = None) -> int:
    import uvicorn

    try:
        require_secure_bind(config.host, config.control_token)
    except MotionSecurityError as exc:
        raise GantryServerError(str(exc)) from exc
    app = build_app(gantry_config or GantryConfig.from_env(), config)
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the gantry motion coordinator.")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Bind host for the gantry coordinator. Non-loopback binds require "
            "OPENDERM_CONTROL_TOKEN."
        ),
    )
    parser.add_argument(
        "--port", type=int, default=8090, help="Bind port for the gantry coordinator."
    )
    return parser


def config_from_args(args: argparse.Namespace) -> GantryServerConfig:
    if args.port <= 0:
        raise GantryServerError("--port must be positive.")
    try:
        control_token = control_token_from_env()
        require_secure_bind(args.host, control_token)
    except MotionSecurityError as exc:
        raise GantryServerError(str(exc)) from exc
    return GantryServerConfig(
        host=args.host,
        port=args.port,
        control_token=control_token,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return serve(config_from_args(args))
    except (GantryServerError, MotionError, MoonrakerError) as exc:
        parser.exit(status=1, message=f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
