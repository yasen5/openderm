from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import logging
import time
from typing import Any

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
except ModuleNotFoundError:  # pragma: no cover - exercised only when optional deps are absent.
    websockets = None
    WebSocketClientProtocol = Any

from .state import AXES, StateSnapshot, StateStore, merge_position, project_position
from .moonraker import MoonrakerError


LOGGER = logging.getLogger(__name__)


class MoonrakerCoordinatorClient:
    def __init__(self, ws_url: str, state_store: StateStore) -> None:
        self.ws_url = ws_url
        self.state_store = state_store
        self._ws: WebSocketClientProtocol | None = None
        self._send_lock = asyncio.Lock()
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._request_id = 0
        self._running = True
        self._connected = asyncio.Event()

    @property
    def connected(self) -> bool:
        return self._connected.is_set() and self._ws is not None

    async def run(self) -> None:
        if websockets is None:
            raise RuntimeError("websockets is required to run the gantry coordinator.")
        while self._running:
            receiver_task: asyncio.Task[None] | None = None
            try:
                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20) as ws:
                    self._ws = ws
                    self._connected.set()
                    receiver_task = asyncio.create_task(
                        self._receive_loop(ws), name="moonraker-ws-recv"
                    )
                    await self._subscribe()
                    await receiver_task
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Moonraker websocket disconnected: %s", exc)
            finally:
                if receiver_task is not None:
                    receiver_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await receiver_task
                self._connected.clear()
                self._ws = None
                await self._mark_stale()
                self._fail_pending("Moonraker websocket disconnected.")
            if self._running:
                await asyncio.sleep(0.5)

    async def close(self) -> None:
        self._running = False
        ws = self._ws
        if ws is not None:
            await ws.close()

    async def wait_until_connected(self, timeout_s: float | None = None) -> None:
        if timeout_s is None:
            await self._connected.wait()
            return
        await asyncio.wait_for(self._connected.wait(), timeout=timeout_s)

    async def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        await self.wait_until_connected(timeout_s=10.0)
        self._request_id += 1
        message_id = self._request_id
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[message_id] = future
        payload = {"jsonrpc": "2.0", "method": method, "id": message_id}
        if params is not None:
            payload["params"] = params
        async with self._send_lock:
            assert self._ws is not None
            await self._ws.send(json.dumps(payload))
        response = await future
        if "error" in response:
            raise MoonrakerError(str(response["error"]))
        return response.get("result", {})

    async def gcode_script(self, script: str) -> dict[str, Any]:
        return await self.call("printer.gcode.script", {"script": script})

    async def emergency_stop(self) -> dict[str, Any]:
        return await self.call("printer.emergency_stop")

    async def firmware_restart(self) -> dict[str, Any]:
        return await self.call("printer.firmware_restart")

    async def query_objects(self, objects: dict[str, Any]) -> dict[str, Any]:
        result = await self.call("printer.objects.query", {"objects": objects})
        status = result.get("status", {})
        await self._publish_status(status, stale=False)
        return status

    async def _subscribe(self) -> None:
        result = await self.call(
            "printer.objects.subscribe",
            {
                "objects": {
                    "toolhead": None,
                    "motion_report": None,
                    "webhooks": None,
                }
            },
        )
        status = result.get("status", {})
        await self._publish_status(status, stale=False)

    async def _handle_message(self, raw_message: str) -> None:
        payload = json.loads(raw_message)
        if "id" in payload:
            future = self._pending.pop(int(payload["id"]), None)
            if future is not None and not future.done():
                future.set_result(payload)
            return
        if payload.get("method") == "notify_status_update":
            params = payload.get("params", [])
            if params:
                status = params[0]
                await self._publish_status(status, stale=False)

    async def _receive_loop(self, ws: WebSocketClientProtocol) -> None:
        async for raw_message in ws:
            await self._handle_message(raw_message)

    async def _publish_status(self, status: dict[str, Any], *, stale: bool) -> None:
        current = self.state_store.get()
        now = time.monotonic()
        toolhead = status.get("toolhead", {})
        motion_report = status.get("motion_report", {})
        webhooks = status.get("webhooks", {})
        live_position = motion_report.get("live_position")
        live_velocity = motion_report.get("live_velocity")
        position_updates = self._extract_position(live_position)
        base_position = (
            project_position(current, now=now) if position_updates is None else current.position
        )
        next_velocity = float(live_velocity) if live_velocity is not None else current.velocity_mm_s
        next_snapshot = StateSnapshot(
            position=merge_position(base_position, position_updates),
            velocity_mm_s=next_velocity,
            timestamp=now,
            is_moving=next_velocity > 0.01,
            last_commanded_target=current.last_commanded_target,
            homed_axes=toolhead.get("homed_axes", current.homed_axes),
            fault=self._fault_from_webhooks(webhooks) if webhooks else current.fault,
            stale=stale,
            enabled=current.enabled,
            active_move_id=current.active_move_id,
            queue_depth=current.queue_depth,
            webhooks_state=webhooks.get("state", current.webhooks_state),
            webhooks_message=webhooks.get("state_message", current.webhooks_message),
            stop_requested=current.stop_requested,
            emergency_latched=current.emergency_latched,
        )
        await self.state_store.publish(next_snapshot)

    async def _mark_stale(self) -> None:
        current = self.state_store.get()
        await self.state_store.publish(
            StateSnapshot(
                position=current.position,
                velocity_mm_s=current.velocity_mm_s,
                timestamp=time.monotonic(),
                is_moving=False,
                last_commanded_target=current.last_commanded_target,
                homed_axes=current.homed_axes,
                fault=current.fault,
                stale=True,
                enabled=current.enabled,
                active_move_id=current.active_move_id,
                queue_depth=current.queue_depth,
                webhooks_state="disconnected",
                webhooks_message=current.webhooks_message,
                stop_requested=current.stop_requested,
                emergency_latched=current.emergency_latched,
            )
        )

    @staticmethod
    def _extract_position(live_position: Any) -> dict[str, float] | None:
        if live_position is None:
            return None
        if isinstance(live_position, dict):
            return {axis: float(live_position[axis]) for axis in AXES if axis in live_position}
        if isinstance(live_position, (list, tuple)) and len(live_position) >= len(AXES):
            return {axis: float(live_position[index]) for index, axis in enumerate(AXES)}
        return None

    @staticmethod
    def _fault_from_webhooks(webhooks: dict[str, Any]) -> str | None:
        state = str(webhooks.get("state", "")).lower()
        if state in {"shutdown", "error"}:
            return "klipper_shutdown"
        return None

    def _fail_pending(self, message: str) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(MoonrakerError(message))
        self._pending.clear()
