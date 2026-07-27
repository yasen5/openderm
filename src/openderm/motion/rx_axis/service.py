"""RX-axis motion, homing, feedback, and safety service."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
import time
from typing import Any

from ...config import GantryConfig
from ..cubemars import (
    DUTY_CYCLE_ZERO_PAYLOAD,
    SERVO_ERROR_CODES,
    CubeMarsFeedback,
    decode_feedback_payload,
    duty_cycle_can_id,
    encode_position_velocity_payload,
    feedback_can_id,
    position_velocity_can_id,
    rpm_can_id,
    set_origin_can_id,
)
from .homing import RxHomingController
from .safety import RxSafetyController
from .streaming import RxStreamingController
from .transport import SocketCanTransport
from .types import (
    CanFrame,
    HomingRecord,
    RxAxisServerConfig,
    RxAxisServerError,
    RxAxisSnapshot,
)


class RxAxisService:
    def __init__(
        self,
        config: GantryConfig,
        server_config: RxAxisServerConfig,
        transport: Any | None = None,
        limit_switch_reader: Any | None = None,
    ) -> None:
        if not config.is_can_axis:
            raise RxAxisServerError(f"RX-axis server requires axis rx, got {config.axis!r}.")
        self.config = config
        self.server_config = server_config
        self.command_can_id = position_velocity_can_id(config.selected_can_node_id)
        self.rpm_command_can_id = rpm_can_id(config.selected_can_node_id)
        self.set_origin_can_id = set_origin_can_id(config.selected_can_node_id)
        self.duty_cycle_can_id = duty_cycle_can_id(config.selected_can_node_id)
        self.feedback_can_id = feedback_can_id(config.selected_can_node_id)
        self.transport = transport or SocketCanTransport(
            config.selected_can_interface,
            timeout_s=server_config.feedback_timeout_s,
        )
        # Serializes active-command state with wire sends. Without this lock,
        # the refresh task can capture a stale nonzero payload and send it
        # after a concurrent stop, safety shutdown, or dead-man zero.
        self._command_lock = asyncio.Lock()
        self._lock = asyncio.Lock()
        self._poll_task: asyncio.Task[None] | None = None
        self._streamer_task: asyncio.Task[None] | None = None
        self._poll_error: str | None = None
        self._stream_error: str | None = None
        self._last_poll_iteration_at: float | None = None
        self._last_stream_iteration_at: float | None = None
        self._feedback: CubeMarsFeedback | None = None
        self._timestamp: float | None = None
        self._last_command: dict[str, Any] | None = None
        self._safety_shutdown: dict[str, Any] | None = None
        self._limit_switch_reader = limit_switch_reader
        self._homing_status: dict[str, Any] | None = None
        self._homed = False
        self._homing_active = False
        # The AK servo position-velocity loop drops out after a single packet;
        # whenever an active move target is set we re-send the same payload at
        # `command_refresh_interval_s` to keep the firmware's loop alive.
        self._active_command_payload: bytes | None = None
        self._active_command_can_id: int = self.command_can_id
        # Active velocity-mode command: {"velocity_rad_s", "commanded_at", "held"}.
        # None = not in velocity mode. Position bounds are checked on feedback;
        # the refresh loop enforces the time-based dead-man independently.
        self._velocity_cmd: dict[str, Any] | None = None
        self.homing = RxHomingController(self)
        self.safety = RxSafetyController(self)
        self.streaming = RxStreamingController(self)

    async def start(self) -> None:
        self._poll_error = None
        self._stream_error = None
        self._last_poll_iteration_at = None
        self._last_stream_iteration_at = None
        self._poll_task = asyncio.create_task(
            self._poll_loop(), name=f"{self.config.axis}-can-poller"
        )
        self._streamer_task = asyncio.create_task(
            self.streaming.run(),
            name=f"{self.config.axis}-can-streamer",
        )

    async def stop(self) -> None:
        for task in (self._poll_task, self._streamer_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._poll_task = None
        self._streamer_task = None
        close = getattr(self.transport, "close", None)
        if close is not None:
            close()
        if self._limit_switch_reader is not None:
            self._limit_switch_reader.close()

    async def poll_once(self) -> CubeMarsFeedback | None:
        # AK servo mode auto-publishes the feedback frame at 1–500 Hz; we just receive it.
        frame = await self._await_feedback()
        if frame is None:
            return None
        feedback = decode_feedback_payload(
            frame.data,
            gear_ratio=self.config.motor_gear_ratio,
            pole_pairs=self.config.motor_pole_pairs,
            position_inverted=self.config.motor_position_inverted,
        )
        self._feedback = feedback
        self._timestamp = time.monotonic()
        await self.safety.check_limit_switch_shutdown()
        await self.safety.check_current_shutdown(feedback)
        await self.safety.check_motor_temperature_shutdown(feedback)
        await self.streaming.check_velocity_guard(feedback)
        return feedback

    async def move_to(
        self,
        position_rad: float,
        velocity_rad_s: float | None = None,
        accel_rad_s2: float | None = None,
        *,
        require_homed: bool = True,
        require_nonnegative: bool = True,
    ) -> dict[str, Any]:
        if require_homed:
            self.homing.require_homed()
        self.safety.require_within_limits(position_rad)
        if require_nonnegative:
            self.safety.require_nonnegative_target(position_rad)
            self.safety.require_command_position_allowed(position_rad)
        velocity = self.safety.resolve_velocity(velocity_rad_s)
        accel = self.safety.resolve_accel(accel_rad_s2)
        self.safety.require_motion_allowed()
        self.safety.require_velocity_allowed(velocity)
        payload = bytes.fromhex(
            encode_position_velocity_payload(
                position_rad,
                velocity,
                accel,
                gear_ratio=self.config.motor_gear_ratio,
                pole_pairs=self.config.motor_pole_pairs,
                position_inverted=self.config.motor_position_inverted,
            )
        )
        async with self._command_lock:
            # Re-check after waiting for the command lock in case a concurrent
            # feedback task latched a safety shutdown.
            self.safety.require_motion_allowed()
            await self._send(self.command_can_id, payload)
            # Activate continuous streaming so the firmware's position loop keeps trying.
            # A position command always cancels velocity mode.
            self._velocity_cmd = None
            self._active_command_payload = payload
            self._active_command_can_id = self.command_can_id
            self._last_command = {
                "type": "move_to",
                "position_rad": position_rad,
                "velocity_rad_s": velocity,
                "accel_rad_s2": accel,
                "payload": payload.hex().upper(),
                "can_id": f"0x{self.command_can_id:08X}",
                "timestamp": time.monotonic(),
            }
            return self.state_payload()

    async def set_velocity(self, velocity_rad_s: float) -> dict[str, Any]:
        return await self.streaming.set_velocity(velocity_rad_s)

    async def stop_motor(self) -> dict[str, Any]:
        # Servo mode has no exit frame; sending duty-cycle = 0 disables drive (motor coasts).
        async with self._command_lock:
            self._velocity_cmd = None
            self._active_command_payload = None
            await self._send(self.duty_cycle_can_id, bytes.fromhex(DUTY_CYCLE_ZERO_PAYLOAD))
            self._last_command = {
                "type": "stop",
                "payload": DUTY_CYCLE_ZERO_PAYLOAD,
                "can_id": f"0x{self.duty_cycle_can_id:08X}",
                "timestamp": time.monotonic(),
            }
            return self.state_payload()

    async def clear_errors(self) -> dict[str, Any]:
        # AK servo mode exposes no clear-errors CAN command; driver-side faults reset on power cycle.
        # Here we only release the software-side safety latch so commands are accepted again.
        # Also drop any stale streaming target so we don't immediately re-issue it.
        async with self._command_lock:
            self._velocity_cmd = None
            self._active_command_payload = None
            self._safety_shutdown = None
            self._last_command = {
                "type": "clear_errors",
                "payload": None,
                "can_id": None,
                "timestamp": time.monotonic(),
            }
            return self.state_payload()

    async def home_axis(self) -> HomingRecord:
        return await self.homing.home_axis()

    @property
    def is_homed(self) -> bool:
        return self._homed

    def snapshot(self) -> RxAxisSnapshot:
        stale = self._timestamp is None or (
            time.monotonic() - self._timestamp > max(1.0, self.server_config.poll_interval_s * 5)
        )
        return RxAxisSnapshot(
            axis=self.config.axis,
            interface=self.config.selected_can_interface,
            motor_can_id=f"0x{self.command_can_id:08X}",
            feedback_can_id=f"0x{self.feedback_can_id:08X}",
            feedback=self._feedback,
            timestamp=self._timestamp,
            stale=stale,
            poll_interval_s=self.server_config.poll_interval_s,
            last_command=self._last_command,
            safety_shutdown=self._safety_shutdown,
            homing=self._homing_status,
            homed=self._homed,
        )

    def state_payload(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        feedback = None if snapshot.feedback is None else asdict(snapshot.feedback)
        error_code = None if snapshot.feedback is None else snapshot.feedback.error_code
        return {
            "axis": snapshot.axis,
            "interface": snapshot.interface,
            "motor_can_id": snapshot.motor_can_id,
            "feedback_can_id": snapshot.feedback_can_id,
            "position_rad": None
            if snapshot.feedback is None
            else round(snapshot.feedback.position_rad, 6),
            "speed_rad_s": None
            if snapshot.feedback is None
            else round(snapshot.feedback.speed_rad_s, 6),
            "current_a": None
            if snapshot.feedback is None
            else round(snapshot.feedback.current_a, 6),
            "error_code": error_code,
            "error_message": None
            if error_code is None
            else SERVO_ERROR_CODES.get(error_code, "unknown"),
            "motor_temperature_c": None
            if snapshot.feedback is None
            else snapshot.feedback.motor_temperature_c,
            "feedback": feedback,
            "timestamp": None if snapshot.timestamp is None else round(snapshot.timestamp, 6),
            "stale": snapshot.stale,
            "poll_interval_s": snapshot.poll_interval_s,
            "last_command": snapshot.last_command,
            "safety_shutdown": snapshot.safety_shutdown,
            "homing": snapshot.homing,
            "homed": snapshot.homed,
            "velocity_command": self._velocity_cmd,
            "max_speed_rad_s": self.server_config.max_speed_rad_s,
            "default_accel_rad_s2": self.server_config.default_accel_rad_s2,
            "current_shutdown_threshold_a": self.server_config.current_shutdown_threshold_a,
            "motor_temperature_shutdown_c": self.server_config.motor_temperature_shutdown_c,
        }

    def health_payload(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        poll_running = self._task_running(self._poll_task)
        stream_running = self._task_running(self._streamer_task)
        motor_error_code = None if snapshot.feedback is None else snapshot.feedback.error_code
        issues: list[str] = []
        if not poll_running:
            issues.append("feedback_poll_loop_not_running")
        if not stream_running:
            issues.append("command_refresh_loop_not_running")
        if snapshot.stale:
            issues.append("feedback_stale")
        if self._poll_error is not None:
            issues.append("feedback_poll_error")
        if self._stream_error is not None:
            issues.append("command_refresh_error")
        if snapshot.safety_shutdown is not None:
            issues.append("safety_shutdown_latched")
        if motor_error_code not in (None, 0):
            issues.append("motor_fault")
        return {
            "ok": not issues,
            "issues": issues,
            "feedback_stale": snapshot.stale,
            "feedback_received": snapshot.timestamp is not None,
            "homed": snapshot.homed,
            "ready_for_motion": not issues and snapshot.homed,
            "motor_error_code": motor_error_code,
            "safety_shutdown": snapshot.safety_shutdown,
            "loops": {
                "feedback_poll": {
                    "running": poll_running,
                    "last_iteration_at": self._last_poll_iteration_at,
                    "error": self._poll_error,
                },
                "command_refresh": {
                    "running": stream_running,
                    "last_iteration_at": self._last_stream_iteration_at,
                    "error": self._stream_error,
                },
            },
        }

    @staticmethod
    def _task_running(task: asyncio.Task[None] | None) -> bool:
        return task is not None and not task.done()

    async def _poll_loop(self) -> None:
        while True:
            try:
                await self.poll_once()
                self._poll_error = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._poll_error = f"{type(exc).__name__}: {exc}"
            finally:
                self._last_poll_iteration_at = time.monotonic()
            await asyncio.sleep(self.server_config.poll_interval_s)

    async def _send(self, can_id: int, payload: bytes) -> None:
        async with self._lock:
            await asyncio.to_thread(self.transport.send, can_id, payload, extended=True)

    async def _await_feedback(self, timeout_s: float | None = None) -> CanFrame | None:
        resolved_timeout = self.server_config.feedback_timeout_s if timeout_s is None else timeout_s
        async with self._lock:
            return await asyncio.to_thread(
                self.transport.transact_receive,
                feedback_can_id=self.feedback_can_id,
                timeout_s=resolved_timeout,
            )

    def limit_switch_payload(self) -> dict[str, Any]:
        return self.homing.limit_switch_payload()
