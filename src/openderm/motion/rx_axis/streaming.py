"""Continuous RX-axis command refresh and velocity dead-man control."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from ..cubemars import CubeMarsFeedback, encode_rpm_payload
from .types import RxAxisServerError

if TYPE_CHECKING:
    from .service import RxAxisService


class RxStreamingController:
    """Own velocity-mode commands and the command refresh watchdog."""

    def __init__(self, service: RxAxisService) -> None:
        self.service = service

    async def set_velocity(self, velocity_rad_s: float) -> dict[str, Any]:
        service = self.service
        async with service._command_lock:
            service.homing.require_homed()
            service.safety.require_motion_allowed()
            velocity = velocity_rad_s
            held: str | None = None
            if velocity != 0.0:
                service.safety.require_velocity_allowed(abs(velocity))
                feedback = service._feedback
                if feedback is None:
                    raise RxAxisServerError(
                        "No CAN feedback yet; refusing to velocity-servo blind."
                    )
                margin = self.velocity_margin(velocity)
                position = feedback.position_rad
                config = service.server_config
                if velocity > 0 and position >= config.max_command_position_rad - margin:
                    velocity, held = 0.0, "max_window"
                elif velocity < 0 and position <= config.min_command_position_rad + margin:
                    velocity, held = 0.0, "min_window"
            payload = bytes.fromhex(
                encode_rpm_payload(
                    velocity,
                    gear_ratio=service.config.motor_gear_ratio,
                    pole_pairs=service.config.motor_pole_pairs,
                    position_inverted=service.config.motor_position_inverted,
                )
            )
            await service._send(service.rpm_command_can_id, payload)
            service._active_command_payload = payload
            service._active_command_can_id = service.rpm_command_can_id
            service._velocity_cmd = {
                "velocity_rad_s": velocity,
                "commanded_at": time.monotonic(),
                "held": held,
            }
            service._last_command = {
                "type": "set_velocity",
                "velocity_rad_s": velocity,
                "held": held,
                "payload": payload.hex().upper(),
                "can_id": f"0x{service.rpm_command_can_id:08X}",
                "timestamp": time.monotonic(),
            }
            result = service.state_payload()
            if held is not None:
                result["held"] = held
            return result

    async def zero_velocity(self, reason: str) -> None:
        async with self.service._command_lock:
            await self.zero_velocity_locked(reason)

    async def zero_velocity_locked(self, reason: str) -> None:
        service = self.service
        payload = bytes.fromhex(
            encode_rpm_payload(
                0.0,
                gear_ratio=service.config.motor_gear_ratio,
                pole_pairs=service.config.motor_pole_pairs,
                position_inverted=service.config.motor_position_inverted,
            )
        )
        await service._send(service.rpm_command_can_id, payload)
        service._active_command_payload = payload
        service._active_command_can_id = service.rpm_command_can_id
        service._velocity_cmd = {
            "velocity_rad_s": 0.0,
            "commanded_at": time.monotonic(),
            "held": reason,
        }
        service._last_command = {
            "type": "velocity_zeroed",
            "reason": reason,
            "payload": payload.hex().upper(),
            "can_id": f"0x{service.rpm_command_can_id:08X}",
            "timestamp": time.monotonic(),
        }

    def velocity_margin(self, velocity_rad_s: float) -> float:
        config = self.service.server_config
        return max(
            config.velocity_window_margin_rad,
            abs(velocity_rad_s) * config.velocity_window_brake_s,
        )

    async def check_velocity_guard(self, feedback: CubeMarsFeedback) -> None:
        service = self.service
        command = service._velocity_cmd
        if command is None or command["velocity_rad_s"] == 0.0:
            return
        velocity = command["velocity_rad_s"]
        margin = self.velocity_margin(velocity)
        config = service.server_config
        reason: str | None = None
        if time.monotonic() - command["commanded_at"] > config.velocity_deadman_s:
            reason = "deadman"
        elif velocity > 0 and feedback.position_rad >= config.max_command_position_rad - margin:
            reason = "max_window"
        elif velocity < 0 and feedback.position_rad <= config.min_command_position_rad + margin:
            reason = "min_window"
        if reason is not None:
            await self.zero_velocity(reason)

    async def run(self) -> None:
        """Refresh active commands and enforce the dead-man without feedback."""
        service = self.service
        refresh_interval = service.server_config.command_refresh_interval_s
        watchdog_interval = min(
            refresh_interval,
            max(0.001, service.server_config.velocity_deadman_s / 4),
        )
        next_refresh_at = 0.0
        while True:
            try:
                async with service._command_lock:
                    now = time.monotonic()
                    velocity_cmd = service._velocity_cmd
                    deadman_expired = (
                        velocity_cmd is not None
                        and velocity_cmd["velocity_rad_s"] != 0.0
                        and now - velocity_cmd["commanded_at"]
                        > service.server_config.velocity_deadman_s
                    )
                    if deadman_expired:
                        await self.zero_velocity_locked("deadman")
                        next_refresh_at = time.monotonic() + refresh_interval
                    elif service._active_command_payload is not None and now >= next_refresh_at:
                        await service._send(
                            service._active_command_can_id,
                            service._active_command_payload,
                        )
                        next_refresh_at = time.monotonic() + refresh_interval
                service._stream_error = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                service._stream_error = f"{type(exc).__name__}: {exc}"
            finally:
                service._last_stream_iteration_at = time.monotonic()
            await asyncio.sleep(watchdog_interval)
