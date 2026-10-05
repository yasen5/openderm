"""Safety policy and shutdown handling for the RX axis."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ..cubemars import DUTY_CYCLE_ZERO_PAYLOAD, CubeMarsFeedback
from .types import RxAxisServerError

if TYPE_CHECKING:
    from .service import RxAxisService


class RxSafetyController:
    """Validate commands and latch hardware safety shutdowns."""

    def __init__(self, service: RxAxisService) -> None:
        self.service = service

    def require_nonnegative_target(self, position_rad: float) -> None:
        service = self.service
        if service._homed and position_rad < 0:
            raise RxAxisServerError("RX-axis target position must be >= 0 after homing.")

    def require_command_position_allowed(self, position_rad: float) -> None:
        service = self.service
        if not service._homed:
            return
        config = service.server_config
        if not config.min_command_position_rad <= position_rad <= config.max_command_position_rad:
            raise RxAxisServerError(
                "RX-axis target position must be between "
                f"{config.min_command_position_rad:.3f}rad and "
                f"{config.max_command_position_rad:.3f}rad after homing."
            )

    def require_within_limits(self, position_rad: float) -> None:
        config = self.service.config
        if not config.can_position_min_rad <= position_rad <= config.can_position_max_rad:
            raise RxAxisServerError(
                f"Requested position {position_rad:.3f}rad is outside soft limits "
                f"[{config.can_position_min_rad:.3f}, {config.can_position_max_rad:.3f}]"
            )

    def require_motion_allowed(self) -> None:
        if self.service._safety_shutdown is not None:
            raise RxAxisServerError(
                "RX-axis safety shutdown is latched. Inspect /state and call "
                "/clear-errors before moving again."
            )

    def resolve_velocity(self, velocity_rad_s: float | None) -> float:
        service = self.service
        if velocity_rad_s is None:
            return min(
                service.config.can_default_speed_rad_s,
                service.server_config.max_speed_rad_s,
            )
        return velocity_rad_s

    def resolve_accel(self, accel_rad_s2: float | None) -> float:
        if accel_rad_s2 is None:
            return self.service.server_config.default_accel_rad_s2
        return accel_rad_s2

    def require_velocity_allowed(self, velocity_rad_s: float) -> None:
        if velocity_rad_s <= 0:
            raise RxAxisServerError("Velocity must be greater than zero in position/velocity mode.")
        max_speed = self.service.server_config.max_speed_rad_s
        if velocity_rad_s > max_speed:
            raise RxAxisServerError(
                f"Requested speed {velocity_rad_s:.3f}rad/s exceeds max speed {max_speed:.3f}rad/s."
            )

    async def disable_drive(self) -> None:
        """Drop active commands before disabling the drive."""
        service = self.service
        async with service._command_lock:
            service._velocity_cmd = None
            service._active_command_payload = None
            await service._send(
                service.duty_cycle_can_id,
                bytes.fromhex(DUTY_CYCLE_ZERO_PAYLOAD),
            )
            service._last_command = {
                "type": "safety_shutdown",
                "payload": DUTY_CYCLE_ZERO_PAYLOAD,
                "can_id": f"0x{service.duty_cycle_can_id:08X}",
                "timestamp": time.monotonic(),
            }

    async def check_current_shutdown(self, feedback: CubeMarsFeedback) -> None:
        service = self.service
        threshold = service.server_config.current_shutdown_threshold_a
        if threshold <= 0 or service._safety_shutdown is not None:
            return
        if abs(feedback.current_a) <= threshold:
            return
        service._safety_shutdown = {
            "reason": "current_threshold_exceeded",
            "current_a": feedback.current_a,
            "threshold_a": threshold,
            "timestamp": time.monotonic(),
        }
        await self.disable_drive()
        raise RxAxisServerError(
            f"Current safety shutdown: |{feedback.current_a:.3f}A| exceeds {threshold:.3f}A."
        )

    async def check_motor_temperature_shutdown(
        self,
        feedback: CubeMarsFeedback,
    ) -> None:
        service = self.service
        threshold = service.server_config.motor_temperature_shutdown_c
        if threshold <= 0 or service._safety_shutdown is not None:
            return
        if feedback.motor_temperature_c < threshold:
            return
        service._safety_shutdown = {
            "reason": "motor_temperature_exceeded",
            "motor_temperature_c": feedback.motor_temperature_c,
            "threshold_c": threshold,
            "timestamp": time.monotonic(),
        }
        await self.disable_drive()
        raise RxAxisServerError(
            f"Motor temperature safety shutdown: "
            f"{feedback.motor_temperature_c}°C reached threshold {threshold:.1f}°C."
        )

    async def check_limit_switch_shutdown(
        self,
        *,
        allow_left: bool = False,
        allow_right: bool = False,
    ) -> None:
        service = self.service
        if service._homing_active or not service._homed or service._safety_shutdown is not None:
            return
        reader = service.homing.get_limit_switch_reader()
        left_pressed = service.homing.axis_left_pressed_fn(reader)()
        right_pressed = service.homing.axis_right_pressed_fn(reader)()
        if (left_pressed and not allow_left) or (right_pressed and not allow_right):
            triggered = []
            if left_pressed and not allow_left:
                triggered.append(f"{service.config.axis}_left")
            if right_pressed and not allow_right:
                triggered.append(f"{service.config.axis}_right")
            service._safety_shutdown = {
                "reason": "limit_switch_triggered",
                "triggered": triggered,
                "timestamp": time.monotonic(),
            }
            await self.disable_drive()
            raise RxAxisServerError(
                f"Limit switch safety shutdown: {', '.join(triggered)} triggered."
            )
