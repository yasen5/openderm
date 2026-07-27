"""Shared RX-axis configuration, frame, snapshot, and homing types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..cubemars import CubeMarsFeedback


class RxAxisServerError(RuntimeError):
    """Raised when the RX-axis server cannot start or process a command."""


@dataclass(frozen=True)
class RxAxisServerConfig:
    host: str = "127.0.0.1"
    port: int = 8091
    axis: str = "rx"
    control_token: str | None = None
    poll_interval_s: float = 0.05
    feedback_timeout_s: float = 0.05
    # The AK servo firmware's position-velocity loop stops trying after one packet,
    # so the service re-sends the active command at this rate to keep it alive.
    command_refresh_interval_s: float = 0.1
    max_speed_rad_s: float = 0.25
    default_accel_rad_s2: float = 10.0
    current_shutdown_threshold_a: float = 5.0
    # Disable drive if the motor's reported temperature reaches or exceeds this many °C.
    # Use <= 0 to disable.
    motor_temperature_shutdown_c: float = 80.0
    homing_fast_speed_rad_s: float = 0.2
    homing_slow_speed_rad_s: float = 0.02
    homing_backoff_rad: float = 0.2
    homing_search_distance_rad: float = 6.0
    homing_timeout_s: float = 45.0
    homing_final_position_rad: float = 0.95
    # This command window matches the mechanically verified safe RX range.
    min_command_position_rad: float = 0.07
    max_command_position_rad: float = 1.76
    # Velocity (speed-loop) mode. The firmware's speed loop knows NOTHING about
    # the command-position window, so the server enforces it: every feedback
    # poll zeroes an active velocity at the window edge (minus the margin), and
    # a velocity command that is not refreshed within the dead-man interval
    # auto-zeroes -- a hung caller can never leave the axis running.
    # The margin SCALES WITH SPEED: margin = max(margin_rad, |v| * brake_s).
    # A flat margin sized for full speed silently shrinks the usable window --
    # a flat 0.05 rad floor margin made rx 0.2..0.25 rad unreachable and the
    # scan skipped every station whose surface wanted 11.5-14 deg of tilt
    # (high-curvature coverage gap); at servo approach speeds the true braking
    # distance is ~0.01 rad.
    velocity_deadman_s: float = 0.4
    velocity_window_margin_rad: float = 0.01
    velocity_window_brake_s: float = 0.1


@dataclass(frozen=True)
class CanFrame:
    can_id: int
    data: bytes


@dataclass(frozen=True)
class RxAxisSnapshot:
    axis: str
    interface: str
    motor_can_id: str
    feedback_can_id: str
    feedback: CubeMarsFeedback | None
    timestamp: float | None
    stale: bool
    poll_interval_s: float
    last_command: dict[str, Any] | None
    safety_shutdown: dict[str, Any] | None
    homing: dict[str, Any] | None
    homed: bool


@dataclass
class HomingRecord:
    status: str
    axis: str
    started_at: float
    completed_at: float | None
    first_trigger_position_rad: float | None = None
    second_trigger_position_rad: float | None = None
    zeroed: bool = False
    final_position_rad: float | None = None
    error: str | None = None

    def to_payload(self) -> dict[str, Any]:
        duration = None if self.completed_at is None else self.completed_at - self.started_at
        return {
            "status": self.status,
            "axis": self.axis,
            "started_at": round(self.started_at, 6),
            "completed_at": None if self.completed_at is None else round(self.completed_at, 6),
            "duration_s": None if duration is None else round(duration, 6),
            "first_trigger_position_rad": self.first_trigger_position_rad,
            "second_trigger_position_rad": self.second_trigger_position_rad,
            "zeroed": self.zeroed,
            "final_position_rad": self.final_position_rad,
            "error": self.error,
        }
