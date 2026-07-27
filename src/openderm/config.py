from __future__ import annotations

from dataclasses import dataclass
import os
from urllib.parse import urlparse


DEFAULT_GANTRY_SERVER_URL = "http://openderm-gantry.local:8090"
DEFAULT_PICO_BRIDGE_PORT = 8095


LINEAR_AXIS_LIMITS: dict[str, tuple[float, float]] = {
    "x": (0.0, 800.0),
    "y": (0.0, 665.0),
    "z": (0.0, 392.0),
}


def default_pico_port(gantry_server_url: str | None = None) -> str:
    """Return the explicit Pico port or derive its TCP bridge from Pi 1.

    Pi 1 hosts both the gantry API and the Pico serial bridge, so one
    ``GANTRY_SERVER_URL`` setting is sufficient.
    ``PICO_PORT`` always wins when the topology differs.
    """
    explicit = os.getenv("PICO_PORT")
    if explicit:
        return explicit
    server_url = gantry_server_url or os.getenv("GANTRY_SERVER_URL", DEFAULT_GANTRY_SERVER_URL)
    host = urlparse(server_url).hostname or "openderm-gantry.local"
    return f"socket://{host}:{DEFAULT_PICO_BRIDGE_PORT}"


def linear_axis_limits(axis: str) -> tuple[float, float]:
    return LINEAR_AXIS_LIMITS.get(axis.lower(), LINEAR_AXIS_LIMITS["x"])


@dataclass(frozen=True)
class GantryConfig:
    moonraker_ws_url: str = "ws://localhost:7125/websocket"
    gantry_server_url: str = DEFAULT_GANTRY_SERVER_URL
    rx_axis_server_url: str = "http://127.0.0.1:8091"
    axis: str = "x"
    travel_min_mm: float = 0.0
    travel_max_mm: float = 800.0
    positioning_timeout_s: float = 30.0
    default_feed_mm_min: float = 1500.0
    home_timeout_s: float = 60.0
    controlled_steppers: tuple[str, ...] = ("stepper_x", "stepper_x1")
    rx_can_interface: str = "can1"
    rx_can_node_id: int = 0x01
    can_position_min_rad: float = -12.5
    can_position_max_rad: float = 12.5
    can_default_speed_rad_s: float = 2.0
    # AK45-36 KV80 has a 36:1 planetary gearbox and 14 pole pairs.
    # Verify both against the values stored in the driver via the R-link upper-computer software.
    motor_pole_pairs: int = 14
    motor_gear_ratio: float = 36.0
    # True when the motor's physical "+ direction" on this axis is toward the home
    # switch. The encode/decode layer flips position and speed signs so that
    # user-facing positive always means "away from the switch", regardless of which
    # way the motor is wired. The production RX motor is inverted; override with
    # MOTOR_POSITION_INVERTED_RX when required by another build.
    rx_motor_position_inverted: bool = True

    @classmethod
    def from_env(cls) -> "GantryConfig":
        axis = os.getenv("GANTRY_AXIS", cls.axis).lower()
        default_travel_min_mm, default_travel_max_mm = linear_axis_limits(axis)
        return cls(
            moonraker_ws_url=os.getenv("MOONRAKER_WS_URL", cls.moonraker_ws_url),
            gantry_server_url=os.getenv("GANTRY_SERVER_URL", cls.gantry_server_url),
            rx_axis_server_url=os.getenv(
                "RX_AXIS_SERVER_URL",
                cls.rx_axis_server_url,
            ),
            axis=axis,
            travel_min_mm=float(os.getenv("GANTRY_TRAVEL_MIN_MM", default_travel_min_mm)),
            travel_max_mm=float(os.getenv("GANTRY_TRAVEL_MAX_MM", default_travel_max_mm)),
            positioning_timeout_s=float(
                os.getenv("GANTRY_POSITIONING_TIMEOUT_S", cls.positioning_timeout_s)
            ),
            default_feed_mm_min=float(
                os.getenv("GANTRY_DEFAULT_FEED_MM_MIN", cls.default_feed_mm_min)
            ),
            home_timeout_s=float(os.getenv("GANTRY_HOME_TIMEOUT_S", cls.home_timeout_s)),
            controlled_steppers=tuple(
                part.strip()
                for part in os.getenv(
                    "GANTRY_CONTROLLED_STEPPERS",
                    ",".join(cls.controlled_steppers),
                ).split(",")
                if part.strip()
            ),
            rx_can_interface=os.getenv("GANTRY_RX_CAN_INTERFACE", cls.rx_can_interface),
            rx_can_node_id=int(os.getenv("GANTRY_RX_CAN_NODE_ID", str(cls.rx_can_node_id)), 0),
            can_position_min_rad=float(
                os.getenv("GANTRY_CAN_POSITION_MIN_RAD", cls.can_position_min_rad)
            ),
            can_position_max_rad=float(
                os.getenv("GANTRY_CAN_POSITION_MAX_RAD", cls.can_position_max_rad)
            ),
            can_default_speed_rad_s=float(
                os.getenv("GANTRY_CAN_DEFAULT_SPEED_RAD_S", cls.can_default_speed_rad_s)
            ),
            motor_pole_pairs=int(os.getenv("MOTOR_POLE_PAIRS", str(cls.motor_pole_pairs))),
            motor_gear_ratio=float(os.getenv("MOTOR_GEAR_RATIO", cls.motor_gear_ratio)),
            rx_motor_position_inverted=os.getenv(
                "MOTOR_POSITION_INVERTED_RX", str(cls.rx_motor_position_inverted)
            )
            .strip()
            .lower()
            in ("1", "true", "yes", "on"),
        )

    def with_overrides(
        self,
        *,
        axis: str | None = None,
        travel_min_mm: float | None = None,
        travel_max_mm: float | None = None,
    ) -> "GantryConfig":
        resolved_axis = (axis or self.axis).lower()
        default_travel_min_mm, default_travel_max_mm = linear_axis_limits(resolved_axis)
        resolved_travel_min_mm = (
            self.travel_min_mm
            if travel_min_mm is None and resolved_axis == self.axis
            else default_travel_min_mm
            if travel_min_mm is None
            else travel_min_mm
        )
        resolved_travel_max_mm = (
            self.travel_max_mm
            if travel_max_mm is None and resolved_axis == self.axis
            else default_travel_max_mm
            if travel_max_mm is None
            else travel_max_mm
        )
        return GantryConfig(
            moonraker_ws_url=self.moonraker_ws_url,
            gantry_server_url=self.gantry_server_url,
            rx_axis_server_url=self.rx_axis_server_url,
            axis=resolved_axis,
            travel_min_mm=resolved_travel_min_mm,
            travel_max_mm=resolved_travel_max_mm,
            positioning_timeout_s=self.positioning_timeout_s,
            default_feed_mm_min=self.default_feed_mm_min,
            home_timeout_s=self.home_timeout_s,
            controlled_steppers=self.controlled_steppers,
            rx_can_interface=self.rx_can_interface,
            rx_can_node_id=self.rx_can_node_id,
            can_position_min_rad=self.can_position_min_rad,
            can_position_max_rad=self.can_position_max_rad,
            can_default_speed_rad_s=self.can_default_speed_rad_s,
            motor_pole_pairs=self.motor_pole_pairs,
            motor_gear_ratio=self.motor_gear_ratio,
            rx_motor_position_inverted=self.rx_motor_position_inverted,
        )

    @property
    def is_can_axis(self) -> bool:
        return self.axis == "rx"

    @property
    def selected_can_node_id(self) -> int:
        if not self.is_can_axis:
            raise ValueError(f"Axis {self.axis!r} is not the RX CAN axis.")
        return self.rx_can_node_id

    @property
    def selected_can_interface(self) -> str:
        if not self.is_can_axis:
            raise ValueError(f"Axis {self.axis!r} is not the RX CAN axis.")
        return self.rx_can_interface

    @property
    def motor_position_inverted(self) -> bool:
        return self.rx_motor_position_inverted

    @property
    def selected_rx_axis_server_url(self) -> str:
        return self.rx_axis_server_url
