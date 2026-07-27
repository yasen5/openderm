"""RX-axis public API facade and server command-line entry point."""

from __future__ import annotations

import argparse

from ...config import GantryConfig
from ..cubemars import CubeMarsError
from ..security import (
    MotionSecurityError,
    control_token_from_env,
    require_secure_bind,
)
from .api import build_app
from .client import RxAxisServerClient
from .service import RxAxisService
from .transport import SocketCanTransport
from .types import (
    CanFrame,
    HomingRecord,
    RxAxisServerConfig,
    RxAxisServerError,
    RxAxisSnapshot,
)

__all__ = [
    "CanFrame",
    "HomingRecord",
    "RxAxisServerClient",
    "RxAxisServerConfig",
    "RxAxisServerError",
    "RxAxisService",
    "RxAxisSnapshot",
    "SocketCanTransport",
    "build_app",
    "build_parser",
    "config_from_args",
    "main",
    "serve",
]


def serve(config: RxAxisServerConfig, gantry_config: GantryConfig | None = None) -> int:
    import uvicorn

    try:
        require_secure_bind(config.host, config.control_token)
    except MotionSecurityError as exc:
        raise RxAxisServerError(str(exc)) from exc
    resolved_gantry_config = (gantry_config or GantryConfig.from_env()).with_overrides(
        axis=config.axis
    )
    print(
        f"[rx_axis_server] axis={resolved_gantry_config.axis} "
        f"node_id=0x{resolved_gantry_config.selected_can_node_id:02X} "
        f"interface={resolved_gantry_config.selected_can_interface} "
        f"gear_ratio={resolved_gantry_config.motor_gear_ratio} "
        f"pole_pairs={resolved_gantry_config.motor_pole_pairs} "
        f"position_inverted={resolved_gantry_config.motor_position_inverted}",
        flush=True,
    )
    app = build_app(resolved_gantry_config, config)
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the RX-axis motion coordinator.")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Bind host for the RX-axis coordinator. Non-loopback binds require "
            "OPENDERM_CONTROL_TOKEN."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Bind port for the RX-axis coordinator. Defaults to 8091.",
    )
    parser.add_argument("--axis", choices=("rx",), default="rx", help="RX axis to control.")
    parser.add_argument(
        "--poll-interval-s", type=float, default=0.05, help="Feedback poll interval in seconds."
    )
    parser.add_argument(
        "--feedback-timeout-s", type=float, default=0.05, help="CAN feedback timeout in seconds."
    )
    parser.add_argument(
        "--max-speed-rad-s",
        type=float,
        default=0.25,
        help="Maximum allowed commanded RX-axis motor speed in rad/s. Defaults to 0.25.",
    )
    parser.add_argument(
        "--velocity-deadman-s",
        type=float,
        default=0.4,
        help=(
            "Velocity-mode dead-man: an unrefreshed /velocity command auto-zeroes "
            "after this many seconds. Defaults to 0.4."
        ),
    )
    parser.add_argument(
        "--velocity-window-margin-rad",
        type=float,
        default=0.01,
        help=(
            "Velocity-mode position guard AT-REST margin: zero the velocity this "
            "far before the command-position window edge (scaled up with speed "
            "by --velocity-window-brake-s). Defaults to 0.01."
        ),
    )
    parser.add_argument(
        "--velocity-window-brake-s",
        type=float,
        default=0.1,
        help=(
            "Velocity-mode braking horizon: the window margin grows to "
            "|velocity| * this many seconds. Defaults to 0.1."
        ),
    )
    parser.add_argument(
        "--default-accel-rad-s2",
        type=float,
        default=10.0,
        help="Default trapezoidal-motion acceleration in rad/s² when a move command omits one. Defaults to 10.",
    )
    parser.add_argument(
        "--current-shutdown-threshold-a",
        type=float,
        default=5.0,
        help="Emergency stop threshold for decoded feedback current magnitude in A. Use <=0 to disable.",
    )
    parser.add_argument(
        "--motor-temperature-shutdown-c",
        type=float,
        default=80.0,
        help="Emergency stop threshold for reported motor temperature in °C. Use <=0 to disable. Defaults to 80.",
    )
    parser.add_argument(
        "--homing-fast-speed-rad-s", type=float, default=0.2, help="Fast homing approach speed."
    )
    parser.add_argument(
        "--homing-slow-speed-rad-s", type=float, default=0.02, help="Slow homing approach speed."
    )
    parser.add_argument(
        "--homing-backoff-rad",
        type=float,
        default=0.2,
        help="Positive backoff after switch trigger.",
    )
    parser.add_argument(
        "--homing-search-distance-rad",
        type=float,
        default=6.0,
        help="Negative search distance used for each homing approach.",
    )
    parser.add_argument(
        "--homing-timeout-s", type=float, default=45.0, help="Timeout for each homing phase."
    )
    parser.add_argument(
        "--homing-final-position-rad",
        type=float,
        default=None,
        help=("Position to move to after setting the limit trigger as zero. Defaults to 0.95 rad."),
    )
    parser.add_argument(
        "--min-command-position-rad",
        type=float,
        default=0.07,
        help="Minimum accepted target position after homing. Defaults to 0.07.",
    )
    parser.add_argument(
        "--max-command-position-rad",
        type=float,
        default=1.76,
        help="Maximum accepted target position after homing. Defaults to 1.76.",
    )
    return parser


DEFAULT_PORT = 8091
DEFAULT_HOMING_FINAL_POSITION_RAD = 0.95


def config_from_args(args: argparse.Namespace) -> RxAxisServerConfig:
    if args.port is None:
        args.port = DEFAULT_PORT
    if args.homing_final_position_rad is None:
        args.homing_final_position_rad = DEFAULT_HOMING_FINAL_POSITION_RAD
    if args.port <= 0:
        raise RxAxisServerError("--port must be positive.")
    if args.poll_interval_s <= 0:
        raise RxAxisServerError("--poll-interval-s must be positive.")
    if args.feedback_timeout_s <= 0:
        raise RxAxisServerError("--feedback-timeout-s must be positive.")
    if args.max_speed_rad_s <= 0:
        raise RxAxisServerError("--max-speed-rad-s must be positive.")
    if args.velocity_deadman_s <= 0:
        raise RxAxisServerError("--velocity-deadman-s must be positive.")
    if args.velocity_window_margin_rad < 0:
        raise RxAxisServerError("--velocity-window-margin-rad must be non-negative.")
    if args.velocity_window_brake_s < 0:
        raise RxAxisServerError("--velocity-window-brake-s must be non-negative.")
    if args.default_accel_rad_s2 <= 0:
        raise RxAxisServerError("--default-accel-rad-s2 must be positive.")
    if args.homing_fast_speed_rad_s <= 0:
        raise RxAxisServerError("--homing-fast-speed-rad-s must be positive.")
    if args.homing_slow_speed_rad_s <= 0:
        raise RxAxisServerError("--homing-slow-speed-rad-s must be positive.")
    if args.homing_backoff_rad <= 0:
        raise RxAxisServerError("--homing-backoff-rad must be positive.")
    if args.homing_search_distance_rad <= 0:
        raise RxAxisServerError("--homing-search-distance-rad must be positive.")
    if args.homing_timeout_s <= 0:
        raise RxAxisServerError("--homing-timeout-s must be positive.")
    if args.min_command_position_rad < 0:
        raise RxAxisServerError("--min-command-position-rad must be non-negative.")
    if args.max_command_position_rad <= args.min_command_position_rad:
        raise RxAxisServerError(
            "--max-command-position-rad must be greater than --min-command-position-rad."
        )
    try:
        control_token = control_token_from_env()
        require_secure_bind(args.host, control_token)
    except MotionSecurityError as exc:
        raise RxAxisServerError(str(exc)) from exc
    return RxAxisServerConfig(
        host=args.host,
        port=args.port,
        axis=args.axis,
        control_token=control_token,
        poll_interval_s=args.poll_interval_s,
        feedback_timeout_s=args.feedback_timeout_s,
        max_speed_rad_s=args.max_speed_rad_s,
        default_accel_rad_s2=args.default_accel_rad_s2,
        current_shutdown_threshold_a=args.current_shutdown_threshold_a,
        motor_temperature_shutdown_c=args.motor_temperature_shutdown_c,
        homing_fast_speed_rad_s=args.homing_fast_speed_rad_s,
        homing_slow_speed_rad_s=args.homing_slow_speed_rad_s,
        homing_backoff_rad=args.homing_backoff_rad,
        homing_search_distance_rad=args.homing_search_distance_rad,
        homing_timeout_s=args.homing_timeout_s,
        homing_final_position_rad=args.homing_final_position_rad,
        min_command_position_rad=args.min_command_position_rad,
        max_command_position_rad=args.max_command_position_rad,
        velocity_deadman_s=args.velocity_deadman_s,
        velocity_window_margin_rad=args.velocity_window_margin_rad,
        velocity_window_brake_s=args.velocity_window_brake_s,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return serve(config_from_args(args))
    except (RxAxisServerError, CubeMarsError) as exc:
        parser.exit(status=1, message=f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
