from __future__ import annotations

import argparse
import json
import os
from typing import Any

from ..config import DEFAULT_GANTRY_SERVER_URL, GantryConfig, default_pico_port
from ..motion.gantry.server import GantryServerClient, GantryServerError
from ..motion.rx_axis.server import RxAxisServerClient, RxAxisServerError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Control X through Klipper, Y/Z through the Raspberry Pi Pico, or RX "
            "through the RX-axis service."
        )
    )
    parser.add_argument(
        "--axis",
        choices=("x", "y", "z", "rx"),
        default=None,
        help="Target axis override. Defaults to GANTRY_AXIS or x.",
    )
    parser.add_argument(
        "--travel-min-mm",
        type=float,
        default=None,
        help="Soft minimum override in millimeters for the selected axis.",
    )
    parser.add_argument(
        "--travel-max-mm",
        type=float,
        default=None,
        help="Soft maximum override in millimeters for the selected axis.",
    )
    parser.add_argument(
        "--gantry-server-url",
        default=os.getenv("GANTRY_SERVER_URL", DEFAULT_GANTRY_SERVER_URL),
        help="URL of the Klipper X-axis command server (env: GANTRY_SERVER_URL).",
    )
    parser.add_argument(
        "--rx-axis-server-url",
        default=None,
        help=(
            "URL of the RX-axis command server. Defaults to "
            "RX_AXIS_SERVER_URL or http://127.0.0.1:8091."
        ),
    )
    parser.add_argument(
        "--pico-port",
        default=default_pico_port(),
        help=(
            "Pico serial port for Y/Z: a local device (/dev/ttyACM0) or a "
            "pyserial URL if the Pico is bridged over the network, e.g. "
            "socket://<pi1-ip>:8095 (served by openderm-pico-bridge). Defaults to PICO_PORT "
            "or the serial bridge on the GANTRY_SERVER_URL host."
        ),
    )
    parser.add_argument(
        "--pico-vmax-mm-s",
        type=float,
        default=None,
        help="Maximum speed (mm/s) for Pico Y/Z moves (default: firmware VMAX).",
    )
    parser.add_argument(
        "--pico-acc-mm-s2",
        type=float,
        default=None,
        help="Acceleration (mm/s^2) for Pico Y/Z moves (default: firmware ACC).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("home", help="Home the configured axis.")

    move_to = subparsers.add_parser("move-to", help="Move the axis to an absolute position.")
    move_to.add_argument(
        "position",
        type=float,
        help="Absolute target in axis units: mm for x/y/z, radians for rx.",
    )
    move_to.add_argument("--feed", type=float, default=None, help="Feed rate in mm/min.")
    move_to.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Target velocity in rad/s for the RX CAN axis.",
    )

    move_by = subparsers.add_parser("move-by", help="Move the axis by a relative offset.")
    move_by.add_argument(
        "delta",
        type=float,
        help="Relative move in axis units: mm for x/y/z, radians for rx.",
    )
    move_by.add_argument("--feed", type=float, default=None, help="Feed rate in mm/min.")
    move_by.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Target velocity in rad/s for the RX CAN axis.",
    )

    subparsers.add_parser("stop", help="Issue an emergency stop.")
    subparsers.add_parser(
        "clear-errors", help="Clear RX-axis motor driver errors and safety shutdown latch."
    )
    subparsers.add_parser("limit-switches", help="Read RX-axis motor limit switch states.")
    subparsers.add_parser("status", help="Fetch the selected axis status.")
    return parser


def serialize_status(status: Any) -> str:
    if isinstance(status, dict):
        return json.dumps(status, indent=2, sort_keys=True)
    return json.dumps(status.__dict__, indent=2, sort_keys=True)


def run_linear_via_server(args: argparse.Namespace, client: GantryServerClient) -> int:
    if args.command == "home":
        print(serialize_status(client.home()))
        return 0
    if args.command == "move-to":
        print(serialize_status(client.move_to(args.position, args.feed)))
        return 0
    if args.command == "move-by":
        status = client.status()
        current_position = float(status["position"][client.axis])
        print(serialize_status(client.move_to(current_position + args.delta, args.feed)))
        return 0
    if args.command == "stop":
        print(serialize_status(client.stop(mode="emergency")))
        return 0
    if args.command == "clear-errors":
        raise GantryServerError("clear-errors is only supported for RX axis.")
    if args.command == "limit-switches":
        raise GantryServerError("limit-switches is only supported for RX axis.")
    if args.command == "status":
        print(serialize_status(client.status()))
        return 0
    raise ValueError(f"Unhandled command: {args.command}")


def run_linear_via_pico(args: argparse.Namespace, axis: str) -> int:
    """Route a Y/Z command to the Raspberry Pi Pico real-time controller.

    PicoAxisClient implements the same command subset as the
    GantryServerClient subset run_linear_via_server uses, so the per-command dispatch is
    shared; this wrapper only owns the shared serial link's lifecycle and translates the
    Pico client's own errors into GantryServerError for main()'s handler.

    The Pico firmware's physical travel limits remain enabled for both axes."""
    if axis not in ("y", "z"):
        raise GantryServerError(f"the Pico controls only Y and Z (got {axis!r})")
    from ..motion.pico.adapter import PicoAxisClient, open_pico_link

    link = None
    try:
        link = open_pico_link(args.pico_port, timeout_s=30.0)
        client = PicoAxisClient(
            link,
            axis,
            enforce_limits=True,
            vmax_mm_s=args.pico_vmax_mm_s,
            acc_mm_s2=args.pico_acc_mm_s2,
            timeout_s=180.0,
        )
        return run_linear_via_server(args, client)
    except GantryServerError:
        raise  # already a clean, main()-handled error (PicoAxisClient translates its own)
    except Exception as exc:  # PicoClientError / serial / ImportError from the link layer
        raise GantryServerError(f"Pico controller error on {args.pico_port}: {exc}")
    finally:
        if link is not None:
            link.close()


def run_can_via_server(args: argparse.Namespace, client: RxAxisServerClient) -> int:
    if args.command == "home":
        print(serialize_status(client.home()))
        return 0
    if args.command == "move-to":
        print(serialize_status(client.move_to(args.position, args.speed)))
        return 0
    if args.command == "move-by":
        print(serialize_status(client.move_by(args.delta, args.speed)))
        return 0
    if args.command == "stop":
        print(serialize_status(client.stop()))
        return 0
    if args.command == "clear-errors":
        print(serialize_status(client.clear_errors()))
        return 0
    if args.command == "limit-switches":
        print(serialize_status(client.limit_switches()))
        return 0
    if args.command == "status":
        print(serialize_status(client.status()))
        return 0
    raise ValueError(f"Unhandled command: {args.command}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = GantryConfig.from_env().with_overrides(
        axis=args.axis,
        travel_min_mm=args.travel_min_mm,
        travel_max_mm=args.travel_max_mm,
    )
    try:
        if not config.is_can_axis:
            if config.axis in ("y", "z"):
                return run_linear_via_pico(args, config.axis)
            if config.axis != "x":
                raise GantryServerError(f"unsupported linear axis: {config.axis!r}")
            server_timeout_s = (
                max(
                    5.0,
                    config.home_timeout_s
                    if args.command == "home"
                    else config.positioning_timeout_s,
                )
                + 5.0
            )
            return run_linear_via_server(
                args,
                GantryServerClient(
                    args.gantry_server_url,
                    axis=config.axis,
                    timeout_s=server_timeout_s,
                ),
            )
        can_url = args.rx_axis_server_url or config.selected_rx_axis_server_url
        return run_can_via_server(
            args,
            RxAxisServerClient(
                can_url,
                timeout_s=max(5.0, config.positioning_timeout_s) + 5.0,
            ),
        )
    except (GantryServerError, RxAxisServerError) as exc:
        parser.exit(status=1, message=f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
