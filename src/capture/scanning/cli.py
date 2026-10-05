"""Command-line interface for the OpenDerm capture workflow.

The rig profile supplies routine settings while allowing explicit tuning
overrides for different hardware and capture geometry.
"""

from __future__ import annotations

import argparse
import shlex
from collections.abc import Sequence

from . import contour as scan


SCAN_PROFILE = (
    "--x-travel-mm",
    "300",
    "--x-step-mm",
    "20",
    "--y-step-mm",
    "15",
    "--target-mm",
    "110",
    "--z-max-mm",
    "392",
    "--z-min-mm",
    "5",
    "--rx-filter-alpha",
    "0.6",
    "--y-pico-acc-mm-s2",
    "150",
    "--rx-speed-rad-s",
    "0.3",
    "--z-pico-acc-mm-s2",
    "150",
    "--period-s",
    "0.01",
    "--rx-accel-rad-s2",
    "2.0",
    "--rx-gain-rad-per-mm",
    "-0.015",
    "--floor-depth-mm",
    "507",
    "--floor-margin-mm",
    "3",
    "--y-pico-vmax-mm-s",
    "85",
    "--z-pico-vmax-mm-s",
    "65",
    "--record",
    "--band-edge-recovery",
    "--feed-mm-min",
    "3000",
    "--record-pause-s",
    "0.5",
    "--simultaneous-sensors",
    "--rx-deadband-mm",
    "0.7",
    "--band-miss-stop-frac",
    "0.30",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openderm-scan",
        description=(
            "Capture an OpenDerm contour-following scan. "
            "Additional scanner flags may be supplied for hardware-specific tuning."
        ),
    )
    parser.add_argument(
        "capture_dir",
        help="New output directory, e.g. captures/subject-001-site-001",
    )
    parser.add_argument("--x-travel-mm", type=float, default=None)
    parser.add_argument("--gantry-server-url", default=None)
    parser.add_argument("--rx-server-url", default=None)
    parser.add_argument("--pico-port", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--no-camera", action="store_true", help="Exercise motion without taking photos."
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print the fully expanded scanner arguments and exit.",
    )
    parser.add_argument(
        "--advanced-help",
        action="store_true",
        help="Show all scanner tuning options.",
    )
    return parser


def expanded_args(args: argparse.Namespace, overrides: Sequence[str]) -> tuple[object, list[str]]:
    scanner = scan
    values = list(SCAN_PROFILE)
    values.extend(("--capture-dir", args.capture_dir))

    if args.x_travel_mm is not None:
        values.extend(("--x-travel-mm", str(args.x_travel_mm)))

    for flag, value in (
        ("--gantry-server-url", args.gantry_server_url),
        ("--rx-server-url", args.rx_server_url),
        ("--pico-port", args.pico_port),
    ):
        if value is not None:
            values.extend((flag, value))
    if args.debug:
        values.append("--debug")
    if args.no_camera:
        values.append("--no-camera")
    values.extend(overrides)
    return scanner, values


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, overrides = parser.parse_known_args(argv)
    if args.advanced_help:
        scan.build_parser().print_help()
        return 0
    scanner, values = expanded_args(args, overrides)
    if args.show_config:
        print(shlex.join(values))
        return 0
    return scanner.run(scanner.build_parser().parse_args(values))


if __name__ == "__main__":
    raise SystemExit(main())
