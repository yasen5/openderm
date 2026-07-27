"""Command-line interface for Canon capture and power-state helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

from .sdk import CanonError
from .session import (
    CanonCaptureConfig,
    capture_images,
    keep_camera_awake,
    release_camera,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Capture still images from a Canon EOS camera over USB using EDSDK.")
    )
    parser.add_argument(
        "--output-dir",
        default="captures",
        help="Directory to store downloaded images.",
    )
    parser.add_argument(
        "--basename",
        default="eos-r7",
        help="Prefix to use when naming captured files.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of still images to capture.",
    )
    parser.add_argument(
        "--interval-s",
        type=float,
        default=0.0,
        help="Delay between captures in seconds.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=15.0,
        help="Seconds to wait for each image transfer from the camera.",
    )
    parser.add_argument(
        "--edsdk-lib",
        default=None,
        help=("Explicit path to libEDSDK.so when it is not in a standard location."),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--keep-awake",
        action="store_true",
        help=(
            "Do not capture; open a camera session and exit WITHOUT closing "
            "it. The camera parks in PC-remote mode (connection icon on a "
            "black screen) where auto power off is inhibited, so later scans "
            "always find it awake; they re-claim it normally. Rerun after any "
            "scan that finishes cleanly (a clean close re-enables the sleep "
            "timer). A camera that already slept must be woken physically "
            "first."
        ),
    )
    mode.add_argument(
        "--release",
        action="store_true",
        help=(
            "Do not capture; the inverse of --keep-awake: cleanly open and "
            "close a session, dropping the camera out of PC-remote mode back "
            "to its normal UI -- its own auto-power-off timer then sleeps it "
            "on the menu schedule. (EDSDK has no direct power-down command; "
            "this is the software way to let the camera sleep.)"
        ),
    )
    return parser


def config_from_args(args: argparse.Namespace) -> CanonCaptureConfig:
    if args.count <= 0:
        raise CanonError("--count must be positive.")
    if args.interval_s < 0:
        raise CanonError("--interval-s must be zero or positive.")
    if args.timeout_s <= 0:
        raise CanonError("--timeout-s must be positive.")
    basename = args.basename.strip()
    if not basename:
        raise CanonError("--basename must not be empty.")
    return CanonCaptureConfig(
        output_dir=Path(args.output_dir),
        basename=basename,
        count=args.count,
        interval_s=args.interval_s,
        timeout_s=args.timeout_s,
        edsdk_lib=args.edsdk_lib,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.keep_awake:
            keep_camera_awake(args.edsdk_lib)
            print(
                "Camera parked in PC-remote mode (connection icon on screen): "
                "auto power off is now inhibited. Scans will re-claim it "
                "normally; rerun this after a scan that finishes cleanly."
            )
        elif args.release:
            release_camera(args.edsdk_lib)
            print(
                "Camera released to its normal UI: its own auto-power-off "
                "timer is running again, so it will sleep on the menu "
                "schedule."
            )
        else:
            capture_images(config_from_args(args))
    except CanonError as exc:
        parser.exit(status=1, message=f"{exc}\n")
    return 0
