"""Two-stage OpenDerm scan registration."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openderm-process",
        description=(
            "Build a canonical 3D skin reconstruction, then check it for "
            "texture doubling and projection ghosts."
        ),
    )
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--quality", choices=("preview", "full"), default="preview")
    parser.add_argument(
        "--fx-full",
        type=float,
        default=39237.0,
        help=(
            "full-resolution focal length in pixels from your camera "
            "intrinsics calibration (default: 39237 for the reference camera)"
        ),
    )
    parser.add_argument(
        "--refit-rig",
        action="store_true",
        help="Re-run the stage-one rig-model fit even when placements3d.json exists.",
    )
    parser.add_argument("--skip-checks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def registration_flags(args: argparse.Namespace) -> list[str]:
    quality = (
        ["--downscale", "3", "--texture-ppmm", "20"]
        if args.quality == "preview"
        else ["--downscale", "1", "--texture-ppmm", "78"]
    )
    shared = [
        "--fx-full",
        f"{args.fx_full:g}",
        "--deformable",
        "--blend",
        "two-band",
        "--focus-weight",
        "4",
        *quality,
    ]
    return [
        *shared,
        "--group-by-row",
        "--contour",
        "on",
        "--contour-smooth",
        "25",
        "--group-feather-mm",
        "4",
        "--max-incidence-deg",
        "65",
        "--mesh-smooth",
        "50",
        "10",
    ]


def commands(args: argparse.Namespace) -> tuple[list[str], list[str], list[list[str]]]:
    capture_dir = args.capture_dir.resolve()
    rigfit_dir = capture_dir / "registration3d-rigfit"
    canonical_dir = capture_dir / "registration3d-canonical"
    flags = registration_flags(args)
    module = [sys.executable, "-m", "processing.register_scan_3d", str(capture_dir)]
    stage_one = [*module, *flags, "--out", str(rigfit_dir)]
    stage_two = [
        *module,
        *flags,
        "--rig-from",
        str(rigfit_dir / "placements3d.json"),
        "--out",
        str(canonical_dir),
    ]
    checks = [
        [sys.executable, "-m", "processing.find_doublings", str(canonical_dir)],
        [sys.executable, "-m", "processing.ghost_check", str(canonical_dir)],
    ]
    return stage_one, stage_two, checks


def _run(command: Sequence[str], *, dry_run: bool) -> None:
    print(f"+ {shlex.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.dry_run and not args.capture_dir.is_dir():
        parser.error(f"capture directory does not exist: {args.capture_dir}")
    stage_one, stage_two, checks = commands(args)

    placements = args.capture_dir.resolve() / "registration3d-rigfit" / "placements3d.json"
    if args.refit_rig or not placements.exists():
        _run(stage_one, dry_run=args.dry_run)
    else:
        print(f"reusing rig-model fit: {placements}")
    _run(stage_two, dry_run=args.dry_run)
    if not args.skip_checks:
        for command in checks:
            _run(command, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
