"""openderm-sim-capture: build a processing-ready capture folder from freehand photos.

Stand-in for src/capture when the gantry is unavailable. COLMAP estimates every
camera pose and a sparse surface from the images; the result is written in the
folder format src/processing reads, with the poses supplied through --poses-from.

Metric scale is NOT observable from the images: it comes from --standoff-mm (the
mean camera-to-surface distance). Millimetre outputs are only as accurate as that
number.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import cv2

from .alignment import canonicalize
from .masks import MASK_SUBDIR, MaskError, check_masks, install_masks, skin_point_filter
from .colmap_runner import DEFAULT_THREADS, ReconstructionError, list_images, reconstruct
from .export import OutputDirectoryError, prepare_output_dir, write_capture_folder
from .process import registration_command, run

MIN_IMAGES = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openderm-sim-capture", description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("image_dir", type=Path, help="folder of freehand photos (top level, jpg/png, one camera)")
    parser.add_argument("--out", type=Path, required=True, help="capture folder to create for src/processing")
    parser.add_argument(
        "--standoff-mm",
        type=float,
        required=True,
        help="approximate mean camera-to-surface distance in mm; sets the metric scale",
    )
    parser.add_argument("--matcher", choices=("auto", "exhaustive", "sequential"), default="auto")
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto", help="COLMAP/processing compute device")
    parser.add_argument(
        "--min-registered-frac",
        type=float,
        default=0.5,
        help="fail if COLMAP registers less than this fraction of the images (default 0.5)",
    )
    parser.add_argument("--force", action="store_true", help="replace an earlier sim-capture run in --out")
    parser.add_argument("--keep-database", action="store_true", help="keep COLMAP's feature database (can be large)")
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help="COLMAP worker threads; SIFT needs about 1 GB per thread at 1440x2560 (default %(default)s)",
    )
    parser.add_argument(
        "--skin-masks",
        type=Path,
        default=None,
        help="directory of <image stem>.png masks (255 = subject, see sim_capture.tools.skin_masks): "
        "scale, surface axes and processing then use only the masked region",
    )
    parser.add_argument("--verbose", action="store_true", help="show COLMAP's own log")
    parser.add_argument("--process", action="store_true", help="then run src/processing on the result")
    parser.add_argument("--quality", choices=("preview", "full"), default="preview")
    parser.add_argument("--dry-run", action="store_true", help="with --process: print the command only")
    return parser


def _image_size(path: Path) -> tuple[int, int]:
    """(width, height) as processing reads it: EXIF orientation ignored."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"cannot read image {path}")
    return int(image.shape[1]), int(image.shape[0])


def validate_inputs(image_dir: Path, out_dir: Path) -> list[str]:
    if not image_dir.is_dir():
        raise ValueError(f"image directory does not exist: {image_dir}")
    if out_dir.resolve() == image_dir.resolve():
        raise ValueError("--out must differ from the image directory")
    names = list_images(image_dir)
    if len(names) < MIN_IMAGES:
        raise ValueError(f"need at least {MIN_IMAGES} images in {image_dir}, found {len(names)}")
    stems = [Path(name).stem for name in names]
    if len(set(stems)) != len(stems):
        raise ValueError("two images share a file stem (e.g. a.jpg and a.png); sidecars would collide")
    sizes = {name: _image_size(image_dir / name) for name in names}
    if len(set(sizes.values())) != 1:
        counts: dict[tuple[int, int], int] = {}
        for size in sizes.values():
            counts[size] = counts.get(size, 0) + 1
        raise ValueError(
            "all images must share one size (processing assumes a single camera and zoom); found "
            + ", ".join(f"{w}x{h} x{n}" for (w, h), n in sorted(counts.items(), key=lambda kv: -kv[1]))
        )
    return names


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        names = validate_inputs(args.image_dir, args.out)
        prepare_output_dir(args.out, args.force)
    except (ValueError, OutputDirectoryError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.standoff_mm <= 0:
        print("error: --standoff-mm must be positive", file=sys.stderr)
        return 2

    print(f"[1/3] COLMAP on {len(names)} images ({args.matcher} matching)", flush=True)
    workspace = args.out / "sim" / "colmap"
    try:
        model = reconstruct(args.image_dir, workspace, names, args.matcher, args.device, args.verbose, args.threads)
    except ReconstructionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        if not args.keep_database:
            (workspace / "database.db").unlink(missing_ok=True)

    expected_size = _image_size(args.image_dir / names[0])
    if (model.width, model.height) != expected_size:
        print(
            f"error: COLMAP read {model.width}x{model.height} but processing reads "
            f"{expected_size[0]}x{expected_size[1]} (EXIF orientation?); strip the orientation tag "
            "from the images and retry",
            file=sys.stderr,
        )
        return 1
    fraction = len(model.image_names) / len(names)
    print(
        f"      registered {len(model.image_names)}/{len(names)} images, {len(model.points)} points, "
        f"reprojection {model.mean_reprojection_error_px:.2f}px, fx={model.fx:.0f}px, k1={model.k1:+.4f}"
    )
    if fraction < args.min_registered_frac:
        print(
            f"error: only {fraction:.0%} of the images registered (--min-registered-frac "
            f"{args.min_registered_frac:g}); the set lacks overlap or texture. Unregistered: "
            + ", ".join(model.unregistered_images[:8])
            + (" ..." if len(model.unregistered_images) > 8 else ""),
            file=sys.stderr,
        )
        return 1
    if model.unregistered_images:
        print(f"      ! unregistered (dropped): {', '.join(model.unregistered_images[:8])}", file=sys.stderr)
    if model.extra_model_sizes:
        print(
            f"      ! COLMAP also built disconnected model(s) of {model.extra_model_sizes} images; "
            "only the largest is used",
            file=sys.stderr,
        )

    print(f"[2/3] metric scale + canonical gauge (standoff {args.standoff_mm:g} mm)", flush=True)
    surface_point_mask = None
    if args.skin_masks is not None:
        try:
            check_masks(args.skin_masks, model.image_names)
            surface_point_mask = skin_point_filter(model, args.skin_masks)
            print(
                f"      {int(surface_point_mask.sum())}/{len(surface_point_mask)} COLMAP points lie on the masked surface"
            )
        except MaskError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
    try:
        canonical = canonicalize(model, args.standoff_mm, surface_point_mask)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    report = write_capture_folder(
        args.out,
        args.image_dir,
        model,
        canonical,
        args.standoff_mm,
        extra_report=None
        if surface_point_mask is None
        else {"skin_masks": str(args.skin_masks), "points_on_mask": int(surface_point_mask.sum())},
    )
    mask_dir = None
    if args.skin_masks is not None:
        mask_dir = args.out / "sim" / MASK_SUBDIR
        install_masks(args.skin_masks, mask_dir, model.image_names)
    for warning in canonical.warnings:
        print(f"      ! {warning}", file=sys.stderr)
    print(
        f"[3/3] wrote {len(model.image_names)} frames to {args.out} "
        f"(scale {canonical.scale_mm_per_colmap_unit:.4g} mm/COLMAP unit, surface "
        f"{report['diagnostics']['surface_extent_x_mm']:.0f}x{report['diagnostics']['surface_extent_y_mm']:.0f} mm)"
    )

    command = registration_command(
        args.out, model.fx, model.width, args.standoff_mm, args.quality, args.device, mask_dir=mask_dir
    )
    if args.process:
        return run(command, args.dry_run)
    print("next: python -m processing.register_scan_3d ... (or re-run with --process):")
    print("  " + " ".join(command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
