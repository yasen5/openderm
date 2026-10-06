"""Write the capture folder src/processing consumes, plus sim-capture's own files.

Layout (processing globs ``*.json`` at the top level only, so everything that is
not a sidecar lives under ``sim/``)::

    <out>/<image>.jpg           hard link / copy of each registered image
    <out>/<image>.json          sidecar in the gantry schema (synthetic values)
    <out>/sim/poses.json        COLMAP poses + intrinsics, read by --poses-from
    <out>/sim/report.json       registration statistics, scale, warnings
    <out>/sim/sparse.ply        sparse surface, millimetres, processing's gauge
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .model import CanonicalPoses, FloatArray, SparseModel

if TYPE_CHECKING:
    from processing.external_poses import ExternalPoseFrameJson, ExternalPosesDocument

SIM_DIR = "sim"


class OutputDirectoryError(RuntimeError):
    """The output folder holds files this run must not mix with."""


def _is_sim_sidecar(path: Path) -> bool:
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return isinstance(document, dict) and document.get("sim") is True


def prepare_output_dir(out_dir: Path, force: bool) -> None:
    """Create ``out_dir``; refuse to mix with foreign sidecars, replace our own on ``force``.

    Processing loads every top-level ``*.json`` as a frame, so a leftover sidecar
    from an earlier run (or from a real gantry scan) would silently join this one.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecars = sorted(out_dir.glob("*.json"))
    foreign = [p for p in sidecars if not _is_sim_sidecar(p)]
    if foreign:
        raise OutputDirectoryError(
            f"{out_dir} already contains non-sim-capture JSON ({foreign[0].name}, ...); "
            "choose an empty output directory"
        )
    if sidecars and not force:
        raise OutputDirectoryError(
            f"{out_dir} already holds a sim-capture run; pass --force to replace it"
        )
    for sidecar in sidecars:
        sidecar.unlink()
    shutil.rmtree(out_dir / SIM_DIR / "colmap", ignore_errors=True)


def _place_image(source: Path, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:  # different filesystem, or links unsupported
        shutil.copy2(source, destination)


def build_sidecar(image_path: Path, station: int, center_mm: FloatArray, depth_mm: float) -> dict[str, Any]:
    """A sidecar in the gantry schema.

    The gantry fields hold the camera centre and zeros. Processing does not use them
    for pose when ``--poses-from`` is given (the poses file is authoritative); they exist
    so ``load_scan_camera_frames`` loads unchanged. ``sim: true`` marks the file as ours.
    """
    return {
        "image": str(image_path),
        "station": station,
        "row": 1,
        "col": station,
        "phase": "+y",
        "x_mm": float(center_mm[0]),
        "y_mm": float(center_mm[1]),
        "z_mm": float(center_mm[2]),
        "rx_rad": 0.0,
        "sensor1_mm": float(depth_mm),
        "sensor2_mm": float(depth_mm),
        "sensor1_in_range": True,
        "sensor2_in_range": True,
        "target_mm": float(depth_mm),
        "settled": True,
        "sim": True,
    }


def write_ply(path: Path, points_mm: FloatArray, colors_rgb: NDArray[np.uint8]) -> None:
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(points_mm)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with path.open("w") as ply:
        ply.write(header)
        for (x, y, z), (r, g, b) in zip(points_mm, colors_rgb):
            ply.write(f"{x:.3f} {y:.3f} {z:.3f} {int(r)} {int(g)} {int(b)}\n")


def write_capture_folder(
    out_dir: Path,
    image_dir: Path,
    model: SparseModel,
    canonical: CanonicalPoses,
    standoff_mm: float,
    extra_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write images, sidecars and the sim/ files. Returns the report dict."""
    sim_dir = out_dir / SIM_DIR
    sim_dir.mkdir(parents=True, exist_ok=True)

    pose_frames: list[ExternalPoseFrameJson] = []
    for index, name in enumerate(model.image_names):
        placed = out_dir / name
        _place_image(image_dir / name, placed)
        sidecar = build_sidecar(placed.resolve(), index + 1, canonical.centers_mm[index], canonical.depths_mm[index])
        (out_dir / f"{Path(name).stem}.json").write_text(json.dumps(sidecar, indent=2))
        pose_frames.append(
            {
                "image": name,
                "R_cam2world": canonical.rotations_cam2world[index].tolist(),
                "C_mm": canonical.centers_mm[index].tolist(),
                "depth_mm": float(canonical.depths_mm[index]),
            }
        )

    poses_document: ExternalPosesDocument = {
        "source": "colmap",
        "image_size": [model.width, model.height],
        "fx_full": model.fx,
        "k1": model.k1,
        "frames": pose_frames,
    }
    (sim_dir / "poses.json").write_text(json.dumps(poses_document, indent=2))

    trusted = canonical.surface_point_mask
    write_ply(sim_dir / "sparse.ply", canonical.points_mm[trusted], model.point_colors[trusted])

    report: dict[str, Any] = {
        "images_registered": len(model.image_names),
        "images_unregistered": model.unregistered_images,
        "other_models_sizes": model.extra_model_sizes,
        "mean_reprojection_error_px": model.mean_reprojection_error_px,
        "sparse_points": int(len(model.points)),
        "surface_points_used": int(trusted.sum()),
        "intrinsics": {"fx_full_px": model.fx, "k1": model.k1, "width": model.width, "height": model.height},
        "standoff_mm": standoff_mm,
        "scale_mm_per_colmap_unit": canonical.scale_mm_per_unit,
        "px_per_mm_at_standoff": model.fx / standoff_mm,
        "diagnostics": canonical.diagnostics,
        "warnings": canonical.warnings,
        **(extra_report or {}),
    }
    (sim_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report
