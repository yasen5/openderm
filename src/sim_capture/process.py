"""Registration command for a sim-capture folder (the sim counterpart of openderm-process).

Single stage: the poses come from COLMAP, so there is no rig fit to bootstrap and
no second ``--rig-from`` pass. Parameters that are in millimetres or pixels are
derived from the scale and resolution of this capture instead of the macro rig's
fixed values.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

Quality = Literal["preview", "full"]

PREVIEW_TARGET_WIDTH_PX = 1280  # analysis resolution for preview
REFERENCE_STANDOFF_MM = 110.0  # the rig's defaults (2 mm surface pitch, 4 mm sigma) assume this
MAX_TEXTURE_PPMM = 78.0  # `openderm-process --quality full`


def registration_command(
    out_dir: Path,
    fx_full: float,
    image_width: int,
    standoff_mm: float,
    quality: Quality = "preview",
    device: str = "auto",
    python: str = sys.executable,
    mask_dir: Path | None = None,
) -> list[str]:
    """argv for ``processing.register_scan_3d`` on a sim-capture folder."""
    downscale = 1 if quality == "full" else max(1, round(image_width / PREVIEW_TARGET_WIDTH_PX))
    full_res_px_per_mm = fx_full / standoff_mm
    texture_ppmm = min(MAX_TEXTURE_PPMM, 0.8 * full_res_px_per_mm / downscale)
    standoff_scale = standoff_mm / REFERENCE_STANDOFF_MM  # mm-valued defaults scale with the subject distance
    command = [
        python,
        "-m",
        "processing.register_scan_3d",
        str(out_dir),
        "--poses-from",
        str(out_dir / "sim" / "poses.json"),
        "--fx-full",
        f"{fx_full:g}",
        "--downscale",
        str(downscale),
        "--texture-ppmm",
        f"{texture_ppmm:.3g}",
        # COLMAP's poses are the prior: nothing to snap back to, and its error is not
        # gantry-encoder small, so loosen the prior and disable the gantry outlier gate
        "--reject-pose-mm",
        "0",
        "--reject-rot-deg",
        "0",
        "--sigma-t",
        f"{0.04 * standoff_mm:.3g}",
        "--sigma-r",
        "2",
        "--surface-pitch",
        f"{2.0 * standoff_scale:.3g}",
        "--mesh-pitch",
        f"{1.0 * standoff_scale:.3g}",
        "--contour",
        "on",
        "--contour-smooth",
        "25",
        "--blend",
        "two-band",
        "--focus-weight",
        "4",
        "--max-incidence-deg",
        "65",
        "--mesh-smooth",
        "50",
        "10",
        "--device",
        device,
        "--out",
        str(out_dir / "registration3d"),
    ]
    if mask_dir is not None:
        command += ["--mask-dir", str(mask_dir)]
    return command


def run(command: Sequence[str], dry_run: bool = False) -> int:
    print(f"+ {shlex.join(command)}", flush=True)
    if dry_run:
        return 0
    return subprocess.run(command, check=False).returncode
