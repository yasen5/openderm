#!/usr/bin/env python3
"""Register a skin-scan capture folder in full 3D.

The public module owns the CLI lifecycle and delegates the four reconstruction
stages to :mod:`processing._reconstruction`. Keeping the orchestration here
preserves the command entry point and lets tests replace individual stages
without importing their implementation details.
"""

from __future__ import annotations

import sys

from ._reconstruction.exporter import export_scan_reconstruction_artifacts
from ._reconstruction.parser import parse_scan_cli_arguments
from ._reconstruction.problem import build_scan_reconstruction_problem
from ._reconstruction.solver import reconstruct_surface_from_camera_frames
from .registration_features import _apply_processing_memory_limit
from .registration_geometry import project_world_points_into_camera

__all__ = ["main", "project_world_points_into_camera"]


def main():
    # The box is shared: a runaway allocation must kill this process cleanly,
    # never invoke the kernel OOM killer on other users.
    cap = _apply_processing_memory_limit()
    try:
        return _main(cap)
    except MemoryError:
        sys.exit(
            "openderm-register: aborted on MemoryError (self-imposed RAM "
            "cap; the kernel OOM killer was NOT invoked). Lower "
            "--texture-ppmm / raise --downscale, or free memory and rerun."
        )


def _main(mem_cap=None):
    args = parse_scan_cli_arguments()
    problem = build_scan_reconstruction_problem(args, mem_cap)
    solution = reconstruct_surface_from_camera_frames(args, problem)
    export_scan_reconstruction_artifacts(args, problem, solution)


if __name__ == "__main__":
    main()
