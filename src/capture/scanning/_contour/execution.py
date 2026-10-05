"""Contour-scan orchestration.

The operational details live in focused modules: lifecycle owns device
startup/shutdown, grid motion owns pivot-compensated traverses, capture owns
camera/metadata handling, and sweep owns the band-march state machine.
"""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

from . import (
    band_motion as contour_band_motion,
    capture as contour_capture,
    grid_motion as contour_grid_motion,
    lifecycle as contour_lifecycle,
    sweep as contour_sweep,
)


def _new_scan_state() -> SimpleNamespace:
    return SimpleNamespace(
        camera=None,
        camera_pending={"info": None},
        cols=0,
        global_k=0,
        last_capture_pose={"x": None, "y": None, "z": None, "rx": None},
        pose_log_path=None,
        rx_start=None,
        x_start=None,
        y_start=None,
    )


def _bind_scan_operations(ctx, state) -> None:
    """Expose mutually recursive scan operations through the runtime context."""
    ctx.move_to_grid_pose = partial(contour_grid_motion.move_to_grid_pose, ctx)
    ctx._band_walk_cruise = partial(
        contour_band_motion._band_walk_cruise,
        ctx,
    )
    state.capture_station = partial(
        contour_capture.capture_station,
        ctx,
        state,
    )


def execute_scan(ctx) -> int:
    """Run one contour scan and always release every opened device."""
    state = _new_scan_state()
    finalize_capture = partial(
        contour_capture.finalize_pending_capture,
        ctx,
        state,
    )
    try:
        if not contour_lifecycle.initialize_scan(ctx, state):
            return 1
        _bind_scan_operations(ctx, state)
        contour_sweep.run_band_sweeps(ctx, state)
        finalize_capture()
        contour_lifecycle.park_head(ctx)
        contour_lifecycle.report_scan(ctx, state)
        return 0
    finally:
        contour_lifecycle.cleanup_scan(ctx, state, finalize_capture)
