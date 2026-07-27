"""Construction and dependency binding for a contour scan runtime."""

from __future__ import annotations

from functools import partial
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

from ..helpers import _read_startup_collision_pose
from . import io as contour_io
from . import motion as contour_motion
from . import regulation as contour_regulation
from . import retreat as contour_retreat


def _bind(ctx, name, function) -> None:
    setattr(ctx, name, partial(function, ctx))


def _install_trace_and_readers(ctx) -> bool:
    args = ctx.args
    ctx.record_on = bool(getattr(args, "record", False))
    ctx.record_path = (
        Path(args.record_path)
        if getattr(args, "record_path", None)
        else Path(args.capture_dir) / "trace.jsonl"
    )
    ctx.pose_last = {"x": None, "y": None, "z": None, "rx": None}
    ctx.rec_ctx = {
        "activity": "startup",
        "station": None,
        "phase": None,
        "col": None,
    }
    ctx._rec = {"fh": None, "t0": None, "n": 0}

    def set_activity(name: str) -> None:
        ctx.rec_ctx["activity"] = name

    ctx.set_activity = set_activity
    ctx.z_last = {"mm": None}
    ctx.rx_last = {"rad": None}
    ctx.level_anchor = {"rad": None}
    _bind(ctx, "record_sample", contour_io.record_sample)
    _bind(ctx, "_close_trace", contour_io._close_trace)
    _bind(ctx, "read_axis", contour_io.read_axis)
    _bind(ctx, "read_position", contour_io.read_position)
    _bind(ctx, "read_rx_rad", contour_io.read_rx_rad)

    if ctx.collision_guard is None:
        return True
    try:
        pose, rx, clearance = _read_startup_collision_pose(
            ctx.collision_guard,
            ctx.read_position,
            ctx.read_rx_rad,
        )
    except Exception as exc:
        print(
            f"Aborting contour scan: self-collision guard cannot verify the parked pose: {exc}.",
            file=sys.stderr,
        )
        ctx._close_trace()
        return False
    if clearance >= ctx.collision_guard.margin:
        return True
    message = (
        f"parked pose x={pose['x']:.0f} y={pose['y']:.0f} "
        f"z={pose['z']:.0f} rx={rx:+.3f}rad has clearance "
        f"{clearance:.0f}mm < {ctx.collision_guard.margin:.0f}mm"
    )
    print(
        f"Aborting contour scan: self-collision guard: {message}. "
        "Move to a safe pose before scanning.",
        file=sys.stderr,
    )
    ctx._close_trace()
    return False


def _install_motion_state(ctx) -> None:
    ctx.z_stream = {"cmd": None, "bad": 0}
    ctx.pico_stream_prev = {}
    ctx.Z_TRUST_BAND_MM = 20.0
    ctx.Z_MAX_BAD = 3
    ctx.Z_NOOP_EPS_MM = 0.02
    ctx.z_sat = {"meas": None, "at_limit": False}
    ctx.rx_limit = {"clamped": False, "axis": False}
    ctx.y_limit = {"hit": False}
    ctx.vel_scale_carry = {"scale": 1.0}
    ctx.pivot_jump = {"hit": False}
    ctx.breadcrumb = []
    ctx.z_rx = {"rad": None}
    ctx.station_rx_bounds = {"min": None, "max": None}
    ctx.pivot_anchor = {"x": None, "y": None, "z": None, "rx": None}
    ctx.grid_anchor = {
        "fx": None,
        "fy": None,
        "gx": None,
        "gy": None,
        "rx": None,
        "valid": False,
    }
    ctx.floor_suspect = {"n": 0, "warned": False}
    ctx.rx_move_fail = {"reason": None}
    ctx.rx_start = None
    ctx.move_to_grid_pose = None


def _bind_runtime_helpers(ctx) -> None:
    motion_helpers = {
        "_clamp_z": contour_motion._clamp_z,
        "step_z": contour_motion.step_z,
        "_move_gantry_arc": contour_motion._move_gantry_arc,
        "breadcrumb_push": contour_motion.breadcrumb_push,
        "breadcrumb_reseed": contour_motion.breadcrumb_reseed,
        "_rx_rejected": contour_motion._rx_rejected,
        "_clamp_z_abs": contour_motion._clamp_z_abs,
        "clamp_rx": contour_motion.clamp_rx,
        "move_regulated_pose": contour_motion.move_regulated_pose,
        "rx_vel_collision_blocked": contour_motion.rx_vel_collision_blocked,
        "_wait_for_swing": contour_motion._wait_for_swing,
        "move_axis_to": contour_motion.move_axis_to,
        "move_x_to": contour_motion.move_x_to,
        "move_y_to": contour_motion.move_y_to,
        "_traverse_arc_rise": contour_motion._traverse_arc_rise,
        "_wait_for_rx": contour_motion._wait_for_rx,
        "set_rx_absolute": contour_motion.set_rx_absolute,
        "attempt_edge_recovery": contour_motion.attempt_edge_recovery,
    }
    io_helpers = {
        "_floor_baseline_z": contour_io._floor_baseline_z,
        "_floor_threshold_mm": contour_io._floor_threshold_mm,
        "_reject_floor": contour_io._reject_floor,
        "read_exclusive": contour_io.read_exclusive,
        "snapshot_for_capture": contour_io.snapshot_for_capture,
    }
    retreat_helpers = {
        "settle_z_standoff": contour_retreat.settle_z_standoff,
        "replay_pose": contour_retreat.replay_pose,
        "retreat_along_breadcrumb": contour_retreat.retreat_along_breadcrumb,
    }
    for name, function in {
        **motion_helpers,
        **io_helpers,
        **retreat_helpers,
    }.items():
        _bind(ctx, name, function)
    _bind(ctx, "regulate_station", contour_regulation.regulate_station)


def build_runtime(
    *,
    args,
    floor_model,
    client_x,
    client_y,
    client_z,
    pico_axes,
    pico_link,
    pico_multi,
    collision_guard,
    rx_client,
    pivot_model,
    controller,
    camera_types,
):
    """Create the shared runtime state consumed by contour scan stages."""
    ctx = SimpleNamespace(
        args=args,
        floor_model=floor_model,
        client_x=client_x,
        client_y=client_y,
        client_z=client_z,
        pico_axes=pico_axes,
        pico_link=pico_link,
        pico_multi=pico_multi,
        collision_guard=collision_guard,
        rx_client=rx_client,
        pivot_model=pivot_model,
        controller=controller,
        lasers_both_on={"on": False},
        stop_state={"requested": False},
        CanonCamera=camera_types[0],
        CanonCaptureConfig=camera_types[1],
        CanonEdsdk=camera_types[2],
        CanonError=camera_types[3],
    )
    ctx._runtime_ctx = ctx

    def handle_sigint(signum, frame):  # noqa: ANN001
        ctx.stop_state["requested"] = True

    signal.signal(signal.SIGINT, handle_sigint)
    if not _install_trace_and_readers(ctx):
        return None
    _install_motion_state(ctx)
    _bind_runtime_helpers(ctx)
    return ctx
