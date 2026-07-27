"""Startup, parking, reporting, and cleanup for contour scans."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from openderm.motion.gantry.server import GantryServerError
from openderm.motion.rx_axis.server import RxAxisServerError
from openderm.sensors.hg_c import HgCSensorError

from ..config import FLOOR_SUSPECT_BAND_MM, PARK_RISE_MM, PARK_RX_RAD
from ..helpers import _fmt_bound


def _open_camera(ctx, state) -> bool:
    if not ctx.args.capture:
        return True
    try:
        sdk = ctx.CanonEdsdk(library_path=ctx.args.edsdk_lib)
        state.camera = ctx.CanonCamera(
            sdk,
            ctx.CanonCaptureConfig(
                output_dir=Path(ctx.args.capture_dir),
                basename=ctx.args.capture_basename,
                timeout_s=ctx.args.capture_timeout_s,
                prefer_raw=False,
                exposure_dwell_s=ctx.args.capture_dwell_s,
            ),
        )
        state.camera.start()
        return True
    except ctx.CanonError as exc:
        print(
            f"Canon camera not available: {exc}\n"
            "Pass --no-camera to run the scan without capturing.",
            file=sys.stderr,
        )
        return False


def _seed_motion_state(ctx, state) -> bool:
    args = ctx.args
    ctx.controller.set_enabled_for_selection("all", False)
    ctx.client_x.stream_start(
        feed_mm_min=args.feed_mm_min,
        tick_s=args.stream_tick_s,
        min_step_mm=args.stream_min_step_mm,
    )
    z_seeds = sorted(
        value for value in (ctx.read_axis(ctx.client_z) for _ in range(5)) if value is not None
    )
    if not z_seeds:
        print(
            "Could not read the current z position to seed continuous motion.",
            file=sys.stderr,
        )
        return False
    ctx.z_stream["cmd"] = z_seeds[len(z_seeds) // 2]

    state.x_start = ctx.read_axis(ctx.client_x)
    if state.x_start is None:
        print("Could not read the current x position.", file=sys.stderr)
        return False
    state.y_start = ctx.read_axis(ctx.client_y)
    if state.y_start is None:
        print(
            "Could not read the current y position (needed for the contour scan).",
            file=sys.stderr,
        )
        return False
    if args.floor_depth_mm is not None or ctx.floor_model is not None:
        ctx.read_axis(ctx.client_z)

    state.rx_start = ctx.read_rx_rad()
    for _ in range(2):
        if state.rx_start is not None:
            break
        time.sleep(0.1)
        state.rx_start = ctx.read_rx_rad()
    if state.rx_start is None:
        print(
            "Aborting contour scan: rx unreadable at start; pivot-compensated "
            "motion cannot be calculated. Check the RX-axis server.",
            file=sys.stderr,
        )
        return False
    ctx.level_anchor["rad"] = state.rx_start
    ctx.z_rx["rad"] = state.rx_start
    ctx.rx_start = state.rx_start
    return True


def _print_scan_configuration(ctx, state) -> None:
    args = ctx.args
    z_desc = (
        f"z stream gain={args.gain_mm_per_mm} "
        f"max-step={args.max_step_mm}mm target={args.target_mm}mm "
        f"deadband={args.deadband_mm}mm"
    )
    pivot_desc = f"pivot-hold xy ({Path(args.rx_pivot_model).name})"
    if args.rx_min_rad is not None or args.rx_max_rad is not None:
        tilt_desc = f"tilt-limit=[{_fmt_bound(args.rx_min_rad)},{_fmt_bound(args.rx_max_rad)}]rad"
    else:
        tilt_desc = "tilt-limit=off"
    rx_desc = (
        f"rx velocity-servo deadband={args.rx_deadband_mm}mm "
        f"alpha={args.rx_filter_alpha} | {tilt_desc} | {pivot_desc}"
    )
    cam_desc = (
        f"camera=JPEG -> {args.capture_dir}/" if args.capture else "camera=disabled (simulated)"
    )
    floor_desc = (
        f"floor-reject=model({Path(args.floor_model).name})-{args.floor_margin_mm:.0f}mm | "
        if ctx.floor_model is not None
        else f"floor-reject z+d>={args.floor_depth_mm - args.floor_margin_mm:.0f}mm | "
        if args.floor_depth_mm is not None
        else "floor-reject=off | "
    )
    print(
        f"contour scan: x start={state.x_start:.2f}mm step={args.x_step_mm}mm "
        f"travel={args.x_travel_mm}mm ({state.cols} positions per sweep) | "
        f"y start={state.y_start:.2f}mm step={args.y_step_mm}mm "
        f"(caps {args.y_max_travel_mm}mm / {args.y_max_rows} sweeps per march) | "
        f"edge tilt<={args.edge_tilt_max_deg:.0f}deg "
        f"step={args.edge_tilt_step_deg:.0f}deg sign={args.edge_tilt_sign:+.0f} "
        f"oor-iters={args.edge_oor_iters} | {floor_desc}"
        f"settle-iters={args.settle_iters} station-timeout={args.station_timeout_s}s "
        f"record-pause={args.record_pause_s}s | {z_desc} | {rx_desc} | {cam_desc} | "
        f"poses -> {state.pose_log_path} | "
        + (f"trace -> {ctx.record_path} | " if ctx.record_on else "")
        + "Ctrl-C to exit"
    )


def initialize_scan(ctx, state) -> bool:
    """Open devices, seed trusted coordinates, and create output state."""
    if not _open_camera(ctx, state) or not _seed_motion_state(ctx, state):
        return False
    state.cols = round(ctx.args.x_travel_mm / ctx.args.x_step_mm) + 1
    capture_dir = Path(ctx.args.capture_dir)
    capture_dir.mkdir(parents=True, exist_ok=True)
    state.pose_log_path = capture_dir / "poses.jsonl"
    _print_scan_configuration(ctx, state)
    return True


def park_head(ctx) -> None:
    """Raise and neutralize the head without traversing back over the body."""
    if ctx.stop_state["requested"]:
        return
    ctx.set_activity("park")
    ctx.record_sample("park:start")
    position = ctx.read_position()
    z_current = ctx.z_stream["cmd"] if ctx.z_stream["cmd"] is not None else position.get("z")
    if z_current is None or position.get("x") is None or position.get("y") is None:
        print("  park: pose unreadable; leaving the head where it is.", file=sys.stderr)
        return

    z_up = float(z_current) - PARK_RISE_MM
    swing_ok = True
    if ctx.collision_guard is not None:
        clearance = ctx.collision_guard.min_clearance(
            float(position["x"]),
            float(position["y"]),
            z_up,
            PARK_RX_RAD,
        )
        swing_ok = clearance >= ctx.collision_guard.margin
    print(
        f"  parking in place: raising z {PARK_RISE_MM:.0f}mm "
        f"({float(z_current):.1f} -> {z_up:.1f}mm)."
    )
    if not ctx._traverse_arc_rise(z_up):
        print("  park: z rise clamped or refused; leaving rx as is.", file=sys.stderr)
    elif swing_ok:
        print(f"  park: rise complete; rx -> {PARK_RX_RAD:+.2f}rad.")
        ctx.set_rx_absolute(PARK_RX_RAD, wait=True)
    else:
        print(
            "  park: rise complete, but the park tilt would sit inside the "
            "collision margin; leaving rx as is.",
            file=sys.stderr,
        )


def report_scan(ctx, state) -> None:
    if not ctx.stop_state["requested"]:
        print(f"contour scan complete: {state.global_k} stations captured.")
    if ctx.floor_suspect["n"]:
        print(
            f"  note: {ctx.floor_suspect['n']} accepted sensor reading(s) fell "
            f"within {FLOOR_SUSPECT_BAND_MM:.0f}mm of the floor-reject threshold "
            "-- the static --floor-depth-mm threshold may be leaking bed reads "
            "at some tilts; consider --floor-model (rx-swept tare) or a larger "
            "--floor-margin-mm.",
            file=sys.stderr,
        )


def _stop_aborted_motion(ctx) -> None:
    try:
        ctx.client_x.stop(mode="soft")
        print("Klipper X soft-stopped (scan aborted).", file=sys.stderr)
    except GantryServerError as exc:
        print(f"warning: gantry soft stop failed: {exc}", file=sys.stderr)
    for axis, client in (("y", ctx.client_y), ("z", ctx.client_z)):
        try:
            client.stop(mode="soft")
            print(f"Pico {axis.upper()} soft-stopped (scan aborted).", file=sys.stderr)
        except Exception as exc:
            print(f"warning: Pico {axis.upper()} stop failed: {exc}", file=sys.stderr)
    if ctx.rx_client is not None:
        try:
            ctx.rx_client.stop()
            print("RX-axis motor stopped (scan aborted).", file=sys.stderr)
        except RxAxisServerError as exc:
            print(f"warning: RX-axis motor stop failed: {exc}", file=sys.stderr)


def cleanup_scan(ctx, state, finalize_capture) -> None:
    """Best-effort shutdown that is safe after partial initialization."""
    try:
        finalize_capture(timeout_s=10.0)
    except Exception:
        pass
    try:
        ctx.client_x.stream_stop()
    except GantryServerError as exc:
        print(f"warning: stream stop failed: {exc}", file=sys.stderr)
    if ctx.stop_state["requested"]:
        _stop_aborted_motion(ctx)
    try:
        ctx.controller.set_enabled_for_selection("all", False)
    except HgCSensorError:
        pass
    try:
        ctx.controller.close()
    except Exception:
        pass
    if state.camera is not None:
        try:
            state.camera.close()
        except Exception:
            pass
    if ctx.pico_link is not None:
        try:
            ctx.pico_link.close()
        except Exception:
            pass
    ctx._close_trace()
