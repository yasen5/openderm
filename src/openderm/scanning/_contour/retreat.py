"""Sensor-guarded Z settling and breadcrumb retreat helpers."""

from __future__ import annotations

import math
import sys
import time

from openderm.motion.gantry.server import GantryServerError

from ..config import BREADCRUMB_SPACING_RAD
from ..helpers import _clamp, _classify_standoff


def settle_z_standoff(ctx, retreat_only: bool = False) -> None:
    """Run the Z-only settle while preserving the caller's trace activity."""
    previous = ctx.rec_ctx["activity"]
    ctx.set_activity("settle_z")
    try:
        _settle_z_standoff_impl(ctx, retreat_only)
    finally:
        ctx.set_activity(previous)


def _settle_z_standoff_impl(ctx, retreat_only: bool = False) -> None:
    """Drive standoff with Z only, failing closed if a near surface persists."""
    args = ctx.args
    on_target = 0
    too_near = False
    for _ in range(args.edge_settle_iters):
        if ctx.stop_state["requested"]:
            return
        first = ctx.read_exclusive("sensor1")
        second = ctx.read_exclusive("sensor2")
        action, error = _classify_standoff(
            first,
            second,
            args.target_mm,
            args.deadband_mm,
            retreat_only,
        )
        too_near = action == "retreat"
        if action == "retreat":
            on_target = 0
            if not ctx.step_z(-abs(args.max_step_mm)):
                ctx.stop_state["requested"] = True
                return
        elif action in ("far", "skip"):
            return
        elif action == "hold":
            on_target += 1
            if on_target >= args.settle_iters:
                return
        else:
            on_target = 0
            if not ctx.step_z(_clamp(args.gain_mm_per_mm * error, args.max_step_mm)):
                ctx.stop_state["requested"] = True
                return
        time.sleep(args.period_s)
    if too_near:
        print(
            "  ERROR: standoff still inside the close limit after "
            f"{args.edge_settle_iters} retreat iterations (z may be pinned at "
            "--z-min-mm, or the surface is rising faster than the retreat can "
            "back off); aborting to avoid driving the head into the surface.",
            file=sys.stderr,
        )
        ctx.stop_state["requested"] = True


def replay_move_to(ctx, target: dict[str, float]) -> bool:
    """Stream an absolute X/Y/Z target and poll until it arrives."""
    if not target:
        return not ctx.stop_state["requested"]
    ctx.record_sample(
        "cmd:replay",
        target={key: round(value, 4) for key, value in target.items()},
    )
    clients = (
        ("x", ctx.client_x),
        ("y", ctx.client_y),
        ("z", ctx.client_z),
    )
    for axis, client in clients:
        if axis not in target:
            continue
        try:
            client.stream_to(target[axis])
        except GantryServerError as exc:
            print(f"  replay move rejected: {exc}", file=sys.stderr)
            return False
        if axis == "z":
            ctx.z_stream["cmd"] = target["z"]
            ctx.z_stream["bad"] = 0
    deadline = time.monotonic() + ctx.args.y_move_timeout_s
    while not ctx.stop_state["requested"]:
        position = ctx.read_position()
        if all(
            position.get(axis) is not None
            and abs(position[axis] - target[axis]) <= ctx.args.x_tolerance_mm
            for axis in target
        ):
            break
        if time.monotonic() >= deadline:
            print(
                "  warning: replay move did not arrive within "
                f"{ctx.args.y_move_timeout_s:.1f}s; continuing.",
                file=sys.stderr,
            )
            break
        time.sleep(ctx.args.period_s)
    return not ctx.stop_state["requested"]


def replay_pose(ctx, pose: dict[str, float]) -> bool:
    """Re-achieve a recorded pose with safe Z ordering and a live crash net."""
    args = ctx.args
    rx_target = pose.get("rx")
    rx_speed_match = None
    if rx_target is not None and ctx.rx_client is not None:
        position = ctx.read_position()
        rx_now = ctx.read_rx_rad()
        speeds = {
            "x": (args.feed_mm_min or 1500.0) / 60.0,
            "y": args.y_pico_vmax_mm_s,
            "z": args.z_pico_vmax_mm_s,
        }
        legs = [
            abs(float(pose[axis]) - float(position[axis])) / max(1e-6, speeds[axis])
            for axis in ("x", "y", "z")
            if pose.get(axis) is not None and position.get(axis) is not None
        ]
        if legs and rx_now is not None:
            leg_time = max(max(legs), 0.2)
            rx_speed_match = max(
                _clamp(
                    abs(float(rx_target) - rx_now) / leg_time,
                    args.rx_speed_rad_s or 0.5,
                ),
                0.05,
            )
    commanded_rx = (
        ctx.set_rx_absolute(
            float(rx_target),
            wait=False,
            speed_rad_s=rx_speed_match,
        )
        if rx_target is not None and ctx.rx_client is not None
        else None
    )
    if ctx.stop_state["requested"]:
        return False
    target = {key: float(pose[key]) for key in ("x", "y", "z") if pose.get(key) is not None}
    if target:
        current_z = (
            ctx.z_stream["cmd"] if ctx.z_stream["cmd"] is not None else ctx.read_axis(ctx.client_z)
        )
        target_z = target.get("z")
        if (
            target_z is not None
            and current_z is not None
            and target_z > current_z + ctx.Z_NOOP_EPS_MM
        ):
            lateral = {key: target[key] for key in ("x", "y") if key in target}
            lateral["z"] = current_z
            if not replay_move_to(ctx, lateral):
                return False
            if not replay_move_to(ctx, {"z": target_z}):
                return False
        elif not replay_move_to(ctx, target):
            return False
    if commanded_rx is not None:
        ctx._wait_for_rx(
            commanded_rx,
            math.radians(args.edge_tilt_max_deg),
        )
    if commanded_rx is not None and "z" in target:
        ctx.z_rx["rad"] = commanded_rx
    settle_z_standoff(ctx, retreat_only=True)
    return not ctx.stop_state["requested"]


def retreat_along_breadcrumb(ctx, why: str) -> bool:
    """Replay the contour-tracked breadcrumb path in reverse."""
    breadcrumb = ctx.breadcrumb
    if not breadcrumb:
        return True
    if len(breadcrumb) == 1:
        seed = breadcrumb[0]
        rx_now = ctx.read_rx_rad()
        if (
            seed.get("rx") is None
            or rx_now is None
            or abs(rx_now - seed["rx"]) < BREADCRUMB_SPACING_RAD
        ):
            return True
    previous_activity = ctx.rec_ctx["activity"]
    ctx.set_activity("backtrack")
    print(
        f"  retreating along the walked path ({len(breadcrumb)} waypoint(s) "
        f"back to the last good pose): {why}."
    )
    ctx.record_sample(
        "backtrack:start",
        target={"waypoints": len(breadcrumb)},
        why=why,
    )
    try:
        for pose in reversed(breadcrumb):
            if ctx.stop_state["requested"]:
                return False
            if not replay_pose(ctx, pose):
                print(
                    "Aborting scan: breadcrumb retreat failed (pose replay).",
                    file=sys.stderr,
                )
                ctx.stop_state["requested"] = True
                return False
    finally:
        ctx.set_activity(previous_activity)
    seed = breadcrumb[0]
    if seed.get("rx") is not None:
        rx_end = ctx.read_rx_rad()
        if rx_end is not None and abs(rx_end - seed["rx"]) > 0.05:
            print(
                f"  warning: retreat finished at rx={rx_end:+.3f}rad but the "
                f"seed pose recorded {seed['rx']:+.3f}rad -- rx restore "
                "failed (RX-axis motor errors?); the traverse arc guard and "
                "--traverse-max-jump-mm cover the residual swing.",
                file=sys.stderr,
            )
    ctx.breadcrumb_reseed(
        seed["x"],
        seed["y"],
        seed["z"],
        seed["rx"],
    )
    return True
