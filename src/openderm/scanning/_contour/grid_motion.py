"""Pivot-compensated station-to-station grid motion."""

from __future__ import annotations

import sys
import time

from ..config import TRAVERSE_ARC_RISE_MIN_MM
from ..helpers import _traverse_arc_delta


def _trusted_z(ctx) -> float | None:
    if ctx.z_stream["cmd"] is not None:
        return float(ctx.z_stream["cmd"])
    return ctx.z_last["mm"]


def _read_pivot_pose(ctx) -> tuple[float | None, float | None]:
    rx_current = ctx.read_rx_rad()
    z_current = _trusted_z(ctx)
    for _ in range(2):
        if rx_current is not None and z_current is not None:
            break
        time.sleep(0.05)
        if rx_current is None:
            rx_current = ctx.read_rx_rad()
        if z_current is None:
            z_current = _trusted_z(ctx)
            if z_current is None:
                z_current = ctx.read_position().get("z")
    return rx_current, z_current


def _guard_pivot_arc(
    ctx,
    rx_current: float,
    z_current: float,
) -> tuple[bool, float]:
    if ctx.z_rx["rad"] is None:
        return True, z_current
    dz_arc, lateral = _traverse_arc_delta(
        ctx.pivot_model,
        {"x": 0.0, "y": 0.0, "z": float(z_current)},
        rx_current,
        ctx.z_rx["rad"],
    )
    ctx.record_sample(
        "traverse:arc",
        target={"dz_arc": round(dz_arc, 2), "lateral": round(lateral, 2)},
    )
    if lateral > ctx.args.traverse_max_jump_mm:
        ctx.pivot_jump["hit"] = True
        print(
            f"  grid move refused: the carried-rx swing "
            f"({ctx.z_rx['rad']:+.3f}->{rx_current:+.3f}rad) demands a "
            f"{lateral:.0f}mm lateral pivot jump (> --traverse-max-jump-mm "
            f"{ctx.args.traverse_max_jump_mm:.0f}mm) -- a computed shortcut "
            "that far can cross the subject regardless of z; retreat along "
            "the walked path instead.",
            file=sys.stderr,
        )
        return False, z_current
    if dz_arc >= -TRAVERSE_ARC_RISE_MIN_MM:
        return True, z_current
    if ctx.args.debug:
        print(
            f"  [dbg] traverse owes arc z {dz_arc:+.1f}mm for the rx swing "
            f"{ctx.z_rx['rad']:+.3f}->{rx_current:+.3f}rad "
            f"(lateral {lateral:.1f}mm); rising first."
        )
    if ctx._traverse_arc_rise(z_current + dz_arc):
        ctx.z_rx["rad"] = rx_current
        z_current += dz_arc
    return True, z_current


def _resolve_grid_target(
    ctx,
    x_grid: float,
    y_grid: float,
) -> tuple[bool, float, float, float | None, float | None]:
    gx, gy = x_grid, y_grid
    rx_current: float | None = None
    z_current: float | None = None
    if ctx.pivot_model is None or ctx.rx_start is None:
        return True, gx, gy, rx_current, z_current

    rx_current, z_current = _read_pivot_pose(ctx)
    if rx_current is None or z_current is None:
        ctx.pivot_jump["hit"] = True
        print(
            "  grid move refused: rx/z unreadable while pivot compensation is "
            "active (a raw-grid fallback could snap the full carried offset "
            "laterally at a stale z); ending this X position.",
            file=sys.stderr,
        )
        return False, gx, gy, rx_current, z_current

    target = ctx.pivot_model.pivot(
        {"x": x_grid, "y": y_grid, "z": float(z_current)},
        rx_current,
        ctx.rx_start,
    )
    gx, gy = target["x"], target["y"]
    ctx.grid_anchor.update(
        fx=x_grid,
        fy=y_grid,
        gx=gx,
        gy=gy,
        rx=rx_current,
        valid=True,
    )
    if ctx.args.debug:
        print(
            f"  [dbg] grid move to focus ({x_grid:.2f}, {y_grid:.2f}) "
            f"at rx={rx_current:+.3f}rad -> gantry ({gx:.2f}, {gy:.2f}) "
            f"(carry-orientation offset dy={gy - y_grid:+.2f}mm)"
        )
    arc_ok, z_current = _guard_pivot_arc(ctx, rx_current, z_current)
    return arc_ok, gx, gy, rx_current, z_current


def _collision_allows(
    ctx,
    gx: float,
    gy: float,
    rx_current: float | None,
    z_current: float | None,
) -> bool:
    if ctx.collision_guard is None:
        return True
    rx_guard = rx_current if rx_current is not None else ctx.read_rx_rad()
    z_guard = z_current if z_current is not None else _trusted_z(ctx)
    if z_guard is None:
        z_guard = ctx.read_position().get("z")
    if rx_guard is None or z_guard is None:
        return True
    clearance = ctx.collision_guard.min_clearance(gx, gy, z_guard, rx_guard)
    if clearance >= ctx.collision_guard.margin:
        return True
    ctx.y_limit["hit"] = True
    print(
        f"  grid move to ({gx:.0f},{gy:.0f}) refused by self-collision "
        f"guard (clearance {clearance:.0f}mm)",
        file=sys.stderr,
    )
    return False


def _move_axis_if_needed(ctx, axis: str, target: float) -> bool:
    client = ctx.client_x if axis == "x" else ctx.client_y
    tolerance = ctx.args.x_tolerance_mm if axis == "x" else ctx.args.y_tolerance_mm
    current = ctx.read_axis(client)
    if current is not None and abs(current - target) <= tolerance:
        return True
    mover = ctx.move_x_to if axis == "x" else ctx.move_y_to
    return mover(target)


def move_to_grid_pose(ctx, x_grid: float, y_grid: float) -> bool:
    """Move to a focus-grid point while carrying the current RX orientation."""
    ctx.y_limit["hit"] = False
    ctx.pivot_jump["hit"] = False
    ctx.set_activity("traverse")
    ctx.record_sample(
        "cmd:grid",
        target={"x": round(x_grid, 4), "y": round(y_grid, 4)},
    )
    ctx.grid_anchor["valid"] = False

    ok, gx, gy, rx_current, z_current = _resolve_grid_target(
        ctx,
        x_grid,
        y_grid,
    )
    if not ok or not _collision_allows(ctx, gx, gy, rx_current, z_current):
        return False
    moved = _move_axis_if_needed(ctx, "x", gx) and _move_axis_if_needed(
        ctx,
        "y",
        gy,
    )
    if moved:
        z_breadcrumb = ctx.z_stream["cmd"] if ctx.z_stream["cmd"] is not None else ctx.z_last["mm"]
        ctx.breadcrumb_push(
            gx,
            gy,
            z_breadcrumb,
            ctx.grid_anchor["rx"] if ctx.grid_anchor.get("valid") else ctx.rx_last["rad"],
        )
    return moved
