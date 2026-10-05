"""Regulated motion between contour scan bands."""

from __future__ import annotations

import sys
import time

from . import band_cruise


def _band_walk_cruise(ctx, y_here: float, y_dest: float) -> tuple[str, float]:
    """Continuously regulate the head while walking to another focus row."""
    return band_cruise.band_walk_cruise(ctx, y_here, y_dest)


def band_transition_walk(
    ctx,
    y_dest: float,
    y_fallback: float,
    x_rails: list[float] | None = None,
) -> tuple[bool, float]:
    """Reach a Y row over the first X rail that retains a safe contour path."""
    cruise = ctx._band_walk_cruise
    args = ctx.args
    reached = y_fallback
    ctx.record_sample("band_walk:start", target={"y_dest": round(y_dest, 4)})
    previous_activity = ctx.rec_ctx["activity"]
    ctx.set_activity("settle")
    try:
        focus_pose = _focus_pose(ctx)
        if focus_pose is None:
            return _unreadable_pose(reached)
        rails = _candidate_rails(
            focus_pose[0],
            x_rails or [],
            args.x_tolerance_mm,
        )
        first = True
        for rail_index, rail in enumerate(rails):
            if ctx.stop_state["requested"]:
                return False, reached
            focus_pose = _focus_pose(ctx)
            if focus_pose is None:
                return _unreadable_pose(reached)
            if not _align_rail(ctx, rail, focus_pose):
                if ctx.stop_state["requested"] or ctx.pivot_jump["hit"]:
                    return False, reached
                continue

            state, _ = cruise(focus_pose[1], focus_pose[1])
            if state == "stop":
                return False, reached
            if state == "dark" and first:
                if not ctx.retreat_along_breadcrumb(
                    "band transition from an off-contour pose (sensors dark)"
                ):
                    return False, reached
                state, _ = cruise(focus_pose[1], focus_pose[1])
                if state == "stop":
                    return False, reached
            first = False
            if state == "collide":
                _retreat_after_collision(
                    ctx,
                    rail,
                    rail_index,
                    len(rails),
                    reached=None,
                )
                continue

            focus_pose = _focus_pose(ctx)
            if focus_pose is None:
                return _unreadable_pose(reached)
            reached = focus_pose[1]
            state, reached = cruise(reached, y_dest)
            if state == "stop":
                return False, reached
            if state == "collide":
                _retreat_after_collision(
                    ctx,
                    rail,
                    rail_index,
                    len(rails),
                    reached=reached,
                )
                continue
            if state == "lost":
                _retreat_after_lost_contour(
                    ctx,
                    rail,
                    rail_index,
                    len(rails),
                    reached,
                )
                continue
            ctx.record_sample(
                "band_walk:done",
                target={"y": round(reached, 4)},
            )
            return True, reached
        print(
            f"  band transition: no active X rail offered a contour path to y={y_dest:.1f}mm.",
            file=sys.stderr,
        )
        return False, reached
    finally:
        ctx.set_activity(previous_activity)


def _focus_pose(ctx) -> tuple[float, float] | None:
    """Derive the live camera focus from gantry pose and pivot offset."""
    rx_current = ctx.read_rx_rad()
    position = ctx.read_position()
    for _ in range(2):
        if (
            (rx_current is not None or ctx.pivot_model is None)
            and position.get("x") is not None
            and position.get("y") is not None
        ):
            break
        time.sleep(0.05)
        if rx_current is None:
            rx_current = ctx.read_rx_rad()
        if position.get("x") is None or position.get("y") is None:
            position = ctx.read_position()
    if (
        (rx_current is None and ctx.pivot_model is not None)
        or position.get("x") is None
        or position.get("y") is None
    ):
        return None
    offset_x = offset_y = 0.0
    if ctx.pivot_model is not None and ctx.rx_start is not None:
        offset = ctx.pivot_model.pivot(
            {"x": 0.0, "y": 0.0, "z": 0.0},
            rx_current,
            ctx.rx_start,
        )
        offset_x = offset.get("x", 0.0)
        offset_y = offset.get("y", 0.0)
    return (
        float(position["x"]) - offset_x,
        float(position["y"]) - offset_y,
    )


def _candidate_rails(
    current_x: float,
    requested: list[float],
    tolerance: float,
) -> list[float]:
    rails = [current_x]
    for candidate in requested:
        if all(abs(candidate - rail) > tolerance for rail in rails):
            rails.append(candidate)
    return rails


def _align_rail(ctx, rail: float, focus_pose: tuple[float, float]) -> bool:
    if abs(focus_pose[0] - rail) <= ctx.args.x_tolerance_mm:
        return True
    if ctx.move_to_grid_pose(rail, focus_pose[1]):
        return True
    if ctx.stop_state["requested"] or ctx.pivot_jump["hit"]:
        return False
    ctx.retreat_along_breadcrumb(f"band walk rail x={rail:.1f}mm align refused")
    return False


def _unreadable_pose(reached: float) -> tuple[bool, float]:
    print(
        "  band transition refused: pose/rx unreadable (cannot derive the head's true focus row).",
        file=sys.stderr,
    )
    return False, reached


def _more_rails_suffix(index: int, count: int) -> str:
    return "; trying the next active rail." if index < count - 1 else "."


def _retreat_after_collision(
    ctx,
    rail: float,
    rail_index: int,
    rail_count: int,
    reached: float | None,
) -> None:
    if reached is None:
        ctx.retreat_along_breadcrumb("band walk establish collision-refused")
        print(
            "  band transition: collision guard refused the establish on "
            f"the x={rail:.1f}mm rail; retreated along the walked path"
            + _more_rails_suffix(rail_index, rail_count),
            file=sys.stderr,
        )
        return
    ctx.retreat_along_breadcrumb(
        f"band walk collision-refused at y={reached:.1f}mm on the x={rail:.1f}mm rail"
    )
    print(
        "  band transition: collision guard refused the cruise at "
        f"y={reached:.1f}mm on the x={rail:.1f}mm rail; retreated along "
        "the walked path" + _more_rails_suffix(rail_index, rail_count),
        file=sys.stderr,
    )


def _retreat_after_lost_contour(
    ctx,
    rail: float,
    rail_index: int,
    rail_count: int,
    reached: float,
) -> None:
    ctx.retreat_along_breadcrumb(
        f"band walk lost the contour at y={reached:.1f}mm on the x={rail:.1f}mm rail"
    )
    print(
        f"  band transition: contour lost at y={reached:.1f}mm on the "
        f"x={rail:.1f}mm rail; retreated along the walked path"
        + _more_rails_suffix(rail_index, rail_count),
        file=sys.stderr,
    )
