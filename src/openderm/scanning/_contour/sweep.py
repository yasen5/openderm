"""X-band scanning and outward Y-march control."""

from __future__ import annotations

import math
import sys
import time
from types import SimpleNamespace

from . import band_motion as contour_band_motion
from ..config import (
    STATION_EDGE,
    STATION_RX_LIMIT,
    STATION_Y_LIMIT,
    STATION_Z_LIMIT,
)
from ..helpers import _edge_side_from_rx


def _approach_band(
    ctx,
    band_idx: int,
    y_focus: float,
    reached_y: float,
    x_targets: list[float],
) -> tuple[bool, float]:
    if not x_targets:
        return True, reached_y
    origin_y = reached_y
    ok, reached_y = contour_band_motion.band_transition_walk(
        ctx,
        y_focus,
        reached_y,
        x_rails=x_targets,
    )
    if ok:
        return True, reached_y
    if ctx.stop_state["requested"]:
        return False, reached_y
    if ctx.pivot_jump["hit"]:
        retreated = ctx.retreat_along_breadcrumb(
            f"band {band_idx + 1} approach refused at y={reached_y:.1f}mm"
        )
        if retreated:
            reached_y = origin_y
    print(
        f"  band {band_idx + 1}: regulated approach to y={y_focus:.1f}mm "
        f"stopped at y={reached_y:.1f}mm; ending this march.",
        file=sys.stderr,
    )
    return False, reached_y


def _reseed_band_breadcrumb(ctx) -> None:
    position = ctx.read_position()
    rx_current = ctx.read_rx_rad()
    for _ in range(2):
        if rx_current is not None and all(
            position.get(axis) is not None for axis in ("x", "y", "z")
        ):
            break
        time.sleep(0.05)
        if rx_current is None:
            rx_current = ctx.read_rx_rad()
        if any(position.get(axis) is None for axis in ("x", "y", "z")):
            position = ctx.read_position()
    if rx_current is not None and all(position.get(axis) is not None for axis in ("x", "y", "z")):
        ctx.breadcrumb_reseed(
            position["x"],
            position["y"],
            position["z"],
            rx_current,
        )
        return
    ctx.breadcrumb.clear()
    print(
        "  warning: band-start pose unreadable; breadcrumb reseeded empty "
        "(retreats cover only this band's own path).",
        file=sys.stderr,
    )


def _classify_station_edge(ctx, march_dir: int) -> int:
    side = _edge_side_from_rx(
        ctx.read_rx_rad(),
        ctx.args.rx_axis_min_rad,
        ctx.args.rx_axis_max_rad,
    )
    return march_dir if side is None else side


def _try_grid_move(ctx, band_idx: int, x_target: float, y_focus: float) -> bool:
    moved = ctx.move_to_grid_pose(x_target, y_focus)
    if not moved and ctx.pivot_jump["hit"] and not ctx.stop_state["requested"]:
        if ctx.retreat_along_breadcrumb(
            f"pivot jump refused at band {band_idx + 1} x={x_target:.1f}mm"
        ):
            moved = ctx.move_to_grid_pose(x_target, y_focus)
    return moved


def _scan_band_targets(
    ctx,
    state,
    band_idx: int,
    y_focus: float,
    x_targets: list[float],
    phase_label: str,
    march_dir: int,
) -> tuple[int, list[tuple[float, int]], int, list[float]]:
    captured = 0
    attempted = 0
    edged_x: list[tuple[float, int]] = []
    band_rx: list[float] = []
    limited_statuses = {
        STATION_EDGE,
        STATION_Z_LIMIT,
        STATION_RX_LIMIT,
        STATION_Y_LIMIT,
    }
    for x_target in x_targets:
        if ctx.stop_state["requested"]:
            break
        ctx.rec_ctx["station"] = state.global_k + 1
        attempted += 1
        if not _try_grid_move(ctx, band_idx, x_target, y_focus):
            if ctx.y_limit["hit"] or ctx.pivot_jump["hit"]:
                if ctx.args.debug:
                    print(
                        f"  [dbg] band {band_idx + 1} (y={y_focus:.1f}mm): "
                        f"skip x={x_target:.1f}mm (grid move refused)."
                    )
                continue
            print("Aborting scan: grid move failed.", file=sys.stderr)
            ctx.stop_state["requested"] = True
            break

        label = f"station {state.global_k + 1} {phase_label} band {band_idx + 1} x-step {captured}"
        print(f"--- {label} | x={x_target:.2f}mm y={y_focus:.2f}mm ---")
        result = ctx.regulate_station(
            label,
            x_target,
            y_focus,
            march_dir,
            edge_recovery=ctx.args.band_edge_recovery,
            restore_rx_on_edge=True,
        )
        if ctx.stop_state["requested"]:
            break
        if result["status"] in limited_statuses:
            side = _classify_station_edge(ctx, march_dir)
            edged_x.append((x_target, side))
            if ctx.args.debug:
                where = "this march" if side == march_dir else "the reverse march"
                print(
                    f"  [dbg] band {band_idx + 1}: edge at "
                    f"x={x_target:.1f}mm ({result['status']}); pruned for "
                    f"{where} ({'+y' if side > 0 else '-y'} flank)."
                )
            continue
        if not state.capture_station(
            x_target,
            y_focus,
            phase_label,
            captured,
            band_idx,
            result,
        ):
            break
        if state.last_capture_pose.get("rx") is not None:
            band_rx.append(float(state.last_capture_pose["rx"]))
        captured += 1
    return captured, edged_x, attempted, band_rx


def sweep_x_band(
    ctx,
    state,
    band_idx: int,
    y_focus: float,
    x_targets: list[float],
    phase_label: str,
    march_dir: int,
    y_from: float | None = None,
) -> tuple[int, float, list, int, list[float]]:
    """Approach and capture one X band."""
    ctx.rec_ctx["phase"] = phase_label
    ctx.rec_ctx["col"] = band_idx + 1
    ctx.record_sample("sweep:start", target={"band_y": round(y_focus, 4)})
    reached_y = y_from if y_from is not None else y_focus
    approached, reached_y = _approach_band(
        ctx,
        band_idx,
        y_focus,
        reached_y,
        x_targets,
    )
    if not approached:
        return (
            0 if ctx.stop_state["requested"] else -1,
            reached_y,
            [],
            0,
            [],
        )
    _reseed_band_breadcrumb(ctx)
    captured, edged_x, attempted, band_rx = _scan_band_targets(
        ctx,
        state,
        band_idx,
        y_focus,
        x_targets,
        phase_label,
        march_dir,
    )
    return captured, y_focus, edged_x, attempted, band_rx


def _miss_fraction_stops_march(
    ctx,
    name: str,
    march_dir: int,
    captured: int,
    attempted: int,
    band_rx: list[float],
) -> bool:
    threshold = ctx.args.band_miss_stop_frac
    if threshold is None or attempted == 0 or (attempted - captured) / attempted < threshold:
        return False
    missed = attempted - captured
    flank_rx = (max(band_rx) if march_dir < 0 else min(band_rx)) if band_rx else None
    in_flank_regime = flank_rx is not None and (
        flank_rx >= ctx.args.band_miss_stop_rx_hi
        if march_dir < 0
        else flank_rx <= ctx.args.band_miss_stop_rx_lo
    )
    if in_flank_regime:
        print(
            f"  band {name} march: {missed}/{attempted} stations "
            f"({100 * missed / attempted:.0f}%) captured no image >= "
            f"{100 * threshold:.0f}% threshold with the band leveled at "
            f"rx={flank_rx:+.3f}rad (the "
            f"{'-y' if march_dir < 0 else '+y'} flank regime); ending this "
            "y sweep.",
            file=sys.stderr,
        )
        return True
    rx_description = (
        "no captures" if flank_rx is None else f"leveled rx {flank_rx:+.3f}rad (mid-range)"
    )
    print(
        f"  band {name} march: {missed}/{attempted} stations "
        f"({100 * missed / attempted:.0f}%) missed, but {rx_description} -- "
        f"not the {'-y' if march_dir < 0 else '+y'} flank regime; continuing "
        "the march (misses look like along-x dropouts; their X positions are "
        "pruned).",
        file=sys.stderr,
    )
    return False


def _run_march(ctx, state, march, driver) -> tuple[str, int]:
    name, k_sequence = march
    march_dir = -1 if name == "descend" else 1
    edged = driver.edged_by_side[march_dir]
    empty_bands = 0
    ending = "cap"
    for k in k_sequence:
        if ctx.stop_state["requested"]:
            ending = "stopped"
            break
        y_focus = state.y_start + k * driver.band_step
        ordered = driver.x_columns if driver.x_forward else list(reversed(driver.x_columns))
        active = [x for x in ordered if x not in edged]
        if not active:
            ending = "all-edged"
            break
        captured, driver.current_y, new_edges, attempted, band_rx = sweep_x_band(
            ctx,
            state,
            driver.band_index,
            y_focus,
            active,
            "+x" if driver.x_forward else "-x",
            march_dir,
            y_from=driver.current_y,
        )
        driver.band_index += 1
        if captured < 0:
            ending = "approach-refused"
            break
        for x_edge, side in new_edges:
            driver.edged_by_side[side].add(x_edge)
        driver.x_forward = not driver.x_forward
        if _miss_fraction_stops_march(
            ctx,
            name,
            march_dir,
            captured,
            attempted,
            band_rx,
        ):
            ending = "miss-frac"
            break
        empty_bands = empty_bands + 1 if captured == 0 else 0
        if empty_bands >= driver.empty_stop:
            ending = "flank"
            break
    return ending, march_dir


def run_band_sweeps(ctx, state) -> None:
    """March X bands outward from the scan-start Y row."""
    if ctx.stop_state["requested"]:
        return
    args = ctx.args
    band_step = args.band_step_mm if args.band_step_mm > 0 else args.y_step_mm
    x_columns = [state.x_start + column * args.x_step_mm for column in range(state.cols)]
    if not x_columns:
        print("  no X positions to sweep.", file=sys.stderr)
        return

    empty_stop = max(1, args.band_empty_stop)
    max_bands = max(4, int(math.ceil(args.y_max_travel_mm / band_step)))
    print(
        f"  contour sweeps (step {band_step:.1f}mm) across "
        f"{len(x_columns)} X positions, marched outward from the start row "
        f"(y={state.y_start:.1f}mm); each x pruned once it edges, march ends "
        f"after {empty_stop} empty band(s); x = fast axis."
    )
    driver = SimpleNamespace(
        band_step=band_step,
        x_columns=x_columns,
        empty_stop=empty_stop,
        band_index=0,
        x_forward=True,
        current_y=state.y_start,
        edged_by_side={1: set(), -1: set()},
    )
    marches = (
        ("descend", range(0, -max_bands, -1)),
        ("ascend", range(1, max_bands)),
    )
    for march in marches:
        ending, _ = _run_march(ctx, state, march, driver)
        print(
            f"  band {march[0]} march ended ({ending}) at y={driver.current_y:.1f}mm.",
            file=sys.stderr,
        )
        if ctx.stop_state["requested"]:
            break
