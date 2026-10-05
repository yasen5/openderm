"""Per-station standoff and RX regulation."""

from __future__ import annotations

import math
import sys
import time
from types import SimpleNamespace

from capture.motion.rx_axis.server import RxAxisServerError

from . import rx_servo
from ..config import (
    PIVOT_FOLLOW_MARGIN,
    RX_OSC_RECOVERY_FACTOR,
    RX_OSC_REVERSALS_FIRST_CUT,
    STATION_EDGE,
    STATION_RECOVERED,
    STATION_RX_LIMIT,
    STATION_SETTLED,
    STATION_TIMEOUT,
    STATION_Y_LIMIT,
    STATION_Z_LIMIT,
)
from ..helpers import _clamp, _fmt_bound, _pivot_rate_budget


def _read_station_anchor(ctx, label: str, x_now: float, y_now: float):
    """Resolve the fixed focus anchor from the commanded grid or live pose."""
    anchor_pos = ctx.read_position()
    use_grid = (
        ctx.grid_anchor.get("valid")
        and ctx.grid_anchor.get("rx") is not None
        and ctx.grid_anchor.get("fx") is not None
        and abs(float(ctx.grid_anchor["fx"]) - x_now) < 1e-6
        and abs(float(ctx.grid_anchor["fy"]) - y_now) < 1e-6
    )
    if use_grid:
        ctx.pivot_anchor["x"] = float(ctx.grid_anchor["gx"])
        ctx.pivot_anchor["y"] = float(ctx.grid_anchor["gy"])
        anchor_rx = float(ctx.grid_anchor["rx"])
    else:
        anchor_rx = ctx.read_rx_rad()
        for _ in range(3):
            if (
                anchor_rx is not None
                and anchor_pos.get("x") is not None
                and anchor_pos.get("y") is not None
            ):
                break
            if ctx.stop_state["requested"]:
                break
            time.sleep(ctx.args.period_s)
            if anchor_rx is None:
                anchor_rx = ctx.read_rx_rad()
            if anchor_pos.get("x") is None or anchor_pos.get("y") is None:
                anchor_pos = ctx.read_position()
        if anchor_rx is None or anchor_pos.get("x") is None or anchor_pos.get("y") is None:
            print(
                f"  warning: {label} could not read the entry pose/rx after "
                "retries; pivot focus-hold disabled for this station "
                "(x/y held, z standoff only).",
                file=sys.stderr,
            )
        ctx.pivot_anchor["x"] = float(anchor_pos["x"]) if anchor_pos.get("x") is not None else x_now
        ctx.pivot_anchor["y"] = float(anchor_pos["y"]) if anchor_pos.get("y") is not None else y_now
    ctx.pivot_anchor["z"] = float(anchor_pos["z"]) if anchor_pos.get("z") is not None else None
    ctx.pivot_anchor["rx"] = anchor_rx
    return anchor_rx


def _set_station_rx_budget(ctx, anchor_rx: float | None) -> None:
    ctx.station_rx_bounds["min"] = None
    ctx.station_rx_bounds["max"] = None
    budget = ctx.args.rx_station_budget_rad
    if budget is None or anchor_rx is None:
        return
    ctx.station_rx_bounds["min"] = anchor_rx - budget
    ctx.station_rx_bounds["max"] = anchor_rx + budget
    if ctx.args.debug:
        print(f"  [dbg] station rx budget: {anchor_rx:+.4f} +/- {budget:.4f} rad")


def _station_pivot_rate_budget(ctx, rx_current: float | None) -> float | None:
    if (
        ctx.pivot_model is None
        or rx_current is None
        or ctx.pivot_anchor["x"] is None
        or ctx.pivot_anchor["y"] is None
    ):
        return None
    step = 0.02
    anchor_x = ctx.pivot_anchor["x"]
    anchor_y = ctx.pivot_anchor["y"]
    arc = ctx.pivot_model.pivot(
        {"x": anchor_x, "y": anchor_y, "z": 0.0},
        rx_current + step,
        rx_current,
    )
    budget = _pivot_rate_budget(
        (arc.get("x", anchor_x) - anchor_x) / step,
        (arc.get("y", anchor_y) - anchor_y) / step,
        arc.get("z", 0.0) / step,
        (ctx.args.feed_mm_min or 1500.0) / 60.0,
        ctx.args.y_pico_vmax_mm_s,
        ctx.args.z_pico_vmax_mm_s,
        PIVOT_FOLLOW_MARGIN,
    )
    if ctx.args.debug and budget is not None:
        print(
            f"  [dbg] pivot-follow rate budget: {budget:.3f} rad/s "
            f"(rx cap {ctx.args.rx_speed_rad_s or 0.25})"
        )
    return budget


def _new_station_state(
    ctx,
    label: str,
    x_now: float,
    y_now: float,
    phase_dir: int,
) -> SimpleNamespace:
    ctx.set_activity("regulate")
    ctx.record_sample(
        "station:start",
        target={"x": round(x_now, 4), "y": round(y_now, 4)},
    )
    anchor_rx = _read_station_anchor(ctx, label, x_now, y_now)
    _set_station_rx_budget(ctx, anchor_rx)
    ctx.pico_stream_prev.clear()
    gain_scale = min(
        1.0,
        ctx.vel_scale_carry["scale"] * RX_OSC_RECOVERY_FACTOR,
    )
    ctx.vel_scale_carry["scale"] = gain_scale
    now = time.monotonic()
    return SimpleNamespace(
        label=label,
        x_now=x_now,
        y_now=y_now,
        phase_dir=phase_dir,
        rx_cur=anchor_rx,
        rx_error_filtered=None,
        on_target_count=0,
        oor_count=0,
        rx_limit_warned=False,
        vel_prev_rx_meas=None,
        vel_integrator=0.0,
        vel_last_t=None,
        vel_prev_abs_err=None,
        vel_w_prev=0.0,
        vel_armed=True,
        vel_rate_budget=_station_pivot_rate_budget(ctx, anchor_rx),
        z_stall=0,
        rx_axis_count=0,
        rx_collision_count=0,
        y_limit_count=0,
        vel_gain_scale=gain_scale,
        vel_err_sign=0,
        vel_reversals=0,
        vel_bumped=False,
        vel_next_cut=RX_OSC_REVERSALS_FIRST_CUT,
        vel_frozen=False,
        iterations_since_report=0,
        last_report_time=now,
        deadline=now + ctx.args.station_timeout_s,
    )


def _read_sensor_sample(ctx) -> SimpleNamespace:
    sensor1 = ctx.read_exclusive("sensor1")
    sensor2 = ctx.read_exclusive("sensor2")
    in_range = [
        reading
        for reading in (sensor1, sensor2)
        if reading.in_range and reading.distance_mm is not None
    ]
    both_in_range = (
        sensor1.in_range
        and sensor1.distance_mm is not None
        and sensor2.in_range
        and sensor2.distance_mm is not None
    )
    return SimpleNamespace(
        sensor1=sensor1,
        sensor2=sensor2,
        in_range=in_range,
        both_in_range=both_in_range,
        d1="--" if sensor1.distance_mm is None else f"{sensor1.distance_mm:.2f}",
        d2="--" if sensor2.distance_mm is None else f"{sensor2.distance_mm:.2f}",
        rx_measured=ctx.read_rx_rad(),
    )


def _stop_rx_velocity(ctx) -> None:
    if ctx.rx_client is None:
        return
    try:
        ctx.rx_client.set_velocity(0.0)
    except RxAxisServerError:
        pass


def _restore_probe_orientation(ctx) -> None:
    if ctx.z_rx["rad"] is None:
        return
    current_rx = ctx.read_rx_rad()
    if current_rx is None or abs(current_rx - ctx.z_rx["rad"]) > 1e-3:
        ctx.set_rx_absolute(ctx.z_rx["rad"])


def _probe_edge(ctx, state, sample, restore_rx_on_edge: bool) -> str:
    level_ref = ctx.level_anchor["rad"]
    if level_ref is None:
        level_ref = ctx.rx_start if ctx.rx_start is not None else ctx.read_rx_rad()
    if level_ref is None:
        print(
            f"  warning: {state.label} out of range but rx unreadable; "
            "cannot run the edge probe -- capturing anyway.",
            file=sys.stderr,
        )
        return STATION_TIMEOUT
    print(
        f"  {state.label}: sensors out of range "
        f"(s1={sample.sensor1.signal_status}, "
        f"s2={sample.sensor2.signal_status}); probing edge from the current "
        f"rx, reach band {level_ref:+.3f}rad +/- "
        f"{ctx.args.edge_tilt_max_deg:.0f}deg (last-leveled anchor)."
    )
    ctx.set_activity("edge_recover")
    ctx.record_sample("edge:start", target={"level_ref": round(level_ref, 4)})
    recovered, rx_used, limited = ctx.attempt_edge_recovery(
        level_ref,
        state.phase_dir,
    )
    ctx.set_activity("regulate")
    if ctx.stop_state["requested"]:
        return STATION_TIMEOUT
    if recovered:
        print(
            f"  {state.label}: recovered at rx={rx_used:+.3f}rad "
            f"({math.degrees(rx_used - level_ref):+.1f}deg tilt); "
            "still on the body."
        )
        ctx.settle_z_standoff()
        ctx.z_rx["rad"] = rx_used
        return STATION_RECOVERED
    if limited:
        print(
            f"  {state.label}: edge recorded, but the RX recovery was LIMITED "
            "by an rx bound/rejection (NOT a confirmed surface edge) -- widen "
            "--rx-min/max-rad or check the RX-axis motor range if this X "
            "position should extend further.",
            file=sys.stderr,
        )
    else:
        print(
            f"  {state.label}: edge of body (no recovery within the "
            f"{ctx.args.edge_tilt_max_deg:.0f}deg cap)."
        )
    if restore_rx_on_edge:
        _restore_probe_orientation(ctx)
    return STATION_EDGE


def _handle_out_of_range(
    ctx,
    state,
    sample,
    edge_recovery: bool,
    restore_rx_on_edge: bool,
) -> str | None:
    state.vel_integrator = 0.0
    _stop_rx_velocity(ctx)
    state.oor_count += 1
    state.rx_error_filtered = None
    state.z_stall = 0
    if state.oor_count >= ctx.args.edge_oor_iters:
        if not edge_recovery:
            if ctx.args.debug:
                print(
                    f"  [dbg] {state.label}: out of range with edge-recovery "
                    "disabled (x-band); reporting edge without an rx probe."
                )
            return STATION_EDGE
        return _probe_edge(ctx, state, sample, restore_rx_on_edge)
    if ctx.args.debug:
        print(
            f"  [dbg] {state.label} not both in range "
            f"(s1={sample.sensor1.signal_status}, "
            f"s2={sample.sensor2.signal_status}); holding z "
            f"({state.oor_count}/{ctx.args.edge_oor_iters} before edge probe)"
        )
    return _timeout_status(ctx, state)


def _standoff_command(ctx, state, sample) -> tuple[bool, float, float]:
    average = sum(reading.distance_mm for reading in sample.in_range) / len(sample.in_range)
    error = average - ctx.args.target_mm
    if abs(error) <= ctx.args.deadband_mm:
        state.z_stall = max(0, state.z_stall - 1)
        if ctx.args.debug:
            used = "+".join(reading.name for reading in sample.in_range)
            print(
                f"  [dbg] d1={sample.d1} d2={sample.d2} avg={average:.2f} "
                f"({used}) err={error:+.3f}mm <= deadband, holding z"
            )
        return True, 0.0, error
    delta = _clamp(
        ctx.args.gain_mm_per_mm * error,
        ctx.args.max_step_mm,
    )
    if ctx.args.debug:
        used = "+".join(reading.name for reading in sample.in_range)
        print(
            f"  [dbg] d1={sample.d1} d2={sample.d2} avg={average:.2f} "
            f"({used}) err={error:+.3f}mm -> dz_standoff={delta:+.3f}mm"
        )
    return False, delta, error


def _dispatch_regulated_move(
    ctx,
    state,
    sample,
    rx_new: float | None,
    standoff_delta: float,
) -> tuple[str | None, tuple[bool, bool, float | None] | None]:
    arc_z_increment = 0.0
    if ctx.pivot_model is not None and state.rx_cur is not None and rx_new is not None:
        arc_z_increment = ctx.pivot_model.pivot(
            {"x": 0.0, "y": 0.0, "z": 0.0},
            rx_new,
            state.rx_cur,
        )["z"]
    result = ctx.move_regulated_pose(
        None,
        standoff_delta + arc_z_increment,
        wait=False,
        rx_meas=sample.rx_measured,
        rx_arc_rad=rx_new,
    )
    if not ctx.y_limit["hit"]:
        state.y_limit_count = 0
        if rx_new is not None:
            state.rx_cur = rx_new
        return None, result

    state.y_limit_count += 1
    if state.y_limit_count >= ctx.args.edge_oor_iters:
        print(
            f"  {state.label}: self-collision guard refused the move for "
            f"{state.y_limit_count} iters -- collision-limited edge.",
            file=sys.stderr,
        )
        return STATION_Y_LIMIT, None
    if ctx.args.debug:
        print(
            f"  [dbg] collision-limited "
            f"({state.y_limit_count}/{ctx.args.edge_oor_iters} before edge)"
        )
    return None, None


def _check_rx_position_limit(ctx, state) -> str | None:
    if not (ctx.rx_limit["clamped"] and ctx.rx_limit["axis"]):
        return None
    state.rx_axis_count += 1
    if state.rx_axis_count < ctx.args.edge_oor_iters:
        return None
    print(
        f"  {state.label}: rx rejected at its axis range for "
        f"{state.rx_axis_count} iters -- reach-limited (rx) edge.",
        file=sys.stderr,
    )
    _stop_rx_velocity(ctx)
    return STATION_RX_LIMIT


def _check_z_limit(
    ctx,
    state,
    z_on_target: bool,
    error: float,
    move_result: tuple[bool, bool, float | None],
) -> str | None:
    hit_high, hit_low, measured_z = move_result
    pushing_bound = not z_on_target and ((hit_high and error > 0) or (hit_low and error < 0))
    if not pushing_bound:
        state.z_stall = max(0, state.z_stall - 1)
        return None
    bound = ctx.args.z_max_mm if hit_high else ctx.args.z_min_mm
    if measured_z is None or bound is None or abs(measured_z - bound) <= 3.0:
        state.z_stall += 1
    if state.z_stall < ctx.args.z_stall_iters:
        return None
    print(
        f"  {state.label}: z at the travel window "
        f"[{_fmt_bound(ctx.args.z_min_mm)}, "
        f"{_fmt_bound(ctx.args.z_max_mm)}]mm with standoff "
        f"err={error:+.2f}mm -- can't reach standoff; reach-limited (z) edge.",
        file=sys.stderr,
    )
    _stop_rx_velocity(ctx)
    return STATION_Z_LIMIT


def _park_rx_for_capture(ctx, state) -> None:
    if ctx.rx_client is not None:
        try:
            ctx.rx_client.set_velocity(0.0)
            hold_rx = ctx.read_rx_rad()
            if hold_rx is not None:
                ctx.rx_client.move_to(
                    hold_rx,
                    speed_rad_s=ctx.args.rx_speed_rad_s,
                    accel_rad_s2=ctx.args.rx_accel_rad_s2,
                )
        except RxAxisServerError as exc:
            print(
                f"  warning: rx park after velocity servo failed: {exc}",
                file=sys.stderr,
            )
    if state.rx_cur is not None:
        ctx.level_anchor["rad"] = float(state.rx_cur)


def _settled_status(
    ctx,
    state,
    z_on_target: bool,
    rx_on_target: bool,
) -> str | None:
    if z_on_target and rx_on_target:
        state.on_target_count += 1
    else:
        state.on_target_count = 0
    if state.on_target_count < ctx.args.settle_iters:
        return None
    _park_rx_for_capture(ctx, state)
    return STATION_SETTLED


def _timeout_status(ctx, state) -> str | None:
    if time.monotonic() < state.deadline:
        return None
    print(
        f"  warning: {state.label} did not settle within "
        f"{ctx.args.station_timeout_s:.1f}s; recording anyway.",
        file=sys.stderr,
    )
    return STATION_TIMEOUT


def _report_control_rate(ctx, state) -> str | None:
    state.iterations_since_report += 1
    now = time.monotonic()
    elapsed = now - state.last_report_time
    if elapsed >= ctx.args.report_interval_s:
        frequency = state.iterations_since_report / elapsed
        print(
            f"  {state.label} x={state.x_now:.2f}mm y={state.y_now:.2f}mm "
            f"control loop: {frequency:.2f} Hz "
            f"({state.iterations_since_report} iters in {elapsed:.2f}s)"
        )
        state.iterations_since_report = 0
        state.last_report_time = now
    return _timeout_status(ctx, state)


def _regulate_in_range(ctx, state, sample) -> str | None:
    state.oor_count = 0
    ctx.set_activity("settle")
    raw_rx_error = sample.sensor1.distance_mm - sample.sensor2.distance_mm
    rx_on_target = True
    rx_new = state.rx_cur
    if ctx.rx_client is not None:
        status, rx_on_target, rx_new = rx_servo.update_rx_servo(
            ctx,
            state,
            raw_rx_error,
            sample.rx_measured,
        )
        if status is not None:
            return status

    z_on_target, standoff_delta, error = _standoff_command(ctx, state, sample)
    status, move_result = _dispatch_regulated_move(
        ctx,
        state,
        sample,
        rx_new,
        standoff_delta,
    )
    if status is not None or move_result is None:
        return status
    status = _check_rx_position_limit(ctx, state)
    if status is not None:
        return status
    status = _check_z_limit(ctx, state, z_on_target, error, move_result)
    if status is not None:
        return status
    status = _settled_status(ctx, state, z_on_target, rx_on_target)
    return status if status is not None else _report_control_rate(ctx, state)


def regulate_station(
    ctx,
    label: str,
    x_now: float,
    y_now: float,
    phase_dir: int,
    edge_recovery: bool = True,
    restore_rx_on_edge: bool = False,
) -> dict:
    """Regulate standoff and tilt at one station and classify the result."""
    state = _new_station_state(ctx, label, x_now, y_now, phase_dir)
    while not ctx.stop_state["requested"]:
        sample = _read_sensor_sample(ctx)
        if sample.both_in_range:
            status = _regulate_in_range(ctx, state, sample)
        else:
            status = _handle_out_of_range(
                ctx,
                state,
                sample,
                edge_recovery,
                restore_rx_on_edge,
            )
        if status is not None:
            return {"status": status}
        time.sleep(ctx.args.period_s)
    return {"status": STATION_TIMEOUT}
