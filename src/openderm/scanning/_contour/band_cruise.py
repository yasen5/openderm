"""Continuous regulated cruise between contour scan bands."""

from __future__ import annotations

import math
import sys
import time
from types import SimpleNamespace

from openderm.motion.rx_axis.server import RxAxisServerError

from ..config import (
    BAND_WALK_MAX_DARK_HOPS,
    PICO_CRUISE_MIN_STEP_MM,
    PIVOT_FOLLOW_MARGIN,
    RX_VELOCITY_REARM_FACTOR,
)
from ..helpers import _clamp, _pivot_rate_budget


def _read_anchor(ctx):
    position = ctx.read_position()
    rx_anchor = ctx.read_rx_rad()
    for _ in range(2):
        if (
            (rx_anchor is not None or ctx.pivot_model is None)
            and position.get("x") is not None
            and position.get("y") is not None
        ):
            break
        time.sleep(ctx.args.period_s)
        if rx_anchor is None:
            rx_anchor = ctx.read_rx_rad()
        if position.get("x") is None or position.get("y") is None:
            position = ctx.read_position()
    if (
        (rx_anchor is None and ctx.pivot_model is not None)
        or position.get("x") is None
        or position.get("y") is None
    ):
        return None, None
    ctx.pivot_anchor.update(
        x=float(position["x"]),
        y=float(position["y"]),
        z=position.get("z"),
        rx=rx_anchor,
    )
    ctx.station_rx_bounds["min"] = None
    ctx.station_rx_bounds["max"] = None
    return position, rx_anchor


def _new_walk_state(ctx, y_here: float, y_dest: float):
    position, rx_anchor = _read_anchor(ctx)
    if position is None:
        return None
    speed = max(ctx.args.band_walk_speed_mm_s, 1.0)
    return SimpleNamespace(
        y_here=y_here,
        y_dest=y_dest,
        speed=speed,
        rx_cur=rx_anchor,
        rx_prev_meas=rx_anchor,
        last_ok_rx=rx_anchor,
        rx_filt=None,
        z_tolerance=max(ctx.args.deadband_mm * 2.0, 2.0),
        rx_tolerance=max(ctx.args.rx_deadband_mm * 2.0, 1.5),
        z_pin_dir=0,
        advance_cap=max(ctx.args.max_step_mm, PICO_CRUISE_MIN_STEP_MM),
        blind_allowance=ctx.args.y_step_mm * BAND_WALK_MAX_DARK_HOPS,
        velocity_previous=0.0,
        velocity_armed=True,
        level_frozen=False,
        restore_tried=False,
        direction=1.0 if y_dest >= y_here else -1.0,
        carrot=y_here,
        last_ok=y_here,
        reached=y_here,
        dark_hold=0,
        collision_count=0,
        settled_count=0,
        lag_hold=0,
        lag_cap=max(ctx.args.edge_oor_iters * 3, 9),
        was_moving=False,
        last_in_range=False,
        last_time=None,
        deadline=time.monotonic() + abs(y_dest - y_here) / speed * 3.0 + 6.0,
    )


def _set_velocity(ctx, state, velocity: float) -> None:
    try:
        ctx.rx_client.set_velocity(velocity)
    except RxAxisServerError:
        pass
    state.velocity_previous = velocity


def _arc_rate_budget(ctx, state, rx_at: float) -> float | None:
    if ctx.pivot_model is None:
        return None
    step = 0.02
    arc = ctx.pivot_model.pivot(
        {"x": 0.0, "y": 0.0, "z": 0.0},
        rx_at + step,
        rx_at,
    )
    y_speed = ctx.args.y_pico_vmax_mm_s
    return _pivot_rate_budget(
        arc.get("x", 0.0) / step,
        arc.get("y", 0.0) / step,
        arc.get("z", 0.0) / step,
        (ctx.args.feed_mm_min or 1500.0) / 60.0,
        max(y_speed - state.speed, y_speed * 0.25),
        ctx.args.z_pico_vmax_mm_s,
        PIVOT_FOLLOW_MARGIN,
    )


def _sample_surface(ctx, state):
    sensor1 = ctx.read_exclusive("sensor1")
    sensor2 = ctx.read_exclusive("sensor2")
    rx_measured = ctx.read_rx_rad()
    now = time.monotonic()
    dt = min(now - state.last_time, 0.4) if state.last_time is not None else ctx.args.period_s
    state.last_time = now
    too_near = any(reading.signal_status == "above_range" for reading in (sensor1, sensor2))
    in_range = [
        reading
        for reading in (sensor1, sensor2)
        if reading.in_range and reading.distance_mm is not None
    ]
    error = (
        sum(reading.distance_mm for reading in in_range) / len(in_range) - ctx.args.target_mm
        if in_range
        else None
    )
    state.last_in_range = bool(in_range)
    if in_range:
        state.dark_hold = 0
        if abs(error) <= 2.0 * state.z_tolerance or state.z_pin_dir != 0:
            state.last_ok = state.carrot
            if rx_measured is not None:
                state.last_ok_rx = rx_measured
    if state.level_frozen and abs(state.carrot - state.last_ok) > 1e-6:
        state.level_frozen = False
    level_active = not too_near and not state.level_frozen and len(in_range) == 2
    if level_active:
        raw_error = sensor1.distance_mm - sensor2.distance_mm
        alpha = ctx.args.rx_filter_alpha
        state.rx_filt = (
            raw_error
            if state.rx_filt is None
            else alpha * raw_error + (1.0 - alpha) * state.rx_filt
        )
    return SimpleNamespace(
        rx_measured=rx_measured,
        dt=dt,
        too_near=too_near,
        error=error,
        level_active=level_active,
    )


def _target_rx_velocity(ctx, state, sample) -> tuple[float, bool]:
    if sample.level_active and state.rx_filt is not None:
        if state.velocity_armed:
            if abs(state.rx_filt) <= state.rx_tolerance:
                state.velocity_armed = False
        elif abs(state.rx_filt) > state.rx_tolerance * RX_VELOCITY_REARM_FACTOR:
            state.velocity_armed = True
    target = 0.0
    pinned = False
    if (
        sample.level_active
        and state.rx_filt is not None
        and state.velocity_armed
        and abs(state.rx_filt) > state.rx_tolerance
        and sample.rx_measured is not None
    ):
        cap = ctx.args.rx_speed_rad_s
        budget = _arc_rate_budget(ctx, state, sample.rx_measured)
        if budget is not None:
            cap = max(
                min(cap, budget),
                ctx.args.rx_velocity_min_rad_s * 1.05,
            )
        target = _clamp(ctx.args.rx_gain_rad_per_mm * state.rx_filt, cap)
        if target != 0.0 and abs(target) < ctx.args.rx_velocity_min_rad_s:
            target = math.copysign(ctx.args.rx_velocity_min_rad_s, target)
        _, hit_bound, _ = ctx.clamp_rx(
            sample.rx_measured,
            math.copysign(1e-3, target),
        )
        if hit_bound:
            target = 0.0
            pinned = True
    if target != 0.0 and ctx.rx_vel_collision_blocked(
        sample.rx_measured,
        target,
    ):
        target = 0.0
        pinned = True
    if ctx.args.rx_velocity_slew_rad_s2 > 0 and sample.dt > 0:
        max_delta = ctx.args.rx_velocity_slew_rad_s2 * sample.dt
        target = min(
            max(
                target,
                state.velocity_previous - max_delta,
            ),
            state.velocity_previous + max_delta,
        )
    return target, pinned


def _update_rx(ctx, state, sample) -> tuple[bool, bool, float]:
    target, pinned = _target_rx_velocity(ctx, state, sample)
    if abs(target) > 1e-9 or abs(state.velocity_previous) > 1e-9:
        _set_velocity(ctx, state, target)
    arc_z = 0.0
    rx_moving = False
    if sample.rx_measured is not None:
        if ctx.pivot_model is not None and state.rx_prev_meas is not None:
            arc_z = ctx.pivot_model.pivot(
                {"x": 0.0, "y": 0.0, "z": 0.0},
                sample.rx_measured,
                state.rx_prev_meas,
            )["z"]
        rx_moving = abs(state.velocity_previous) > 1e-9 or (
            state.rx_prev_meas is not None and abs(sample.rx_measured - state.rx_prev_meas) > 0.003
        )
        state.rx_prev_meas = sample.rx_measured
        state.rx_cur = sample.rx_measured
    settled = not rx_moving and (
        not sample.level_active
        or state.rx_filt is None
        or not state.velocity_armed
        or abs(state.rx_filt) <= state.rx_tolerance
        or pinned
    )
    return rx_moving, settled, arc_z


def _z_delta(ctx, state, sample) -> float:
    if sample.too_near:
        return -abs(ctx.args.max_step_mm)
    if sample.error is None or abs(sample.error) <= state.z_tolerance:
        return 0.0
    delta = _clamp(
        ctx.args.gain_mm_per_mm * sample.error,
        ctx.args.max_step_mm,
    )
    if state.z_pin_dir != 0 and math.copysign(1.0, delta) == state.z_pin_dir:
        return 0.0
    state.z_pin_dir = 0
    if abs(delta) < PICO_CRUISE_MIN_STEP_MM:
        delta = math.copysign(PICO_CRUISE_MIN_STEP_MM, delta)
    return delta


def _restore_last_tracked_rx(ctx, state) -> bool:
    if (
        state.restore_tried
        or state.level_frozen
        or state.last_ok_rx is None
        or state.rx_cur is None
        or abs(state.rx_cur - state.last_ok_rx) <= 1e-3
    ):
        return False
    state.restore_tried = True
    _set_velocity(ctx, state, 0.0)
    arc_back = 0.0
    if ctx.pivot_model is not None:
        arc_back = ctx.pivot_model.pivot(
            {"x": 0.0, "y": 0.0, "z": 0.0},
            state.last_ok_rx,
            state.rx_cur,
        )["z"]
    ctx.move_regulated_pose(state.last_ok_rx, arc_back)
    state.rx_cur = state.last_ok_rx
    state.rx_prev_meas = state.last_ok_rx
    state.rx_filt = None
    state.level_frozen = True
    state.dark_hold = 0
    state.settled_count = 0
    return True


def _advance_carrot(ctx, state, sample) -> tuple[str | None, float, bool, bool]:
    remaining = abs(state.y_dest - state.carrot)
    servo_lagging = (
        sample.error is not None
        and abs(sample.error) > 2.0 * state.z_tolerance
        and state.z_pin_dir == 0
    )
    if servo_lagging:
        state.lag_hold += 1
    elif sample.error is not None and abs(sample.error) <= 2.0 * state.z_tolerance:
        state.lag_hold = 0
    lag_released = state.lag_hold >= state.lag_cap
    blind = sample.error is None and not sample.too_near
    advance = 0.0
    if sample.too_near or (servo_lagging and not lag_released):
        pass
    elif blind and abs(state.carrot - state.last_ok) >= state.blind_allowance:
        state.dark_hold += 1
        if state.dark_hold >= ctx.args.edge_oor_iters:
            return "lost", remaining, False, lag_released
    elif blind and remaining <= 1e-9:
        state.dark_hold += 1
        if state.dark_hold >= ctx.args.edge_oor_iters and _restore_last_tracked_rx(ctx, state):
            return "restored", remaining, False, lag_released
    elif remaining > 1e-9:
        advance = min(
            max(state.speed * sample.dt, PICO_CRUISE_MIN_STEP_MM),
            state.advance_cap,
            remaining,
        )
        if servo_lagging:
            advance = min(advance, PICO_CRUISE_MIN_STEP_MM)
        if abs(advance - remaining) < 1e-9 and advance >= PICO_CRUISE_MIN_STEP_MM:
            split = max(
                remaining - PICO_CRUISE_MIN_STEP_MM / 2.0,
                0.0,
            )
            advance = split if split > 1e-9 else remaining
        state.carrot += state.direction * advance
        ctx.pivot_anchor["y"] += state.direction * advance
        state.reached = state.carrot
    return None, remaining, advance > 0.0, lag_released


def _dispatch_motion(
    ctx,
    state,
    sample,
    moving: bool,
    rx_moving: bool,
    dz: float,
    arc_z: float,
) -> str | None:
    if not (moving or rx_moving or abs(dz) > 1e-9 or state.was_moving):
        return None
    hit_high, hit_low, _ = ctx.move_regulated_pose(
        None,
        dz + arc_z,
        wait=False,
        rx_meas=sample.rx_measured,
        rx_arc_rad=(sample.rx_measured if sample.rx_measured is not None else state.rx_cur),
    )
    if ctx.y_limit["hit"]:
        state.collision_count += 1
        if state.collision_count >= ctx.args.edge_oor_iters:
            print(
                "  band transition: self-collision guard refused the cruise "
                "pose; backing out along the walked path.",
                file=sys.stderr,
            )
            return "collide"
    else:
        state.collision_count = 0
        if (
            abs(dz) > 1e-9
            and not sample.too_near
            and sample.error is not None
            and ((hit_high and sample.error > 0) or (hit_low and sample.error < 0))
        ):
            state.z_pin_dir = 1 if hit_high else -1
    return None


def _settled_result(
    state,
    sample,
    remaining: float,
    moving: bool,
    rx_settled: bool,
    dz: float,
    lag_released: bool,
) -> tuple[str, float] | None:
    if remaining > 1e-9 or moving or state.was_moving:
        return None
    z_ok = (
        sample.error is None
        or abs(sample.error) <= state.z_tolerance
        or state.z_pin_dir != 0
        or lag_released
    )
    if rx_settled and z_ok and (abs(dz) <= 1e-9 or lag_released):
        state.settled_count += 1
        if state.settled_count >= 2:
            return (
                "ok" if sample.error is not None else "dark",
                state.reached,
            )
    else:
        state.settled_count = 0
    return None


def _deadline_result(state) -> tuple[str, float] | None:
    if time.monotonic() < state.deadline:
        return None
    if abs(state.y_dest - state.y_here) < 1e-9:
        print(
            "  band transition: settle budget expired; proceeding best-effort.",
            file=sys.stderr,
        )
        return ("ok" if state.last_in_range else "dark"), state.reached
    print(
        "  band transition: cruise budget expired on this X rail; retreating to try the next.",
        file=sys.stderr,
    )
    return "lost", state.reached


def band_walk_cruise(ctx, y_here: float, y_dest: float) -> tuple[str, float]:
    """Continuously regulate Z/RX while advancing a focus-row target."""
    state = _new_walk_state(ctx, y_here, y_dest)
    if state is None:
        return "stop", y_here
    try:
        while not ctx.stop_state["requested"]:
            deadline = _deadline_result(state)
            if deadline is not None:
                return deadline
            sample = _sample_surface(ctx, state)
            rx_moving, rx_settled, arc_z = _update_rx(ctx, state, sample)
            dz = _z_delta(ctx, state, sample)
            status, remaining, moving, lag_released = _advance_carrot(
                ctx,
                state,
                sample,
            )
            if status == "restored":
                time.sleep(ctx.args.period_s)
                continue
            if status is not None:
                return status, state.reached
            status = _dispatch_motion(
                ctx,
                state,
                sample,
                moving,
                rx_moving,
                dz,
                arc_z,
            )
            if status is not None:
                return status, state.reached
            settled = _settled_result(
                state,
                sample,
                remaining,
                moving,
                rx_settled,
                dz,
                lag_released,
            )
            if settled is not None:
                return settled
            state.was_moving = moving
            time.sleep(ctx.args.period_s)
        return "stop", state.reached
    finally:
        _set_velocity(ctx, state, 0.0)
