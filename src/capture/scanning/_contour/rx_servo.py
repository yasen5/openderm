"""Velocity-servo policy used by per-station contour regulation."""

from __future__ import annotations

import math
import sys
import time

from capture.motion.rx_axis.server import RxAxisServerError

from ..config import (
    RX_OSC_BACKSWING_DEADBANDS,
    RX_OSC_FREEZE_REVERSALS,
    RX_OSC_MIN_GAIN_SCALE,
    RX_OSC_REVERSALS_PER_CUT,
    RX_VELOCITY_INTEGRATOR_CLAMP_FRACTION,
    RX_VELOCITY_INTEGRATOR_LEAK_PER_S,
    RX_VELOCITY_REARM_FACTOR,
    STATION_RX_LIMIT,
    STATION_Y_LIMIT,
)
from ..helpers import _clamp, _fmt_bound


def _filter_error(ctx, state, raw_error: float) -> float:
    if state.rx_error_filtered is None:
        state.rx_error_filtered = raw_error
    else:
        alpha = ctx.args.rx_filter_alpha
        state.rx_error_filtered = alpha * raw_error + (1.0 - alpha) * state.rx_error_filtered
    return state.rx_error_filtered


def _record_reversal(ctx, state, error: float) -> None:
    if abs(error) <= ctx.args.rx_deadband_mm:
        return
    sign = 1 if error > 0 else -1
    reversal = False
    if state.vel_err_sign == 0:
        state.vel_err_sign = sign
    elif sign != state.vel_err_sign:
        state.vel_err_sign = sign
        state.vel_bumped = False
        reversal = True
    elif (
        state.vel_reversals >= 1
        and not state.vel_bumped
        and abs(error) > ctx.args.rx_deadband_mm * RX_OSC_BACKSWING_DEADBANDS
    ):
        state.vel_bumped = True
        reversal = True
    if not reversal:
        return

    state.vel_reversals += 1
    if (
        not state.vel_frozen
        and state.vel_reversals >= RX_OSC_FREEZE_REVERSALS
        and state.vel_gain_scale <= RX_OSC_MIN_GAIN_SCALE
    ):
        state.vel_frozen = True
        print(
            f"  {state.label}: rx still oscillating at "
            f"{state.vel_gain_scale:.0%} gain "
            f"({state.vel_reversals} reversals); freezing leveling, "
            f"settling best-effort at d1-d2={error:+.2f}mm.",
            file=sys.stderr,
        )
    elif state.vel_reversals >= state.vel_next_cut and state.vel_gain_scale > RX_OSC_MIN_GAIN_SCALE:
        state.vel_next_cut = state.vel_reversals + RX_OSC_REVERSALS_PER_CUT
        state.vel_gain_scale = max(
            state.vel_gain_scale * 0.5,
            RX_OSC_MIN_GAIN_SCALE,
        )
        ctx.vel_scale_carry["scale"] = state.vel_gain_scale
        state.vel_integrator = 0.0
        print(
            f"  {state.label}: rx oscillating ({state.vel_reversals} "
            "reversals); reducing rx velocity gain and rate cap to "
            f"{state.vel_gain_scale:.0%}.",
            file=sys.stderr,
        )


def _velocity_cap(ctx, state) -> float:
    cap = ctx.args.rx_speed_rad_s or 0.25
    if state.vel_rate_budget is not None:
        cap = min(cap, state.vel_rate_budget)
    if state.vel_gain_scale < 1.0:
        cap = max(
            cap * state.vel_gain_scale,
            ctx.args.rx_velocity_min_rad_s * 1.05,
        )
    return cap


def _update_armed_state(ctx, state, error: float) -> None:
    if state.vel_armed and abs(error) <= ctx.args.rx_deadband_mm:
        state.vel_armed = False
    elif not state.vel_armed and abs(error) > ctx.args.rx_deadband_mm * RX_VELOCITY_REARM_FACTOR:
        state.vel_armed = True
    if state.vel_frozen:
        state.vel_armed = False


def _desired_velocity(
    ctx,
    state,
    error: float,
    dt: float,
    cap: float,
) -> tuple[float, bool]:
    if not state.vel_armed:
        state.vel_integrator *= 0.5
        if abs(state.vel_integrator) < 0.005:
            state.vel_integrator = 0.0
        state.vel_w_prev = 0.0
        return 0.0, True

    proportional = ctx.args.rx_gain_rad_per_mm * state.vel_gain_scale * error
    stalled = state.vel_prev_abs_err is not None and abs(error) > state.vel_prev_abs_err - 0.05
    if stalled and abs(proportional + state.vel_integrator) < cap:
        state.vel_integrator += (
            ctx.args.rx_velocity_ki_rad_s_per_mm_s * state.vel_gain_scale * error * dt
        )
    state.vel_integrator = _clamp(
        state.vel_integrator * (1.0 - RX_VELOCITY_INTEGRATOR_LEAK_PER_S * dt),
        RX_VELOCITY_INTEGRATOR_CLAMP_FRACTION * cap,
    )
    desired = _clamp(proportional + state.vel_integrator, cap)
    if ctx.args.rx_velocity_slew_rad_s2 > 0 and dt > 0:
        max_delta = ctx.args.rx_velocity_slew_rad_s2 * dt
        command = state.vel_w_prev + _clamp(
            desired - state.vel_w_prev,
            max_delta,
        )
    else:
        command = desired
    if command != 0.0 and abs(command) < ctx.args.rx_velocity_min_rad_s:
        command = (
            math.copysign(ctx.args.rx_velocity_min_rad_s, command) if command * desired > 0 else 0.0
        )
    return command, False


def _motion_block(ctx, state, rx_measured, command: float) -> str | None:
    blocked = None
    if rx_measured is not None and command != 0.0:
        _, hit_bound, axis_bound = ctx.clamp_rx(
            rx_measured,
            math.copysign(1e-3, command),
        )
        if hit_bound:
            blocked = "axis" if axis_bound else "user"
    if blocked is None and ctx.rx_vel_collision_blocked(rx_measured, command):
        blocked = "collision"
    if blocked is None:
        return None

    state.vel_integrator = 0.0
    state.vel_w_prev = 0.0
    if blocked == "user" and not state.rx_limit_warned:
        print(
            f"  note: rx held at tilt limit "
            f"[{_fmt_bound(ctx.args.rx_min_rad)}, "
            f"{_fmt_bound(ctx.args.rx_max_rad)}] rad (velocity servo); "
            f"residual d1-d2={state.rx_error_filtered:+.2f}mm not levelled out.",
            file=sys.stderr,
        )
        state.rx_limit_warned = True
    return blocked


def _dispatch_velocity(ctx, state, command: float, blocked: str | None) -> str | None:
    if blocked is not None:
        command = 0.0
    try:
        response = ctx.rx_client.set_velocity(command)
    except RxAxisServerError as exc:
        ctx._rx_rejected(exc)
    else:
        held = response.get("held") if isinstance(response, dict) else None
        if held in ("max_window", "min_window") and command != 0.0:
            blocked = "axis"
            state.vel_integrator = 0.0
            state.vel_w_prev = 0.0
    state.velocity_command = command
    return blocked


def _blocked_status(ctx, state, blocked: str | None) -> str | None:
    if blocked == "axis":
        state.rx_axis_count += 1
        if state.rx_axis_count >= ctx.args.edge_oor_iters:
            print(
                f"  {state.label}: rx blocked at its axis window for "
                f"{state.rx_axis_count} iters (velocity servo), residual "
                f"d1-d2={state.rx_error_filtered:+.2f}mm -- reach-limited "
                "(rx) edge.",
                file=sys.stderr,
            )
            return STATION_RX_LIMIT
    else:
        state.rx_axis_count = 0

    if blocked == "collision":
        state.rx_collision_count += 1
        if state.rx_collision_count >= ctx.args.edge_oor_iters:
            print(
                f"  {state.label}: rx velocity gated by the self-collision "
                f"guard for {state.rx_collision_count} iters, residual "
                f"d1-d2={state.rx_error_filtered:+.2f}mm -- "
                "collision-limited edge.",
                file=sys.stderr,
            )
            return STATION_Y_LIMIT
    else:
        state.rx_collision_count = 0
    return None


def update_rx_servo(
    ctx,
    state,
    raw_error: float,
    rx_measured: float | None,
) -> tuple[str | None, bool, float | None]:
    """Advance the RX velocity controller by one sensor sample."""
    error = _filter_error(ctx, state, raw_error)
    rx_new = rx_measured if rx_measured is not None else state.rx_cur
    now = time.monotonic()
    dt = 0.0 if state.vel_last_t is None else min(0.4, max(0.0, now - state.vel_last_t))
    state.vel_last_t = now
    _record_reversal(ctx, state, error)
    _update_armed_state(ctx, state, error)
    command, on_target = _desired_velocity(
        ctx,
        state,
        error,
        dt,
        _velocity_cap(ctx, state),
    )
    blocked = _motion_block(ctx, state, rx_measured, command)
    if blocked is not None:
        command = 0.0
        if blocked == "user":
            on_target = True
    blocked = _dispatch_velocity(ctx, state, command, blocked)
    status = _blocked_status(ctx, state, blocked)

    if (
        on_target
        and rx_measured is not None
        and state.vel_prev_rx_meas is not None
        and abs(rx_measured - state.vel_prev_rx_meas) > 0.005
    ):
        on_target = False
    state.vel_prev_rx_meas = rx_measured
    state.vel_prev_abs_err = abs(error)
    state.vel_w_prev = command
    if ctx.args.debug:
        measured = "--" if rx_measured is None else f"{rx_measured:+.3f}"
        print(
            f"  [dbg] VELSERVO d1-d2 raw={raw_error:+.3f} "
            f"filt={error:+.3f}mm -> w={command:+.3f}rad/s "
            f"(i={state.vel_integrator:+.3f}"
            f"{', ARMED' if state.vel_armed else ''}"
            f"{', ' + blocked if blocked else ''}) @ rx={measured}rad"
        )
    return status, on_target, rx_new
