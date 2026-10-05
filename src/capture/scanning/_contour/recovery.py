"""Edge recovery and retreat movement for contour scans."""

from __future__ import annotations

import math
import sys
import time

from capture.motion.rx_axis.server import RxAxisServerError

from ..config import *  # noqa: F401,F403


def _traverse_arc_rise(ctx, z_target: float) -> bool:
    """RAISE z to z_target (an arc-owed retreat) and WAIT for arrival, BEFORE a
    lateral grid traverse -- so a post-probe swing translates the head along the
    focus orbit instead of sweeping toward the body at the stale (deep) z (the
    stale-anchor hazard). Rise only: callers never pass a descend.
    Returns True only when z actually REACHED the unclamped target -- a rise
    clamped short by --z-min-mm (or a failed/unverified move) returns False so
    the caller can refuse a large lateral jump instead of running it unsafely.
    Keeps the stream accumulator (z_stream) in sync like _replay_move_to."""
    _clamp_z = ctx._clamp_z
    args = ctx.args
    client_z = ctx.client_z
    collision_guard = ctx.collision_guard
    move_axis_to = ctx.move_axis_to
    read_axis = ctx.read_axis
    read_position = ctx.read_position
    read_rx_rad = ctx.read_rx_rad
    record_sample = ctx.record_sample
    y_limit = ctx.y_limit
    z_stream = ctx.z_stream
    clamped, _ = _clamp_z(z_target, -1.0)  # delta<0: rising (retreat direction)
    # Self-collision guard: a ~100mm rise at a tilted rx can reach the frame.
    # Best-effort (a failed read never blocks, matching the other guard sites).
    if collision_guard is not None:
        _p = read_position()
        _r = read_rx_rad()
        if _r is not None and all(_p.get(a) is not None for a in ("x", "y")):
            _clr = collision_guard.min_clearance(float(_p["x"]), float(_p["y"]), clamped, _r)
            if _clr < collision_guard.margin:
                print(
                    f"  arc rise to z={clamped:.1f}mm refused by the "
                    f"self-collision guard (clearance {_clr:.0f}mm).",
                    file=sys.stderr,
                )
                return False
    prev_cmd = z_stream["cmd"]
    z_stream["cmd"] = clamped
    z_stream["bad"] = 0
    ok = move_axis_to(
        client_z,
        clamped,
        args.x_tolerance_mm,
        args.x_move_timeout_s,
        "z arc-rise",
    )
    if not ok:
        # The dispatch was rejected: roll the accumulator back.
        z_stream["cmd"] = prev_cmd
        # move_axis_to flags any GantrySoftLimitError as y_limit; a z-rise
        # clamp is not a Y edge; clear it so the scanner records the
        # honest 'pivot-jump' reason instead.
        y_limit["hit"] = False
        return False
    # Verify arrival at the UNCLAMPED target: move_axis_to returns True on a
    # poll timeout (warn-and-continue semantics), and _clamp_z may have pinned
    # the rise at --z-min-mm -- both must read as NOT-risen here.
    z_now = read_axis(client_z)
    for _ in range(2):
        if z_now is not None:
            break
        time.sleep(0.05)
        z_now = read_axis(client_z)
    arrived = z_now is not None and abs(z_now - z_target) <= max(1.0, args.x_tolerance_mm * 2.0)
    record_sample(
        "traverse:rise",
        target={"z": round(z_target, 4)},
        arrived=arrived,
    )
    return arrived


def _wait_for_rx(ctx, target_rad: float, swing_rad: float) -> None:
    """Block until RX reaches target_rad (or a swing-scaled timeout). Unlike
    _wait_for_swing (tuned for the tiny per-iteration leveling steps and
    hard-capped at 3s), a recovery/restore swing can be the full 20 deg
    (~7s at the default 0.05 rad/s), so the deadline scales with the swing
    magnitude -- otherwise the head traverses x while still mid-swing."""
    args = ctx.args
    read_rx_rad = ctx.read_rx_rad
    stop_state = ctx.stop_state
    speed = args.rx_speed_rad_s or 0.05
    deadline = time.monotonic() + min(20.0, max(0.5, swing_rad / speed * 1.5 + 1.0))
    while not stop_state["requested"]:
        value = read_rx_rad()
        if value is not None and abs(value - target_rad) <= 0.005:
            return
        if time.monotonic() >= deadline:
            if args.debug:
                print("  [dbg] rx swing wait timed out; continuing")
            return
        time.sleep(0.02)


def set_rx_absolute(
    ctx, target_rad: float, wait: bool = True, speed_rad_s: float | None = None
) -> float | None:
    """Move RX to an absolute angle, clamped to --rx-min/max-rad, with NO
    pivot compensation -- the recovery tilt is meant to swing the view back
    onto the body, so we deliberately do NOT hold the viewed point. With wait=True
    (default) it blocks until the (possibly large) swing lands so the caller's next
    sensor read is not taken mid-move; with wait=False it fires the move and returns
    immediately (so it can run CONCURRENTLY with a gantry move -- the caller waits
    later). Returns the commanded angle, or None if rx is unreadable or the move is
    rejected; rx_move_fail['reason'] then says which ('unreadable' / 'rejected')."""
    _wait_for_rx = ctx._wait_for_rx
    args = ctx.args
    read_rx_rad = ctx.read_rx_rad
    record_sample = ctx.record_sample
    rx_client = ctx.rx_client
    rx_move_fail = ctx.rx_move_fail
    rx_move_fail["reason"] = None
    rx_now = read_rx_rad()
    if rx_now is None:
        rx_move_fail["reason"] = "unreadable"
        print("  warning: rx angle unreadable; cannot command rx.", file=sys.stderr)
        return None
    # Clamp to the user safety window AND the RX-axis motor's hard command window so
    # we never command a target the server would 400-reject.
    tgt = target_rad
    lo = max([v for v in (args.rx_min_rad, args.rx_axis_min_rad) if v is not None])
    hi = min([v for v in (args.rx_max_rad, args.rx_axis_max_rad) if v is not None])
    tgt = max(lo, min(hi, tgt))
    try:
        rx_client.move_to(
            tgt,
            speed_rad_s=speed_rad_s if speed_rad_s is not None else args.rx_speed_rad_s,
            accel_rad_s2=args.rx_accel_rad_s2,
        )
    except RxAxisServerError as exc:
        rx_move_fail["reason"] = "rejected"
        print(
            f"  rx move rejected (soft limit / command window / shutdown): {exc}",
            file=sys.stderr,
        )
        return None
    record_sample("cmd:rx", target={"rx": round(tgt, 6)})
    if wait:
        _wait_for_rx(tgt, abs(tgt - rx_now))
    return tgt


def attempt_edge_recovery(ctx, level_ref: float, phase_dir: int) -> tuple[bool, float, bool]:
    """Probe whether an out-of-range reading is a recoverable steep surface or
    the body edge. Tilt RX toward the body -- back toward the start y -- in
    --edge-tilt-step-deg increments, STARTING AT THE CURRENT carried rx and
    BOUNDED to level_ref +/- the --edge-tilt-max-deg hard cap (<= 20 deg).
    level_ref is the LAST rx at which a station settled via leveling
    (level_anchor) -- the most recent verifiably normal-to-skin orientation
    -- so recovery tilts cannot RATCHET past the edge (recovered stations do
    not move the anchor), while the ramp begins where the head actually is:
    probing from a scan-global reference can snap the camera back toward
    rx_start when the carried angle has walked far from it on a curved
    subject, testing an
    angular band that could not even contain the body. Reads both sensors
    after each step. Returns (recovered, rx_used, limited):
      recovered=True (with the tilt that worked) as soon as BOTH sensors are
        back in range NEAR the target standoff (average <= --target-mm +
        --edge-recover-window-mm). In-range alone is NOT recovery: the bed
        under the body edge can sit inside the sensor window (observed on
        hardware reading ~127mm after a 2deg tilt) and must not fake a
        recovery -- a too-far in-range reading keeps the ramp going.
      recovered=False otherwise (rx left where the probe ended). limited=True
        means the ramp was BLOCKED (anchor band edge / RX-axis motor command window /
        --rx-min/max-rad / rx unreadable) before exhausting its reach, rather
        than the surface simply being gone -- a too-tight rx range can
        masquerade as a body edge, so the caller warns.
    The tilt direction is edge_tilt_sign * phase_dir. phase_dir is the Y-march
    direction (descend=-1, ascend=+1), not the X-sweep parity. Flip
    --edge-tilt-sign if it tilts the wrong way on this rig."""
    args = ctx.args
    read_exclusive = ctx.read_exclusive
    read_rx_rad = ctx.read_rx_rad
    rx_move_fail = ctx.rx_move_fail
    set_rx_absolute = ctx.set_rx_absolute
    stop_state = ctx.stop_state
    sign = args.edge_tilt_sign * float(phase_dir)
    cap_rad = math.radians(args.edge_tilt_max_deg)
    step_rad = math.radians(args.edge_tilt_step_deg)
    rx_now = read_rx_rad()
    if rx_now is None:
        time.sleep(0.05)
        rx_now = read_rx_rad()
    if rx_now is None:
        # NEVER assume rx = level_ref: the snap guard below compares targets
        # to rx_now, so a wrong assumption (head actually far outside the
        # band) would permit a one-command swing next to the surface.
        # Without a trusted rx there is no safe swing to make.
        print(
            "  edge probe skipped: rx unreadable, cannot bound the swing; reach-limited.",
            file=sys.stderr,
        )
        return False, level_ref, True
    # Reach band: anchor +/- cap. The ramp runs from the CURRENT rx toward
    # the band edge in the probe direction; after a long leveled contour walk
    # the current rx can sit anywhere in (or even outside) the band, so the
    # step count comes from the actual distance to cover, not the cap.
    band_lo = level_ref - cap_rad
    band_hi = level_ref + cap_rad
    # NEVER SNAP INTO THE BAND. If the
    # current rx sits so far outside the band that even the first clamped step
    # is a multi-step jump, the surface here is beyond the probe's designed
    # <=20deg reach -- skip the swing entirely and report reach-limited; the
    # caller's breadcrumb retreat walks the head back along its own path.
    first_target = max(band_lo, min(band_hi, rx_now + sign * step_rad))
    if abs(first_target - rx_now) > 2.0 * step_rad:
        print(
            f"  edge probe skipped: current rx {rx_now:+.3f}rad is "
            f"{math.degrees(abs(first_target - rx_now)):.1f}deg outside the "
            f"anchor band [{band_lo:+.3f}, {band_hi:+.3f}]rad -- a one-command "
            "swing into the band is unsafe near the surface; reach-limited.",
            file=sys.stderr,
        )
        return False, rx_now, True
    reach = (rx_now - band_lo) if sign < 0 else (band_hi - rx_now)
    steps = max(1, int(math.ceil(max(0.0, reach) / step_rad)))
    limited = False
    far_warned = False  # one bed/floor-in-window warning per probe
    prev_cmd = rx_now
    for i in range(1, steps + 1):
        if stop_state["requested"]:
            break
        target = max(band_lo, min(band_hi, rx_now + sign * i * step_rad))
        cmd = set_rx_absolute(target)
        if cmd is None:
            # Tilt physically impossible (rejected) or rx unreadable: we cannot
            # probe further in this direction. STOP -- do not skip to a larger,
            # equally-impossible tilt and silently call it an edge.
            limited = True
            print(
                f"  warning: edge probe stopped at rx={target:+.3f}rad: "
                f"rx move {rx_move_fail['reason']}; could not test further tilt.",
                file=sys.stderr,
            )
            break
        pinned = abs(cmd - prev_cmd) < 1e-9
        prev_cmd = cmd
        r1 = read_exclusive("sensor1")
        r2 = read_exclusive("sensor2")
        both = (
            r1.in_range
            and r1.distance_mm is not None
            and r2.in_range
            and r2.distance_mm is not None
        )
        # In-range alone is NOT recovery: the bed/floor under the body edge
        # can sit inside the sensor window (hardware: bed read ~127mm 'ok'
        # after a 2deg tilt -> false recovery -> z descended toward the bed).
        # The lost skin was at ~target, so require the reading NEAR it.
        avg = None
        near = False
        if both:
            avg = (r1.distance_mm + r2.distance_mm) / 2.0
            near = avg <= args.target_mm + args.edge_recover_window_mm
        if args.debug:
            d1 = "--" if r1.distance_mm is None else f"{r1.distance_mm:.2f}"
            d2 = "--" if r2.distance_mm is None else f"{r2.distance_mm:.2f}"
            verdict = (
                "BOTH in range near target: RECOVERED"
                if near
                else "in range but TOO FAR (bed/floor?), not a recovery"
                if both
                else "still out"
            )
            print(
                f"  [dbg] edge probe tilt {math.degrees(cmd - rx_now):+.1f}deg "
                f"(rx={cmd:+.3f}rad): d1={d1} d2={d2} {verdict}"
            )
        if both and near:
            return True, cmd, False
        if both and not far_warned:
            far_warned = True
            print(
                f"  edge probe: sensors in range at {avg:.1f}mm but past "
                f"target+{args.edge_recover_window_mm:.0f}mm -- a deeper surface "
                "(bed/floor), not the lost skin; continuing the tilt ramp.",
                file=sys.stderr,
            )
        if pinned:
            # The commanded rx did not move: pinned at the anchor band edge,
            # the RX-axis motor command window, or a --rx-min/max-rad bound -- more
            # tilt in this direction is impossible. Stop (limited: a tight rx
            # range can masquerade as a body edge, so the caller warns).
            limited = True
            print(
                f"  warning: edge probe pinned at rx={cmd:+.3f}rad (anchor band "
                f"[{band_lo:+.3f}, {band_hi:+.3f}]rad / rx limits) before "
                "exhausting its reach; could not test further tilt.",
                file=sys.stderr,
            )
            break
    # Edge (or limited): leave RX where the probe ended. The caller may carry
    # this contour tilt to the next position because snapping RX back to level
    # is an RX-only swing that can collide with a steep surface.
    return False, prev_cmd, limited
