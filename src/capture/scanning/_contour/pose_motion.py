"""Collision-aware pose movement for contour scans."""

from __future__ import annotations

import math
import sys
import time

from capture.motion.gantry.server import (
    GantryServerClient,
    GantryServerError,
    GantrySoftLimitError,
)
from capture.motion.rx_axis.server import RxAxisServerError

from ..config import *  # noqa: F401,F403
from ..helpers import _cruise_batch, _next_z_cmd


def _clamp_z(ctx, target: float, delta_mm: float) -> tuple[float, bool]:
    """Clamp a z setpoint to the usable --z-min/max-mm travel window. Returns
    (clamped_target, at_limit) where at_limit is True when the clamp blocked
    further motion in the requested direction."""
    args = ctx.args
    at_limit = False
    if args.z_max_mm is not None and target > args.z_max_mm:
        target = args.z_max_mm
        at_limit = delta_mm > 0
    if args.z_min_mm is not None and target < args.z_min_mm:
        target = args.z_min_mm
        at_limit = delta_mm < 0
    return target, at_limit


def step_z(ctx, delta_mm: float) -> bool:
    """Advance the streamed Z setpoint by ``delta_mm``."""
    Z_MAX_BAD = ctx.Z_MAX_BAD
    Z_TRUST_BAND_MM = ctx.Z_TRUST_BAND_MM
    _clamp_z = ctx._clamp_z
    args = ctx.args
    client_z = ctx.client_z
    read_axis = ctx.read_axis
    record_sample = ctx.record_sample
    z_sat = ctx.z_sat
    z_stream = ctx.z_stream
    z = read_axis(client_z)
    if z is None:
        z_sat["meas"] = None
        return False
    z_sat["meas"] = z
    try:
        prev = z_stream["cmd"]
        new_cmd, z_stream["bad"], hold = _next_z_cmd(
            prev, z_stream["bad"], z, delta_mm, Z_TRUST_BAND_MM, Z_MAX_BAD
        )
        new_cmd, z_sat["at_limit"] = _clamp_z(new_cmd, delta_mm)
        z_stream["cmd"] = new_cmd
        if z_stream["bad"] and prev is not None:
            print(
                f"  warning: ignoring implausible z read {z:.2f}mm (command "
                f"{prev:.2f}mm, {z_stream['bad']} in a row); "
                f"{'holding z' if hold else 'advancing from command'}.",
                file=sys.stderr,
            )
        if args.debug:
            print(
                f"  [dbg] step_z stream: z_meas={z:.3f} delta={delta_mm:+.3f} "
                f"-> cmd={new_cmd:.3f}{' HOLD' if hold else ''}"
                f"{' ZLIMIT' if z_sat['at_limit'] else ''}"
            )
        client_z.stream_to(new_cmd)
        record_sample("cmd:z", target={"z": round(new_cmd, 4)})
        return True
    except GantryServerError as exc:
        print(f"  z move rejected (likely travel limit): {exc}", file=sys.stderr)
        return False


def _move_gantry_arc(ctx, target: dict[str, float]) -> bool:
    """Stream a regulated X/Y/Z pose. Non-fatal: warns on rejection
    (a y soft-limit hit instead flags a reach-limited (y) edge). Returns True
    when the dispatch was ACCEPTED (or was a genuine no-op), False when it was
    refused -- callers must not breadcrumb/pair a refused pose (the head never
    went there; replaying it would deterministically re-refuse and kill the
    retreat that safety depends on)."""
    args = ctx.args
    client_x = ctx.client_x
    client_y = ctx.client_y
    client_z = ctx.client_z
    pico_multi = ctx.pico_multi
    pico_stream_prev = ctx.pico_stream_prev
    y_limit = ctx.y_limit
    try:
        pico_targets = {axis: target[axis] for axis in ("y", "z") if axis in target}
        if "x" in target:
            client_x.stream_to(target["x"])
        cruise = _cruise_batch(pico_stream_prev, pico_targets, PICO_CRUISE_MIN_STEP_MM)
        if pico_multi is not None and len(pico_targets) >= 2:
            pico_multi.stream_to(pico_targets, continuous=cruise)
        elif pico_targets:
            for axis in ("y", "z"):
                if axis in pico_targets:
                    (client_y if axis == "y" else client_z).stream_to(
                        pico_targets[axis], continuous=cruise
                    )
        pico_stream_prev.update(pico_targets)
    except GantrySoftLimitError as exc:
        # Backstop (move_regulated_pose normally pre-checks gy and skips before
        # this): a y soft-limit hit is a reach-limited (y) EDGE, not an abort --
        # flag it so regulate_station ends this station and the scan continues.
        y_limit["hit"] = True
        if args.debug:
            print(f"  [dbg] y soft-limit on the regulated move: {exc}", file=sys.stderr)
        return False
    except GantryServerError as exc:
        print(
            f"  warning: regulated gantry move rejected (likely travel limit): {exc}",
            file=sys.stderr,
        )
        return False
    return True


def breadcrumb_push(ctx, x, y, z, rx) -> None:
    """Append a walked pose to the trail (decimated; no-op if x/y/z unknown;
    rx optional -- pose-only crumbs still make a safe retreat path). Cheap
    enough for the per-iteration control path."""
    breadcrumb = ctx.breadcrumb
    if x is None or y is None or z is None:
        return
    if breadcrumb:
        last = breadcrumb[-1]
        drx = abs(rx - last["rx"]) if (rx is not None and last.get("rx") is not None) else 0.0
        if (
            abs(x - last["x"]) < BREADCRUMB_SPACING_MM
            and abs(y - last["y"]) < BREADCRUMB_SPACING_MM
            and abs(z - last["z"]) < BREADCRUMB_SPACING_MM
            and drx < BREADCRUMB_SPACING_RAD
        ):
            return
    breadcrumb.append(
        {
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "rx": float(rx) if rx is not None else None,
        }
    )
    if len(breadcrumb) > BREADCRUMB_MAX:
        # Halve the interior resolution; keep the seed and the newest exact.
        breadcrumb[1:-1] = breadcrumb[1:-1][::2]


def breadcrumb_reseed(ctx, x, y, z, rx) -> None:
    """Restart the trail at a known-good pose (a capture / sweep start)."""
    breadcrumb = ctx.breadcrumb
    breadcrumb.clear()
    if None not in (x, y, z):
        breadcrumb.append(
            {
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "rx": float(rx) if rx is not None else None,
            }
        )


def _rx_rejected(ctx, exc: RxAxisServerError) -> bool:
    """Handle a rejected rx move. We pre-clamp every command to the RX-axis motor's
    command window, so a rejection here is almost always rx asking to tilt past
    its axis range to follow the surface -- a body edge, not a fatal error. Flag
    it as an axis limit (clamped + axis) and return True so the regulator stops
    that X position gracefully (STATION_RX_LIMIT) instead of aborting the scan."""
    rx_limit = ctx.rx_limit
    print(
        f"  rx move rejected (treating as rx-axis edge): {exc}",
        file=sys.stderr,
    )
    rx_limit["clamped"] = True
    rx_limit["axis"] = True
    return True


def _clamp_z_abs(ctx, z: float) -> tuple[float, bool, bool]:
    """Clamp an ABSOLUTE z command to the --z-min/max-mm travel window. Returns
    (z_clamped, hit_high, hit_low) where the flags say the request was outside
    the window on that side (used for reach-limited z-edge detection)."""
    args = ctx.args
    hit_high = args.z_max_mm is not None and z > args.z_max_mm
    hit_low = args.z_min_mm is not None and z < args.z_min_mm
    if hit_high:
        z = args.z_max_mm
    if hit_low:
        z = args.z_min_mm
    return z, hit_high, hit_low


def clamp_rx(ctx, rx_from: float, rx_delta: float) -> tuple[float, bool, bool]:
    """Clamp rx_from + rx_delta to the effective rx window (the axis hardware
    window --rx-axis-min/max-rad intersected with --rx-min/max-rad and the
    station budget). Returns (rx_new, hit_bound, axis): hit_bound True if the
    controller wanted to tilt past a bound and could not move; axis True if the
    binding bound was the rx AXIS hardware window (a reach-limited edge) rather
    than only a user/budget bound (settle there)."""
    args = ctx.args
    station_rx_bounds = ctx.station_rx_bounds
    # Track each bound WITH whether it is the axis-hardware bound, so a
    # user/budget bound that happens to equal the axis value is still
    # classified as non-axis (settle), not an edge. is_axis True only when the
    # binding bound's source is the rx axis window.
    min_cands = [
        (args.rx_min_rad, False),
        (station_rx_bounds["min"], False),
        (args.rx_axis_min_rad, True),
    ]
    max_cands = [
        (args.rx_max_rad, False),
        (station_rx_bounds["max"], False),
        (args.rx_axis_max_rad, True),
    ]
    # eff_min = the LARGEST lower bound; on a tie prefer the non-axis source
    # (sort key puts is_axis False first at equal value -> max keeps last, so
    # invert: pick max value, and among equals choose non-axis).
    min_set = [(v, ax) for v, ax in min_cands if v is not None]
    max_set = [(v, ax) for v, ax in max_cands if v is not None]
    eff_min, min_is_axis = (None, False)
    if min_set:
        top = max(v for v, _ in min_set)
        eff_min = top
        min_is_axis = all(ax for v, ax in min_set if abs(v - top) < 1e-9)
    eff_max, max_is_axis = (None, False)
    if max_set:
        bot = min(v for v, _ in max_set)
        eff_max = bot
        max_is_axis = all(ax for v, ax in max_set if abs(v - bot) < 1e-9)
    target = rx_from + rx_delta
    if eff_min is not None:
        target = max(eff_min, target)
    if eff_max is not None:
        target = min(eff_max, target)
    hit_bound = abs(target - rx_from) < 1e-6  # could not move from here
    axis = False
    if hit_bound:
        axis = min_is_axis if rx_delta < 0 else max_is_axis
    return target, hit_bound, axis


def move_regulated_pose(
    ctx,
    rx_target: float | None,
    total_dz_mm: float,
    *,
    wait: bool = True,
    rx_meas: float | None = None,
    rx_arc_rad: float | None = None,
) -> tuple[bool, bool, float | None]:
    """ONE coordinated move that holds the station's focus AND trims the
    standoff. x/y go to the ABSOLUTE pivot arc from the entry anchor
    (pivot(anchor, rx_target, anchor_rx)); z is a RELATIVE command advanced by
    total_dz_mm = standoff_delta + arc_z_increment, where arc_z_increment is the
    pivot's z motion for this rx step (it CANCELS the anchor z, so no raw z read
    ever feeds the z command). Routing z as a relative delta keeps it inside the
    glitch-protected accumulator (_next_z_cmd / z_stream) in stream mode -- the
    same guard that prevents a transient position-read glitch from plunging z
    -- instead of an absolute anchor-z baseline. So the head orbits the focus at
    constant standoff as rx tilts, AND z stays glitch-safe. Returns
    (z_hit_high, z_hit_low, z_meas) for saturation detection. ``wait=False``
    skips the post-dispatch swing wait so the caller keeps reading mid-motion;
    ``rx_meas`` passes an already-measured rx to skip a second RX-axis server
    round trip. rx_arc_rad (velocity-servo mode): follow the pivot arc AT
    THIS rx WITHOUT dispatching any rx position move -- the RX-axis motor is under
    continuous velocity control and a position command would cancel it
    server-side; the arc chases the MEASURED camera angle instead."""
    Z_MAX_BAD = ctx.Z_MAX_BAD
    Z_TRUST_BAND_MM = ctx.Z_TRUST_BAND_MM
    _clamp_z_abs = ctx._clamp_z_abs
    _move_gantry_arc = ctx._move_gantry_arc
    _rx_rejected = ctx._rx_rejected
    _wait_for_swing = ctx._wait_for_swing
    args = ctx.args
    breadcrumb_push = ctx.breadcrumb_push
    client_z = ctx.client_z
    collision_guard = ctx.collision_guard
    pivot_anchor = ctx.pivot_anchor
    pivot_model = ctx.pivot_model
    pose_last = ctx.pose_last
    read_axis = ctx.read_axis
    read_position = ctx.read_position
    read_rx_rad = ctx.read_rx_rad
    record_sample = ctx.record_sample
    rx_client = ctx.rx_client
    rx_last = ctx.rx_last
    rx_limit = ctx.rx_limit
    y_limit = ctx.y_limit
    z_rx = ctx.z_rx
    z_stream = ctx.z_stream
    rx_limit["clamped"] = False
    rx_limit["axis"] = False
    y_limit["hit"] = False
    record_sample("cmd:regulated", target={"rx": rx_target, "dz_mm": round(total_dz_mm, 4)})
    a = pivot_anchor

    # x/y: absolute pivot hold (anchor-based, x/y only -- z is a relative command
    # so the anchor z is not needed here). x/y reads are not subject to the z
    # trapq glitch, so the anchor x/y are trusted. Compute this BEFORE moving rx,
    # so a y target outside the soft travel limit refuses the WHOLE move (rx too)
    # -- otherwise rx would tilt to a pose the gantry can't translate to.
    gx = gy = None
    arc_rx = rx_target if rx_target is not None else rx_arc_rad
    if (
        pivot_model is not None
        and arc_rx is not None
        and a["x"] is not None
        and a["y"] is not None
        and a["rx"] is not None
    ):
        base = pivot_model.pivot({"x": a["x"], "y": a["y"]}, arc_rx, a["rx"])
        gx, gy = base["x"], base["y"]
    else:
        gx, gy = a["x"], a["y"]  # hold at anchor (may be None -> omit from move)

    if rx_meas is None:
        rx_meas = read_rx_rad()

    # Self-collision guard (replaces the static y soft-limit): refuse the WHOLE move
    # -- rx included -- if the target pose would bring a moving part within the
    # collision margin of the frame. Full pose = target x/y (pivot arc) + target z
    # (current z + this step's dz) + target rx. Checked BEFORE the rx dispatch so rx
    # never tilts to a pose the gantry cannot safely translate to. On a violation the
    # move is held and flagged as a
    # collision-limited edge, and the scan continues. Reuses the
    # y_limit["hit"] skip mechanism. See docs/collision_guard.md.
    if collision_guard is not None and gx is not None and gy is not None:
        rx_chk = arc_rx if arc_rx is not None else rx_meas
        z_base = z_stream["cmd"] if z_stream.get("cmd") is not None else a.get("z")
        z_chk = (z_base + total_dz_mm) if z_base is not None else None
        if rx_chk is not None and z_chk is not None:
            clr = collision_guard.min_clearance(gx, gy, z_chk, rx_chk)
            if clr < collision_guard.margin:
                y_limit["hit"] = True
                return False, False, read_axis(client_z)

    if rx_target is not None and (rx_meas is None or abs(rx_target - rx_meas) > 1e-6):
        try:
            rx_client.move_to(
                rx_target,
                speed_rad_s=args.rx_speed_rad_s,
                accel_rad_s2=args.rx_accel_rad_s2,
            )
        except RxAxisServerError as exc:
            _rx_rejected(exc)

    # Z is a relative command through the glitch-protected accumulator.
    z_meas = read_axis(client_z)
    hit_high = hit_low = False
    gz: float | None = None
    # During continuous dispatch, measured Z can lag the standing
    # command, so two setpoint-bounce sources need suppressing:
    #   (a) a zero-delta hold iteration would rebase cmd := meas, CANCELLING
    #       the in-flight z; skip the z update entirely instead;
    #   (b) meas + delta can land BEHIND the standing command while the
    #       error keeps its sign; clamp monotone (below, in the branch).
    continuous_step = not wait
    if continuous_step and abs(total_dz_mm) <= 1e-9:
        pass
    else:
        prev = z_stream["cmd"]
        # Never fabricate a Z baseline. If both the accumulator and the
        # current reading are unavailable, skip this update.
        if prev is None and z_meas is None:
            if args.debug:
                print("  [dbg] z unseeded and unreadable; skipping z this iter")
        else:
            new_cmd, z_stream["bad"], hold = _next_z_cmd(
                prev,
                z_stream["bad"],
                z_meas if z_meas is not None else prev,
                total_dz_mm,
                Z_TRUST_BAND_MM,
                Z_MAX_BAD,
            )
            if continuous_step and prev is not None:
                if total_dz_mm < 0:
                    new_cmd = min(new_cmd, prev)
                else:
                    new_cmd = max(new_cmd, prev)
            gz, hit_high, hit_low = _clamp_z_abs(new_cmd)
            z_stream["cmd"] = gz
            if z_stream["bad"] and prev is not None and args.debug:
                print(
                    f"  warning: ignoring implausible z read {z_meas} (cmd {prev}); "
                    f"{'holding' if hold else 'advancing from cmd'}.",
                    file=sys.stderr,
                )

    target: dict[str, float] = {}
    if gx is not None:
        target["x"] = gx
    if gy is not None:
        target["y"] = gy
    if gz is not None:
        target["z"] = gz
    if args.debug:
        rxs = "--" if rx_target is None else f"{rx_target:+.3f}"
        xs = "--" if "x" not in target else f"{target['x']:.2f}"
        ys = "--" if "y" not in target else f"{target['y']:.2f}"
        zs = "--" if "z" not in target else f"{target['z']:.2f}"
        print(
            f"  [dbg] regulate pose rx={rxs}rad -> gantry x={xs} y={ys} z={zs} "
            f"(dz={total_dz_mm:+.2f}mm)"
        )
    if target:
        dispatched = _move_gantry_arc(target)
        if dispatched and "z" in target and arc_rx is not None:
            # z and rx were commanded as one orbit step: record the pairing so
            # a later grid traverse can tell whether the carried rx has since
            # moved WITHOUT z (edge probe) and owes the arc z (see z_rx).
            z_rx["rad"] = arc_rx
        # Drop a breadcrumb at the commanded pose: continuous regulation can
        # walk the head around a flank; the retreat must be able to walk
        # back along exactly that path.
        # ONLY for ACCEPTED dispatches: a soft-limit-refused pose was never
        # reached, and replaying it would deterministically re-refuse -- one
        # phantom crumb at a routine reach-limited edge would otherwise kill
        # the retreat (and with it the scan).
        if dispatched:
            breadcrumb_push(
                target.get("x", pose_last["x"]),
                target.get("y", pose_last["y"]),
                target.get("z", z_stream["cmd"]),
                rx_target if rx_target is not None else rx_last["rad"],
            )
    if wait:
        pos = read_position()
        _wait_for_swing(
            rx_target,
            abs((rx_target or 0.0) - (rx_meas or 0.0)) if rx_meas is not None else 0.0,
            target,
            pos,
        )
    return hit_high, hit_low, z_meas


def rx_vel_collision_blocked(ctx, rx_now: float | None, w: float) -> bool:
    """Velocity-servo counterpart of move_regulated_pose's collision check:
    would holding rate w for RX_VEL_GUARD_LOOKAHEAD_S rotate the head into
    a pose the guard refuses? set_velocity is dispatched straight to the
    RX-axis server -- it never passes through move_regulated_pose, so the
    pre-dispatch check cannot veto it. Without this gate the camera keeps
    tilting while the gantry's arc-follow dispatches are being refused,
    and the head ends parked in a sub-margin pose that NO dispatch (not
    even a hold-in-place) can leave. Probes the pivot-arc pose (x/y arc + arc-z
    from the live anchor, standing z command) at the lookahead angle; a
    rotation that IMPROVES clearance is never blocked, so a head already
    inside the margin can still rotate back out."""
    clamp_rx = ctx.clamp_rx
    collision_guard = ctx.collision_guard
    pivot_anchor = ctx.pivot_anchor
    pivot_model = ctx.pivot_model
    z_stream = ctx.z_stream
    if collision_guard is None or pivot_model is None or rx_now is None or w == 0.0:
        return False
    a = pivot_anchor
    if a["x"] is None or a["y"] is None or a["rx"] is None:
        return False
    z_base = z_stream["cmd"] if z_stream.get("cmd") is not None else a.get("z")
    if z_base is None:
        return False
    look = math.copysign(min(abs(w) * RX_VEL_GUARD_LOOKAHEAD_S, RX_VEL_GUARD_MAX_LOOK_RAD), w)
    rx_look = clamp_rx(rx_now, look)[0]
    if abs(rx_look - rx_now) < 1e-9:
        return False  # pinned at a window bound; the window logic owns it
    arc = pivot_model.pivot({"x": a["x"], "y": a["y"]}, rx_look, a["rx"])
    dz_look = pivot_model.pivot({"x": 0.0, "y": 0.0, "z": 0.0}, rx_look, rx_now)["z"]
    clr_look = collision_guard.min_clearance(arc["x"], arc["y"], z_base + dz_look, rx_look)
    if clr_look >= collision_guard.margin:
        return False
    here = pivot_model.pivot({"x": a["x"], "y": a["y"]}, rx_now, a["rx"])
    clr_here = collision_guard.min_clearance(here["x"], here["y"], z_base, rx_now)
    return clr_look <= clr_here + 1e-9


def _wait_for_swing(
    ctx,
    rx_target: float | None,
    delta_rad: float,
    target: dict[str, float] | None,
    pos0: dict[str, float] | None,
) -> None:
    # pos0 is the gantry position already read immediately after the move was
    # dispatched (the read_position() at the move_regulated_pose call site). Reuse it
    # for the FIRST arrival check so a held/settled pose costs ZERO extra position
    # reads here; only re-read (the expensive Pico-bridge round trip) on later passes.
    args = ctx.args
    read_position = ctx.read_position
    read_rx_rad = ctx.read_rx_rad
    stop_state = ctx.stop_state
    y_now = None if not pos0 else pos0.get("y")
    # Deadline covers whichever is slower: the rx swing, or the pivot y
    # travel at the stream feed (the y excursion can be ~8mm when the
    # gantry is on the far side of the arc from the new target).
    speed = args.rx_speed_rad_s or 0.05
    rx_time = delta_rad / speed
    y_time = 0.0
    if target is not None and target.get("y") is not None and y_now is not None:
        stream_feed_mm_s = (args.feed_mm_min or 600.0) / 60.0
        y_time = abs(target["y"] - float(y_now)) / max(1e-6, stream_feed_mm_s)
    deadline = time.monotonic() + min(3.0, max(0.5, max(rx_time, y_time) * 1.5 + 0.3))
    need_y = target is not None and "y" in target
    pos = pos0 if pos0 else {}
    while not stop_state["requested"]:
        rx_ok = True
        if rx_target is not None:
            value = read_rx_rad()
            rx_ok = value is not None and abs(value - rx_target) <= 0.005
        y_ok = True
        if need_y:
            y_val = pos.get("y")
            y_ok = y_val is not None and abs(float(y_val) - target["y"]) <= args.x_tolerance_mm
        if rx_ok and y_ok:
            return
        if time.monotonic() >= deadline:
            if args.debug:
                print("  [dbg] swing wait timed out; continuing with possibly-moving pose")
            return
        time.sleep(0.05)
        if need_y:
            pos = read_position()  # refresh for the next pass (first pass reused pos0)


def move_axis_to(
    ctx,
    client: GantryServerClient,
    target: float,
    tolerance_mm: float,
    move_timeout_s: float,
    label: str,
) -> bool:
    """Stream an absolute axis target and poll until it arrives."""
    args = ctx.args
    read_axis = ctx.read_axis
    record_sample = ctx.record_sample
    stop_state = ctx.stop_state
    y_limit = ctx.y_limit
    try:
        client.stream_to(target)
    except GantrySoftLimitError as exc:
        y_limit["hit"] = True
        print(f"  {label}: {exc}", file=sys.stderr)
        return False
    except GantryServerError as exc:
        print(f"  {label} move rejected: {exc}", file=sys.stderr)
        return False
    record_sample(f"cmd:{label}", target={label: round(target, 4)})
    deadline = time.monotonic() + move_timeout_s
    while not stop_state["requested"]:
        value = read_axis(client)
        if value is not None and abs(value - target) <= tolerance_mm:
            return True
        if time.monotonic() >= deadline:
            print(
                f"  warning: {label} did not reach {target:.2f} mm within "
                f"{move_timeout_s:.1f}s; continuing.",
                file=sys.stderr,
            )
            return True
        time.sleep(args.period_s)
    return False


def move_x_to(ctx, target_x: float) -> bool:
    args = ctx.args
    client_x = ctx.client_x
    move_axis_to = ctx.move_axis_to
    return move_axis_to(
        client_x,
        target_x,
        args.x_tolerance_mm,
        args.x_move_timeout_s,
        "x",
    )


def move_y_to(ctx, target_y: float) -> bool:
    args = ctx.args
    client_y = ctx.client_y
    move_axis_to = ctx.move_axis_to
    return move_axis_to(
        client_y,
        target_y,
        args.y_tolerance_mm,
        args.y_move_timeout_s,
        "y",
    )
