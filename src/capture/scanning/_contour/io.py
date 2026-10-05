"""Tracing, pose reads, sensor sampling, and capture snapshots."""

from __future__ import annotations

import dataclasses
import json
import math
import sys
import time

from capture.motion.gantry.server import (
    GantryServerClient,
    GantryServerError,
)
from capture.motion.rx_axis.server import RxAxisServerError

from ..config import *  # noqa: F401,F403


def record_sample(ctx, event: str, *, target=None, **extra) -> None:
    """Append one timestamped pose sample to the --record trace (no-op when off).
    Reads only the cached pose_last -- never the hardware -- so it is safe to call
    from inside the control loops. ``event`` names what produced the sample
    (read:pose / read:z / cmd:rx / capture / ...); ``target`` optionally carries a
    commanded setpoint; ``extra`` adds any further fields."""
    _rec = ctx._rec
    pose_last = ctx.pose_last
    rec_ctx = ctx.rec_ctx
    record_on = ctx.record_on
    record_path = ctx.record_path
    if not record_on or _rec.get("disabled"):
        return
    now = time.monotonic()
    if _rec["t0"] is None:
        _rec["t0"] = now
    if _rec["fh"] is None:
        try:
            record_path.parent.mkdir(parents=True, exist_ok=True)
            _rec["fh"] = record_path.open("w")
        except OSError as exc:
            print(
                f"  warning: could not open --record trace {record_path}: {exc}",
                file=sys.stderr,
            )
            _rec["disabled"] = True
            return
    sample = {
        "seq": _rec["n"],
        "t_s": round(now - _rec["t0"], 4),  # seconds since the first sample
        "t_wall": round(time.time(), 4),  # epoch seconds
        "event": event,
        "activity": rec_ctx["activity"],
        "station": rec_ctx["station"],
        "phase": rec_ctx["phase"],
        "col": rec_ctx["col"],
        "x_mm": pose_last["x"],
        "y_mm": pose_last["y"],
        "z_mm": pose_last["z"],
        "rx_rad": pose_last["rx"],
    }
    if target is not None:
        sample["target"] = target
    if extra:
        sample.update(extra)
    try:
        _rec["fh"].write(json.dumps(sample) + "\n")
    # Swallow EVERYTHING (not just OSError): a bad write or a stray
    # non-serializable value must never crash the control loops the recorder
    # is only observing. Passivity beats a complete trace.
    except Exception as exc:
        print(f"  warning: --record trace write failed: {exc}", file=sys.stderr)
    _rec["n"] += 1


def _close_trace(
    ctx,
) -> None:
    """Flush and close the trace file if open. Idempotent -- safe to call from an
    early abort (e.g. the self-collision startup gate returns before the scan's
    try/finally) AND again from that finally."""
    _rec = ctx._rec
    record_path = ctx.record_path
    fh = _rec["fh"]
    if fh is None:
        return
    _rec["fh"] = None
    try:
        fh.close()
    except Exception:
        pass
    print(f"scan trace ({_rec['n']} samples) -> {record_path}")


def read_axis(ctx, client: GantryServerClient) -> float | None:
    pose_last = ctx.pose_last
    record_sample = ctx.record_sample
    z_last = ctx.z_last
    try:
        snapshot = client.status()
    except GantryServerError as exc:
        print(f"warning: {client.axis} position read failed: {exc}", file=sys.stderr)
        return None
    position = snapshot.get("position") or {}
    value = position.get(client.axis)
    if value is None:
        return None
    if client.axis == "z":
        z_last["mm"] = float(value)
    pose_last[client.axis] = float(value)
    record_sample(f"read:{client.axis}")
    return float(value)


def read_position(
    ctx,
) -> dict[str, float]:
    """Return the complete pose from Klipper X plus Pico Y/Z state."""
    client_x = ctx.client_x
    client_y = ctx.client_y
    client_z = ctx.client_z
    pico_axes = ctx.pico_axes
    pico_multi = ctx.pico_multi
    pose_last = ctx.pose_last
    record_sample = ctx.record_sample
    try:
        snapshot = client_x.status()
    except GantryServerError as exc:
        print(f"warning: gantry position read failed: {exc}", file=sys.stderr)
        return {}
    pos = dict(snapshot.get("position") or {})
    if pico_multi is not None:
        try:
            pos.update(pico_multi.positions())  # all Pico axes in ONE round trip
        except GantryServerError as exc:
            print(f"warning: pico position read failed: {exc}", file=sys.stderr)
            for axis in pico_axes:
                pos.pop(axis, None)
    else:
        for axis in pico_axes:
            cl = client_y if axis == "y" else client_z
            try:
                pos[axis] = float(cl.status()["position"][axis])
            except (GantryServerError, KeyError, TypeError):
                pos.pop(axis, None)
    for _axis in ("x", "y", "z"):
        if pos.get(_axis) is not None:
            pose_last[_axis] = float(pos[_axis])
    if any(pos.get(a) is not None for a in ("x", "y", "z")):
        record_sample("read:pose")
    return pos


def read_rx_rad(
    ctx,
) -> float | None:
    """Measured RX angle in rad from the RX-axis server, or None if unavailable."""
    pose_last = ctx.pose_last
    record_sample = ctx.record_sample
    rx_client = ctx.rx_client
    rx_last = ctx.rx_last
    if rx_client is None:
        return None
    try:
        snapshot = rx_client.status()
    except RxAxisServerError as exc:
        print(f"warning: rx position read failed: {exc}", file=sys.stderr)
        return None
    value = snapshot.get("position_rad")
    if value is None:
        return None
    rx_last["rad"] = float(value)
    pose_last["rx"] = float(value)
    record_sample("read:rx")
    return float(value)


def _floor_baseline_z(
    ctx,
) -> float | None:
    """Best current z for the floor filter: the stream accumulator (the
    commanded setpoint -- glitch-protected, and it LEADS the axis during a
    descent, so the filter errs toward calling floor EARLIER: safe), else the
    last trusted z read."""
    Z_TRUST_BAND_MM = ctx.Z_TRUST_BAND_MM
    z_last = ctx.z_last
    z_stream = ctx.z_stream
    if z_stream["cmd"] is not None:
        cmd = float(z_stream["cmd"])
        meas = z_last["mm"]
        # Conservative in BOTH motion directions: during a descent the
        # commanded z LEADS (max = cmd -> rejects earlier); during a retreat
        # the measured z is the deeper one. Trust-band-guarded so a trapq
        # glitch read cannot inflate the baseline.
        if meas is not None and abs(cmd - float(meas)) <= Z_TRUST_BAND_MM:
            return max(cmd, float(meas))
        return cmd
    return z_last["mm"]


def _floor_threshold_mm(
    ctx,
) -> float | None:
    """Bed z+d threshold base at the CURRENT tilt: the tare's rx-sweep fit
    evaluated at the last-read rx -- clamped into the fitted range, which errs
    toward rejection (steeper-than-fitted tilts read the bed even DEEPER, so a
    clamped threshold sits below the true bed) -- the model's constant when
    there is no fit or rx is unknown, else the constant --floor-depth-mm.
    None = floor rejection off."""
    args = ctx.args
    floor_model = ctx.floor_model
    rx_last = ctx.rx_last
    if floor_model is not None:
        coeffs = floor_model["coeffs"]
        rx = rx_last["rad"]
        if coeffs is not None and rx is not None:
            r = min(max(rx, floor_model["rx_min"]), floor_model["rx_max"])
            return coeffs[0] + coeffs[1] * r + coeffs[2] * r * r
        return floor_model["const"]
    return args.floor_depth_mm


def _reject_floor(ctx, reading):
    """Reclassify a reading OF THE BED as out-of-range (signal_status
    'floor'). The bed is at a fixed machine height, so gantry z + distance is
    ~constant for bed hits (~--floor-depth-mm) and always LARGER than for
    skin, which sits above the bed. Applied at the read chokepoint so EVERY
    consumer -- the rx leveler (which otherwise normals itself to the bed
    when both spots land on it past a body edge: the observed wrong-way
    rotation), the standoff loop, the edge probe's recovery test and the
    capture gate -- treats bed hits as 'surface gone', which is what they
    are. Fail-open: filter off, reading already out of range, or no z
    baseline -> the reading passes through untouched."""
    _floor_baseline_z = ctx._floor_baseline_z
    _floor_threshold_mm = ctx._floor_threshold_mm
    args = ctx.args
    floor_model = ctx.floor_model
    floor_suspect = ctx.floor_suspect
    rx_last = ctx.rx_last
    if not reading.in_range or reading.distance_mm is None:
        return reading
    depth = _floor_threshold_mm()
    if depth is None:
        return reading
    z = _floor_baseline_z()
    if z is None:
        return reading
    # Project the beam onto the VERTICAL before the depth test. The bed
    # lives at a fixed machine DEPTH, and "z + d" equals that depth only
    # for a beam pointing straight down: a genuine bed hit at tilt rx has
    # path length D/cos(rx), so z + d*cos(rx) recovers the true depth at
    # ANY angle -- while a sideways skin reading contributes ~nothing
    # vertically and can never trip the test. Without the projection, the
    # newly reachable near-horizontal tilts rejected a whole band of real
    # skin (shin-top-1: z 378.7 deep + sideways d ~122 summed to 500 >=
    # the 497 line; at rx=100 deg the beam cannot even see the bed).
    # Applies to the STATIC threshold; a --floor-model fit is already an
    # empirical function of rx in raw z+d terms and keeps the raw sum.
    d_eff = reading.distance_mm
    if floor_model is None:
        _rxv = rx_last["rad"]
        if _rxv is not None:
            d_eff = reading.distance_mm * math.cos(_rxv)
    if z + d_eff >= depth - args.floor_margin_mm:
        if args.debug:
            print(
                f"  [dbg] {reading.name}: z+d_vert = {z:.1f}+{d_eff:.1f}"
                f" = {z + d_eff:.1f}mm >= floor threshold "
                f"{depth:.0f}-{args.floor_margin_mm:.0f}mm -> FLOOR"
            )
        return dataclasses.replace(reading, in_range=False, signal_status="floor")
    # FLOOR-SUSPECT: the reading passed, but sits within a small band under the
    # reject line. With the STATIC --floor-depth-mm threshold the bed's apparent
    # z+d follows the tilt (beam obliquity), so at some rx the bed leaks a few
    # mm below the threshold and reads as 'body' -- stale-anchor
    # accepted bed reads at z+d 494-496mm against a 497mm line, tracked the
    # floor, and drove z to bed level. Count (and warn once) so the operator
    # learns the threshold is marginal; --floor-model makes it rx-dependent.
    if z + d_eff >= depth - args.floor_margin_mm - FLOOR_SUSPECT_BAND_MM:
        floor_suspect["n"] += 1
        if not floor_suspect["warned"]:
            floor_suspect["warned"] = True
            print(
                f"  warning: {reading.name} accepted at z+d_vert = "
                f"{z + d_eff:.1f}mm, within "
                f"{FLOOR_SUSPECT_BAND_MM:.0f}mm of the floor-reject threshold "
                f"({depth - args.floor_margin_mm:.0f}mm). If this is the bed "
                "leaking through (its apparent depth follows the tilt), the "
                "head will normalize to the FLOOR: use --floor-model (rx-swept "
                "tare) or widen --floor-margin-mm.",
                file=sys.stderr,
            )
    return reading


def read_exclusive(ctx, sensor_name: str):
    """Light a single sensor, read it, and switch it back off so the two
    laser spots never interfere (the enable waits settle_time_s). No settle
    on disable: nothing reads while both lasers are off, and the NEXT
    sensor's enable settle already covers this one's turn-off -- the
    disable-side sleep was 40ms/iteration of pure dead time. Readings AT THE
    BED's absolute depth are reclassified out-of-range here (_reject_floor),
    so no controller ever mistakes the bed for skin.

    --simultaneous-sensors (EXPERIMENTAL): both lasers stay ON and the read
    skips the enable-settle dance entirely -- the two settles are most of
    the control period, so this roughly doubles the loop rate. The spots
    MAY optically cross-bias the analog readings (the interference the
    exclusive dance exists to prevent); captures still darken both lasers
    (capture_station resets lasers_both_on)."""
    _reject_floor = ctx._reject_floor
    args = ctx.args
    controller = ctx.controller
    lasers_both_on = ctx.lasers_both_on
    if args.simultaneous_sensors:
        if not lasers_both_on["on"]:
            controller.set_enabled("sensor1", True, settle=False)
            controller.set_enabled("sensor2", True)  # ONE settle covers both
            lasers_both_on["on"] = True
        return _reject_floor(controller.read_sensor(sensor_name, samples=args.samples))
    controller.set_enabled(sensor_name, True)
    try:
        return _reject_floor(controller.read_sensor(sensor_name, samples=args.samples))
    finally:
        controller.set_enabled(sensor_name, False, settle=False)


def snapshot_for_capture(ctx):
    """Pre-capture sensor snapshot, optionally breath-gated.

    Without --capture-gate-mm, one read is taken from each sensor. With it,
    keep re-reading (head held still) until the
    average standoff is within the gate of --target-mm, so the shutter
    fires at the same surface phase the settle gate verified -- on a live
    subject the surface breathes +/-1-2mm, and the seconds between settling
    and the shutter are otherwise spent drifting off the verified pose.
    Gives up after --capture-gate-timeout-s (about one breath cycle) and
    captures anyway, flagged in the metadata.

    Returns (s1, s2, gated, wait_s): ``gated`` is None when the gate is
    disabled, True when the reading passed, False on timeout. The returned
    readings are the ones that passed (or the last attempt), so the sidecar
    metadata describes the actual capture pose."""
    args = ctx.args
    read_exclusive = ctx.read_exclusive
    stop_state = ctx.stop_state
    t0 = time.monotonic()
    while True:
        s1 = read_exclusive("sensor1")
        s2 = read_exclusive("sensor2")
        if args.capture_gate_mm is None:
            return s1, s2, None, time.monotonic() - t0
        in_range = [r for r in (s1, s2) if r.in_range and r.distance_mm is not None]
        if in_range:
            avg = sum(r.distance_mm for r in in_range) / len(in_range)
            if abs(avg - args.target_mm) <= args.capture_gate_mm:
                return s1, s2, True, time.monotonic() - t0
        if stop_state["requested"] or time.monotonic() - t0 >= args.capture_gate_timeout_s:
            print(
                f"  warning: capture gate not satisfied within "
                f"{args.capture_gate_timeout_s:.1f}s; capturing anyway.",
                file=sys.stderr,
            )
            return s1, s2, False, time.monotonic() - t0
        time.sleep(args.period_s)
