"""Control-math and I/O helpers for contour scanning."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _classify_standoff(r1, r2, target_mm: float, deadband_mm: float, retreat_only: bool):
    """Pure per-iteration decision for settle_z_standoff. Returns (action, error):

      ('retreat', None) -- crash-imminent: EITHER sensor reads 'above_range' (nearer
                           than the close limit). Checked FIRST, before averaging, so
                           a split reading (one spot too near on a steep local rise,
                           the other still on farther skin) cannot be averaged away
                           and driven into the surface. Caller backs z off.
      ('far', None)     -- no sensor in range and none too near (below_range only):
                           benign, nothing to do.
      ('hold', error)   -- in range and within the deadband.
      ('skip', error)   -- retreat_only and the only correction would ADVANCE the head
                           toward the surface (error>0 == surface farther than target):
                           never approach on the way home.
      ('step', error)   -- in range, off target: caller applies step_z(clamp(gain*error)).

    Sign: increasing z moves the head CLOSER (reading decreases), so error<0 (too near)
    yields a negative step (retreat) and error>0 (too far) a positive step (approach)."""
    if any(r.signal_status == "above_range" for r in (r1, r2)):
        return ("retreat", None)
    in_range = [r for r in (r1, r2) if r.in_range and r.distance_mm is not None]
    if not in_range:
        return ("far", None)
    error = sum(r.distance_mm for r in in_range) / len(in_range) - target_mm
    if abs(error) <= deadband_mm:
        return ("hold", error)
    if retreat_only and error > 0:
        return ("skip", error)
    return ("step", error)


def _fmt_bound(value: float | None) -> str:
    return "unset" if value is None else f"{value:.3f}"


def _read_startup_collision_pose(
    collision_guard,
    read_position,
    read_rx_rad,
) -> tuple[dict[str, float], float, float]:
    """Read and validate the full startup pose for the fail-closed guard."""
    try:
        raw_position = read_position()
        raw_rx = read_rx_rad()
        if not isinstance(raw_position, dict):
            raise ValueError("linear-axis position is unavailable")
        missing = [axis for axis in ("x", "y", "z") if raw_position.get(axis) is None]
        if missing:
            raise ValueError("missing position for " + ", ".join(missing))
        if raw_rx is None:
            raise ValueError("RX angle is unavailable")
        position = {axis: float(raw_position[axis]) for axis in ("x", "y", "z")}
        rx = float(raw_rx)
        if not all(math.isfinite(value) for value in (*position.values(), rx)):
            raise ValueError("pose contains a non-finite value")
        clearance = collision_guard.min_clearance(
            position["x"],
            position["y"],
            position["z"],
            rx,
        )
    except Exception as exc:
        raise RuntimeError(f"collision startup pose is unreadable: {exc}") from exc
    return position, rx, clearance


def _move_completed(record: dict, label: str) -> bool:
    """Check a blocking move's record. The gantry server returns HTTP 200 even
    when the job FAILED server-side (status='failed', e.g. error='timeout' when
    its done-gate stalls), so the client must inspect the record; otherwise
    failed moves pass silently and the head pose is not where the controller
    thinks. Non-fatal: warn and let the regulator re-correct."""
    status = record.get("status")
    if status == "completed":
        return True
    print(
        f"  warning: {label} move {status!r}: error={record.get('error')!r}",
        file=sys.stderr,
    )
    return False


def _next_z_cmd(
    cmd: float | None,
    bad: int,
    z_meas: float,
    delta_mm: float,
    trust_band_mm: float,
    max_bad: int,
) -> tuple[float, int, bool]:
    """Stream-mode z accumulator step (pure; unit-tested in the simulation).

    Returns (new_cmd, new_bad, hold). The commanded z setpoint is OUR running
    value, advanced by the controller delta (which is driven by the optical
    sensor error / search step) -- it is NEVER snapped to a raw read. The live
    read is used only to TRACK the real axis when it AGREES with the command
    (|z_meas-cmd| <= trust_band: self-correcting, no windup, bounded overshoot).
    A read that disagrees is the Klipper trapq-drain glitch (live_position ~0
    while the gantry is at ~300mm) and is IGNORED -- we advance from cmd instead,
    so a corrupt read can never become a setpoint and cause a plunge. After
    `max_bad` disagreeing reads in a row we stop advancing (hold=True -> caller
    re-sends cmd) to bound open-loop drift. cmd=None seeds from z_meas.
    """
    if cmd is None or abs(z_meas - cmd) <= trust_band_mm:
        return z_meas + delta_mm, 0, False  # trust the live axis
    bad += 1
    if bad > max_bad:
        return cmd, bad, True  # sustained mismatch -> hold
    return cmd + delta_mm, bad, False  # ignore glitch, advance from cmd


def _edge_side_from_rx(
    rx_rad: float | None, axis_min_rad: float, axis_max_rad: float
) -> int | None:
    """Which body FLANK an edge/reach-limit belongs to, from the camera tilt
    (pure; unit-tested). The tilt tracks the surface normal around a curved subject,
    so the settled rx encodes the side being viewed: near the axis FLOOR the
    +y flank and near the CEILING the -y flank. Split at the window
    midpoint; returns +1 (+y side), -1 (-y side), or None when rx is unknown.

    Used so a band prunes an x only when the edge belongs to the flank the
    march is actually walking toward: an rx-floor reach-limit hit during a -y
    march is the +y flank's limit, not this march's edge -- pruning it there
    can carve a coverage gap out of every later band.

    A tilt within 10% of the window half-width of the midpoint is AMBIGUOUS
    (returns None -> callers fall back to the march-direction heuristic): the
    side signal comes from the tilt sitting clearly toward one bound, and at
    mid-window the classification would be sign noise."""
    if rx_rad is None:
        return None
    mid = 0.5 * (axis_min_rad + axis_max_rad)
    dead = 0.1 * 0.5 * (axis_max_rad - axis_min_rad)
    if abs(rx_rad - mid) <= dead:
        return None
    return 1 if rx_rad < mid else -1


def _pivot_rate_budget(
    dx_per_rad: float,
    dy_per_rad: float,
    dz_per_rad: float,
    vx_mm_s: float,
    vy_mm_s: float,
    vz_mm_s: float,
    margin: float,
) -> float | None:
    """Max rx angular rate (rad/s) the gantry can FOLLOW while holding the
    pivot focus (pure; unit-tested). Each rad/s of rx drags the head along the
    pivot arc at |d(axis)/d(rx)| mm/s per axis; the sustainable rate is the
    tightest axis's vmax over its lever, derated by margin. Commanding faster
    than this makes the focus hold kinematically impossible -- the spots sweep
    the skin, the measured tilt error becomes lag artifact, and the loop feeds
    its own error. Returns None when no axis
    constrains (all levers ~0)."""
    budget = None
    for lever, vmax in ((dx_per_rad, vx_mm_s), (dy_per_rad, vy_mm_s), (dz_per_rad, vz_mm_s)):
        lv = abs(lever)
        if lv < 1e-6 or vmax <= 0:
            continue
        b = margin * vmax / lv
        budget = b if budget is None else min(budget, b)
    return budget


def _cruise_batch(
    prev_sent: dict[str, float],
    targets_mm: dict[str, float],
    min_step_mm: float,
) -> bool:
    """Whether a batch of streamed Pico setpoints should CRUISE (MOVEC/MOVEM C)
    instead of stop-at-target (pure; unit-tested).

    Cruising blends the ~10Hz regulated retargets into one continuous motion --
    but ONLY when the axis will still be in flight when the next setpoint
    lands: a cruise waypoint that completes with no fresh target hard-stops
    from speed (firmware EVT UNDERRUN, step-skip risk; see
    src/pico/gantry_firmware.py).
    So a batch cruises only when EVERY axis in it moves at least min_step_mm
    from its PREVIOUS streamed setpoint (at 150mm/s2 a 1mm move is still
    accelerating ~115ms later, longer than a loop period). An axis with no
    recorded previous setpoint (station entry, or motion through a
    non-streamed path) disqualifies the batch: its true in-flight distance is
    unknown, and a plain MOVE is always safe."""
    if not targets_mm:
        return False
    for axis, mm in targets_mm.items():
        prev = prev_sent.get(axis)
        if prev is None or abs(mm - prev) < min_step_mm:
            return False
    return True


def _traverse_arc_delta(
    pivot_model, pose: dict, rx_cur: float, rx_z_ref: float
) -> tuple[float, float]:
    """Pivot-arc displacement OWED by a grid traverse because the carried rx moved
    since z was last sensor-established (pure; unit-tested). Returns
    (dz_arc, lateral_mm):
      dz_arc      -- z component of the arc from rx_z_ref to rx_cur. NEGATIVE
                     means the head must RISE to stay on the focus orbit -- the
                     stale-anchor hazard: a probe can change rx while leaving z
                     untouched, making the next grid move unsafe if this rise
                     is discarded.
      lateral_mm  -- the x/y swing magnitude of that same arc: how far the
                     traverse jumps laterally BECAUSE of the rx change (the grid
                     step itself is not included)."""
    arc = pivot_model.pivot(pose, rx_cur, rx_z_ref)
    dz_arc = arc.get("z", pose["z"]) - pose["z"]
    lateral = max(
        abs(arc.get("x", pose["x"]) - pose["x"]),
        abs(arc.get("y", pose["y"]) - pose["y"]),
    )
    return dz_arc, lateral


def _load_floor_model(path: Path) -> dict:
    """Load a src/scripts/calibration/floor_depth_tare.py JSON. Returns
    {'const', 'coeffs', 'rx_min', 'rx_max'}: 'coeffs' is the [c0, c1, c2] of the
    rx-sweep fit z+d = c0 + c1*rx + c2*rx^2 valid over [rx_min, rx_max], or None
    for a single-point tare (constant threshold 'const')."""
    data = json.loads(path.read_text())
    model = {
        "const": float(data["floor_depth_mm"]),
        "coeffs": None,
        "rx_min": 0.0,
        "rx_max": 0.0,
    }
    fit = data.get("fit")
    if fit and fit.get("coeffs") is not None:
        coeffs = [float(v) for v in fit["coeffs"]]
        if len(coeffs) != 3:
            raise ValueError(f"fit.coeffs must have 3 entries, got {len(coeffs)}")
        rx_min, rx_max = float(fit["rx_min"]), float(fit["rx_max"])
        if rx_max <= rx_min:
            raise ValueError("fit.rx_max must exceed fit.rx_min")
        model.update(coeffs=coeffs, rx_min=rx_min, rx_max=rx_max)
    return model
