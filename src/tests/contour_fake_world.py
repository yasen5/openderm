"""Shared hardware-free simulation fixtures for the contour scan test suite.

A virtual body (``FakeWorld``) whose Y extent varies with X, an optical
sensor that drops out of range off the body and swings the spot back under an rx
tilt, and fake gantry / RX-axis / Pico / controller clients that drive the real
scan ``run()`` loop against them.

These live here rather than in a TestCase module so importing them does not make
unittest re-collect a suite. ``test_scan.py`` imports them from here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# The Y soft-travel limit the fake gantry/pico clients enforce. Canonical source
# (both scan scripts import it from here too), so the fakes stay in sync with the
# real client without importing either scan script.
from capture.motion.gantry.server import (
    Y_SOFT_MAX_MM,
    Y_SOFT_MIN_MM,
)

BODY = {
    0.0: (-12.0, 18.0),
    10.0: (-12.0, 28.0),
    20.0: (-12.0, 8.0),
}
TILT_GAIN = 18.0  # mm of spot shift per rad of rx (so 20deg ~= 6.3mm reach)
Z_AT_TARGET = 308.0  # gantry z at which the standoff reads exactly target
START_Z = 300.0
TARGET_MM = 110.0
DISTANCE_MIN_MM = 65.0  # HG-C near limit: closer than this reads above_range
DISTANCE_MAX_MM = 135.0  # HG-C far limit (floor model only): farther is below_range


class FakeWorld:
    """Shared mutable pose + the body model the fake sensor reads against.

    body: per-X-position (ylo, yhi) where the surface exists.
    recede/flank_y: model a flank that drops away -- beyond |y|>flank_y the z
        needed to hit the standoff grows by `recede` mm per mm of y, so far rows
        need more z travel (used to exercise the reach-limited edge).
    z_ceiling: a hard physical z travel limit the gantry cannot exceed (used to
        exercise the EMPIRICAL z-saturation detector without --z-max-mm).
    """

    def __init__(
        self,
        body=BODY,
        recede=0.0,
        flank_y=0.0,
        z_ceiling=None,
        k_diff=25.0,
        rx_level=0.0,
        ideal_standoff_model=None,
        floor_dist0=None,
        floor_rx_gain=0.0,
        slew_rx_rad_s=None,
        slew_xyz_mm_s=None,
        slew_dt_s=0.02,
        rx_meas_offset_rad=0.0,
        diff_noise_mm=0.0,
        rx_accel_rad_s2=None,
        rx_backlash_rad=0.0,
        level_follow_per_tick=0.0,
        level_follow_max=None,
        phantom_tilt_mm_per_rad=0.0,
        rx_speed_floor_rad_s=0.0,
        rx_level_y_gain=0.0,
    ) -> None:
        # k_diff default is NON-zero: real leveling actively pulls rx back to the
        # surface normal (rx_level), and the edge probe's reach band is anchored
        # at the last LEVELED rx -- with k_diff=0 every orientation reads as
        # "normal" (d1-d2 == 0), so settled stations would launder carried probe
        # tilts into the anchor and X positions would ratchet past their edges, a
        # behaviour real leveling prevents. 25 mm/rad puts the rx deadband at
        # 0.02 rad, tight enough that the anchor tracks the true normal.
        self.pos = {"x": 0.0, "y": 0.0, "z": START_Z}
        self.rx = 0.0
        self.body = body
        self.recede = recede
        self.flank_y = flank_y
        self.z_ceiling = z_ceiling
        # Ideal-pivot leveling model: d1-d2 = k_diff * (rx - rx_level), i.e. the
        # difference reflects ONLY the head tilt vs the surface normal (the focus
        # is perfectly held), so rx levels to rx_level. k_diff=0 -> d1-d2==0.
        self.k_diff = k_diff
        self.rx_level = rx_level
        # Optional y-DEPENDENT surface normal (rad per mm of y): models the
        # cross-contour tilt profile rx_level(y). 0 = flat.
        self.rx_level_y_gain = rx_level_y_gain
        # Ideal-pivot STANDOFF model: when set, the sensor average is held ONLY if
        # the gantry z follows the full pivot arc (z = anchor_z + arc_z(rx) + trim).
        # The standoff reads TARGET + START_ERR - (pos.z - anchor_z - arc_z(rx)), so
        # the arc_z cancels iff z carried it -- the new combined move does, the old
        # x/y-only pivot did not (its standoff would diverge as rx tilts). anchor_z
        # and rx0 are the station-1 entry pose (= the initial pose here).
        self.ideal_standoff_model = ideal_standoff_model
        # Optional BED/FLOOR under the body edge: when the spot is off the body,
        # instead of always reading
        # below_range the sensor sees the bed at
        #   floor_dist0 + floor_rx_gain*(rx - rx_level) - (z - START_Z)
        # -- following z (a descending head closes on it) and rx (tilting shortens
        # the beam path, which is how the real bed entered the window mid-probe).
        # In [DISTANCE_MIN_MM, DISTANCE_MAX_MM] it reads 'ok' (in range!); farther
        # is below_range. None keeps the floor out of range.
        self.floor_dist0 = floor_dist0
        self.floor_rx_gain = floor_rx_gain
        self.anchor_z = START_Z
        self.rx0 = 0.0
        self.avg_trace: list[float] = []
        # Optional (lo, hi) y soft limits for the startup-guard test; None = the fake
        # does not enforce (the real GantryServerClient's limit is unit-tested in
        # test_gantry_server). Test coords are centered near 0, not the rig's [50,570].
        self.y_limits: tuple[float, float] | None = None
        # When True, the fake clients enforce the y soft-limit on MOVES (raising
        # GantrySoftLimitError) using Y_SOFT_MIN_MM/MAX -- tests that exercise the
        # y-limit-as-edge behavior set this AND patch those module constants to test
        # coords. (Default off so other tests, centered near 0, are unaffected.)
        self.enforce_y_limits: bool = False
        self.rx_cmd_log: list[float] = []  # every absolute rx target commanded
        self.move_log: list[tuple[str, float]] = []
        self.arc_x: list[float] = []  # x targets seen by move_xyz (pivot arc)
        # Optional first-order axis dynamics (default None = teleport). With slew
        # rates set, commands write targets and the
        # positions advance toward them by rate*slew_dt on every sensor/status
        # read -- a deterministic virtual clock, so in-flight readings exist and
        # continuous-control behavior can be tested with in-flight readings.
        self.slew_rx = slew_rx_rad_s
        self.slew_xyz = slew_xyz_mm_s
        self.slew_dt = slew_dt_s
        self.targets = dict(self.pos)
        self.rx_target = self.rx
        self.clock = 0.0
        # Constant offset on the reported rx (encoder vs command). This reproduces
        # the condition that makes a measured-base equality test
        # measured-base bound detection sensitive to encoder offset.
        self.rx_meas_offset = rx_meas_offset_rad
        # Optional second-order RX dynamics and transmission backlash. With
        # rx_accel set, the rotor
        # (self.rx -- what the rotor-side encoder reports) accelerates/brakes at
        # this rate instead of slewing at constant speed, so direction reversals
        # cost real time (the AK45's --rx-accel-rad-s2). With rx_backlash set,
        # the CAMERA (self.cam_rx -- what d1-d2 actually measures) is dragged
        # through a dead zone of this TOTAL width: on every rotor reversal the
        # camera stands still for backlash_rad of rotor travel, the rig's ~5 deg
        # loose-coupling backlash the encoder is blind to (rx-axis-backlash).
        self.rx_accel = rx_accel_rad_s2
        self.rx_backlash = rx_backlash_rad
        self.rx_vel = 0.0
        self.cam_rx = self.rx
        # Velocity (speed-loop) mode: a commanded angular rate integrated per
        # tick. None = position mode. A position command cancels it (as the
        # real RX-axis server does). rx_vel_log records every commanded rate.
        # rx_speed_floor: commanded rates below the AK45's measured stall floor
        # produce no motion.
        self.rx_vel_cmd: float | None = None
        self.rx_vel_log: list[float] = []
        self.rx_speed_floor = rx_speed_floor_rad_s
        # Optional (lo, hi, margin) window for the velocity mode's server-side
        # guard: set_velocity toward a bound the rotor is within `margin` of
        # returns held (max_window/min_window) and does NOT move -- the real
        # RX-axis server's behavior. The scan must turn held streaks into a
        # reach-limited edge.
        self.rx_vel_window: tuple[float, float, float] | None = None
        # Optional CURVED-SKIN TREADMILL (off by default): tilting rx also
        # walks the sensor spots across the surface, so the local normal
        # (rx_level) recedes AHEAD of the leveling motion -- the level shifts
        # by this FRACTION of every camera tilt increment (same direction), up
        # to level_follow_max (the curvature runs out). Effective plant slope
        # becomes k_diff*(1-fraction), which can make d(d1-d2)/d(rx) several
        # times smaller than the geometric k_diff.
        self.level_follow = level_follow_per_tick
        self.level_follow_max = level_follow_max
        self._prev_cam_rx = self.cam_rx
        # Optional PHANTOM TILT (off by default): each sensor is read at a
        # DIFFERENT pose while a swing is in flight (the two exclusive reads
        # are ~50ms apart on hardware), so each reading picks up a distance
        # bias proportional to the camera motion since the previous read --
        # and d1-d2 inherits the difference. This is motion-proportional
        # corruption in the measurement path.
        self.phantom_tilt = phantom_tilt_mm_per_rad
        self._prev_read_cam_rx = self.cam_rx
        # DETERMINISTIC pseudo-noise on the d1-d2 tilt signal (amplitude, mm):
        # amp * sin(2.1 * n) over the read counter. This reproduces raw-error
        # sign flips near equilibrium.
        self.diff_noise = diff_noise_mm
        self._nreads = 0

    @property
    def dynamic(self) -> bool:
        return self.slew_rx is not None or self.slew_xyz is not None

    def command_axis(self, axis: str, value: float) -> None:
        """Route a commanded axis value: teleport or set a dynamic target."""
        value = float(value)
        if self.dynamic:
            self.targets[axis] = value
        else:
            self.pos[axis] = value

    def command_rx(self, value: float) -> None:
        value = float(value)
        self.rx_vel_cmd = None  # a position command cancels velocity mode
        if self.dynamic:
            self.rx_target = value
        else:
            self.rx = value
            self.cam_rx = value

    def command_rx_velocity(self, value: float) -> None:
        self.rx_vel_cmd = float(value)
        self.rx_vel_log.append(float(value))

    def _drag_cam_rx(self) -> None:
        """Backlash: the camera is DRAGGED by the rotor through a dead zone of
        rx_backlash total width -- it only moves once the rotor takes up the
        slack, so every rotor reversal costs backlash_rad of dead travel."""
        if not self.rx_backlash:
            self.cam_rx = self.rx
            return
        half = self.rx_backlash / 2.0
        if self.rx - self.cam_rx > half:
            self.cam_rx = self.rx - half
        elif self.cam_rx - self.rx > half:
            self.cam_rx = self.rx + half

    def advance(self) -> None:
        """One virtual-clock tick: slew each axis toward its target. Called from
        every sensor read and every gantry/RX-axis status read."""
        if not self.dynamic:
            return
        self.clock += self.slew_dt
        if self.slew_xyz is not None:
            for a in ("x", "y", "z"):
                p, t = self.pos[a], self.targets[a]
                step = self.slew_xyz * self.slew_dt
                self.pos[a] = t if abs(t - p) <= step else p + math.copysign(step, t - p)
        if self.slew_rx is not None and self.rx_vel_cmd is not None:
            # Speed loop: integrate the commanded rate (clamped to the slew cap,
            # the fake's stand-in for the server speed limit); camera follows
            # through the backlash dead zone exactly as in position mode.
            # Sub-floor commands STALL (no motion at all), as on hardware.
            v = max(-self.slew_rx, min(self.slew_rx, self.rx_vel_cmd))
            if abs(v) < self.rx_speed_floor:
                v = 0.0
            self.rx += v * self.slew_dt
            self._drag_cam_rx()
            if self.level_follow:
                self.rx_level += self.level_follow * (self.cam_rx - self._prev_cam_rx)
                if self.level_follow_max is not None:
                    self.rx_level = min(self.rx_level, self.level_follow_max)
                self._prev_cam_rx = self.cam_rx
        elif self.slew_rx is not None:
            d = self.rx_target - self.rx
            if self.rx_accel:
                # Accel-limited rotor: chase the velocity that cruises at
                # slew_rx but brakes in time to stop at the target
                # (v = sqrt(2*a*d), the standard trapezoid decel envelope).
                v_des = (
                    0.0
                    if d == 0.0
                    else math.copysign(
                        min(self.slew_rx, math.sqrt(2.0 * self.rx_accel * abs(d))), d
                    )
                )
                dv_max = self.rx_accel * self.slew_dt
                dv = v_des - self.rx_vel
                self.rx_vel += math.copysign(min(abs(dv), dv_max), dv)
                step = self.rx_vel * self.slew_dt
                if abs(d) <= abs(step):
                    self.rx = self.rx_target
                    self.rx_vel = 0.0
                else:
                    self.rx += step
            else:
                step = self.slew_rx * self.slew_dt
                self.rx = self.rx_target if abs(d) <= step else self.rx + math.copysign(step, d)
            self._drag_cam_rx()
            if self.level_follow:
                self.rx_level += self.level_follow * (self.cam_rx - self._prev_cam_rx)
                if self.level_follow_max is not None:
                    self.rx_level = min(self.rx_level, self.level_follow_max)
                self._prev_cam_rx = self.cam_rx

    def _column_extent(self, x: float) -> tuple[float, float]:
        key = min(self.body, key=lambda k: abs(k - x))
        return self.body[key]

    def reading(self) -> tuple[float | None, float, str]:
        # Each sensor read is one tick of the virtual clock (dynamics mode).
        self.advance()
        # The spot shifts in y with the rx tilt; both sensors share the spot for
        # the in/out-of-range test, then split by the d1-d2 tilt model. All the
        # camera-physical terms use cam_rx (== rx unless backlash is enabled):
        # the sensors ride the CAMERA, not the rotor encoder.
        spot_y = self.pos["y"] + TILT_GAIN * self.cam_rx
        ylo, yhi = self._column_extent(self.pos["x"])
        if ylo <= spot_y <= yhi:
            if self.ideal_standoff_model is not None:
                arc_z = (
                    self.ideal_standoff_model.pivot(
                        {"x": 0.0, "y": 0.0, "z": self.anchor_z}, self.cam_rx, self.rx0
                    )["z"]
                    - self.anchor_z
                )
                avg = TARGET_MM + (Z_AT_TARGET - START_Z) - (self.pos["z"] - self.anchor_z - arc_z)
            else:
                # z needed to reach standoff grows on the receding flank.
                z_needed = Z_AT_TARGET + self.recede * max(0.0, abs(spot_y) - self.flank_y)
                avg = TARGET_MM + (z_needed - self.pos["z"])
            level_eff = self.rx_level + self.rx_level_y_gain * self.pos["y"]
            diff = self.k_diff * (self.cam_rx - level_eff)
            if self.diff_noise:
                self._nreads += 1
                diff += self.diff_noise * math.sin(2.1 * self._nreads)
            if self.phantom_tilt:
                # In-flight measurement corruption: this read's distance is
                # biased by the camera motion since the PREVIOUS sensor read
                # (the reads are not simultaneous). avg carries it too, but
                # d1-d2 is the signal the leveler chases.
                diff += self.phantom_tilt * (self.cam_rx - self._prev_read_cam_rx)
                self._prev_read_cam_rx = self.cam_rx
            self.avg_trace.append(avg)  # records the real standoff, incl. crashes
            # Near-side physical limit: driving the head NEARER than the close limit
            # reads above_range (crash-imminent). The far side is left in-range so the
            # recede tests model z-unreachability via avg drift, not sensor dropout.
            # (The asymmetric split-sensor case is covered by the _classify_standoff
            # unit tests, which exercise that decision directly.)
            if avg < DISTANCE_MIN_MM:
                return None, 0.0, "above_range"
            return avg, diff, "ok"
        # Spot off the body: optionally the bed is visible (see floor_dist0 above)
        # -- IN RANGE when close enough, exactly the false-recovery hardware case.
        if self.floor_dist0 is not None:
            cosr = math.cos(self.cam_rx)
            if cosr > 0.05:  # a near-horizontal beam cannot see the bed at all
                # BEAM OBLIQUITY: the bed lives at a fixed VERTICAL depth, so a
                # beam tilted by rx reads the slant path (vertical)/cos(rx) --
                # the physics the scan's projected floor test (z + d*cos(rx))
                # inverts. floor_rx_gain remains an empirical-leak
                # knob on the vertical depth itself.
                d = (
                    self.floor_dist0
                    + self.floor_rx_gain * (self.cam_rx - self.rx_level)
                    - (self.pos["z"] - START_Z)
                ) / cosr
                if d < DISTANCE_MIN_MM:
                    return None, 0.0, "above_range"
                if d <= DISTANCE_MAX_MM:
                    return d, 0.0, "ok"
        return None, 0.0, "below_range"  # spot off the body (floor): too far


@dataclass
class FakeReading:
    name: str
    distance_mm: float | None
    in_range: bool
    signal_status: str


class FakeGantryClient:
    def __init__(self, world: FakeWorld, axis: str) -> None:
        self.world = world
        self.axis = axis
        # Mirrors the real client attribute the scan's pre-check reads.
        self.enforce_y_limits = world.enforce_y_limits

    def status(self) -> dict:
        self.world.advance()
        return {
            "homed_axes": ["x", "y", "z"],
            "position": dict(self.world.pos),
        }

    def _guard_y(self, value: float) -> None:
        # Mirror GantryServerClient: refuse a y target outside the (patched) module
        # limits when enforcing, so the y-limit-as-edge path is exercised in the sim.
        if self.enforce_y_limits and not (Y_SOFT_MIN_MM <= value <= Y_SOFT_MAX_MM):
            from capture.motion.gantry.server import GantrySoftLimitError

            raise GantrySoftLimitError(
                f"refusing to move y to {value:.2f}mm: outside the soft travel limit "
                f"[{Y_SOFT_MIN_MM:.0f}, {Y_SOFT_MAX_MM:.0f}]mm."
            )

    def _apply(self, target: float) -> None:
        # Enforce a hard physical z ceiling if the world has one (the axis simply
        # cannot go past it), which is what the empirical z-stall detector sees.
        if self.axis == "z" and self.world.z_ceiling is not None:
            target = min(target, self.world.z_ceiling)
        self.world.command_axis(self.axis, target)

    def move_to(self, target, *, feed_mm_min=None, tolerance_mm=None, blocking=True):
        if self.axis == "y":
            self._guard_y(float(target))
        self._apply(float(target))
        self.world.move_log.append((self.axis, float(target)))
        return {"status": "completed", "error": None}

    def move_xyz(self, target, *, feed_mm_min=None, tolerance_mm=None, blocking=True):
        if "y" in target:
            self._guard_y(float(target["y"]))
        for axis, v in target.items():
            v = float(v)
            # The z axis can't pass a hard physical ceiling (z_ceiling), same as
            # move_to -- the coordinated move must honour it too.
            if axis == "z" and self.world.z_ceiling is not None:
                v = min(v, self.world.z_ceiling)
            self.world.command_axis(axis, v)
        if "x" in target:
            self.world.arc_x.append(float(target["x"]))
        if "y" in target:
            self.world.move_log.append(("y", float(target["y"])))
        return {"status": "completed", "error": None}

    # Stream API (unused in blocking-mode tests, present for completeness).
    def stream_start(self, **kwargs):  # pragma: no cover
        pass

    def stream_stop(self):  # pragma: no cover
        pass

    def stream_to(self, target):  # pragma: no cover
        if self.axis == "y":
            self._guard_y(float(target))
        self._apply(float(target))

    def stop(self, mode="soft"):  # pragma: no cover
        pass


class FakePicoAxisClient:
    """Stand-in for capture.motion.pico.adapter.PicoAxisClient, backed by the
    SAME FakeWorld as FakeGantryClient. It mirrors the adapter subset the contour scan
    uses (status/move_to/stream_to/home/stop/drain_events)."""

    def __init__(self, world: FakeWorld, axis: str) -> None:
        self.world = world
        self.axis = axis
        # Mirror FakeGantryClient: the real Pico Y client enforces [50, 570]mm, but the
        # sim runs y near 0, so honour world.enforce_y_limits (False unless a test opts
        # in) -- otherwise every near-0 y read would falsely trip the soft-limit guard.
        self.enforce_y_limits = world.enforce_y_limits
        self.homed = True
        self.continuous_calls = 0  # stream_to(..., continuous=True) count (smooth return)

    def status(self) -> dict:
        self.world.advance()
        return {
            "position": {self.axis: float(self.world.pos[self.axis])},
            "homed_axes": [self.axis] if self.homed else [],
            "moving": False,
        }

    def _guard_y(self, value: float) -> None:
        if (
            self.axis == "y"
            and self.enforce_y_limits
            and not (Y_SOFT_MIN_MM <= value <= Y_SOFT_MAX_MM)
        ):
            from capture.motion.gantry.server import GantrySoftLimitError

            raise GantrySoftLimitError(
                f"refusing to move y to {value:.2f}mm: outside the soft travel limit "
                f"[{Y_SOFT_MIN_MM:.0f}, {Y_SOFT_MAX_MM:.0f}]mm."
            )

    def _apply(self, target: float) -> None:
        target = float(target)
        if self.axis == "z" and self.world.z_ceiling is not None:
            target = min(target, self.world.z_ceiling)
        self.world.command_axis(self.axis, target)

    def move_to(self, target, *, feed_mm_min=None, tolerance_mm=None) -> dict:
        self._guard_y(float(target))
        self._apply(target)
        self.world.move_log.append((self.axis, float(self.world.pos[self.axis])))
        return {"status": "completed"}

    def stream_to(self, target, continuous: bool = False) -> dict:
        self._guard_y(float(target))
        self._apply(target)
        if continuous:
            self.continuous_calls += 1
        return {"status": "queued"}

    def home(self, fast_mm_s=None, slow_mm_s=None) -> dict:
        self.world.pos[self.axis] = 0.0
        self.homed = True
        return {"status": "homed"}

    def drain_events(self) -> list:
        return []

    def stop(self, mode: str = "soft") -> dict:
        return {"status": "stopped"}

    def close(self) -> None:
        pass


class FakeRxAxisClient:
    def __init__(self, world: FakeWorld) -> None:
        self.world = world

    def status(self) -> dict:
        self.world.advance()
        return {"position_rad": self.world.rx + self.world.rx_meas_offset}

    def move_to(self, position_rad, *, speed_rad_s=None, accel_rad_s2=None):
        self.world.command_rx(position_rad)
        self.world.rx_cmd_log.append(float(position_rad))
        return {"status": "completed"}

    def set_velocity(self, velocity_rad_s):
        v = float(velocity_rad_s)
        win = self.world.rx_vel_window
        if win is not None and v != 0.0:
            lo, hi, margin = win
            if v > 0 and self.world.rx >= hi - margin:
                self.world.command_rx_velocity(0.0)
                return {"velocity_command": {"velocity_rad_s": 0.0}, "held": "max_window"}
            if v < 0 and self.world.rx <= lo + margin:
                self.world.command_rx_velocity(0.0)
                return {"velocity_command": {"velocity_rad_s": 0.0}, "held": "min_window"}
        self.world.command_rx_velocity(v)
        return {"velocity_command": {"velocity_rad_s": v}}

    def move_by(self, delta_rad, *, speed_rad_s=None, accel_rad_s2=None):  # pragma: no cover
        self.world.rx += float(delta_rad)
        return {"status": "completed"}

    def stop(self):  # pragma: no cover
        return {"status": "ok"}


class FakeController:
    def __init__(self, world: FakeWorld) -> None:
        self.world = world
        # (name-or-selection, enabled) in call order -- lets tests assert the
        # laser choreography (exclusive dance vs both-on with photo blackouts).
        self.enable_log: list[tuple[str, bool]] = []

    def set_enabled(self, name, enabled, *, settle=True):
        self.enable_log.append((name, bool(enabled)))

    def set_enabled_for_selection(self, selection, enabled):
        self.enable_log.append((selection, bool(enabled)))

    def read_sensor(self, name, *, samples=None) -> FakeReading:
        avg, diff, status = self.world.reading()
        if avg is None:
            return FakeReading(name=name, distance_mm=None, in_range=False, signal_status=status)
        dist = avg + (diff / 2.0 if name == "sensor1" else -diff / 2.0)
        return FakeReading(name=name, distance_mm=dist, in_range=True, signal_status=status)

    def close(self):
        pass
