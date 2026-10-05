#!/usr/bin/env python3
"""Off-hardware simulation of the Pico gantry firmware.

Run as a STANDALONE process (src/tests/test_gantry_firmware.py invokes it via subprocess) so the
MicroPython module stubs (machine / rp2 / uselect / time) it installs can't contaminate the
rest of the test suite. Drives Axis.emit_block() block-by-block with the PIO FIFO drained each
tick (steady state: the PIO consumes one block per slice) and records the (pos_mm, speed_mm/s)
trajectory, then asserts the cruise (MOVEC look-ahead) behaviour and that every safety clamp
still holds. Prints "ALL OK" and exits 0 on success, else raises (non-zero exit).
"""

import sys
import types
from pathlib import Path

import time as _rt

# --- MicroPython module stubs (this is a throwaway subprocess) -----------------------------
_machine = types.ModuleType("machine")


class _Pin:
    OUT = 1
    IN = 0
    PULL_UP = 2

    def __init__(self, pin, mode=None, pull=None, value=0):
        self._v = value

    def value(self, v=None):
        if v is None:
            return self._v
        self._v = v


_machine.Pin = _Pin
sys.modules["machine"] = _machine

_rp2 = types.ModuleType("rp2")


def _asm_pio(*a, **k):
    def deco(fn):
        return fn

    return deco


class _PIO:
    OUT_LOW = 0


class _SM:
    def __init__(self, sm_id, prog, freq=None, set_base=None):
        self.q = []

    def active(self, v=None):
        return 1 if v is None else None

    def tx_fifo(self):
        return len(self.q)

    def put(self, w):
        self.q.append(w)


_rp2.asm_pio = _asm_pio
_rp2.PIO = _PIO
_rp2.StateMachine = _SM
sys.modules["rp2"] = _rp2

_uselect = types.ModuleType("uselect")
_uselect.POLLIN = 1


class _Poll:
    def register(self, *a, **k):
        pass

    def poll(self, t=0):
        return []


_uselect.poll = lambda: _Poll()
sys.modules["uselect"] = _uselect

_time = types.ModuleType("time")
_time.sleep_us = lambda n: None
_time.sleep_ms = lambda n: None
_time.sleep = lambda n: None
_time.ticks_ms = lambda: int(_rt.monotonic() * 1000)
_time.ticks_us = lambda: int(_rt.monotonic() * 1e6)
_time.ticks_diff = lambda a, b: a - b
sys.modules["time"] = _time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pico"))
import gantry_firmware as firmware  # noqa: E402

SPM = 640  # steps per mm (Y/Z)

# Capture firmware _send() lines (EVT / OK) so we can assert UNDERRUN / CLAMP signalling.
SENT = []
firmware._send = SENT.append


def fresh(vmax=20.0, acc=1000.0, vstart=1.0, soft=False, soft_min=0.0, soft_max=1000.0):
    ax = firmware.Y
    ax.activate()  # fresh fake StateMachine each scenario
    ax.steps_per_mm = SPM
    ax.vmax = vmax * SPM
    ax.acc = acc * SPM
    ax.vstart = vstart * SPM
    ax.recalc_brake()
    ax.pos = 0
    ax.speed = 0.0
    ax.cmd_vel = 0.0
    ax.mode = firmware.POSITION
    ax.cruise = False
    ax.homed = True
    ax.clamped = False
    ax.soft = soft
    ax.soft_min = int(soft_min * SPM)
    ax.soft_max = int(soft_max * SPM)
    ax.lim = (False, False)
    ax.cur_dir = ax.dir_end
    return ax


def drive(ax, max_blocks=40000, retarget=None):
    # Run a few extra idle blocks after motion stops: reaching a limit returns idle one
    # block BEFORE emit_block sets the CLAMP flag, so don't bail on the very first idle.
    traj = []
    idle = 0
    for i in range(max_blocks):
        ax.sm.q = []  # PIO consumed the queued jobs (steady state)
        if retarget:
            retarget(i, ax)
        moved = ax.emit_block()
        traj.append((ax.pos / SPM, ax.speed / SPM))
        if not moved and ax.speed == 0.0:
            idle += 1
            if idle >= 5:
                break
        else:
            idle = 0
    return traj


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


# --- 1. Normal MOVE: decel envelope -> clean stop exactly at target ------------------------
ax = fresh()
ax.cruise = False
ax.target = int(50 * SPM)
SENT.clear()
t = drive(ax)
assert approx(t[-1][0], 50.0), ("move land", t[-1])
assert t[-1][1] == 0.0, ("move final speed", t[-1])
peak = max(s for _, s in t)
assert 19.0 <= peak <= 20.001, ("move peak ~vmax", peak)
near = [s for p, s in t if 49.3 <= p < 50.0]
assert near and min(near) < 19.0, ("move decelerates near target", near[:5])
assert not any("UNDERRUN" in s for s in SENT), ("normal MOVE must not underrun", SENT)

# --- 2. Cruise look-ahead: speed stays ~vmax THROUGH streamed waypoints --------------------
ax = fresh()
ax.cruise = True
ax.target = int(20 * SPM)
state = {"s": 0}


def rt(i, a):
    p = a.pos / SPM
    if state["s"] == 0 and p >= 19.0:
        a.target = int(40 * SPM)
        a.cruise = True
        state["s"] = 1
    elif state["s"] == 1 and p >= 39.0:
        a.target = int(60 * SPM)
        a.cruise = False
        state["s"] = 2  # final = normal stop


SENT.clear()
t = drive(ax, retarget=rt)
assert state["s"] == 2, ("cruise retargets fired", state)
assert approx(t[-1][0], 60.0) and t[-1][1] == 0.0, ("cruise final stop at 60", t[-1])
cruising = [s for p, s in t if 10.0 <= p <= 58.0]  # past accel ramp, before final decel
assert min(cruising) >= 0.95 * 20.0, ("cruise never dipped at waypoints 20/40", min(cruising))
# Well-paced cruise (retargeted before each waypoint, final = normal stop) never hard-stops.
assert not any("UNDERRUN" in s for s in SENT), ("paced cruise must not underrun", SENT)

# --- 3. Cruise never overshoots a stale target; signals UNDERRUN on the hard-stop ---------
ax = fresh()
ax.cruise = True
ax.target = int(30 * SPM)
SENT.clear()
t = drive(ax)
assert max(p for p, _ in t) <= 30.0 + 1e-9, ("cruise no overshoot", max(p for p, _ in t))
assert approx(t[-1][0], 30.0) and t[-1][1] == 0.0, ("cruise lands at stale target", t[-1])
assert any(s.startswith("EVT UNDERRUN y") for s in SENT), (
    "stale cruise target emits UNDERRUN",
    SENT,
)

# --- 4. Soft limit braking is NOT bypassed by cruise --------------------------------------
ax = fresh(soft=True, soft_min=0.0, soft_max=40.0)
ax.cruise = True
ax.target = int(100 * SPM)  # past soft_max (bypass the accept-gate, test the emit_block clamp)
t = drive(ax)
assert max(p for p, _ in t) <= 40.0 + 1e-9, ("cruise respects soft_max", max(p for p, _ in t))
assert t[-1][1] == 0.0, ("cruise soft-stop", t[-1])
assert ax.clamped, "cruise hitting soft_max raises EVT CLAMP"
braked = [s for p, s in t if 39.3 <= p <= 40.0]
assert braked and min(braked) < 19.0, ("cruise brakes (sqrt) into soft_max", braked[:5])

# --- 5. Hardware limit switch stops cruise ------------------------------------------------
ax = fresh()
ax.cruise = True
ax.lim = (False, True)  # end switch pressed
ax.target = int(50 * SPM)
t = drive(ax, max_blocks=50)
assert max(p for p, _ in t) <= 1e-9, ("cruise blocked by pressed end limit", max(p for p, _ in t))

# --- 6. Protocol: MOVEC sets cruise; MOVE / JOG / STOP clear it ----------------------------
ax = fresh()
firmware._handle("MOVEC Y 100")
assert firmware.Y.cruise is True, "MOVEC sets cruise"
firmware._handle("MOVE Y 50")
assert firmware.Y.cruise is False, "MOVE clears cruise"
firmware._handle("MOVEC Y 80")
assert firmware.Y.cruise is True
firmware._handle("STOP Y")
assert firmware.Y.cruise is False, "STOP clears cruise"
firmware._handle("MOVEC Y 80")
firmware._handle("JOG Y 5")
assert firmware.Y.cruise is False, "JOG clears cruise"

# --- 7. Z axis cruise works too (multi-axis) ----------------------------------------------
z = firmware.Z
z.activate()
z.steps_per_mm = SPM
z.vmax = 20 * SPM
z.acc = 1000 * SPM
z.vstart = 1 * SPM
z.recalc_brake()
z.pos = 0
z.speed = 0.0
z.mode = firmware.POSITION
z.cruise = True
z.homed = True
z.soft = False
z.lim = (False, False)
z.cur_dir = z.dir_end
z.target = int(25 * SPM)
tz = drive(z)
assert approx(tz[-1][0], 25.0) and tz[-1][1] == 0.0, ("z cruise lands", tz[-1])
assert max(p for p, _ in tz) <= 25.0 + 1e-9, "z cruise no overshoot"

print("ALL OK")
