# gantry_firmware.py -- Multi-axis (Y + Z) motion server for the Raspberry Pi Pico.
# ruff: noqa: F821
#
# The rp2.asm_pio decorator injects assembler instructions such as pull(),
# mov(), and jmp() while compiling _step_prog; they are not Python globals.
#
# Per-axis PIO step generation (block mode) + an accel-limited velocity/position controller
# + a line-based USB-serial command protocol for low-latency Y/Z target streaming.
#
# AXES (both CL57Y, 640 steps/mm, NC limit switches, MF opto OFF = enabled):
#   Y: step GP2, dir GP3, MF GP4, home GP14, end GP10   (PIO0 SM0)   -- hardware-verified
#   Z: step GP6, dir GP7, MF GP8, home GP21, end GP17   (PIO1 SM4)   -- hardware-verified
#
# PROTOCOL (newline-terminated; replies OK/ERR; async EVT). Commands are addressed to an
# axis with an optional Y/Z token; commands without an axis token default to Y:
#   PING                          -> OK PONG
#   EN  [axis] <0|1>              -> enable/disable (free) that axis
#   HOME [axis] [fast] [slow]     -> two-stage home, then zero (blocks)
#   VMAX [axis] <mm_s>            -> set max velocity
#   ACC  [axis] <mm_s2>           -> set acceleration
#   MOVE [axis] <mm>              -> absolute target (position mode), returns immediately
#   MOVEM [C] <ax> <mm> [..]      -> multi-axis absolute target in ONE command (saves a
#                                    round trip when streaming Y+Z); leading C = cruise
#                                    (MOVEC). e.g. MOVEM Y 263.98 Z 309.97. All-or-nothing:
#                                    if any axis isn't homed / is past its soft limit, none move.
#   JOG  [axis] <mm_s>            -> continuous signed velocity until STOP/limit
#   STOP [axis]                   -> decelerate to a stop
#   POS? [axis]                   -> OK POS [axis] <mm> <homed> <moving>
#   SOFT [axis] <0|1>             -> enable/disable soft-limit enforcement
#   BENCH [axis] [period_us]      -> step-rate benchmark (motor freed; re-HOME after)
#   STATUS?                       -> OK STATUS y ... | z ...   (all axes)
#   BYE                           -> stop the server

import rp2
from machine import Pin
import sys
import uselect
import math
import time

# ---------------- PIO clock / timing (shared) ----------------
PIO_HZ = 1_000_000  # 1 cycle = 1 us
PIO_OVERHEAD = 7  # ~fixed cycles per pulse in the job loop
MIN_DELAY = 1
FIFO_DEPTH = 4
JOB_WORDS = 2  # words per (count-1, delay) job
IO_MS = 3  # serial/EVT servicing period (ms)
DEBOUNCE_MS = 4  # limit-switch debounce (ms): rejects stepper-driver EMI spikes on
# the high-impedance switch inputs (a real press holds; noise does not)
BLOCK_MS = 3  # motion planned in fixed time-slices
BLOCK_S = BLOCK_MS / 1000.0

VELOCITY = 0
POSITION = 1

# ---------------- per-axis config ----------------
# Y -- hardware-verified.
Y_STEP, Y_DIR, Y_MF, Y_HOME, Y_END, Y_SM = 2, 3, 4, 14, 10, 0
Y_STEPS_PER_MM = 640
Y_DIR_HOME = 1  # dir value that moves toward the home switch (y decreases)
Y_HOME_NC = True  # NC switches: idle LOW, pressed/open HIGH
Y_END_NC = True
Y_SOFT_MIN_MM, Y_SOFT_MAX_MM = 0.0, 665.0  # Physical-travel backstop.
# Self-collision is enforced separately by the host collision guard.
Y_VMAX_MM_S, Y_ACC_MM_S2, Y_VSTART_MM_S = 20.0, 1000.0, 1.0
Y_HOME_FAST_MM_S, Y_HOME_SLOW_MM_S = 20.0, 5.0
Y_HOME_BACKOFF_MM, Y_HOME_MAX_MM = 3.0, 670.0

# Z -- hardware-verified reference pinout, direction, and switch polarity.
Z_STEP, Z_DIR, Z_MF, Z_HOME, Z_END, Z_SM = 6, 7, 8, 21, 17, 4
Z_STEPS_PER_MM = 640  # Tune from the Z drive mechanics and verify measured travel.
Z_DIR_HOME = 1  # Direction value that moves toward the home switch.
Z_HOME_NC = True  # NC switches: idle LOW, pressed/open HIGH.
Z_END_NC = True
Z_SOFT_MIN_MM, Z_SOFT_MAX_MM = 0.0, 392.0  # physical travel backstop (app clamps tighter)
Z_VMAX_MM_S, Z_ACC_MM_S2, Z_VSTART_MM_S = 20.0, 1000.0, 1.0
Z_HOME_FAST_MM_S, Z_HOME_SLOW_MM_S = 20.0, 5.0
Z_HOME_BACKOFF_MM, Z_HOME_MAX_MM = 3.0, 400.0


# ---------------- PIO step-pulse generator (block mode) ----------------
# Each job = two TX-FIFO words: (count-1, delay). Emits `count` HIGH pulses (3 us) spaced
# `delay` cycles apart, then pulls the next job. Empty FIFO -> stalls with the pin LOW.
@rp2.asm_pio(set_init=rp2.PIO.OUT_LOW)
def _step_prog():
    pull(block)
    mov(y, osr)
    pull(block)
    mov(isr, osr)
    label("loop")
    mov(x, isr)
    set(pins, 1)[2]
    set(pins, 0)
    label("dly")
    jmp(x_dec, "dly")
    jmp(y_dec, "loop")


def _send(line):
    sys.stdout.write(line)
    sys.stdout.write("\n")


class Axis:
    def __init__(
        self,
        name,
        sm_id,
        step_pin,
        dir_pin,
        mf_pin,
        home_pin,
        end_pin,
        steps_per_mm,
        dir_home,
        home_nc,
        end_nc,
        soft_min_mm,
        soft_max_mm,
        vmax_mm_s,
        acc_mm_s2,
        vstart_mm_s,
        home_fast_mm_s,
        home_slow_mm_s,
        home_backoff_mm,
        home_max_mm,
    ):
        self.name = name
        self.steps_per_mm = steps_per_mm
        self.dir_home = dir_home
        self.dir_end = 1 - dir_home
        self.home_nc = home_nc
        self.end_nc = end_nc
        self.soft_min = int(soft_min_mm * steps_per_mm)
        self.soft_max = int(soft_max_mm * steps_per_mm)
        self.vstart = vstart_mm_s * steps_per_mm
        self.vmax = vmax_mm_s * steps_per_mm
        self.acc = acc_mm_s2 * steps_per_mm
        self.brake_steps = self.vmax * self.vmax / (2.0 * self.acc)
        self.home_fast_mm_s = home_fast_mm_s
        self.home_slow_mm_s = home_slow_mm_s
        self.home_backoff = int(home_backoff_mm * steps_per_mm)
        self.home_max = int(home_max_mm * steps_per_mm)
        self._step_pin = step_pin
        self._sm_id = sm_id
        self.dr = Pin(dir_pin, Pin.OUT, value=self.dir_end)
        self.mf = Pin(mf_pin, Pin.OUT, value=0)  # opto OFF = enabled
        self.home_sw = Pin(home_pin, Pin.IN, Pin.PULL_UP)
        self.end_sw = Pin(end_pin, Pin.IN, Pin.PULL_UP)
        self.sm = None
        self.pos = 0
        self.target = 0
        self.speed = 0.0
        self.cmd_vel = 0.0
        self.cur_dir = self.dir_end
        self.mode = VELOCITY
        self.enabled = True
        self.homed = False
        self.clamped = False
        self.soft = True
        self.cruise = False  # POSITION look-ahead: cruise THROUGH the target at vmax
        # (no decel-to-zero), for streamed waypoints. Set by MOVEC,
        # cleared by MOVE/JOG/STOP/HOME. Overshoot is still impossible
        # (n is clamped to the remaining distance); a target reached
        # with no fresh waypoint just hard-stops there.
        self.last_lim = (False, False)  # last reported (debounced) state, for EVT LIMIT
        self.lim = (False, False)  # debounced limit state used by motion/homing/EVT
        self._lim_cand = (False, False)  # candidate raw state awaiting the debounce window
        self._lim_t = 0

    def activate(self):
        # (Re)create the SM each run so the FIFO/shift state starts clean (an orphaned
        # half-job from a mid-pair Ctrl-C cannot carry into the next run).
        self.sm = rp2.StateMachine(
            self._sm_id, _step_prog, freq=PIO_HZ, set_base=Pin(self._step_pin, Pin.OUT)
        )
        self.sm.active(1)
        self.set_enabled(True)

    def deactivate(self):
        if self.sm is not None:
            self.sm.active(0)

    @staticmethod
    def _pressed(level, nc):
        return (level == 1) if nc else (level == 0)

    def limits(self):
        """Raw (un-debounced) limit read: (home_pressed, end_pressed)."""
        return (
            self._pressed(self.home_sw.value(), self.home_nc),
            self._pressed(self.end_sw.value(), self.end_nc),
        )

    def poll_limits(self):
        """Debounced limit state. A change is committed only once the raw read holds steady
        for DEBOUNCE_MS, which rejects stepper-driver EMI spikes on the switch inputs while a
        real (held) press still registers. Returns the debounced (home_pressed, end_pressed)."""
        raw = self.limits()
        now = time.ticks_ms()
        if raw != self._lim_cand:
            self._lim_cand = raw
            self._lim_t = now
        elif raw != self.lim and time.ticks_diff(now, self._lim_t) >= DEBOUNCE_MS:
            self.lim = raw
        return self.lim

    def pos_mm(self):
        return self.pos / self.steps_per_mm

    def moving(self):
        return abs(self.speed) >= self.vstart

    def set_enabled(self, on):
        self.enabled = bool(on)
        self.mf.value(0 if self.enabled else 1)  # opto OFF = enabled
        if not self.enabled:
            self.speed = 0.0
            self.cmd_vel = 0.0

    def recalc_brake(self):
        self.brake_steps = self.vmax * self.vmax / (2.0 * self.acc)

    def _set_dir_when_drained(self, new_dir):
        if new_dir == self.cur_dir:
            return True
        if self.sm.tx_fifo() != 0:
            return False
        self.cur_dir = new_dir
        self.dr.value(new_dir)
        time.sleep_us(20)  # DIR setup time
        return True

    def emit_block(self):
        """Plan one BLOCK_S time-slice and push it as a (count-1, delay) job. Returns True
        if a block was emitted, False if idle / waiting for the FIFO to drain."""
        DIR_HOME, DIR_END = self.dir_home, self.dir_end
        if self.mode == POSITION:
            d = self.target - self.pos
            if d == 0:
                cmd = 0.0
            else:
                ad = abs(d)
                if self.cruise or ad >= self.brake_steps:
                    # cruise: hold vmax THROUGH the waypoint (no decel envelope) so a
                    # streamed sequence flows continuously; the n<=rem clamp below still
                    # prevents overshoot, and a stale waypoint just hard-stops at target.
                    cmd = self.vmax if d > 0 else -self.vmax
                else:
                    v = math.sqrt(2.0 * self.acc * ad)
                    cmd = v if d > 0 else -v
        else:
            cmd = self.cmd_vel

        hp, ep = self.lim
        limited = False
        if cmd < 0 and hp:
            cmd = 0.0
            limited = True
        if cmd > 0 and ep:
            cmd = 0.0
            limited = True
        if self.soft and self.homed:
            if cmd < 0:
                rem = self.pos - self.soft_min
                if rem <= 0:
                    cmd = 0.0
                    limited = True
                elif rem < self.brake_steps:
                    vs = math.sqrt(2.0 * self.acc * rem)
                    if -cmd > vs:
                        cmd = -vs
            elif cmd > 0:
                rem = self.soft_max - self.pos
                if rem <= 0:
                    cmd = 0.0
                    limited = True
                elif rem < self.brake_steps:
                    vs = math.sqrt(2.0 * self.acc * rem)
                    if cmd > vs:
                        cmd = vs

        dv = self.acc * BLOCK_S
        if cmd > self.speed:
            self.speed = min(cmd, self.speed + dv)
        else:
            self.speed = max(cmd, self.speed - dv)

        if self.mode == POSITION and not self.cruise:
            d = self.target - self.pos
            ad = abs(d)
            if d != 0 and ad < self.brake_steps:
                vs = math.sqrt(2.0 * self.acc * ad)
                if abs(self.speed) > vs:
                    self.speed = vs if self.speed > 0 else -vs

        if abs(self.speed) < self.vstart:
            if cmd == 0.0:
                self.speed = 0.0
                if (
                    self.mode == POSITION
                    and self.target != self.pos
                    and limited
                    and not self.clamped
                ):
                    self.clamped = True
                    _send(
                        "EVT CLAMP %s pos=%.3f tgt=%.3f"
                        % (self.name, self.pos_mm(), self.target / self.steps_per_mm)
                    )
                return False
            new_dir = DIR_END if cmd > 0 else DIR_HOME
            if not self._set_dir_when_drained(new_dir):
                return False
            self.speed = self.vstart if cmd > 0 else -self.vstart

        want_dir = DIR_END if self.speed > 0 else DIR_HOME
        if not self._set_dir_when_drained(want_dir):
            return False

        spd = abs(self.speed)
        n = int(spd * BLOCK_S)
        if n < 1:
            n = 1
        if self.mode == POSITION:
            rem = abs(self.target - self.pos)
            if rem == 0:
                # A cruise (MOVEC) waypoint reached with no fresh target = stream underrun:
                # the axis is still at speed and hard-stops here (vmax->0 in one block), which
                # can skip steps and desync commanded vs actual. Emit ONCE (the next block's
                # speed is already 0, so the guard won't re-fire) so the host can detect it
                # and re-home if needed. A normal MOVE has decelerated to ~vstart by now, but
                # the self.cruise guard keeps it from ever tripping this.
                if self.cruise and abs(self.speed) >= self.vstart:
                    _send("EVT UNDERRUN %s pos=%.3f" % (self.name, self.pos_mm()))
                self.speed = 0.0
                return False
            if n > rem:
                n = rem
        if self.soft and self.homed:
            room = (
                (self.soft_max - self.pos)
                if self.cur_dir == DIR_END
                else (self.pos - self.soft_min)
            )
            if room <= 0:
                self.speed = 0.0
                return False
            if n > room:
                n = room
        delay = int(1_000_000.0 / spd) - PIO_OVERHEAD
        if delay < MIN_DELAY:
            delay = MIN_DELAY
        self.sm.put(n - 1)
        self.sm.put(delay)
        self.pos += n if self.cur_dir == DIR_END else -n
        return True

    def feed(self):
        if not self.enabled:
            return
        while self.sm.tx_fifo() <= FIFO_DEPTH - JOB_WORDS:
            if not self.emit_block():
                break

    # --- homing (blocking; abort_cb() lets the host cancel) ---
    def _home_run(
        self,
        direction,
        max_steps,
        period_us,
        abort_cb,
        until_home_pressed=False,
        until_home_released=False,
        n=8,
    ):
        while self.sm.tx_fifo() != 0:
            pass
        time.sleep_us(800)  # let the prev job finish before DIR flip
        while not self._set_dir_when_drained(direction):
            pass
        delay = max(MIN_DELAY, period_us - PIO_OVERHEAD)
        inc = n if direction == self.dir_end else -n
        emitted = 0
        i = 0
        while emitted < max_steps:
            if (i & 15) == 0 and abort_cb():
                _send("EVT ABORT %s host-stop" % self.name)
                return False
            i += 1
            hp, ep = self.poll_limits()
            if until_home_pressed and hp:
                return True
            if until_home_released and not hp:
                return True
            if ep:
                _send("EVT ABORT %s end-switch" % self.name)
                return False
            if direction == self.dir_home and hp and not until_home_pressed:
                _send("EVT ABORT %s home-switch" % self.name)
                return False
            while self.sm.tx_fifo() >= JOB_WORDS:
                pass
            self.sm.put(n - 1)
            self.sm.put(delay)
            self.pos += inc
            emitted += n
        return not (until_home_pressed or until_home_released)

    def do_home(self, abort_cb, fast_mm_s=None, slow_mm_s=None):
        if not self.enabled:
            return False
        # Park every OTHER axis before this blocking home: while do_home busy-waits, the
        # other axis isn't fed and its FIFO drains (it physically stops). Zeroing its speed
        # here makes it re-accelerate from rest afterwards instead of jumping back to a stale
        # vmax (a velocity discontinuity / missed-step jolt).
        for other in AXES:
            if other is not self:
                other.cmd_vel = 0.0
                other.speed = 0.0
                other.mode = VELOCITY
                other.target = other.pos
        fast_mm_s = self.home_fast_mm_s if fast_mm_s is None else fast_mm_s
        slow_mm_s = self.home_slow_mm_s if slow_mm_s is None else slow_mm_s
        self.homed = False
        self.clamped = False
        self.cruise = False
        self.speed = 0.0
        self.cmd_vel = 0.0
        self.mode = VELOCITY
        fast_us = int(1_000_000 / (fast_mm_s * self.steps_per_mm))
        slow_us = int(1_000_000 / (slow_mm_s * self.steps_per_mm))
        clear = self.home_backoff + 10 * self.steps_per_mm
        if self.lim[0]:
            if not self._home_run(self.dir_end, clear, slow_us, abort_cb, until_home_released=True):
                return False
            if not self._home_run(self.dir_end, self.home_backoff, slow_us, abort_cb):
                return False
        if not self._home_run(
            self.dir_home, self.home_max, fast_us, abort_cb, until_home_pressed=True, n=8
        ):
            return False
        if not self._home_run(self.dir_end, self.home_backoff, slow_us, abort_cb, n=8):
            return False
        if not self._home_run(
            self.dir_home,
            self.home_backoff + 2 * self.steps_per_mm,
            slow_us,
            abort_cb,
            until_home_pressed=True,
            n=1,
        ):
            return False
        while self.sm.tx_fifo() != 0:
            pass
        time.sleep_ms(5)
        self.pos = 0
        self.target = 0
        self.homed = True
        return True

    def bench(self, per):
        was = self.enabled
        self.set_enabled(False)
        total, nblk = 20000, 128
        delay = max(MIN_DELAY, per - PIO_OVERHEAD)
        done = 0
        t0 = time.ticks_us()
        while done < total:
            while self.sm.tx_fifo() > FIFO_DEPTH - JOB_WORDS:
                pass
            self.sm.put(nblk - 1)
            self.sm.put(delay)
            done += nblk
        while self.sm.tx_fifo() != 0:
            pass
        us = time.ticks_diff(time.ticks_us(), t0)
        self.set_enabled(was)
        return done, us


# ---------------- axes + shared state ----------------
Y = Axis(
    "y",
    Y_SM,
    Y_STEP,
    Y_DIR,
    Y_MF,
    Y_HOME,
    Y_END,
    Y_STEPS_PER_MM,
    Y_DIR_HOME,
    Y_HOME_NC,
    Y_END_NC,
    Y_SOFT_MIN_MM,
    Y_SOFT_MAX_MM,
    Y_VMAX_MM_S,
    Y_ACC_MM_S2,
    Y_VSTART_MM_S,
    Y_HOME_FAST_MM_S,
    Y_HOME_SLOW_MM_S,
    Y_HOME_BACKOFF_MM,
    Y_HOME_MAX_MM,
)
Z = Axis(
    "z",
    Z_SM,
    Z_STEP,
    Z_DIR,
    Z_MF,
    Z_HOME,
    Z_END,
    Z_STEPS_PER_MM,
    Z_DIR_HOME,
    Z_HOME_NC,
    Z_END_NC,
    Z_SOFT_MIN_MM,
    Z_SOFT_MAX_MM,
    Z_VMAX_MM_S,
    Z_ACC_MM_S2,
    Z_VSTART_MM_S,
    Z_HOME_FAST_MM_S,
    Z_HOME_SLOW_MM_S,
    Z_HOME_BACKOFF_MM,
    Z_HOME_MAX_MM,
)
AXES = (Y, Z)
BY_NAME = {"y": Y, "z": Z}

_running = True
_poll = uselect.poll()
_poll.register(sys.stdin, uselect.POLLIN)
_home_line = ""
_pending = []  # lines received during a blocking HOME; run() dispatches them after


def _is_abort(line, homing_name):
    """True if `line` is a stop aimed at the axis currently homing (or a global STOP/BYE)."""
    parts = line.split()
    if not parts:
        return False
    c = parts[0].upper()
    if c == "BYE":
        return True
    rest = parts[1:]
    if rest and rest[0].upper() in ("Y", "Z"):
        name = rest[0].lower()
        rest = rest[1:]
    else:
        name = "y"
    if c == "STOP":
        return name == homing_name
    if c == "EN":
        return name == homing_name and bool(rest) and rest[0] == "0"
    return False


def _abort_requested(homing_name):
    """Non-blocking: read any complete lines that arrived during a blocking HOME. Buffer them
    all in _pending (run() dispatches them afterwards -- nothing is dropped) and return True if
    one is a stop aimed at the homing axis (or a global STOP/BYE)."""
    global _home_line
    aborted = False
    while _poll.poll(0):
        ch = sys.stdin.read(1)
        if not ch:
            break
        if ch in ("\n", "\r"):
            ln = _home_line.strip()
            _home_line = ""
            if ln:
                _pending.append(ln)
                if _is_abort(ln, homing_name):
                    aborted = True
        else:
            _home_line += ch
            if len(_home_line) > 80:
                _home_line = ""
    return aborted


def _tag(explicit, name):
    return (name.upper() + " ") if explicit else ""


def _handle(line):
    global _running
    parts = line.split()
    if not parts:
        return
    c = parts[0].upper()
    if c == "PING":
        _send("OK PONG")
        return
    if c == "BYE":
        _running = False
        _send("OK BYE")
        return
    if c == "STATUS?":
        out = "OK STATUS"
        for ax in AXES:
            hp, ep = ax.lim
            out += " %s pos=%.3f tgt=%.3f vel=%.3f homed=%d en=%d home=%d end=%d" % (
                ax.name,
                ax.pos_mm(),
                ax.target / ax.steps_per_mm,
                ax.speed / ax.steps_per_mm,
                1 if ax.homed else 0,
                1 if ax.enabled else 0,
                1 if hp else 0,
                1 if ep else 0,
            )
        _send(out)
        return
    if c == "MOVEM":
        # Multi-axis MOVE/MOVEC in ONE command: "MOVEM [C] <axis> <mm> [<axis> <mm> ...]".
        # A leading C selects cruise (MOVEC) semantics. All-or-nothing: if any axis is not
        # homed or its target is past the soft limit, NOTHING moves (keeps Y/Z coordinated).
        a = parts[1:]
        cruise = bool(a) and a[0].upper() == "C"
        if cruise:
            a = a[1:]
        if not a or (len(a) % 2) != 0:
            _send("ERR BADARG")
            return
        pairs = []
        try:
            i = 0
            while i < len(a):
                axx = BY_NAME.get(a[i].lower())
                if axx is None:
                    _send("ERR BADAXIS")
                    return
                pairs.append((axx, float(a[i + 1])))
                i += 2
        except (IndexError, ValueError):
            _send("ERR BADARG")
            return
        for axx, tmm in pairs:
            if not axx.homed:
                _send("ERR NOTHOMED %s" % axx.name)
                return
            if axx.soft and not (
                axx.soft_min / axx.steps_per_mm <= tmm <= axx.soft_max / axx.steps_per_mm
            ):
                _send("ERR SOFTLIMIT %s" % axx.name)
                return
        out = "OK MOVEM"
        for axx, tmm in pairs:
            axx.mode = POSITION
            axx.target = int(round(tmm * axx.steps_per_mm))
            axx.cruise = cruise
            axx.clamped = False
            out += " %s %.3f" % (axx.name.upper(), tmm)
        _send(out)
        return

    # axis-addressed commands: optional Y/Z token, else default Y (back-compat)
    rest = parts[1:]
    explicit = bool(rest) and rest[0].upper() in ("Y", "Z")
    name = rest[0].lower() if explicit else "y"
    if explicit:
        rest = rest[1:]
    ax = BY_NAME.get(name)
    if ax is None:
        _send("ERR BADAXIS")
        return
    tag = _tag(explicit, name)
    try:
        if c == "EN":
            ax.set_enabled(rest[0] == "1")
            _send("OK EN %s%d" % (tag, 1 if ax.enabled else 0))
        elif c == "HOME":
            fast = float(rest[0]) if len(rest) > 0 else None
            slow = float(rest[1]) if len(rest) > 1 else None
            ok = ax.do_home(lambda: _abort_requested(ax.name), fast, slow)
            _send(("OK HOMED %s%.3f" % (tag, ax.pos_mm())) if ok else ("ERR HOME %s" % name))
        elif c == "VMAX":
            ax.vmax = float(rest[0]) * ax.steps_per_mm
            ax.recalc_brake()
            _send("OK VMAX %s%.3f" % (tag, ax.vmax / ax.steps_per_mm))
        elif c == "ACC":
            ax.acc = float(rest[0]) * ax.steps_per_mm
            ax.recalc_brake()
            _send("OK ACC %s%.3f" % (tag, ax.acc / ax.steps_per_mm))
        elif c == "MOVE" or c == "MOVEC":
            # MOVE = stop-at-target (decel to v=0); MOVEC = cruise THROUGH the target
            # (look-ahead for streamed waypoints). Same accept-gate (homed + soft limit).
            t = float(rest[0])
            tmm = t
            if not ax.homed:
                _send("ERR NOTHOMED %s" % name)
            elif ax.soft and not (
                ax.soft_min / ax.steps_per_mm <= tmm <= ax.soft_max / ax.steps_per_mm
            ):
                _send("ERR SOFTLIMIT %s" % name)
            else:
                ax.mode = POSITION
                ax.target = int(round(tmm * ax.steps_per_mm))
                ax.cruise = c == "MOVEC"
                ax.clamped = False
                _send("OK %s %s%.3f" % (c, tag, tmm))
        elif c == "JOG":
            v = float(rest[0]) * ax.steps_per_mm
            ax.cmd_vel = max(-ax.vmax, min(ax.vmax, v))
            ax.mode = VELOCITY
            ax.cruise = False
            _send("OK JOG %s%.3f" % (tag, ax.cmd_vel / ax.steps_per_mm))
        elif c == "STOP":
            ax.mode = VELOCITY
            ax.cmd_vel = 0.0
            ax.target = ax.pos
            ax.cruise = False
            ax.clamped = False
            _send("OK STOP %s" % tag if tag else "OK STOP")
        elif c == "POS?":
            hp, ep = ax.limits()
            _send(
                "OK POS %s%.3f %d %d"
                % (tag, ax.pos_mm(), 1 if ax.homed else 0, 1 if ax.moving() else 0)
            )
        elif c == "SOFT":
            ax.soft = rest[0] == "1"
            _send("OK SOFT %s%d" % (tag, 1 if ax.soft else 0))
        elif c == "BENCH":
            per = int(rest[0]) if len(rest) > 0 else 10
            done, us = ax.bench(per)
            _send(
                "OK BENCH %s%d us=%d %d steps/s %.1f mm/s"
                % (tag, done, us, (done * 1000000) // us, (done * 1000000.0 / us) / ax.steps_per_mm)
            )
        else:
            _send("ERR UNKNOWN %s" % c)
    except (IndexError, ValueError):
        _send("ERR BADARG")


def run():
    global _running
    _running = True
    for ax in AXES:
        ax.activate()
    line = ""
    overflow = False
    last_io = time.ticks_ms()
    _send("OK READY")
    try:
        while _running:
            for ax in AXES:
                ax.poll_limits()  # debounce the switch inputs every pass
                ax.feed()
            if time.ticks_diff(time.ticks_ms(), last_io) >= IO_MS:
                last_io = time.ticks_ms()
                while _poll.poll(0):
                    ch = sys.stdin.read(1)
                    if not ch:
                        break
                    if ch not in ("\n", "\r"):
                        if not overflow:
                            line += ch
                            if len(line) > 80:
                                line = ""
                                overflow = True
                    else:
                        if overflow:
                            overflow = False
                        elif line:
                            _handle(line)
                            while _pending:  # commands buffered during a blocking HOME
                                _handle(_pending.pop(0))
                        line = ""
                for ax in AXES:
                    if ax.lim != ax.last_lim:
                        _send(
                            "EVT LIMIT %s home=%d end=%d"
                            % (ax.name, 1 if ax.lim[0] else 0, 1 if ax.lim[1] else 0)
                        )
                        ax.last_lim = ax.lim
    except KeyboardInterrupt:
        pass
    finally:
        for ax in AXES:
            ax.cmd_vel = 0.0
            ax.speed = 0.0
            ax.mode = VELOCITY
            ax.target = ax.pos
            ax.set_enabled(False)
            ax.deactivate()
        _send("OK STOPPED")


if __name__ == "__main__":
    run()
