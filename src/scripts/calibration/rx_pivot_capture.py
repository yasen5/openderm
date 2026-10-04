#!/usr/bin/env python3
"""Record same-point poses for RX-pivot calibration.

The utility moves RX to a requested angle and regulates Z to keep the average
laser standoff at the target distance. Use the arrow keys to keep one fixed
surface feature centered: Left/Right adjust Y, Up/Down adjust X, and [ / ]
change the jog step.

Pressing Enter records the current pose -- x, y, z, rx, sensor1, sensor2 (plus
the average and a timestamp) -- as one JSON line, then prompts for the next RX
angle. Record at least six well-spaced angles across the intended scan range.
Blank keeps the current angle; q at the prompt quits.

The two laser spots interfere when lit together, so the sensors are never on at
the same time; each reading lights one, reads it, and turns it off.

SAFETY: X/Y jogs move before Z re-regulates. Over a rising surface, a large jog
can drive the head into the target. Use an inert target and keep
`jog_step_mm` in config/scripts.json small relative to surface variation.

X is controlled by Klipper. Y and Z are controlled by one Raspberry Pi Pico
over the shared serial connection in config/scripts.json. That file also
controls whether to home the Pico axes at startup.

Requirements:
  - `openderm-gantry-server` is running on Pi #1 with X homed.
  - The Pico firmware is reachable at the configured port with Y/Z homed
    (configure home_y/home_z or use ``openderm --axis y|z home``).
  - The RX-axis server is running and rx homed.
  - Run on the host that owns the ADS1115 + sensor enable GPIOs (Pi #2).
  - A TTY-attached keyboard (run it in a real terminal, not a pipe).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import select
import sys
import termios
import time
import tty
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from openderm.calibration import AxisSetup, CalibrationSession
from openderm.motion.gantry.server import (
    GantryServerClient,
    GantryServerError,
)
from openderm.script_config import calibration_options
from openderm.motion.rx_axis.server import (
    RxAxisServerClient,
    RxAxisServerError,
)
from openderm.sensors.hg_c import build_sensor_controller


ARROW_PREFIX = "\x1b"


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _read_key(fd: int) -> str:
    """Read one key (collapsing a CSI arrow escape into a 3-char token)."""
    first = os.read(fd, 1).decode("utf-8", "ignore")
    if first != ARROW_PREFIX:
        return first
    # Pull the rest of an escape sequence if it is already buffered.
    rest = b""
    while select.select([fd], [], [], 0)[0]:
        rest += os.read(fd, 1)
        if len(rest) >= 2:
            break
    return first + rest.decode("utf-8", "ignore")


class _PivotCaptureWorkflow:
    """Interactive RX-pivot capture over an initialized calibration session."""

    def __init__(self, session: CalibrationSession, args: argparse.Namespace) -> None:
        self.session = session
        self.args = args
        self.fd = sys.stdin.fileno()
        self.previous_termios = termios.tcgetattr(self.fd)
        self.target_rx_rad = math.radians(args.target_rx) if args.degrees else args.target_rx
        self.jog_step_mm = args.jog_step_mm
        self.on_target = 0
        self.last_report = time.monotonic()
        self.record_count = 0
        self.record_path = Path(args.record_file)

    @staticmethod
    def emit(message: str) -> None:
        """Print a complete line while the terminal is in raw mode."""
        sys.stdout.write("\r\x1b[K" + message + "\r\n")
        sys.stdout.flush()

    def _goto_rx(self, radians: float) -> bool:
        self.emit(f"moving rx -> {radians:+.4f} rad ({math.degrees(radians):+.2f} deg) ...")
        if not self.session.move_rx_to(
            radians,
            speed_rad_s=self.args.rx_speed_rad_s,
        ):
            return False
        landed = self.session.read_rx()
        if landed is not None:
            self.emit(f"rx at {landed:+.4f} rad ({math.degrees(landed):+.2f} deg)")
        return True

    def _move_axis(self, axis: str, delta_mm: float) -> bool:
        return self.session.move_axis_relative(
            axis,
            delta_mm,
            feed_mm_min=(self.args.feed_mm_min if axis == "z" else self.args.xy_feed_mm_min),
            prefer_move_by=axis == "x",
        )

    def _record_pose(self) -> None:
        sensor1, sensor2 = self.session.read_pair(samples=self.args.samples)
        rx = self.session.read_rx()
        record = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "x_mm": self.session.read_axis("x"),
            "y_mm": self.session.read_axis("y"),
            "z_mm": self.session.read_axis("z"),
            "rx_rad": rx,
            "rx_deg": None if rx is None else math.degrees(rx),
            "sensor1_mm": sensor1.distance_mm,
            "sensor2_mm": sensor2.distance_mm,
            "sensor1_in_range": sensor1.in_range,
            "sensor2_in_range": sensor2.in_range,
            "avg_mm": self.session.average_distance((sensor1, sensor2)),
            "target_mm": self.args.target_mm,
        }
        try:
            with self.record_path.open("a") as log:
                log.write(json.dumps(record) + "\n")
        except OSError as exc:
            self.emit(f"  warning: failed to write record: {exc}")
            return

        self.record_count += 1
        rx_text = "--" if record["rx_deg"] is None else f"{record['rx_deg']:+.2f}deg"
        self.emit(
            f"recorded pose #{self.record_count} -> x={record['x_mm']} "
            f"y={record['y_mm']} z={record['z_mm']} rx={rx_text}"
        )

    def _prompt_new_rx(self) -> float | None:
        unit = "deg" if self.args.degrees else "rad"
        shown = math.degrees(self.target_rx_rad) if self.args.degrees else self.target_rx_rad
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.previous_termios)
        try:
            while True:
                sys.stdout.write(
                    f"\nNew target rx in {unit} (current {shown:+.4f}, blank to keep, q to quit): "
                )
                sys.stdout.flush()
                line = sys.stdin.readline()
                if not line:
                    return None
                text = line.strip()
                if text.lower() == "q":
                    return None
                if not text:
                    return self.target_rx_rad
                try:
                    value = float(text)
                except ValueError:
                    print("  not a number; try again.")
                    continue
                return math.radians(value) if self.args.degrees else value
        finally:
            tty.setraw(self.fd)

    def _read_keyboard(self) -> tuple[float, float, str | None]:
        dx = 0.0
        dy = 0.0
        action: str | None = None
        ready = select.select([self.fd], [], [], self.args.period_s)[0]
        while ready:
            key = _read_key(self.fd)
            if key in ("q", "Q", "\x03"):
                action = "quit"
                break
            if key in ("\r", "\n"):
                action = "record"
                break
            if key in ("\x1b[D", "a", "A"):
                dy += self.jog_step_mm
            elif key in ("\x1b[C", "d", "D"):
                dy -= self.jog_step_mm
            elif key == "\x1b[A":
                dx -= self.jog_step_mm
            elif key == "\x1b[B":
                dx += self.jog_step_mm
            elif key in ("[", ","):
                self.jog_step_mm = max(0.001, self.jog_step_mm / 2.0)
                self.emit(f"  x/y jog step -> {self.jog_step_mm:.3f} mm")
            elif key in ("]", "."):
                self.jog_step_mm *= 2.0
                self.emit(f"  x/y jog step -> {self.jog_step_mm:.3f} mm")
            ready = select.select([self.fd], [], [], 0)[0]
        return dx, dy, action

    def _regulate_z(self):
        sensor1, sensor2 = self.session.read_pair(samples=self.args.samples)
        average = self.session.average_distance((sensor1, sensor2))
        if average is None:
            self.on_target = 0
            moved = self._move_axis("z", self.args.search_step_mm)
        else:
            error = average - self.args.target_mm
            if abs(error) <= self.args.deadband_mm:
                self.on_target += 1
                moved = True
            else:
                self.on_target = 0
                delta = _clamp(
                    self.args.gain_mm_per_mm * error,
                    self.args.max_step_mm,
                )
                moved = self._move_axis("z", delta)
        return sensor1, sensor2, average, moved

    def _render_status(self, sensor1, sensor2, average: float | None) -> None:
        now = time.monotonic()
        if now - self.last_report < self.args.report_interval_s:
            return
        positions = {axis: self.session.read_axis(axis) for axis in ("x", "y", "z")}
        axis_text = {
            axis: "--" if value is None else f"{value:8.2f}" for axis, value in positions.items()
        }
        sensor1_text = "--" if sensor1.distance_mm is None else f"{sensor1.distance_mm:6.2f}"
        sensor2_text = "--" if sensor2.distance_mm is None else f"{sensor2.distance_mm:6.2f}"
        average_text = "--" if average is None else f"{average:6.2f}"
        error_text = "  --  " if average is None else f"{average - self.args.target_mm:+6.2f}"
        flag = "ON-TGT" if self.on_target > 0 else "      "
        sys.stdout.write(
            f"\r\x1b[K x={axis_text['x']} y={axis_text['y']} "
            f"z={axis_text['z']} | s1={sensor1_text} s2={sensor2_text} "
            f"avg={average_text} err={error_text}mm {flag} | "
            f"step={self.jog_step_mm:.3f}mm "
        )
        sys.stdout.flush()
        self.last_report = now

    def _handle_action(self, action: str | None) -> bool:
        if action == "quit":
            self.emit("quit")
            return False
        if action != "record":
            return True

        self._record_pose()
        new_target = self._prompt_new_rx()
        if new_target is None:
            self.emit("quit")
            return False
        self.target_rx_rad = new_target
        self._goto_rx(new_target)
        self.last_report = time.monotonic()
        return True

    def _control_loop(self) -> None:
        while True:
            dx, dy, action = self._read_keyboard()
            if not self._handle_action(action):
                break
            if action == "record":
                continue
            if dx and self._move_axis("x", dx):
                self.emit(f"  x {dx:+.3f} mm")
            if dy and self._move_axis("y", dy):
                self.emit(f"  y {dy:+.3f} mm")

            sensor1, sensor2, average, moved = self._regulate_z()
            if not moved:
                self.emit("z move failed; stopping.")
                break
            self._render_status(sensor1, sensor2, average)

    def run(self) -> int:
        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        self.emit(f"recording poses to {self.record_path}")
        if not self._goto_rx(self.target_rx_rad):
            return 1

        self.emit(
            f"regulating z to avg standoff target={self.args.target_mm:.2f} mm | "
            f"x/y jog step={self.args.jog_step_mm:.3f} mm"
        )
        self.emit(
            "keys: Left/A = y+ Right/D = y- Up = x- Down = x+ "
            "[ / ] = step -/+ Enter = record + new rx q = quit"
        )
        self.session.set_logger(self.emit)
        tty.setraw(self.fd)
        try:
            self._control_loop()
            return 0
        finally:
            termios.tcsetattr(
                self.fd,
                termios.TCSADRAIN,
                self.previous_termios,
            )
            sys.stdout.write("\r\n")
            sys.stdout.flush()


def _connect_session(args: argparse.Namespace) -> CalibrationSession | None:
    from openderm.motion.pico.adapter import PicoAxisClient, open_pico_link

    session = CalibrationSession()
    session.add_axis(
        "x",
        GantryServerClient(args.gantry_server_url, axis="x", timeout_s=60.0),
    )
    try:
        print(f"connecting to Pico (Y+Z) on {args.pico_port} ...", file=sys.stderr)
        session.connect_pico_axes(
            args.pico_port,
            {
                "y": AxisSetup(
                    vmax_mm_s=args.y_pico_vmax_mm_s,
                    acc_mm_s2=args.y_pico_acc_mm_s2,
                    home=args.home_y,
                ),
                "z": AxisSetup(
                    vmax_mm_s=args.z_pico_vmax_mm_s,
                    acc_mm_s2=args.z_pico_acc_mm_s2,
                    home=args.home_z,
                ),
            },
            open_link=open_pico_link,
            axis_client_factory=PicoAxisClient,
            timeout_s=60.0,
        )
        print(f"Pico link up (Y+Z) on {args.pico_port}.", file=sys.stderr)
    except Exception as exc:
        print(f"could not start the Y/Z Pico on {args.pico_port}: {exc}", file=sys.stderr)
        session.close()
        return None

    try:
        for axis in ("x", "y", "z"):
            if not session.is_homed(axis):
                print(f"{axis}-axis is not homed.", file=sys.stderr)
                session.close()
                return None
    except GantryServerError as exc:
        print(f"axis status check failed: {exc}", file=sys.stderr)
        session.close()
        return None

    try:
        rx = session.connect_rx(
            args.rx_server_url,
            client_factory=RxAxisServerClient,
            timeout_s=60.0,
        )
        rx.status()
    except RxAxisServerError as exc:
        print(
            f"RX-axis server not reachable or rx is not homed at {args.rx_server_url}: {exc}",
            file=sys.stderr,
        )
        session.close()
        return None

    try:
        session.connect_sensors(build_sensor_controller)
    except Exception:
        session.close()
        raise
    return session


def run(args: argparse.Namespace) -> int:
    try:
        args = calibration_options("rx_pivot_capture", args)
        if not math.isfinite(args.target_rx):
            raise ValueError("target_rx must be finite.")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if not sys.stdin.isatty():
        print(
            "This script needs a TTY-attached keyboard; run it in a real terminal.",
            file=sys.stderr,
        )
        return 1

    session = _connect_session(args)
    if session is None:
        return 1
    try:
        return _PivotCaptureWorkflow(session, args).run()
    finally:
        session.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record fixed-point poses across RX angles. Persistent settings "
            "are loaded from config/scripts.json."
        )
    )
    parser.add_argument(
        "target_rx",
        type=float,
        help="Initial RX angle (radians, or degrees when configured in scripts.json).",
    )
    parser.add_argument(
        "--record-file",
        required=True,
        type=Path,
        help="JSONL file to append this run's poses to.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
