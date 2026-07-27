#!/usr/bin/env python3
"""Regulate the gantry z-axis so the sensor1/sensor2 average distance hits a target.

Runs on Raspberry Pi #2: it reads the laser sensors locally (ADS1115 + GPIOs)
and commands the Pico-controlled Z axis through the serial bridge on Raspberry
Pi #1.

The two laser spots optically interfere when lit at the same time, so the
sensors are NEVER on together. Each control iteration reads them one at a time
(enable sensor1 -> read -> disable; enable sensor2 -> read -> disable) and
regulates on the average of whichever sensors are in range.

Two control dimensions are regulated simultaneously:

  z-height: from the AVERAGE of the in-range sensors.
    Plant sign (fixed by the mechanical setup): increasing z DECREASES the
    sensor distance reading. A single proportional law on
    ``error = average_reading - target``:
      - reading > target (too high) -> increase z   (error > 0  -> +z)
      - reading < target (too low)  -> decrease z   (error < 0  -> -z)
    If neither sensor is in range, the loop always increases z.

  RX angle: from the DIFFERENCE between the two sensors.
    The sensors are mounted a fixed distance apart, so ``d1 - d2`` is
    proportional to the tilt about the RX axis. A proportional law nudges RX
    to drive ``d1 - d2`` to zero (sensor1 == sensor2), leveling the head.
    RX is only adjusted when BOTH sensors are in range (the difference is
    meaningless otherwise). RX runs on the separate RX-axis server and the
    axis must be homed.

    NOTE: the RX plant sign depends on how the RX-axis motor is mounted relative to
    the sensors and is NOT known a priori. During bring-up, if RX diverges
    (the head tilts further instead of leveling), flip the sign of
    --rx-gain-rad-per-mm.

Requirements:
  - The Pico firmware is reachable and Z is homed.
  - For RX regulation (on by default; disable with --no-rx), the RX-axis server
    must be running and the RX axis homed.
  - This script must run on the host that owns the ADS1115 + sensor enable GPIOs
    (Raspberry Pi #2).

Select the Pico connection with --pico-port and the RX-axis service with
--rx-server-url or RX_AXIS_SERVER_URL.
Defaults intentionally gentle for initial testing.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time

from openderm.config import default_pico_port
from openderm.motion.gantry.server import (
    GantryServerError,
)
from openderm.motion.rx_axis.server import (
    RxAxisServerClient,
    RxAxisServerError,
)
from openderm.sensors.hg_c import (
    HgCSensorError,
    build_sensor_controller,
)


DEFAULT_RX_SERVER_URL = os.getenv("RX_AXIS_SERVER_URL", "http://127.0.0.1:8091")
DEFAULT_TARGET_MM = 110.0
DEFAULT_GAIN_MM_PER_MM = 0.5
DEFAULT_MAX_STEP_MM = 2.0
DEFAULT_SEARCH_STEP_MM = 2.0
DEFAULT_DEADBAND_MM = 0.5
DEFAULT_PERIOD_S = 0.1
DEFAULT_SAMPLES = 1
DEFAULT_FEED_MM_MIN: float | None = None
DEFAULT_REPORT_INTERVAL_S = 1.0

# Reference-rig profile used by the documented OpenDerm hardware workflow.
# The sign remains mount-dependent; flip it during bring-up if error diverges.
DEFAULT_RX_GAIN_RAD_PER_MM = -0.005
DEFAULT_RX_MAX_STEP_RAD = 0.004
DEFAULT_RX_DEADBAND_MM = 0.5
DEFAULT_RX_SPEED_RAD_S: float | None = 0.02
# Exponential moving-average weight on the RX error (d1 - d2). Each loop the
# filtered error becomes alpha*raw + (1-alpha)*previous. Lower = smoother but
# laggier; 1.0 disables filtering. The raw difference is noisy, so smoothing it
# is the main lever against RX jitter.
DEFAULT_RX_FILTER_ALPHA = 0.2


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def run(args: argparse.Namespace) -> int:
    if not args.regulate_z and not args.regulate_rx:
        print(
            "Nothing to regulate: --no-z and --no-rx cannot both be set.",
            file=sys.stderr,
        )
        return 1
    if not 0.0 < args.rx_filter_alpha <= 1.0:
        print("--rx-filter-alpha must be in (0, 1].", file=sys.stderr)
        return 1

    # Z is controlled by the Raspberry Pi Pico. Keep its firmware [0, 392] mm
    # travel backstop enabled; this regulator has no separate Z window.
    pico_link = None
    if args.regulate_z:
        try:
            from openderm.motion.pico.adapter import PicoAxisClient, open_pico_link

            print(f"connecting to Pico (Z) on {args.pico_port} ...", file=sys.stderr)
            pico_link = open_pico_link(args.pico_port, timeout_s=30.0)
            client = PicoAxisClient(
                pico_link,
                "z",
                enforce_limits=True,
                vmax_mm_s=args.pico_vmax_mm_s,
                acc_mm_s2=args.pico_acc_mm_s2,
                timeout_s=30.0,
            )
            if args.home_z:
                print(f"homing Z via Pico on {args.pico_port} ...", file=sys.stderr)
                client.home()
            print(f"Pico link up (Z) on {args.pico_port}.", file=sys.stderr)
        except Exception as exc:  # ImportError(pyserial) / serial / firmware not responding
            print(f"could not start the Z Pico on {args.pico_port}: {exc}", file=sys.stderr)
            return 1

    if args.regulate_z:
        try:
            state = client.status()
        except GantryServerError as exc:
            print(f"Z Pico not reachable at {args.pico_port}: {exc}", file=sys.stderr)
            return 1
        if "z" not in state.get("homed_axes", []):
            print(
                "z-axis is not homed. Re-run with --home-z, or home it first with "
                "`openderm --axis z home`.",
                file=sys.stderr,
            )
            return 1

    rx_client: RxAxisServerClient | None = None
    if args.regulate_rx:
        rx_client = RxAxisServerClient(args.rx_server_url, timeout_s=30.0)
        try:
            # /state returns 409 (raised as RxAxisServerError) until the axis
            # is homed, so a successful status() doubles as the homed check.
            rx_client.status()
        except RxAxisServerError as exc:
            print(
                f"RX-axis server not reachable or rx axis not homed at "
                f"{args.rx_server_url}: {exc}\n"
                "Home it with `openderm --axis rx home`, or pass --no-rx to "
                "regulate z only.",
                file=sys.stderr,
            )
            return 1

    controller = build_sensor_controller()
    stop_requested = False

    def handle_sigint(signum, frame):  # noqa: ANN001
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, handle_sigint)

    def read_z() -> float | None:
        try:
            snapshot = client.status()
        except GantryServerError as exc:
            print(f"warning: z position read failed: {exc}", file=sys.stderr)
            return None
        position = snapshot.get("position") or {}
        z = position.get("z")
        return None if z is None else float(z)

    def step_z(delta_mm: float) -> bool:
        """Move z by delta_mm. In stream mode, update the live setpoint (no wait);
        otherwise issue a blocking relative move. Returns False if rejected."""
        z = read_z()
        if z is None:
            return False
        try:
            if args.stream:
                # Push the new absolute setpoint; the server's stream worker
                # chases it continuously without settling between corrections.
                client.stream_to(z + delta_mm)
            else:
                client.move_to(
                    z + delta_mm,
                    feed_mm_min=args.feed_mm_min,
                    tolerance_mm=args.tolerance_mm,
                )
            return True
        except GantryServerError as exc:
            # Most likely cause: target outside the server's z travel limits.
            print(f"  z move rejected (likely travel limit): {exc}", file=sys.stderr)
            return False

    def step_rx(delta_rad: float) -> bool:
        """Nudge the RX axis by delta_rad (relative). move_by reads the current
        position from the server and issues an absolute target. Returns False if
        rejected (soft limit, not homed, or a latched safety shutdown)."""
        try:
            rx_client.move_by(delta_rad, speed_rad_s=args.rx_speed_rad_s)
            return True
        except RxAxisServerError as exc:
            print(
                f"  rx move rejected (likely soft limit or safety shutdown): {exc}",
                file=sys.stderr,
            )
            return False

    def read_exclusive(sensor_name: str):
        """Light a single sensor, read it, and switch it back off.

        The two laser spots interfere when lit together, so we never leave more
        than one enabled. set_enabled() already waits settle_time_s after each
        toggle, so the reading is taken after the laser has stabilized and the
        sensor is dark again before the next one is lit."""
        controller.set_enabled(sensor_name, True)
        try:
            return controller.read_sensor(sensor_name, samples=args.samples)
        finally:
            controller.set_enabled(sensor_name, False)

    try:
        # Start with both sensors disabled; read_exclusive() lights exactly one
        # at a time so the two laser spots never interfere.
        controller.set_enabled_for_selection("all", False)
        z_desc = (
            f"z={args.pico_port} mode={'stream' if args.stream else 'blocking'} "
            f"gain={args.gain_mm_per_mm} mm/mm max-step={args.max_step_mm} mm "
            f"search-step={args.search_step_mm} mm deadband={args.deadband_mm} mm "
            f"target={args.target_mm} mm"
            if args.regulate_z
            else "z=disabled"
        )
        rx_desc = (
            f"rx={args.rx_server_url} gain={args.rx_gain_rad_per_mm} rad/mm "
            f"max-step={args.rx_max_step_rad} rad deadband={args.rx_deadband_mm} mm"
            if args.regulate_rx
            else "rx=disabled"
        )
        print(f"regulating | {z_desc} | {rx_desc} | samples={args.samples} | Ctrl-C to exit")

        iterations_since_report = 0
        last_report_time = time.monotonic()
        # Low-pass state for the RX error; reset to None whenever the estimate
        # becomes invalid (a sensor drops out of range) so we don't resume from
        # stale data.
        rx_error_filtered: float | None = None

        while not stop_requested:
            # Read the sensors one at a time so the laser spots never interfere.
            # z regulates on the average of whichever sensors are in range; RX
            # regulates on their difference and needs BOTH in range.
            r1 = read_exclusive("sensor1")
            r2 = read_exclusive("sensor2")
            in_range = [r for r in (r1, r2) if r.in_range and r.distance_mm is not None]
            d1 = "--" if r1.distance_mm is None else f"{r1.distance_mm:.2f}"
            d2 = "--" if r2.distance_mm is None else f"{r2.distance_mm:.2f}"

            # --- z-height: average of in-range sensors -> target ---
            if args.regulate_z:
                if in_range:
                    avg = sum(r.distance_mm for r in in_range) / len(in_range)
                    error = avg - args.target_mm
                    if args.debug:
                        used = "+".join(r.name for r in in_range)
                    if abs(error) <= args.deadband_mm:
                        if args.debug:
                            print(
                                f"  [dbg] d1={d1} d2={d2} avg={avg:.2f} ({used}) "
                                f"err={error:+.3f}mm <= deadband, holding z"
                            )
                    else:
                        # error > 0 (too high) -> increase z; error < 0 -> decrease z.
                        delta = _clamp(args.gain_mm_per_mm * error, args.max_step_mm)
                        if args.debug:
                            print(
                                f"  [dbg] d1={d1} d2={d2} avg={avg:.2f} ({used}) "
                                f"err={error:+.3f}mm -> dz={delta:+.3f}mm"
                            )
                        if not step_z(delta):
                            break
                else:
                    # Neither sensor in range: always increase z.
                    delta = args.search_step_mm
                    if args.debug:
                        print(
                            f"  [dbg] both out-of-range "
                            f"(s1={r1.signal_status}, s2={r2.signal_status}) -> dz={delta:+.3f}mm"
                        )
                    if not step_z(delta):
                        break

            # --- RX angle: difference sensor1 - sensor2 -> 0 ---
            # Only when BOTH sensors are in range; the tilt estimate is otherwise
            # undefined. Independent of the z action above (separate actuator).
            if args.regulate_rx:
                if (
                    r1.in_range
                    and r1.distance_mm is not None
                    and r2.in_range
                    and r2.distance_mm is not None
                ):
                    rx_error_raw = r1.distance_mm - r2.distance_mm
                    # Exponential moving average to reject the per-read sensor
                    # noise that would otherwise dither the RX-axis motor.
                    if rx_error_filtered is None:
                        rx_error_filtered = rx_error_raw
                    else:
                        rx_error_filtered = (
                            args.rx_filter_alpha * rx_error_raw
                            + (1.0 - args.rx_filter_alpha) * rx_error_filtered
                        )
                    if abs(rx_error_filtered) <= args.rx_deadband_mm:
                        if args.debug:
                            print(
                                f"  [dbg] d1-d2 raw={rx_error_raw:+.3f} "
                                f"filt={rx_error_filtered:+.3f}mm <= rx-deadband, holding rx"
                            )
                    else:
                        rx_delta = _clamp(
                            args.rx_gain_rad_per_mm * rx_error_filtered, args.rx_max_step_rad
                        )
                        if args.debug:
                            print(
                                f"  [dbg] d1-d2 raw={rx_error_raw:+.3f} "
                                f"filt={rx_error_filtered:+.3f}mm -> drx={rx_delta:+.4f}rad"
                            )
                        if not step_rx(rx_delta):
                            break
                else:
                    # Tilt estimate undefined: drop the filter so it restarts
                    # cleanly when both sensors return.
                    rx_error_filtered = None
                    if args.debug:
                        print("  [dbg] rx: need both sensors in range, holding rx")

            iterations_since_report += 1
            now = time.monotonic()
            elapsed = now - last_report_time
            if elapsed >= args.report_interval_s:
                hz = iterations_since_report / elapsed
                print(
                    f"  control loop: {hz:.2f} Hz ({iterations_since_report} iters in {elapsed:.2f}s)"
                )
                iterations_since_report = 0
                last_report_time = now

            time.sleep(args.period_s)
    finally:
        try:
            controller.set_enabled_for_selection("all", False)
        except HgCSensorError:
            pass
        try:
            controller.close()
        except Exception:
            pass
        if pico_link is not None:
            try:
                pico_link.close()
            except Exception:
                pass
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Regulate the gantry z-axis so the sensor1/sensor2 average distance "
            "reaches a target (default 110 mm), and the RX angle so sensor1 == "
            "sensor2 (level)."
        )
    )
    parser.add_argument(
        "--pico-port",
        default=default_pico_port(),
        help=(
            "Pico serial port for Z: a local device (/dev/ttyACM0) or a "
            "pyserial URL if the Pico is bridged over the network, e.g. "
            "socket://<pi1-ip>:8095 (see openderm-pico-bridge). Defaults to PICO_PORT "
            "or the serial bridge on the GANTRY_SERVER_URL host."
        ),
    )
    parser.add_argument(
        "--pico-vmax-mm-s",
        type=float,
        default=None,
        help="Z max speed (mm/s) for the Pico controller (default: firmware VMAX).",
    )
    parser.add_argument(
        "--pico-acc-mm-s2",
        type=float,
        default=None,
        help="Z acceleration (mm/s^2) for the Pico controller (default: firmware ACC).",
    )
    parser.add_argument(
        "--home-z",
        action="store_true",
        help="Home Z on the Pico before regulating.",
    )
    parser.add_argument(
        "--target-mm",
        type=float,
        default=DEFAULT_TARGET_MM,
        help="Target sensor-average distance in mm (default %(default)s).",
    )
    parser.add_argument(
        "--no-z",
        dest="regulate_z",
        action="store_false",
        help="Disable z-height regulation (e.g. to debug RX motion on its own).",
    )
    parser.set_defaults(regulate_z=True)
    parser.add_argument(
        "--rx-server-url",
        default=DEFAULT_RX_SERVER_URL,
        help=(
            "Base URL of the RX-axis server controlling the rx axis (default from "
            "RX_AXIS_SERVER_URL or %(default)s)."
        ),
    )
    parser.add_argument(
        "--no-rx",
        dest="regulate_rx",
        action="store_false",
        help="Disable RX (tilt) regulation and regulate the z-height only.",
    )
    parser.set_defaults(regulate_rx=True)
    parser.add_argument(
        "--rx-gain-rad-per-mm",
        type=float,
        default=DEFAULT_RX_GAIN_RAD_PER_MM,
        help=(
            "Proportional gain for RX: rx step = gain * (sensor1 - sensor2), clamped "
            "to ±rx-max-step. Flip the sign if RX diverges (default %(default)s)."
        ),
    )
    parser.add_argument(
        "--rx-max-step-rad",
        type=float,
        default=DEFAULT_RX_MAX_STEP_RAD,
        help="Maximum per-iteration RX move in rad (default %(default)s).",
    )
    parser.add_argument(
        "--rx-deadband-mm",
        type=float,
        default=DEFAULT_RX_DEADBAND_MM,
        help="sensor1-sensor2 difference magnitude (mm) treated as level (default %(default)s).",
    )
    parser.add_argument(
        "--rx-filter-alpha",
        type=float,
        default=DEFAULT_RX_FILTER_ALPHA,
        help=(
            "EMA weight (0-1] on the RX error to smooth jitter: filtered = "
            "alpha*raw + (1-alpha)*prev. Lower is smoother but laggier; 1.0 "
            "disables filtering (default %(default)s)."
        ),
    )
    parser.add_argument(
        "--rx-speed-rad-s",
        type=float,
        default=DEFAULT_RX_SPEED_RAD_S,
        help="Speed in rad/s for RX moves (default: RX-axis server default).",
    )
    parser.add_argument(
        "--gain-mm-per-mm",
        type=float,
        default=DEFAULT_GAIN_MM_PER_MM,
        help=(
            "Proportional gain: z step = gain * (reading - target), clamped to "
            "±max-step (default %(default)s)."
        ),
    )
    parser.add_argument(
        "--max-step-mm",
        type=float,
        default=DEFAULT_MAX_STEP_MM,
        help="Maximum per-iteration z move in mm while in range (default %(default)s).",
    )
    parser.add_argument(
        "--search-step-mm",
        type=float,
        default=DEFAULT_SEARCH_STEP_MM,
        help="Fixed z step in mm used while both sensors are out of range (default %(default)s).",
    )
    parser.add_argument(
        "--deadband-mm",
        type=float,
        default=DEFAULT_DEADBAND_MM,
        help="Average-error magnitude (mm) treated as on-target (default %(default)s).",
    )
    parser.add_argument(
        "--period-s",
        type=float,
        default=DEFAULT_PERIOD_S,
        help="Control loop period in seconds (default %(default)s).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="ADC samples to average per reading (default %(default)s).",
    )
    parser.add_argument(
        "--feed-mm-min",
        type=float,
        default=DEFAULT_FEED_MM_MIN,
        help="Feed rate in mm/min for z moves (default: server default).",
    )
    parser.add_argument(
        "--tolerance-mm",
        type=float,
        default=None,
        help=(
            "Position tolerance for a blocking move to count as complete (default: "
            "server default of 0.05 mm). Looser values finish each move sooner and "
            "raise the loop rate; ignored in --stream mode."
        ),
    )
    parser.add_argument(
        "--report-interval-s",
        type=float,
        default=DEFAULT_REPORT_INTERVAL_S,
        help="How often to print the measured control loop frequency (default %(default)s).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help=(
            "Use the gantry server's streaming mode (continuous setpoint chasing, "
            "no per-move settle wait) instead of blocking moves. Much higher loop rate."
        ),
    )
    parser.add_argument(
        "--stream-tick-s",
        type=float,
        default=0.05,
        help="Stream worker tick period in seconds; only used with --stream (default %(default)s).",
    )
    parser.add_argument(
        "--stream-min-step-mm",
        type=float,
        default=0.01,
        help="Minimum setpoint change that triggers a new stream move (default %(default)s).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-loop controller decisions (reading, error, z step).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
