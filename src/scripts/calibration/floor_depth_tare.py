#!/usr/bin/env python3
"""Tare the bed/floor absolute depth for the scanner's --floor-depth-mm filter.

The scanner can reject sensor readings of the bed by their absolute depth: the
bed is at a fixed machine height, so gantry z + sensor distance is approximately
constant for any bed hit and larger than for skin above it. This utility
measures that reference.

Two modes:

SINGLE-POINT (default). You park the head over BARE BED (no body part under the
lasers) at a tilt representative of scanning, the script regulates z so the
sensor average sits at --target-mm (scan-like geometry), then medians
--tare-samples rounds of z + d per sensor. RX is NOT moved; the current angle is
only recorded. Pass the printed value to the scans as --floor-depth-mm.

RX SWEEP (--rx-sweep lo:hi:N, e.g. 0.35:1.5:6). The sensor distance is measured
along the BEAM, so the bed's z+d drifts with the tilt (path obliquity + the
sensor origin riding the rx lever arm) -- tens of mm across a wide working range,
more than the scans' --floor-margin-mm. The sweep steps rx through N angles,
re-settles z and takes a single-point tare AT EACH, then least-squares fits
    z + d  =  c0 + c1*rx + c2*rx^2
over the reachable samples and stores the fit in the output JSON. Point the
scans at that file with --floor-model captures/floor_depth.json and the
rejection threshold follows the live rx. A sample where z cannot reach the
standoff (the bed drops out of z travel as the tilt steepens -- holding
--target-mm along a steeper beam needs a DEEPER z) is SKIPPED with a warning and
the fit covers the reachable range; the scans clamp rx into that range when
evaluating (steeper tilts read the bed even deeper, so clamping errs toward
rejection). Fewer than 3 reachable samples -> no fit, constant fallback.

The Z regulator is bounded for calibration: a crash-imminent ``above_range`` on
either sensor retreats Z, search travel is limited by ``--max-travel-mm`` per
sample, and every settle is limited by ``--timeout-s``.

Run on the host that owns the sensors (Pi 2), with Pico Z homed. The RX-axis
server is required for --rx-sweep (it commands rx
over the bare bed; make sure the swept range is clear) and optional otherwise
(only records the tilt).

Typical use:
  python src/scripts/calibration/floor_depth_tare.py \
      --pico-port socket://openderm-gantry.local:8095 \
      --target-mm 120 --rx-sweep 0.35:1.5:6
then scan with:
  --floor-model captures/floor_depth.json --floor-margin-mm 10
keeping the margin below the body's thickness-above-bed at the edges. (Taring at
--target-mm 120 while scanning at 110 is fine: z+d is a property of the bed, not
the standoff; the second-order error is a few mm at the steepest tilts.)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from openderm.calibration import AxisSetup, CalibrationSession
from openderm.motion.rx_axis.server import (
    RxAxisServerClient,
    RxAxisServerError,
)
from openderm.sensors.hg_c import build_sensor_controller


DEFAULT_RX_SERVER_URL = os.getenv("RX_AXIS_SERVER_URL", "http://127.0.0.1:8091")
# Regulation defaults match openderm-regulate and openderm-scan.
DEFAULT_TARGET_MM = 110.0
DEFAULT_GAIN_MM_PER_MM = 0.5
DEFAULT_MAX_STEP_MM = 2.0
DEFAULT_SEARCH_STEP_MM = 2.0
DEFAULT_DEADBAND_MM = 0.5
DEFAULT_PERIOD_S = 0.1
DEFAULT_SAMPLES = 5
DEFAULT_SETTLE_ITERS = 5
# Tare bounds. The head is parked near the bed by hand, so the floor should be
# found within a short travel; a larger excursion means the head is NOT over
# bare bed (or the pose is wrong) -- abort/skip rather than hunt.
DEFAULT_MAX_TRAVEL_MM = 80.0
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_TARE_SAMPLES = 10
DEFAULT_RX_SPEED_RAD_S = 0.1
DEFAULT_RX_ACCEL_RAD_S2: float | None = None
DEFAULT_RX_SETTLE_TOL_RAD = 0.01
DEFAULT_OUT = str(REPO_ROOT / "captures" / "floor_depth.json")


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _parse_rx_sweep(spec: str) -> list[float]:
    """Parse 'lo:hi:N' into N evenly spaced rx targets, inclusive of both ends."""
    try:
        lo_s, hi_s, n_s = spec.split(":")
        lo, hi, n = float(lo_s), float(hi_s), int(n_s)
    except ValueError as exc:
        raise ValueError(f"--rx-sweep must be lo:hi:N (rad:rad:count), got {spec!r}") from exc
    if n < 2:
        raise ValueError("--rx-sweep needs N >= 2 sample points")
    if hi <= lo:
        raise ValueError("--rx-sweep needs hi > lo")
    return [lo + i * (hi - lo) / (n - 1) for i in range(n)]


def _polyfit2(xs: list[float], ys: list[float]) -> tuple[list[float], float]:
    """Pure-python least-squares quadratic fit. Returns ([c0, c1, c2], rms_mm).
    Normal equations solved by Gaussian elimination with partial pivoting --
    3x3, a handful of samples, no numpy dependency needed."""
    s = [0.0] * 5
    for x in xs:
        p = 1.0
        for k in range(5):
            s[k] += p
            p *= x
    t = [0.0] * 3
    for x, y in zip(xs, ys):
        t[0] += y
        t[1] += x * y
        t[2] += x * x * y
    a = [
        [s[0], s[1], s[2], t[0]],
        [s[1], s[2], s[3], t[1]],
        [s[2], s[3], s[4], t[2]],
    ]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-12:
            raise ValueError("degenerate fit (rx samples too clustered)")
        a[col], a[piv] = a[piv], a[col]
        for r in range(col + 1, 3):
            f = a[r][col] / a[col][col]
            for c in range(col, 4):
                a[r][c] -= f * a[col][c]
    coeffs = [0.0] * 3
    for r in (2, 1, 0):
        v = a[r][3] - sum(a[r][k] * coeffs[k] for k in range(r + 1, 3))
        coeffs[r] = v / a[r][r]
    n = len(xs)
    rms = (
        sum((coeffs[0] + coeffs[1] * x + coeffs[2] * x * x - y) ** 2 for x, y in zip(xs, ys)) / n
    ) ** 0.5
    return coeffs, rms


class _FloorTareWorkflow:
    """Floor-depth calibration logic over an initialized hardware session."""

    def __init__(
        self,
        session: CalibrationSession,
        args: argparse.Namespace,
        *,
        sweep_targets: list[float] | None,
        initial_rx_rad: float | None,
    ) -> None:
        self.session = session
        self.args = args
        self.sweep_targets = sweep_targets
        self.initial_rx_rad = initial_rx_rad

    def _step_z(self, delta_mm: float) -> bool:
        return self.session.move_axis_relative(
            "z",
            delta_mm,
            feed_mm_min=self.args.feed_mm_min,
            tolerance_mm=self.args.tolerance_mm,
        )

    def _read_pair(self):
        return self.session.read_pair(samples=self.args.samples)

    def _settle_to_target(self) -> str:
        """Regulate Z until the bed reading is stable at the target distance."""
        z0 = self.session.read_axis("z")
        if z0 is None:
            return "z-limit"
        on_target = 0
        deadline = time.monotonic() + self.args.timeout_s

        while not self.session.stop_requested:
            r1, r2 = self._read_pair()
            d1 = "--" if r1.distance_mm is None else f"{r1.distance_mm:.2f}"
            d2 = "--" if r2.distance_mm is None else f"{r2.distance_mm:.2f}"
            in_range = [
                reading
                for reading in (r1, r2)
                if reading.in_range and reading.distance_mm is not None
            ]

            if any(reading.signal_status == "above_range" for reading in (r1, r2)):
                on_target = 0
                if self.args.debug:
                    print(f"  [dbg] d1={d1} d2={d2} above_range -> retreat")
                if not self._step_z(-abs(self.args.max_step_mm)):
                    return "z-limit"
            elif not in_range:
                on_target = 0
                if self.args.debug:
                    print(
                        f"  [dbg] out of range (s1={r1.signal_status}, "
                        f"s2={r2.signal_status}) -> dz=+{self.args.search_step_mm:.1f}"
                    )
                if not self._step_z(self.args.search_step_mm):
                    return "z-limit"
            else:
                average = sum(reading.distance_mm for reading in in_range) / len(in_range)
                error = average - self.args.target_mm
                if abs(error) <= self.args.deadband_mm:
                    on_target += 1
                    if self.args.debug:
                        print(
                            f"  [dbg] d1={d1} d2={d2} avg={average:.2f} "
                            f"err={error:+.3f} on-target "
                            f"{on_target}/{self.args.settle_iters}"
                        )
                    if on_target >= self.args.settle_iters:
                        return "settled"
                else:
                    on_target = 0
                    delta = _clamp(
                        self.args.gain_mm_per_mm * error,
                        self.args.max_step_mm,
                    )
                    if self.args.debug:
                        print(
                            f"  [dbg] d1={d1} d2={d2} avg={average:.2f} "
                            f"err={error:+.3f} -> dz={delta:+.3f}"
                        )
                    if not self._step_z(delta):
                        return "z-limit"

            z_now = self.session.read_axis("z")
            if z_now is not None and abs(z_now - z0) > self.args.max_travel_mm:
                return "travel"
            if time.monotonic() >= deadline:
                return "timeout"
            time.sleep(self.args.period_s)
        return "abort"

    def _measure(self) -> tuple[float, float, int, int] | None:
        depths1: list[float] = []
        depths2: list[float] = []
        for _ in range(self.args.tare_samples):
            if self.session.stop_requested:
                return None
            r1, r2 = self._read_pair()
            z = self.session.read_axis("z")
            if z is not None:
                if r1.in_range and r1.distance_mm is not None:
                    depths1.append(z + r1.distance_mm)
                if r2.in_range and r2.distance_mm is not None:
                    depths2.append(z + r2.distance_mm)
            time.sleep(self.args.period_s)

        if len(depths1) < 3 or len(depths2) < 3:
            print(
                f"  too few valid samples (s1 {len(depths1)}, s2 {len(depths2)} "
                f"of {self.args.tare_samples}) -- readings dropped out of range "
                "after settling.",
                file=sys.stderr,
            )
            return None
        return (
            statistics.median(depths1),
            statistics.median(depths2),
            len(depths1),
            len(depths2),
        )

    def _write_out(self, payload: dict) -> Path:
        out = Path(self.args.out)
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2))
        except OSError as exc:
            print(f"warning: could not write {out}: {exc}", file=sys.stderr)
        return out

    def _settle_and_measure(self) -> tuple[float, float, int, int] | None:
        status = self._settle_to_target()
        if status == "abort":
            return None
        if status != "settled":
            reason = {
                "z-limit": "z move rejected (bed beyond z travel at this target?)",
                "travel": (
                    f"z moved > --max-travel-mm {self.args.max_travel_mm:.0f}mm "
                    "without settling -- is the head really over bare bed?"
                ),
                "timeout": (f"did not settle within --timeout-s {self.args.timeout_s:.0f}s"),
            }[status]
            print(f"Aborting tare: {reason}.", file=sys.stderr)
            return None
        return self._measure()

    def _run_single_point(self) -> int:
        measurement = self._settle_and_measure()
        if measurement is None:
            print("Aborting tare: no measurement recorded.", file=sys.stderr)
            return 1

        med1, med2, n1, n2 = measurement
        floor_depth = (med1 + med2) / 2.0
        split = abs(med1 - med2)
        z_end = self.session.read_axis("z")
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mode": "point",
            "floor_depth_mm": round(floor_depth, 2),
            "sensor1_depth_mm": round(med1, 2),
            "sensor2_depth_mm": round(med2, 2),
            "sensor_split_mm": round(split, 2),
            "samples_used": {"sensor1": n1, "sensor2": n2},
            "z_mm": None if z_end is None else round(z_end, 2),
            "target_mm": self.args.target_mm,
            "rx_rad": (None if self.initial_rx_rad is None else round(self.initial_rx_rad, 4)),
        }
        out = self._write_out(payload)
        print(
            f"\nfloor depth (z + d): {floor_depth:.1f}mm   "
            f"[s1 {med1:.1f}, s2 {med2:.1f}, split {split:.1f}mm] -> {out}"
        )
        if split > 5.0:
            print(
                f"  warning: the two beams disagree by {split:.1f}mm -- "
                "re-check the bare, flat bed tare pose.",
                file=sys.stderr,
            )
        print(f"scan with:  --floor-depth-mm {floor_depth:.0f} --floor-margin-mm 10")
        return 0

    def _collect_sweep_samples(self) -> tuple[list[dict], list[dict]]:
        samples: list[dict] = []
        skipped: list[dict] = []
        assert self.sweep_targets is not None

        for target in self.sweep_targets:
            if self.session.stop_requested:
                break
            print(f"--- sweep point rx={target:+.3f}rad ---")
            moved = self.session.move_rx_to(
                target,
                speed_rad_s=self.args.rx_speed_rad_s,
                accel_rad_s2=self.args.rx_accel_rad_s2,
                settle_tolerance_rad=self.args.rx_settle_tol_rad,
                poll_period_s=self.args.period_s,
            )
            if not moved:
                skipped.append({"rx_rad": round(target, 4), "reason": "rx-move"})
                continue

            status = self._settle_to_target()
            if status == "abort":
                break
            if status != "settled":
                print(
                    f"  skipping rx={target:+.3f}rad: {status} "
                    "(bed unreachable within the z travel/time bounds).",
                    file=sys.stderr,
                )
                skipped.append({"rx_rad": round(target, 4), "reason": status})
                continue

            measurement = self._measure()
            if measurement is None:
                skipped.append({"rx_rad": round(target, 4), "reason": "sensor-dropout"})
                continue
            med1, med2, n1, n2 = measurement
            measured_rx = self.session.read_rx()
            used_rx = target if measured_rx is None else measured_rx
            z_end = self.session.read_axis("z")
            depth = (med1 + med2) / 2.0
            samples.append(
                {
                    "rx_rad": round(used_rx, 4),
                    "depth_mm": round(depth, 2),
                    "sensor1_depth_mm": round(med1, 2),
                    "sensor2_depth_mm": round(med2, 2),
                    "samples_used": {"sensor1": n1, "sensor2": n2},
                    "z_mm": None if z_end is None else round(z_end, 2),
                }
            )
            print(f"  rx={used_rx:+.3f}rad: z+d = {depth:.1f}mm [s1 {med1:.1f}, s2 {med2:.1f}]")
        return samples, skipped

    @staticmethod
    def _fit_sweep(samples: list[dict]) -> dict | None:
        if len(samples) < 3:
            print(
                f"warning: only {len(samples)} reachable sweep point(s) (< 3): "
                "no rx fit; using a constant floor depth.",
                file=sys.stderr,
            )
            return None
        xs = [sample["rx_rad"] for sample in samples]
        ys = [sample["depth_mm"] for sample in samples]
        try:
            coeffs, rms = _polyfit2(xs, ys)
        except ValueError as exc:
            print(f"warning: fit failed ({exc}); constant fallback only.", file=sys.stderr)
            return None
        return {
            "kind": "poly2",
            "coeffs": [round(coefficient, 6) for coefficient in coeffs],
            "rx_min": round(min(xs), 4),
            "rx_max": round(max(xs), 4),
            "rms_mm": round(rms, 3),
        }

    def _run_sweep(self) -> int:
        samples, skipped = self._collect_sweep_samples()
        if self.session.stop_requested:
            print("Aborted (Ctrl-C) mid-sweep; nothing recorded.", file=sys.stderr)
            return 1
        if not samples:
            print(
                "Aborting tare: no sweep point was reachable. Check the pose or "
                "adjust the target and travel bounds.",
                file=sys.stderr,
            )
            return 1

        fit = self._fit_sweep(samples)
        constant_depth = sum(sample["depth_mm"] for sample in samples) / len(samples)
        max_split = max(
            abs(sample["sensor1_depth_mm"] - sample["sensor2_depth_mm"]) for sample in samples
        )
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mode": "sweep",
            "floor_depth_mm": round(constant_depth, 2),
            "sensor_split_mm": round(max_split, 2),
            "target_mm": self.args.target_mm,
            "rx_sweep": self.args.rx_sweep,
            "samples": samples,
            "skipped": skipped,
            "fit": fit,
        }
        out = self._write_out(payload)
        print(f"\nsweep: {len(samples)} point(s) sampled, {len(skipped)} skipped -> {out}")
        if fit is None:
            print(f"scan with:  --floor-depth-mm {constant_depth:.0f} --floor-margin-mm 10")
        else:
            print(f"scan with:  --floor-model {out} --floor-margin-mm 10")
        if max_split > 5.0:
            print(
                f"  warning: the two beams disagree by up to {max_split:.1f}mm across the sweep.",
                file=sys.stderr,
            )
        return 0

    def run(self) -> int:
        z_start = self.session.read_axis("z")
        if z_start is None:
            print("Could not read the current z position.", file=sys.stderr)
            return 1
        mode = (
            f"rx-sweep {self.sweep_targets[0]:+.2f}.."
            f"{self.sweep_targets[-1]:+.2f}rad ({len(self.sweep_targets)} pts)"
            if self.sweep_targets is not None
            else "single point"
        )
        print(
            f"floor tare: {mode} | target={self.args.target_mm}mm "
            f"z start={z_start:.2f}mm | park over BARE BED | Ctrl-C to abort"
        )
        if self.sweep_targets is None:
            return self._run_single_point()
        return self._run_sweep()


def _validate_args(args: argparse.Namespace) -> list[float] | None:
    if args.settle_iters < 1:
        raise ValueError("--settle-iters must be at least 1.")
    if args.tare_samples < 3:
        raise ValueError("--tare-samples must be at least 3 (median needs a few).")
    if args.max_travel_mm <= 0:
        raise ValueError("--max-travel-mm must be positive.")
    return _parse_rx_sweep(args.rx_sweep) if args.rx_sweep else None


def _connect_session(
    args: argparse.Namespace,
    sweep_targets: list[float] | None,
) -> tuple[CalibrationSession, float | None] | None:
    from openderm.motion.pico.adapter import PicoAxisClient, open_pico_link

    session = CalibrationSession()
    try:
        print(f"connecting to Pico (Z) on {args.pico_port} ...", file=sys.stderr)
        session.connect_pico_axes(
            args.pico_port,
            {
                "z": AxisSetup(
                    vmax_mm_s=args.pico_vmax_mm_s,
                    acc_mm_s2=args.pico_acc_mm_s2,
                    home=args.home_z,
                )
            },
            open_link=open_pico_link,
            axis_client_factory=PicoAxisClient,
            timeout_s=30.0,
        )
        print(f"Pico link up (Z) on {args.pico_port}.", file=sys.stderr)
        if not session.is_homed("z"):
            print(
                "z-axis is not homed. Re-run with --home-z, or home it first.",
                file=sys.stderr,
            )
            session.close()
            return None
    except Exception as exc:
        print(f"could not start the Z Pico on {args.pico_port}: {exc}", file=sys.stderr)
        session.close()
        return None

    initial_rx_rad: float | None = None
    try:
        rx = session.connect_rx(
            args.rx_server_url,
            client_factory=RxAxisServerClient,
            timeout_s=10.0,
        )
        initial_rx_rad = float(rx.status().get("position_rad"))
    except (RxAxisServerError, TypeError, ValueError) as exc:
        session.rx = None
        if sweep_targets is not None:
            print(
                f"--rx-sweep needs the RX-axis server at {args.rx_server_url}: {exc}",
                file=sys.stderr,
            )
            session.close()
            return None
        print(f"note: rx angle not recorded ({exc}).", file=sys.stderr)

    try:
        session.connect_sensors(build_sensor_controller)
        session.install_sigint_handler()
    except Exception:
        session.close()
        raise
    return session, initial_rx_rad


def run(args: argparse.Namespace) -> int:
    try:
        sweep_targets = _validate_args(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    connected = _connect_session(args, sweep_targets)
    if connected is None:
        return 1
    session, initial_rx_rad = connected
    try:
        return _FloorTareWorkflow(
            session,
            args,
            sweep_targets=sweep_targets,
            initial_rx_rad=initial_rx_rad,
        ).run()
    finally:
        session.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the bed/floor absolute depth (gantry z + sensor distance) for "
            "the contour scans' floor-rejection filter: park the head over BARE "
            "BED, and this regulates z to the target standoff on the bed, samples "
            "both sensors, and records the median z + d -- at the current tilt "
            "(default), or swept over an rx range and fitted (--rx-sweep) for the "
            "scans' --floor-model rx-dependent threshold."
        )
    )
    parser.add_argument(
        "--pico-port",
        default=os.getenv("PICO_PORT", "/dev/ttyACM0"),
        help=(
            "Pico serial port for Z: a local device or a pyserial URL "
            "(e.g. socket://<pi1-ip>:8095). Defaults to PICO_PORT or /dev/ttyACM0."
        ),
    )
    parser.add_argument(
        "--pico-vmax-mm-s",
        type=float,
        default=None,
        help="Z max speed (mm/s) for the Pico (default: firmware VMAX).",
    )
    parser.add_argument(
        "--pico-acc-mm-s2",
        type=float,
        default=None,
        help="Z acceleration (mm/s^2) for the Pico (default: firmware ACC).",
    )
    parser.add_argument(
        "--home-z",
        action="store_true",
        help="Home Z on the Pico before taring.",
    )
    parser.add_argument(
        "--rx-server-url",
        default=DEFAULT_RX_SERVER_URL,
        help=(
            "RX-axis server URL (default from RX_AXIS_SERVER_URL or "
            "%(default)s). REQUIRED (rx homed) for "
            "--rx-sweep; otherwise only used to record the tare's tilt "
            "(unreachable tolerated)."
        ),
    )
    parser.add_argument(
        "--rx-sweep",
        default=None,
        metavar="LO:HI:N",
        help=(
            "Sweep mode: sample the floor depth at N rx angles from LO to HI rad "
            "(e.g. 0.35:1.5:6), settle z at each, and fit z+d = c0 + c1*rx + "
            "c2*rx^2 for the scans' --floor-model. Angles the bed cannot be "
            "reached at (z travel) are skipped and the fit covers the reachable "
            "range. The head MUST be over bare bed for the whole swept range. "
            "Default: off (single point at the current rx, which is never moved)."
        ),
    )
    parser.add_argument(
        "--rx-speed-rad-s",
        type=float,
        default=DEFAULT_RX_SPEED_RAD_S,
        help="Speed for sweep rx moves (default %(default)s).",
    )
    parser.add_argument(
        "--rx-accel-rad-s2",
        type=float,
        default=DEFAULT_RX_ACCEL_RAD_S2,
        help="Max acceleration for sweep rx moves (default: RX-axis server default).",
    )
    parser.add_argument(
        "--rx-settle-tol-rad",
        type=float,
        default=DEFAULT_RX_SETTLE_TOL_RAD,
        help="Arrival tolerance for sweep rx moves (default %(default)s).",
    )
    parser.add_argument(
        "--target-mm",
        type=float,
        default=DEFAULT_TARGET_MM,
        help=(
            "Standoff to settle at on the bed before taring (default %(default)s). "
            "Use a scan-like value; taring at e.g. 120 for a scan run at 110 is "
            "fine (z+d is a property of the bed -- the second-order error is a few "
            "mm at the steepest tilts)."
        ),
    )
    parser.add_argument(
        "--gain-mm-per-mm",
        type=float,
        default=DEFAULT_GAIN_MM_PER_MM,
        help="Proportional z gain (default %(default)s).",
    )
    parser.add_argument(
        "--max-step-mm",
        type=float,
        default=DEFAULT_MAX_STEP_MM,
        help="Maximum per-iteration z move in mm (default %(default)s).",
    )
    parser.add_argument(
        "--search-step-mm",
        type=float,
        default=DEFAULT_SEARCH_STEP_MM,
        help="Fixed descent step while both sensors read below range (default %(default)s).",
    )
    parser.add_argument(
        "--deadband-mm",
        type=float,
        default=DEFAULT_DEADBAND_MM,
        help="Average-error magnitude treated as on-target (default %(default)s).",
    )
    parser.add_argument(
        "--settle-iters",
        type=int,
        default=DEFAULT_SETTLE_ITERS,
        help="Consecutive in-deadband reads required before taring (default %(default)s).",
    )
    parser.add_argument(
        "--tare-samples",
        type=int,
        default=DEFAULT_TARE_SAMPLES,
        help="Measurement rounds per point; the median is used (default %(default)s).",
    )
    parser.add_argument(
        "--max-travel-mm",
        type=float,
        default=DEFAULT_MAX_TRAVEL_MM,
        help=(
            "Skip/abort if z moves farther than this from a settle's starting "
            "position without settling (default %(default)s) -- the head should be "
            "parked NEAR the bed, so a long hunt means a wrong pose."
        ),
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="Per-settle timeout (default %(default)s).",
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
        help="ADC samples to average per sensor reading (default %(default)s).",
    )
    parser.add_argument(
        "--feed-mm-min",
        type=float,
        default=None,
        help="Feed rate in mm/min for z moves (default: server default).",
    )
    parser.add_argument(
        "--tolerance-mm",
        type=float,
        default=None,
        help="Position tolerance for blocking z moves (default: server default).",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help="Where to write the tare JSON (default %(default)s).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-loop controller decisions.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
