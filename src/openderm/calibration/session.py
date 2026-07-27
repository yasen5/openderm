"""Shared hardware lifecycle and I/O helpers for calibration scripts."""

from __future__ import annotations

import signal
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from openderm.motion.gantry.server import GantryServerError
from openderm.motion.rx_axis.server import RxAxisServerError
from openderm.sensors.hg_c import HgCSensorError


@dataclass(frozen=True)
class AxisSetup:
    """Configuration for one axis on a shared Pico connection."""

    vmax_mm_s: float | None = None
    acc_mm_s2: float | None = None
    home: bool = False


class CalibrationSession:
    """Own calibration hardware and expose consistent, safe I/O operations.

    The scripts retain their calibration-specific control loops; this class owns
    the repeated connection, homing checks, exclusive laser reads, relative
    movement, interrupt handling, and best-effort shutdown.
    """

    def __init__(self, *, log: Callable[[str], None] | None = None) -> None:
        self.axes: dict[str, Any] = {}
        self.rx: Any | None = None
        self.sensors: Any | None = None
        self.stop_requested = False
        self._resources: list[Any] = []
        self._previous_sigint_handler: Any | None = None
        self._log = log or (lambda message: print(message, file=sys.stderr))

    def set_logger(self, log: Callable[[str], None]) -> None:
        self._log = log

    def add_axis(self, axis: str, client: Any) -> Any:
        self.axes[axis] = client
        return client

    def connect_pico_axes(
        self,
        port: str,
        setups: Mapping[str, AxisSetup],
        *,
        open_link: Callable[..., Any],
        axis_client_factory: Callable[..., Any],
        timeout_s: float,
    ) -> None:
        """Open one Pico link and configure all requested axes on it."""
        link = open_link(port, timeout_s=timeout_s)
        self._resources.append(link)
        for axis, setup in setups.items():
            client = axis_client_factory(
                link,
                axis,
                enforce_limits=True,
                vmax_mm_s=setup.vmax_mm_s,
                acc_mm_s2=setup.acc_mm_s2,
                timeout_s=timeout_s,
            )
            if setup.home:
                self._log(f"homing {axis.upper()} via Pico on {port} ...")
                client.home()
            self.add_axis(axis, client)

    def connect_rx(
        self,
        url: str,
        *,
        client_factory: Callable[..., Any],
        timeout_s: float,
    ) -> Any:
        self.rx = client_factory(url, timeout_s=timeout_s)
        return self.rx

    def connect_sensors(self, factory: Callable[[], Any]) -> Any:
        self.sensors = factory()
        self.sensors.set_enabled_for_selection("all", False)
        return self.sensors

    def install_sigint_handler(self) -> None:
        if self._previous_sigint_handler is not None:
            return
        self._previous_sigint_handler = signal.getsignal(signal.SIGINT)

        def request_stop(_signum: int, _frame: Any) -> None:
            self.stop_requested = True

        signal.signal(signal.SIGINT, request_stop)

    def is_homed(self, axis: str) -> bool:
        state = self.axes[axis].status()
        return axis in (state.get("homed_axes") or [])

    def read_axis(self, axis: str) -> float | None:
        try:
            state = self.axes[axis].status()
        except GantryServerError as exc:
            self._log(f"warning: {axis} position read failed: {exc}")
            return None
        value = (state.get("position") or {}).get(axis)
        return None if value is None else float(value)

    def read_rx(self) -> float | None:
        if self.rx is None:
            return None
        try:
            value = self.rx.status().get("position_rad")
        except RxAxisServerError as exc:
            self._log(f"warning: rx position read failed: {exc}")
            return None
        return None if value is None else float(value)

    def move_axis_relative(
        self,
        axis: str,
        delta_mm: float,
        *,
        feed_mm_min: float | None = None,
        tolerance_mm: float | None = None,
        prefer_move_by: bool = False,
    ) -> bool:
        client = self.axes[axis]
        try:
            if prefer_move_by:
                client.move_by(delta_mm, feed_mm_min=feed_mm_min)
            else:
                position = self.read_axis(axis)
                if position is None:
                    return False
                kwargs = {"feed_mm_min": feed_mm_min}
                if tolerance_mm is not None:
                    kwargs["tolerance_mm"] = tolerance_mm
                client.move_to(position + delta_mm, **kwargs)
        except GantryServerError as exc:
            self._log(f"  {axis} move rejected (likely travel limit): {exc}")
            return False
        return True

    def move_rx_to(
        self,
        target_rad: float,
        *,
        speed_rad_s: float | None = None,
        accel_rad_s2: float | None = None,
        settle_tolerance_rad: float | None = None,
        timeout_s: float = 30.0,
        poll_period_s: float = 0.1,
    ) -> bool:
        if self.rx is None:
            return False
        try:
            kwargs = {"speed_rad_s": speed_rad_s}
            if accel_rad_s2 is not None:
                kwargs["accel_rad_s2"] = accel_rad_s2
            self.rx.move_to(target_rad, **kwargs)
        except RxAxisServerError as exc:
            self._log(f"  rx move to {target_rad:+.3f}rad rejected: {exc}")
            return False

        if settle_tolerance_rad is None:
            return True
        deadline = time.monotonic() + timeout_s
        while not self.stop_requested:
            position = self.read_rx()
            if position is not None and abs(position - target_rad) <= settle_tolerance_rad:
                return True
            if time.monotonic() >= deadline:
                self._log(
                    f"  warning: rx did not reach {target_rad:+.3f}rad in "
                    f"{timeout_s:g}s; continuing at the measured angle."
                )
                return True
            time.sleep(max(poll_period_s, 0.02))
        return False

    def read_exclusive(self, sensor_name: str, *, samples: int) -> Any:
        """Read one sensor with the other laser guaranteed to remain off."""
        if self.sensors is None:
            raise RuntimeError("sensor controller is not connected")
        self.sensors.set_enabled(sensor_name, True)
        try:
            return self.sensors.read_sensor(sensor_name, samples=samples)
        finally:
            self.sensors.set_enabled(sensor_name, False)

    def read_pair(self, *, samples: int) -> tuple[Any, Any]:
        return (
            self.read_exclusive("sensor1", samples=samples),
            self.read_exclusive("sensor2", samples=samples),
        )

    @staticmethod
    def average_distance(readings: Iterable[Any]) -> float | None:
        valid = [
            reading.distance_mm
            for reading in readings
            if reading.in_range and reading.distance_mm is not None
        ]
        return None if not valid else sum(valid) / len(valid)

    def close(self) -> None:
        """Leave lasers off and close owned resources, even after partial setup."""
        if self._previous_sigint_handler is not None:
            signal.signal(signal.SIGINT, self._previous_sigint_handler)
            self._previous_sigint_handler = None
        if self.sensors is not None:
            try:
                self.sensors.set_enabled_for_selection("all", False)
            except HgCSensorError:
                pass
            try:
                self.sensors.close()
            except Exception:
                pass
            self.sensors = None
        while self._resources:
            resource = self._resources.pop()
            try:
                resource.close()
            except Exception:
                pass

    def __enter__(self) -> CalibrationSession:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()
