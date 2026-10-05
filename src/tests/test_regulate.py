"""Hardware-free routing tests for Pico Z regulation.

Drives the real run() loop against fake Pico/RX-axis motor/sensor clients, auto-stopping
after one control iteration (the fake sensor controller fires the captured SIGINT handler).
Verifies Z is homed and moved through the Pico and that the link is closed.
"""

from __future__ import annotations

import contextlib
import io
import unittest
from dataclasses import dataclass
from unittest import mock

from openderm.scanning import regulate


@dataclass
class FakeReading:
    name: str
    distance_mm: float | None
    in_range: bool
    signal_status: str


class FakePicoZClient:
    """Pico Z client implementing the regulator's required motion methods."""

    def __init__(self) -> None:
        self.axis = "z"
        self.z = 300.0
        self.homed_flag = True
        self.homed_calls = 0
        self.stream_calls = 0
        self.move_calls = 0

    def status(self):
        return {
            "position": {"z": self.z},
            "homed_axes": (["z"] if self.homed_flag else []),
            "moving": False,
        }

    def move_to(self, pos, feed_mm_min=None, tolerance_mm=None):
        self.z = float(pos)
        self.move_calls += 1
        return {"status": "completed"}

    def stream_to(self, pos, continuous=False):
        self.z = float(pos)
        self.stream_calls += 1
        return {"status": "queued"}

    def home(self, fast_mm_s=None, slow_mm_s=None):
        self.z = 0.0
        self.homed_flag = True
        self.homed_calls += 1
        return {"status": "homed"}

    def stop(self, mode="soft"):
        return {"status": "stopped"}


class FakeLink:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeRxAxis:
    def status(self):
        return {"position_rad": 0.0}

    def move_by(self, delta_rad, speed_rad_s=None):
        return {"status": "completed"}


class FakeSensors:
    """Reads in-range near the target; after `stop_after` reads, fires the SIGINT handler so
    run()'s loop exits cleanly (rc 0)."""

    def __init__(self, stop_after, on_stop) -> None:
        self.reads = 0
        self.stop_after = stop_after
        self.on_stop = on_stop

    def set_enabled(self, name, on, settle=True):
        pass

    def set_enabled_for_selection(self, sel, on):
        pass

    def read_sensor(self, name, samples=None):
        self.reads += 1
        if self.reads >= self.stop_after:
            self.on_stop()
        return FakeReading(name=name, distance_mm=112.0, in_range=True, signal_status="in_range")

    def close(self):
        pass


def _run(extra_args, pico_client=None):
    captured = {}
    link = FakeLink()

    def stop():
        if "h" in captured:
            captured["h"](2, None)  # SIGINT

    sensors = FakeSensors(stop_after=2, on_stop=stop)
    regulate.RxAxisServerClient = lambda url, *, timeout_s=30.0: FakeRxAxis()
    regulate.build_sensor_controller = lambda: sensors

    args = regulate.build_parser().parse_args(
        [
            "--no-rx",
            "--period-s=0",
            "--report-interval-s=1e9",
            "--samples=1",
            *extra_args,
        ]
    )
    import openderm.motion.pico.adapter as pico_mod

    with (
        mock.patch.object(regulate.signal, "signal", lambda s, h: captured.__setitem__("h", h)),
        mock.patch.object(pico_mod, "open_pico_link", lambda port, timeout_s=30.0: link),
        mock.patch.object(pico_mod, "PicoAxisClient", lambda *a, **k: pico_client),
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        rc = regulate.run(args)
    return rc, link


class ZRegulatePicoTests(unittest.TestCase):
    def test_parser_has_no_z_controller_override(self) -> None:
        parser = regulate.build_parser()
        self.assertFalse(any(action.dest == "z_backend" for action in parser._actions))

    def test_pico_homes_streams_and_closes(self) -> None:
        pico = FakePicoZClient()
        rc, link = _run(
            ["--home-z", "--stream", "--pico-port=dummy://pico"],
            pico_client=pico,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(pico.homed_calls, 1, "--home-z should home Z on the Pico")
        self.assertGreater(pico.stream_calls, 0, "z should be regulated via the Pico (stream_to)")
        self.assertTrue(link.closed, "the shared Pico link must be closed")

    def test_pico_blocking(self) -> None:
        pico = FakePicoZClient()
        rc, link = _run(
            ["--pico-port=dummy://pico"],
            pico_client=pico,
        )
        self.assertEqual(rc, 0)
        self.assertGreater(pico.move_calls, 0, "blocking z regulation goes through move_to")
        self.assertTrue(link.closed)


if __name__ == "__main__":
    unittest.main()
