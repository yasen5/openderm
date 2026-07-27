"""Hardware-free simulation of floor_depth_tare.run().

Drives the tare against the FakeWorld from contour_fake_world.py with a body
spanning everything -- i.e. a flat "bed" whose reading follows z exactly like the
real bed. The invariant under test: the fake surface satisfies
z + avg = Z_AT_TARGET + TARGET_MM = 418 at ANY z, so the tare must regulate from
a wrong starting z to the target standoff and report floor_depth_mm = 418
regardless of where z started.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import floor_depth_tare as fdt
from contour_fake_world import (
    TARGET_MM,
    Z_AT_TARGET,
    FakeController,
    FakeRxAxisClient,
    FakePicoAxisClient,
    FakeWorld,
)


def _run_tare(world, extra_args=None):
    fdt.RxAxisServerClient = lambda url, *, timeout_s=5.0: FakeRxAxisClient(world)
    fdt.build_sensor_controller = lambda: FakeController(world)
    import openderm.motion.pico.adapter as pico_mod

    class FakeLink:
        def close(self):
            pass

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "floor_depth.json"
        args = fdt.build_parser().parse_args(
            [
                f"--target-mm={TARGET_MM}",
                "--period-s=0",
                "--settle-iters=3",
                "--tare-samples=5",
                f"--out={out}",
                *(extra_args or []),
            ]
        )
        with (
            mock.patch.object(pico_mod, "open_pico_link", return_value=FakeLink()),
            mock.patch.object(
                pico_mod,
                "PicoAxisClient",
                side_effect=lambda _link, axis, **_kwargs: FakePicoAxisClient(world, axis),
            ),
        ):
            rc = fdt.run(args)
        payload = json.loads(out.read_text()) if out.exists() else None
    return rc, payload


class FloorDepthTareSweepTests(unittest.TestCase):
    """--rx-sweep: sample the bed at several rx angles and fit z+d vs rx.

    World: no body under the head. The fake sensor models beam obliquity, so
    z+d varies smoothly and nonlinearly with RX; the quadratic floor model must
    fit every reachable sample with a small residual."""

    def test_sweep_fits_rx_dependence(self) -> None:
        world = FakeWorld(
            body={0.0: (500.0, 600.0)},  # far away: every read is the bed
            floor_dist0=130.0,
            floor_rx_gain=20.0,
        )
        rc, payload = _run_tare(
            world,
            ["--rx-sweep=0.10:1.20:6", "--target-mm=120", "--timeout-s=5"],
        )
        self.assertEqual(rc, 0)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["mode"], "sweep")
        self.assertEqual(len(payload["samples"]), 6)
        self.assertEqual(payload["skipped"], [])
        # rx sample points span the requested range.
        rxs = [s["rx_rad"] for s in payload["samples"]]
        self.assertAlmostEqual(rxs[0], 0.10, delta=0.02)
        self.assertAlmostEqual(rxs[-1], 1.20, delta=0.02)
        # The quadratic accurately models the sampled oblique-beam depths.
        fit = payload["fit"]
        self.assertIsNotNone(fit)
        self.assertLess(fit["rms_mm"], 1.0)
        self.assertAlmostEqual(fit["rx_min"], 0.10, delta=0.02)
        self.assertAlmostEqual(fit["rx_max"], 1.20, delta=0.02)

    def test_sweep_skips_unreachable_tilts(self) -> None:
        # Holding the standoff at steeper tilt needs deeper z; a z ceiling at
        # 330 leaves only rx=0.35 reachable. The sweep must
        # SKIP the unreachable points (per-sample timeout), keep the one good
        # sample, and fall back to a constant (no fit with < 3 points).
        world = FakeWorld(
            body={0.0: (500.0, 600.0)},
            floor_dist0=130.0,
            floor_rx_gain=20.0,
            z_ceiling=330.0,
        )
        rc, payload = _run_tare(
            world,
            ["--rx-sweep=0.35:1.5:6", "--target-mm=120", "--timeout-s=2"],
        )
        self.assertEqual(rc, 0)  # partial success, loudly warned -- not an abort
        self.assertEqual(len(payload["samples"]), 1)
        self.assertEqual(len(payload["skipped"]), 5)
        self.assertIsNone(payload["fit"])
        self.assertAlmostEqual(payload["floor_depth_mm"], 445.0, delta=2.0)


class FloorDepthTareTests(unittest.TestCase):
    def test_tare_settles_and_reports_absolute_depth(self) -> None:
        # Flat surface everywhere; start z=300 (reads 118, off target by 8mm).
        world = FakeWorld(body={0.0: (-1000.0, 1000.0)})
        rc, payload = _run_tare(world)
        self.assertEqual(rc, 0)
        self.assertIsNotNone(payload)
        # z regulated to the standoff...
        self.assertAlmostEqual(world.pos["z"], Z_AT_TARGET, delta=1.5)
        # ...and the recorded absolute depth is the invariant z + d = 418,
        # independent of the starting z.
        self.assertAlmostEqual(payload["floor_depth_mm"], Z_AT_TARGET + TARGET_MM, delta=1.5)
        self.assertAlmostEqual(payload["sensor1_depth_mm"], payload["sensor2_depth_mm"], delta=0.1)
        self.assertEqual(payload["rx_rad"], 0.0)

    def test_tare_aborts_when_no_surface_within_travel_bound(self) -> None:
        # No body anywhere near: every read is below_range, the bounded search
        # descends and must ABORT at --max-travel-mm, not hunt forever.
        world = FakeWorld(body={0.0: (500.0, 600.0)})  # far away in y: never hit
        rc, payload = _run_tare(world, ["--max-travel-mm=20"])
        self.assertEqual(rc, 1)
        self.assertIsNone(payload)
        # z never descended past the bound (start 300 + 20 + one step slack).
        self.assertLessEqual(world.pos["z"], 300.0 + 20.0 + 2.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
