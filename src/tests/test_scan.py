"""Hardware-free tests for the OpenDerm contour scanner."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock

os.environ.setdefault("OPENDERM_COLLISION_MODE", "off")

from capture.motion.rx_pivot import HarmonicFit, RxPivotModel
from capture.scanning import contour as scan

from contour_fake_world import (
    TARGET_MM,
    FakeController,
    FakeGantryClient,
    FakeRxAxisClient,
    FakePicoAxisClient,
    FakeWorld,
)


def _identity_pivot_model(path: Path) -> None:
    RxPivotModel(
        axis_coeffs={axis: HarmonicFit(0.0, 0.0, 0.0) for axis in ("x", "y", "z")},
        metadata={"purpose": "test"},
    ).save(path)


def _run_sim(
    output_dir: str,
    extra_args: list[str] | None = None,
    *,
    world: FakeWorld | None = None,
    controller_box: dict[str, FakeController] | None = None,
) -> tuple[int, list[dict], FakeWorld, str, str]:
    world = world or FakeWorld()
    extra_args = list(extra_args or [])

    scan.GantryServerClient = lambda _url, *, axis="x", timeout_s=30.0: FakeGantryClient(
        world, axis
    )
    scan.RxAxisServerClient = lambda _url, *, timeout_s=30.0: FakeRxAxisClient(world)

    def make_controller() -> FakeController:
        controller = FakeController(world)
        if controller_box is not None:
            controller_box["controller"] = controller
        return controller

    scan.build_sensor_controller = make_controller

    if not any(arg.startswith("--rx-pivot-model") for arg in extra_args):
        model_path = Path(output_dir) / "pivot-model.json"
        _identity_pivot_model(model_path)
        extra_args.append(f"--rx-pivot-model={model_path}")

    args = scan.build_parser().parse_args(
        [
            "--no-camera",
            "--pico-port=dummy://pico",
            f"--target-mm={TARGET_MM}",
            "--x-step-mm=10",
            "--x-travel-mm=20",
            "--y-step-mm=5",
            "--y-max-travel-mm=80",
            "--y-max-rows=30",
            "--period-s=0",
            "--record-pause-s=0",
            "--station-timeout-s=5",
            "--settle-iters=3",
            "--deadband-mm=0.5",
            "--gain-mm-per-mm=0.5",
            "--max-step-mm=2",
            "--edge-tilt-max-deg=20",
            "--edge-tilt-step-deg=2",
            "--edge-oor-iters=3",
            "--rx-speed-rad-s=1.0",
            "--rx-axis-min-rad=-10",
            "--rx-axis-max-rad=10",
            f"--capture-dir={output_dir}",
            *extra_args,
        ]
    )

    class FakeLink:
        def close(self) -> None:
            pass

    class FakePicoMultiAxis:
        def __init__(self, clients: dict[str, FakePicoAxisClient]) -> None:
            self.clients = clients

        def positions(self) -> dict[str, float]:
            return {axis: float(client.world.pos[axis]) for axis, client in self.clients.items()}

        def stream_to(self, targets: dict[str, float], continuous: bool = False) -> dict[str, str]:
            for axis, target in targets.items():
                self.clients[axis].stream_to(target, continuous=continuous)
            return {"status": "queued"}

    import capture.motion.pico.adapter as pico_adapter

    stdout = io.StringIO()
    stderr = io.StringIO()
    with (
        mock.patch.object(pico_adapter, "open_pico_link", return_value=FakeLink()),
        mock.patch.object(
            pico_adapter,
            "PicoAxisClient",
            side_effect=lambda _link, axis, **_kwargs: FakePicoAxisClient(world, axis),
        ),
        mock.patch.object(pico_adapter, "PicoMultiAxis", FakePicoMultiAxis),
        contextlib.redirect_stdout(stdout),
        contextlib.redirect_stderr(stderr),
    ):
        result = scan.run(args)

    poses_path = Path(output_dir) / "poses.jsonl"
    captures = (
        [json.loads(line) for line in poses_path.read_text().splitlines()]
        if poses_path.exists()
        else []
    )
    return result, captures, world, stdout.getvalue(), stderr.getvalue()


class ScanTraversalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        (
            cls.result,
            cls.captures,
            cls.world,
            cls.stdout,
            cls.stderr,
        ) = _run_sim(cls.temp_dir.name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_scan_completes_and_captures(self) -> None:
        self.assertEqual(self.result, 0, self.stderr)
        self.assertTrue(self.captures)

    def test_x_is_the_fast_axis(self) -> None:
        by_sweep: dict[int, list[dict]] = defaultdict(list)
        for capture in self.captures:
            by_sweep[capture["col"]].append(capture)

        for sweep, captures in by_sweep.items():
            y_positions = {round(capture["y_mm"], 6) for capture in captures}
            self.assertEqual(
                len(y_positions),
                1,
                f"sweep {sweep} spans multiple Y positions: {y_positions}",
            )
            x_positions = [capture["x_mm"] for capture in captures]
            expected = (
                sorted(x_positions)
                if captures[0]["phase"] == "+x"
                else sorted(x_positions, reverse=True)
            )
            self.assertEqual(x_positions, expected)

    def test_sweeps_march_toward_both_flanks(self) -> None:
        y_positions = [capture["y_mm"] for capture in self.captures]
        self.assertLess(min(y_positions), 0.0)
        self.assertGreater(max(y_positions), 0.0)
        self.assertEqual(
            {capture["phase"] for capture in self.captures},
            {"+x", "-x"},
        )

    def test_robot_parks_after_the_scan(self) -> None:
        self.assertAlmostEqual(self.world.rx, scan.PARK_RX_RAD)


class ScanSafetyAndControlTests(unittest.TestCase):
    def test_edge_recovery_can_follow_a_y_flank(self) -> None:
        world = FakeWorld(body={0.0: (100.0, 200.0)})
        with tempfile.TemporaryDirectory() as output_dir:
            result, captures, _world, stdout, stderr = _run_sim(
                output_dir,
                [
                    "--x-travel-mm=0",
                    "--y-max-rows=1",
                    "--band-edge-recovery",
                ],
                world=world,
            )
        self.assertEqual(result, 0, stderr)
        self.assertFalse(captures)
        self.assertIn("probing edge", stdout)

    def test_floor_filter_prevents_bed_capture(self) -> None:
        world = FakeWorld(
            body={0.0: (100.0, 200.0)},
            floor_dist0=130.0,
            floor_rx_gain=10.0,
        )
        with tempfile.TemporaryDirectory() as output_dir:
            result, captures, _world, _stdout, stderr = _run_sim(
                output_dir,
                [
                    "--x-travel-mm=0",
                    "--y-max-rows=1",
                    "--floor-depth-mm=430",
                    "--floor-margin-mm=6",
                    "--band-edge-recovery",
                ],
                world=world,
            )
        self.assertEqual(result, 0, stderr)
        self.assertFalse(captures, "bed readings were captured as skin")

    def test_velocity_servo_settles_at_surface_normal(self) -> None:
        world = FakeWorld(
            body={0.0: (-400.0, 400.0)},
            k_diff=40.0,
            rx_level=0.3,
            slew_rx_rad_s=0.5,
            slew_xyz_mm_s=50.0,
            slew_dt_s=0.02,
            rx_speed_floor_rad_s=0.033,
        )
        with tempfile.TemporaryDirectory() as output_dir:
            result, captures, _world, _stdout, stderr = _run_sim(
                output_dir,
                [
                    "--x-travel-mm=0",
                    "--y-max-rows=1",
                    "--rx-speed-rad-s=0.5",
                    "--rx-velocity-slew-rad-s2=0",
                    "--station-timeout-s=3",
                    "--period-s=0",
                    "--rx-gain-rad-per-mm=-0.01",
                    "--rx-filter-alpha=0.4",
                ],
                world=world,
            )
        self.assertEqual(result, 0, stderr)
        self.assertTrue(captures)
        self.assertTrue(
            all(capture["station_status"] == scan.STATION_SETTLED for capture in captures)
        )
        self.assertAlmostEqual(captures[0]["rx_rad"], 0.3, delta=0.08)

    def test_simultaneous_sensor_mode_darkens_lasers_for_capture(self) -> None:
        controller_box: dict[str, FakeController] = {}
        with tempfile.TemporaryDirectory() as output_dir:
            result, captures, _world, _stdout, stderr = _run_sim(
                output_dir,
                [
                    "--x-travel-mm=0",
                    "--y-max-rows=1",
                    "--simultaneous-sensors",
                ],
                controller_box=controller_box,
            )
        self.assertEqual(result, 0, stderr)
        self.assertTrue(captures)
        log = controller_box["controller"].enable_log
        self.assertIn(("sensor1", True), log)
        self.assertIn(("sensor2", True), log)
        self.assertGreaterEqual(log.count(("all", False)), 2)

    def test_missing_pivot_model_fails_before_motion(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            missing = Path(output_dir) / "missing.json"
            result, captures, _world, _stdout, stderr = _run_sim(
                output_dir,
                [f"--rx-pivot-model={missing}"],
            )
        self.assertEqual(result, 1)
        self.assertFalse(captures)
        self.assertIn("rx-pivot model", stderr)


class ScanRecordingTests(unittest.TestCase):
    def test_record_writes_monotonic_trace_with_capture_and_park(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            result, captures, _world, _stdout, stderr = _run_sim(
                output_dir,
                ["--x-travel-mm=0", "--y-max-rows=1", "--record"],
            )
            trace_path = Path(output_dir) / "trace.jsonl"
            records = [json.loads(line) for line in trace_path.read_text().splitlines()]

        self.assertEqual(result, 0, stderr)
        self.assertTrue(captures)
        self.assertTrue(records)
        self.assertEqual(
            [record["seq"] for record in records],
            list(range(len(records))),
        )
        times = [record["t_s"] for record in records]
        self.assertEqual(times, sorted(times))
        activities = {record["activity"] for record in records}
        self.assertIn("capture", activities)
        self.assertIn("park", activities)

    def test_record_is_off_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            result, _captures, _world, _stdout, stderr = _run_sim(
                output_dir,
                ["--x-travel-mm=0", "--y-max-rows=1"],
            )
            self.assertEqual(result, 0, stderr)
            self.assertFalse((Path(output_dir) / "trace.jsonl").exists())


class ScanMathTests(unittest.TestCase):
    def test_collision_startup_pose_read_fails_closed(self) -> None:
        guard = mock.Mock()
        guard.min_clearance.return_value = 100.0

        with self.assertRaisesRegex(RuntimeError, "unreadable"):
            scan._read_startup_collision_pose(
                guard,
                lambda: {"x": 1.0, "y": 2.0},
                lambda: 0.5,
            )
        with self.assertRaisesRegex(RuntimeError, "unreadable"):
            scan._read_startup_collision_pose(
                guard,
                lambda: (_ for _ in ()).throw(OSError("position link down")),
                lambda: 0.5,
            )

        position, rx, clearance = scan._read_startup_collision_pose(
            guard,
            lambda: {"x": 1.0, "y": 2.0, "z": 3.0},
            lambda: 0.5,
        )
        self.assertEqual(position, {"x": 1.0, "y": 2.0, "z": 3.0})
        self.assertEqual(rx, 0.5)
        self.assertEqual(clearance, 100.0)

    def test_edge_side_from_rx(self) -> None:
        self.assertEqual(scan._edge_side_from_rx(0.25, 0.2, 1.6), 1)
        self.assertEqual(scan._edge_side_from_rx(1.55, 0.2, 1.6), -1)
        self.assertIsNone(scan._edge_side_from_rx(0.9, 0.2, 1.6))
        self.assertIsNone(scan._edge_side_from_rx(None, 0.2, 1.6))

    def test_pivot_rate_budget_uses_tightest_axis(self) -> None:
        self.assertAlmostEqual(
            scan._pivot_rate_budget(0.0, 257.0, 40.0, 50.0, 100.0, 100.0, 0.8),
            0.8 * 100.0 / 257.0,
        )
        self.assertIsNone(scan._pivot_rate_budget(0.0, 0.0, 0.0, 50.0, 100.0, 100.0, 0.8))

    def test_pico_batch_cruises_only_when_every_axis_advances(self) -> None:
        previous = {"y": 100.0, "z": 300.0}
        self.assertTrue(scan._cruise_batch(previous, {"y": 103.0, "z": 304.0}, 1.0))
        self.assertFalse(scan._cruise_batch(previous, {"y": 103.0, "z": 300.3}, 1.0))


if __name__ == "__main__":
    unittest.main()
