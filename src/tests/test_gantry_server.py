from __future__ import annotations

import argparse
import asyncio
import json
import os
import unittest
from unittest import mock
from urllib import error


from capture.config import GantryConfig
from capture.motion.gantry.server import (
    GantryCoordinatorService,
    GantryServerClient,
    GantryServerError,
    _receive_stream_messages,
    config_from_args,
    position_payload,
    state_payload,
    x_only_gantry_config,
)
from capture.motion.gantry.state import StateStore, project_position
from capture.motion.gantry.manager import MotionError, MotionManager
from capture.motion.gantry.moonraker_coordinator import MoonrakerCoordinatorClient


class FakeMoonraker:
    def __init__(self) -> None:
        self.scripts: list[str] = []
        self.emergency_stops = 0
        self.firmware_restarts = 0

    async def gcode_script(self, script: str) -> dict[str, object]:
        self.scripts.append(script)
        return {"ok": True}

    async def emergency_stop(self) -> dict[str, object]:
        self.emergency_stops += 1
        return {"ok": True}

    async def firmware_restart(self) -> dict[str, object]:
        self.firmware_restarts += 1
        return {"ok": True}


class MotionManagerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.config = GantryConfig(axis="x")
        self.state_store = StateStore()
        self.moonraker = FakeMoonraker()
        self.manager = MotionManager(self.config, self.state_store, self.moonraker)
        await self.state_store.update(
            position={"x": 10.0},
            homed_axes="x",
            stale=False,
            enabled=True,
            webhooks_state="ready",
        )
        await self.manager.start()

    async def asyncTearDown(self) -> None:
        await self.manager.stop()

    async def test_soft_stop_cancels_queued_moves(self) -> None:
        await self.manager.enqueue_move(
            target={"x": 25.0}, feed_mm_s=10.0, tolerance_mm=0.05, commander_id=None
        )
        second = await self.manager.enqueue_move(
            target={"x": 40.0}, feed_mm_s=10.0, tolerance_mm=0.05, commander_id=None
        )
        result = await self.manager.request_soft_stop()
        self.assertEqual(result["stopped"], "soft")
        self.assertIn(second.move_id, result["cancelled_move_ids"])

    async def test_move_after_soft_stop_clears_the_latch_and_runs(self) -> None:
        # The Ctrl-C trap (regression): an aborted scan soft-stops the server,
        # which latches stop_requested -- and the worker loop cancelled EVERY
        # subsequent move at dequeue ('soft_stop') because the only move-side
        # clear was unreachable past that check. A NEW move enqueued after the
        # stop must clear the latch and execute, like start_stream always did.
        await self.manager.request_soft_stop()
        self.assertTrue(self.state_store.get().stop_requested)
        record = await self.manager.enqueue_move(
            target={"x": 25.0}, feed_mm_s=10.0, tolerance_mm=0.05, commander_id=None
        )
        self.assertFalse(self.state_store.get().stop_requested)
        # The fake Moonraker never reports arrival, so the move can't complete;
        # what matters is that it STARTS EXECUTING (status leaves 'queued' for
        # 'active') instead of being cancelled 'soft_stop' at dequeue.
        for _ in range(100):
            if record.status != "queued":
                break
            await asyncio.sleep(0.01)
        self.assertNotEqual(
            (record.status, record.error),
            ("cancelled", "soft_stop"),
            "move after soft stop was cancelled by the stale latch",
        )
        self.assertIn(record.status, ("active", "completed"))

    async def test_emergency_stop_preempts_active_move_and_cancels_queue(self) -> None:
        command_started = asyncio.Event()

        async def blocking_gcode(script: str) -> dict[str, object]:
            self.moonraker.scripts.append(script)
            command_started.set()
            await asyncio.Event().wait()
            return {"ok": True}

        self.moonraker.gcode_script = blocking_gcode
        active = await self.manager.enqueue_move(
            target={"x": 25.0},
            feed_mm_s=10.0,
            tolerance_mm=0.05,
            commander_id=None,
        )
        await asyncio.wait_for(command_started.wait(), timeout=1.0)
        self.assertEqual(active.status, "active")

        queued = await self.manager.enqueue_move(
            target={"x": 40.0},
            feed_mm_s=10.0,
            tolerance_mm=0.05,
            commander_id=None,
        )
        result = await self.manager.request_emergency_stop()

        self.assertEqual(self.moonraker.emergency_stops, 1)
        self.assertEqual(result["active_move_id"], active.move_id)
        self.assertIn(queued.move_id, result["cancelled_move_ids"])
        self.assertEqual(queued.status, "cancelled")
        await asyncio.wait_for(active.waiter.wait(), timeout=1.0)
        self.assertEqual(active.status, "cancelled")
        self.assertEqual(active.error, "emergency_stop")
        self.assertTrue(self.manager.emergency_latched)
        self.assertEqual(self.state_store.get().fault, "emergency_stop")
        self.assertTrue(self.state_store.get().emergency_latched)
        self.assertFalse(self.manager._worker_task.done())

    async def test_emergency_stop_stops_stream_and_blocks_motion_until_reset(self) -> None:
        await self.manager.start_stream(feed_mm_s=10.0, tick_s=0.01)
        await self.manager.request_emergency_stop()

        self.assertFalse(self.manager.is_streaming)
        self.assertFalse(self.state_store.get().streaming)
        with self.assertRaisesRegex(MotionError, "emergency_stop_latched"):
            await self.manager.enqueue_move(
                target={"x": 25.0},
                feed_mm_s=10.0,
                tolerance_mm=0.05,
                commander_id=None,
            )
        with self.assertRaisesRegex(MotionError, "emergency_stop_latched"):
            await self.manager.enqueue_home(("x",))
        with self.assertRaisesRegex(MotionError, "emergency_stop_latched"):
            await self.manager.start_stream(feed_mm_s=10.0)

    async def test_successful_reset_clears_emergency_latch_and_requires_rehoming(self) -> None:
        await self.manager.request_emergency_stop()
        reset = await self.manager.enqueue_control("reset")

        for _ in range(100):
            if self.state_store.get().webhooks_state == "restarting":
                break
            await asyncio.sleep(0.01)
        self.assertEqual(self.state_store.get().webhooks_state, "restarting")
        await self.state_store.update(
            webhooks_state="ready",
            stale=False,
            fault=None,
        )
        await asyncio.wait_for(reset.waiter.wait(), timeout=1.0)

        self.assertEqual(reset.status, "completed")
        self.assertFalse(self.manager.emergency_latched)
        self.assertFalse(self.state_store.get().emergency_latched)
        self.assertEqual(self.state_store.get().homed_axes, "")
        self.assertEqual(self.moonraker.firmware_restarts, 1)

    async def test_emergency_delivery_failure_stays_latched(self) -> None:
        self.moonraker.emergency_stop = mock.AsyncMock(
            side_effect=RuntimeError("moonraker offline")
        )

        with self.assertRaisesRegex(MotionError, "delivery_failed"):
            await self.manager.request_emergency_stop()

        self.assertTrue(self.manager.emergency_latched)
        self.assertTrue(self.state_store.get().emergency_latched)
        self.assertEqual(
            self.state_store.get().fault,
            "emergency_stop_delivery_failed",
        )

    async def test_disable_rejects_while_moving(self) -> None:
        await self.state_store.update(is_moving=True)
        record = await self.manager.enqueue_control("disable")
        await record.waiter.wait()
        self.assertEqual(record.status, "failed")
        self.assertEqual(record.error, "disable_conflict")

    async def test_move_start_seeds_projection_with_commanded_velocity(self) -> None:
        await self.state_store.update(last_commanded_target={"x": 50.0}, is_moving=True)
        await self.state_store.update(velocity_mm_s=10.0, timestamp=10.0)
        projected = project_position(self.state_store.get(), now=12.0)
        self.assertGreater(projected["x"], 10.0)

    async def test_stream_emits_gcode_for_target_without_blocking(self) -> None:
        result = await self.manager.start_stream(feed_mm_s=10.0, tick_s=0.01)
        self.assertTrue(result["streaming"])
        self.assertTrue(self.manager.is_streaming)
        self.assertTrue(self.state_store.get().streaming)
        await self.manager.update_stream_target({"x": 25.0})
        # Give the stream worker a few ticks to emit the move.
        await asyncio.sleep(0.05)
        await self.manager.stop_stream()
        self.assertFalse(self.manager.is_streaming)
        self.assertFalse(self.state_store.get().streaming)
        self.assertTrue(any("X25" in script for script in self.moonraker.scripts))

    async def test_stream_target_clamps_to_travel_limits(self) -> None:
        await self.manager.start_stream(feed_mm_s=10.0, tick_s=0.01)
        resolved = await self.manager.update_stream_target({"x": 1e9})
        await self.manager.stop_stream()
        # x is not the configured axis here, so it clamps to linear_axis_limits.
        self.assertLess(resolved["x"], 1e9)

    async def test_start_stream_rejects_when_blocking_move_queued(self) -> None:
        await self.manager.enqueue_move(
            target={"x": 99.0}, feed_mm_s=1.0, tolerance_mm=0.05, commander_id=None
        )
        with self.assertRaises(Exception):
            await self.manager.start_stream(feed_mm_s=10.0)

    async def test_update_stream_target_requires_active_stream(self) -> None:
        with self.assertRaises(Exception):
            await self.manager.update_stream_target({"x": 5.0})


class MoonrakerCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_partial_status_update_keeps_projected_motion_continuous(self) -> None:
        store = StateStore()
        client = MoonrakerCoordinatorClient("ws://example.test", store)
        await store.update(
            position={"x": 20.0},
            velocity_mm_s=10.0,
            timestamp=10.0,
            is_moving=True,
            last_commanded_target={"x": 50.0},
            stale=False,
        )
        with mock.patch(
            "capture.motion.gantry.moonraker_coordinator.time.monotonic", return_value=11.0
        ):
            await client._publish_status({"motion_report": {"live_velocity": 10.0}}, stale=False)
        self.assertAlmostEqual(store.get().position["x"], 30.0)


class GantryServiceHealthTests(unittest.IsolatedAsyncioTestCase):
    async def test_health_reports_dependencies_and_faults(self) -> None:
        service = GantryCoordinatorService(GantryConfig(axis="x"))
        initial = service.health_payload()
        self.assertFalse(initial["ok"])
        self.assertIn("moonraker_loop_not_running", initial["issues"])
        self.assertIn("motion_worker_not_running", initial["issues"])
        self.assertIn("state_stale", initial["issues"])

        keep_running = asyncio.Event()
        service._moonraker_task = asyncio.create_task(keep_running.wait())
        service.moonraker._ws = mock.AsyncMock()
        service.moonraker._connected.set()
        await service.motion.start()
        await service.state_store.update(
            stale=False,
            webhooks_state="ready",
            fault=None,
            emergency_latched=False,
        )
        try:
            healthy = service.health_payload()
            self.assertTrue(healthy["ok"])
            self.assertEqual(healthy["issues"], [])

            await service.state_store.update(fault="klipper_shutdown")
            degraded = service.health_payload()
            self.assertFalse(degraded["ok"])
            self.assertIn("klipper_fault", degraded["issues"])
        finally:
            service._moonraker_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await service._moonraker_task
            await service.motion.stop()


class StreamReceiveTests(unittest.IsolatedAsyncioTestCase):
    async def test_websocket_disconnect_exits_cleanly(self) -> None:
        class FakeDisconnect(Exception):
            pass

        class DisconnectedWebSocket:
            async def receive_text(self) -> str:
                raise FakeDisconnect()

        motion = mock.Mock()
        await _receive_stream_messages(
            DisconnectedWebSocket(),
            motion,
            FakeDisconnect,
        )
        motion.register_commander_ping.assert_not_called()


class PayloadTests(unittest.TestCase):
    def test_position_payload_projects_expected_shape(self) -> None:
        store = StateStore()
        snapshot = store.get()
        payload = position_payload(snapshot)
        self.assertIn("position", payload)
        self.assertIn("velocity", payload)
        self.assertIn("timestamp", payload)

    def test_state_payload_includes_fault_and_queue(self) -> None:
        store = StateStore()
        snapshot = store.get()
        payload = state_payload(snapshot)
        self.assertIn("fault", payload)
        self.assertIn("queue_depth", payload)
        self.assertIn("emergency_latched", payload)
        self.assertNotIn("enabled_motion_axes", payload)


class ClientTests(unittest.TestCase):
    def test_client_status_parses_json(self) -> None:
        client = GantryServerClient("http://example.test", axis="x")
        fake_response = mock.MagicMock()
        fake_response.read.return_value = b'{"position": {"x": 12.5}}'
        fake_response.__enter__.return_value = fake_response
        fake_response.__exit__.return_value = False
        with mock.patch(
            "capture.motion.gantry.server.request.urlopen", return_value=fake_response
        ):
            status = client.status()
        self.assertEqual(status["position"]["x"], 12.5)

    def test_client_sends_control_token(self) -> None:
        client = GantryServerClient(
            "http://example.test",
            axis="x",
            control_token="test-control-token",
        )
        fake_response = mock.MagicMock()
        fake_response.read.return_value = b'{"ok": true}'
        fake_response.__enter__.return_value = fake_response
        fake_response.__exit__.return_value = False
        with mock.patch(
            "capture.motion.gantry.server.request.urlopen",
            return_value=fake_response,
        ) as urlopen:
            client.status()
        sent_request = urlopen.call_args.args[0]
        self.assertEqual(
            sent_request.get_header("Authorization"),
            "Bearer test-control-token",
        )

    def test_client_wraps_network_errors(self) -> None:
        client = GantryServerClient("http://example.test", axis="x")
        with mock.patch(
            "capture.motion.gantry.server.request.urlopen", side_effect=error.URLError("boom")
        ):
            with self.assertRaises(GantryServerError):
                client.status()

    def _capture_move_payload(self, **move_kwargs: object) -> dict:
        client = GantryServerClient("http://example.test", axis="x")
        fake_response = mock.MagicMock()
        fake_response.read.return_value = b'{"status": "queued"}'
        fake_response.__enter__.return_value = fake_response
        fake_response.__exit__.return_value = False
        with mock.patch(
            "capture.motion.gantry.server.request.urlopen", return_value=fake_response
        ) as urlopen:
            client.move_to(50.0, **move_kwargs)
        sent_request = urlopen.call_args.args[0]
        return json.loads(sent_request.data.decode("utf-8"))

    def test_move_to_includes_tolerance_when_set(self) -> None:
        body = self._capture_move_payload(feed_mm_min=900, tolerance_mm=0.3)
        self.assertEqual(body["x"], 50.0)
        self.assertTrue(body["blocking"])
        self.assertEqual(body["feed_mm_s"], 15.0)
        self.assertEqual(body["tolerance_mm"], 0.3)

    def test_move_to_omits_tolerance_when_unset(self) -> None:
        body = self._capture_move_payload(feed_mm_min=900)
        self.assertNotIn("tolerance_mm", body)

    def test_stop_defaults_to_emergency_mode(self) -> None:
        client = GantryServerClient("http://example.test", axis="x")
        fake_response = mock.MagicMock()
        fake_response.read.return_value = b'{"stopped": "emergency"}'
        fake_response.__enter__.return_value = fake_response
        fake_response.__exit__.return_value = False
        with mock.patch(
            "capture.motion.gantry.server.request.urlopen",
            return_value=fake_response,
        ) as urlopen:
            client.stop()
        sent_request = urlopen.call_args.args[0]
        self.assertEqual(
            json.loads(sent_request.data.decode("utf-8")),
            {"mode": "emergency"},
        )

    def test_client_rejects_non_x_axis(self) -> None:
        with self.assertRaisesRegex(ValueError, "controls X only"):
            GantryServerClient("http://example.test", axis="y")

    def test_move_xyz_rejects_pico_axes(self) -> None:
        client = GantryServerClient("http://example.test", axis="x")
        with self.assertRaisesRegex(GantryServerError, "controls X only"):
            client.move_xyz({"x": 10.0, "z": 20.0})


class ConfigTests(unittest.TestCase):
    def test_config_from_args_builds_expected_config(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--host")
        parser.add_argument("--port", type=int)
        args = parser.parse_args(["--host", "127.0.0.1", "--port", "9005"])
        config = config_from_args(args)
        self.assertEqual(config.host, "127.0.0.1")
        self.assertEqual(config.port, 9005)

    def test_config_rejects_unauthenticated_remote_bind(self) -> None:
        args = argparse.Namespace(host="0.0.0.0", port=8090)
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(GantryServerError, "Refusing to bind"):
                config_from_args(args)

    def test_config_accepts_authenticated_remote_bind(self) -> None:
        args = argparse.Namespace(host="0.0.0.0", port=8090)
        with mock.patch.dict(
            os.environ,
            {"OPENDERM_CONTROL_TOKEN": "test-control-token"},
            clear=True,
        ):
            config = config_from_args(args)
        self.assertEqual(config.control_token, "test-control-token")

    def test_coordinator_is_x_only_even_with_generic_config(self) -> None:
        config = x_only_gantry_config(GantryConfig(axis="z"))
        self.assertEqual(config.axis, "x")
        self.assertEqual((config.travel_min_mm, config.travel_max_mm), (0.0, 800.0))


if __name__ == "__main__":
    unittest.main()
