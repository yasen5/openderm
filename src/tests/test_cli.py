from __future__ import annotations

import argparse
import io
import unittest
from unittest import mock


from openderm.app.cli import (
    build_parser,
    main,
    run_can_via_server,
    run_linear_via_pico,
    run_linear_via_server,
    serialize_status,
)
from openderm.motion.gantry.server import GantryServerError


class FakeServerClient:
    def __init__(self, axis: str = "z") -> None:
        self.calls: list[tuple[str, object, object]] = []
        self.axis = axis

    def status(self) -> dict[str, object]:
        self.calls.append(("status", None, None))
        return {"position": {self.axis: 100.0}}

    def move_to(self, position_mm: float, feed_mm_min: float | None = None) -> dict[str, object]:
        self.calls.append(("move_to", position_mm, feed_mm_min))
        return {"move_id": "mv_test", "status": "completed", "target": {self.axis: position_mm}}

    def home(self) -> dict[str, object]:
        self.calls.append(("home", None, None))
        return {"ok": True}

    def stop(self, mode: str = "emergency") -> dict[str, object]:
        self.calls.append(("stop", mode, None))
        return {"ok": True}


class FakeRxAxisClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object, object]] = []

    def status(self) -> dict[str, object]:
        self.calls.append(("status", None, None))
        return {"position_rad": 1.25}

    def home(self) -> dict[str, object]:
        self.calls.append(("home", None, None))
        return {"status": "completed", "final_position_rad": 0.56}

    def limit_switches(self) -> dict[str, object]:
        self.calls.append(("limit_switches", None, None))
        return {"rx_left_pressed": False, "rx_right_pressed": False}

    def move_to(self, position_rad: float, speed_rad_s: float | None = None) -> dict[str, object]:
        self.calls.append(("move_to", position_rad, speed_rad_s))
        return {"position_rad": position_rad, "last_command": {"velocity_rad_s": speed_rad_s}}

    def move_by(self, delta_rad: float, speed_rad_s: float | None = None) -> dict[str, object]:
        self.calls.append(("move_by", delta_rad, speed_rad_s))
        return {"position_rad": 1.25 + delta_rad}

    def stop(self) -> dict[str, object]:
        self.calls.append(("stop", None, None))
        return {"ok": True}

    def clear_errors(self) -> dict[str, object]:
        self.calls.append(("clear_errors", None, None))
        return {"ok": True}


class CliTests(unittest.TestCase):
    def test_serialize_status_accepts_dict(self) -> None:
        rendered = serialize_status({"ok": True})
        self.assertIn('"ok": true', rendered)

    def test_run_linear_via_server_move_by_reads_status_then_moves(self) -> None:
        client = FakeServerClient()
        args = argparse.Namespace(command="move-by", delta=25.0, feed=600.0)
        with mock.patch("sys.stdout", new_callable=io.StringIO):
            run_linear_via_server(args, client)
        self.assertEqual(client.calls[0], ("status", None, None))
        self.assertEqual(client.calls[1], ("move_to", 125.0, 600.0))

    def test_run_linear_stop_requests_emergency_mode(self) -> None:
        client = FakeServerClient(axis="x")
        args = argparse.Namespace(command="stop")
        with mock.patch("sys.stdout", new_callable=io.StringIO):
            run_linear_via_server(args, client)
        self.assertEqual(client.calls, [("stop", "emergency", None)])

    def test_run_can_via_server_move_to_uses_rx_axis_client(self) -> None:
        client = FakeRxAxisClient()
        args = argparse.Namespace(command="move-to", position=2.0, speed=0.5)
        with mock.patch("sys.stdout", new_callable=io.StringIO):
            run_can_via_server(args, client)
        self.assertEqual(client.calls[0], ("move_to", 2.0, 0.5))

    def test_run_can_via_server_move_by_uses_rx_axis_client(self) -> None:
        client = FakeRxAxisClient()
        args = argparse.Namespace(command="move-by", delta=-0.25, speed=0.5)
        with mock.patch("sys.stdout", new_callable=io.StringIO):
            run_can_via_server(args, client)
        self.assertEqual(client.calls[0], ("move_by", -0.25, 0.5))

    def test_run_can_via_server_home_uses_rx_axis_client(self) -> None:
        client = FakeRxAxisClient()
        args = argparse.Namespace(command="home")
        with mock.patch("sys.stdout", new_callable=io.StringIO):
            run_can_via_server(args, client)
        self.assertEqual(client.calls[0], ("home", None, None))

    def test_run_can_via_server_clear_errors_uses_rx_axis_client(self) -> None:
        client = FakeRxAxisClient()
        args = argparse.Namespace(command="clear-errors")
        with mock.patch("sys.stdout", new_callable=io.StringIO):
            run_can_via_server(args, client)
        self.assertEqual(client.calls[0], ("clear_errors", None, None))

    def test_run_can_via_server_limit_switches_uses_rx_axis_client(self) -> None:
        client = FakeRxAxisClient()
        args = argparse.Namespace(command="limit-switches")
        with mock.patch("sys.stdout", new_callable=io.StringIO):
            run_can_via_server(args, client)
        self.assertEqual(client.calls[0], ("limit_switches", None, None))

    def test_main_prefers_gantry_server_for_linear_axes(self) -> None:
        # X is the only axis routed through the Klipper gantry server.
        fake_client = FakeServerClient(axis="x")
        with mock.patch("openderm.app.cli.GantryServerClient", return_value=fake_client):
            with mock.patch("sys.argv", ["openderm", "--axis", "x", "move-to", "250"]):
                with mock.patch("sys.stdout", new_callable=io.StringIO):
                    result = main()
        self.assertEqual(result, 0)
        self.assertEqual(fake_client.calls[0], ("move_to", 250.0, None))

    def test_main_exits_when_gantry_server_is_unreachable(self) -> None:
        with mock.patch(
            "openderm.app.cli.GantryServerClient", side_effect=GantryServerError("offline")
        ):
            with mock.patch("sys.argv", ["openderm", "--axis", "x", "move-to", "250"]):
                with mock.patch("sys.stdout", new_callable=io.StringIO):
                    with self.assertRaises(SystemExit) as exc:
                        main()
        self.assertEqual(exc.exception.code, 1)

    def test_main_uses_rx_axis_server_for_can_axes(self) -> None:
        fake_client = FakeRxAxisClient()
        with mock.patch("openderm.app.cli.RxAxisServerClient", return_value=fake_client):
            with mock.patch(
                "sys.argv", ["openderm", "--axis", "rx", "move-to", "1.5", "--speed", "0.5"]
            ):
                with mock.patch("sys.stdout", new_callable=io.StringIO):
                    result = main()
        self.assertEqual(result, 0)
        self.assertEqual(fake_client.calls[0], ("move_to", 1.5, 0.5))

    def test_parser_exposes_gantry_server_url(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--gantry-server-url", "http://127.0.0.1:9000", "status"])
        self.assertEqual(args.gantry_server_url, "http://127.0.0.1:9000")

    def test_parser_exposes_rx_axis_server_url(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--rx-axis-server-url", "http://127.0.0.1:9001", "status"])
        self.assertEqual(args.rx_axis_server_url, "http://127.0.0.1:9001")


class FakePicoClient:
    """Drop-in for PicoAxisClient over the CLI subset (status/move_to/home/stop + axis)."""

    def __init__(self, axis: str = "y") -> None:
        self.axis = axis
        self.calls: list[tuple[str, object, object]] = []

    def status(self) -> dict[str, object]:
        self.calls.append(("status", None, None))
        return {"position": {self.axis: 100.0}, "homed_axes": [self.axis], "moving": False}

    def move_to(self, position_mm, feed_mm_min=None, tolerance_mm=None) -> dict[str, object]:
        self.calls.append(("move_to", position_mm, feed_mm_min))
        return {"status": "completed"}

    def home(self) -> dict[str, object]:
        self.calls.append(("home", None, None))
        return {"status": "homed"}

    def stop(self, mode: str = "soft") -> dict[str, object]:
        self.calls.append(("stop", mode, None))
        return {"status": "stopped"}


class FakeLink:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _pico_args(command: str, **kw) -> argparse.Namespace:
    base = dict(
        command=command,
        pico_port="socket://pico:8095",
        pico_vmax_mm_s=35.0,
        pico_acc_mm_s2=None,
        feed=None,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def _patch_pico(link, client):
    return (
        mock.patch("openderm.motion.pico.adapter.open_pico_link", return_value=link),
        mock.patch("openderm.motion.pico.adapter.PicoAxisClient", return_value=client),
    )


class PicoCliTests(unittest.TestCase):
    def test_pico_z_keeps_firmware_backstop(self) -> None:
        link, client = FakeLink(), FakePicoClient("z")
        p1, p2 = _patch_pico(link, client)
        with p1, p2 as pac, mock.patch("sys.stdout", new_callable=io.StringIO):
            run_linear_via_pico(_pico_args("home"), "z")
        self.assertTrue(
            pac.call_args.kwargs["enforce_limits"],
            "Z must keep the firmware [0,392] travel backstop on",
        )

    def test_pico_y_keeps_firmware_backstop(self) -> None:
        link, client = FakeLink(), FakePicoClient("y")
        p1, p2 = _patch_pico(link, client)
        with p1, p2 as pac, mock.patch("sys.stdout", new_callable=io.StringIO):
            run_linear_via_pico(_pico_args("home"), "y")
        self.assertTrue(
            pac.call_args.kwargs["enforce_limits"],
            "Y must keep the firmware physical-travel backstop on",
        )

    def test_pico_move_to_routes_and_closes_link(self) -> None:
        link, client = FakeLink(), FakePicoClient("y")
        p1, p2 = _patch_pico(link, client)
        with p1, p2, mock.patch("sys.stdout", new_callable=io.StringIO):
            rc = run_linear_via_pico(_pico_args("move-to", position=150.0), "y")
        self.assertEqual(rc, 0)
        self.assertEqual(client.calls[0], ("move_to", 150.0, None))
        self.assertTrue(link.closed, "shared Pico link must be closed")

    def test_pico_home_routes(self) -> None:
        link, client = FakeLink(), FakePicoClient("z")
        p1, p2 = _patch_pico(link, client)
        with p1, p2, mock.patch("sys.stdout", new_callable=io.StringIO):
            run_linear_via_pico(_pico_args("home"), "z")
        self.assertEqual(client.calls[0], ("home", None, None))
        self.assertTrue(link.closed)

    def test_pico_move_by_reads_status_then_moves(self) -> None:
        link, client = FakeLink(), FakePicoClient("y")
        p1, p2 = _patch_pico(link, client)
        with p1, p2, mock.patch("sys.stdout", new_callable=io.StringIO):
            run_linear_via_pico(_pico_args("move-by", delta=10.0), "y")
        self.assertEqual(client.calls[0], ("status", None, None))
        self.assertEqual(client.calls[1], ("move_to", 110.0, None))

    def test_pico_rejects_non_yz_axis(self) -> None:
        with self.assertRaises(GantryServerError):
            run_linear_via_pico(_pico_args("home"), "x")

    def test_pico_closes_link_and_translates_error(self) -> None:
        link = FakeLink()
        p1 = mock.patch("openderm.motion.pico.adapter.open_pico_link", return_value=link)
        p2 = mock.patch(
            "openderm.motion.pico.adapter.PicoAxisClient", side_effect=RuntimeError("link stalled")
        )
        with p1, p2:
            with self.assertRaises(GantryServerError):
                run_linear_via_pico(_pico_args("home"), "y")
        self.assertTrue(link.closed, "link must be closed even when setup fails")

    def test_main_always_routes_y_to_pico(self) -> None:
        link, client = FakeLink(), FakePicoClient("y")
        p1, p2 = _patch_pico(link, client)
        with (
            p1,
            p2,
            mock.patch("openderm.app.cli.GantryServerClient") as gsc,
            mock.patch(
                "sys.argv",
                ["openderm", "--axis", "y", "--pico-port", "socket://pico:8095", "move-to", "150"],
            ),
            mock.patch("sys.stdout", new_callable=io.StringIO),
        ):
            rc = main()
        self.assertEqual(rc, 0)
        self.assertEqual(client.calls[0], ("move_to", 150.0, None))
        gsc.assert_not_called()
        self.assertTrue(link.closed)

    def test_parser_has_no_controller_override(self) -> None:
        parser = build_parser()
        self.assertFalse(any(action.dest == "backend" for action in parser._actions))

    def test_parser_exposes_pico_port(self) -> None:
        args = build_parser().parse_args(["--pico-port", "socket://1.2.3.4:8095", "home"])
        self.assertEqual(args.pico_port, "socket://1.2.3.4:8095")


if __name__ == "__main__":
    unittest.main()
