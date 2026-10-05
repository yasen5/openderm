from __future__ import annotations

import os
import unittest
from unittest import mock

from rx_axis_test_support import (
    RxAxisServerClient,
    RxAxisServerError,
    config_from_args,
    make_server_args,
)


class RxAxisConfigTests(unittest.TestCase):
    def test_config_from_args_validates_port(self) -> None:
        with self.assertRaises(RxAxisServerError):
            config_from_args(make_server_args(port=0))

    def test_config_from_args_validates_default_accel(self) -> None:
        with self.assertRaises(RxAxisServerError):
            config_from_args(make_server_args(default_accel_rad_s2=0.0))

    def test_config_rejects_unauthenticated_remote_bind(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RxAxisServerError, "Refusing to bind"):
                config_from_args(make_server_args(host="0.0.0.0"))

    def test_config_accepts_authenticated_remote_bind(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"OPENDERM_CONTROL_TOKEN": "test-control-token"},
            clear=True,
        ):
            config = config_from_args(make_server_args(host="0.0.0.0"))
        self.assertEqual(config.control_token, "test-control-token")

    def test_config_from_args_rx_port_default(self) -> None:
        rx = config_from_args(make_server_args(axis="rx", port=None))
        self.assertEqual(rx.port, 8091)

    def test_config_from_args_rx_homing_final_position(self) -> None:
        rx = config_from_args(make_server_args(axis="rx", homing_final_position_rad=None))
        self.assertAlmostEqual(rx.homing_final_position_rad, 0.95)

    def test_client_sends_control_token(self) -> None:
        client = RxAxisServerClient(
            "http://example.test",
            control_token="test-control-token",
        )
        fake_response = mock.MagicMock()
        fake_response.read.return_value = b'{"ok": true}'
        fake_response.__enter__.return_value = fake_response
        fake_response.__exit__.return_value = False
        with mock.patch(
            "openderm.motion.rx_axis.client.request.urlopen",
            return_value=fake_response,
        ) as urlopen:
            client.status()
        sent_request = urlopen.call_args.args[0]
        self.assertEqual(
            sent_request.get_header("Authorization"),
            "Bearer test-control-token",
        )
