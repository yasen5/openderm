from __future__ import annotations

import socket
import threading
import unittest

from capture.motion.pico.bridge import authenticate_connection


class PicoBridgeAuthenticationTests(unittest.TestCase):
    def _authenticate(self, line: bytes) -> tuple[bool, bytes]:
        server_socket, client_socket = socket.socketpair()
        result: list[bool] = []

        def run_server() -> None:
            result.append(
                authenticate_connection(
                    server_socket,
                    "test-control-token",
                    timeout_s=1.0,
                )
            )

        thread = threading.Thread(target=run_server)
        thread.start()
        try:
            client_socket.sendall(line)
            response = client_socket.recv(128)
        finally:
            thread.join(timeout=2.0)
            server_socket.close()
            client_socket.close()
        self.assertFalse(thread.is_alive())
        return result[0], response

    def test_valid_token_opens_bridge(self) -> None:
        authenticated, response = self._authenticate(b"AUTH test-control-token\n")
        self.assertTrue(authenticated)
        self.assertEqual(response, b"OK AUTH\n")

    def test_invalid_token_is_rejected(self) -> None:
        authenticated, response = self._authenticate(b"AUTH wrong-token\n")
        self.assertFalse(authenticated)
        self.assertEqual(response, b"ERR AUTH\n")


if __name__ == "__main__":
    unittest.main()
