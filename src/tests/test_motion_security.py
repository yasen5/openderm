from __future__ import annotations

import unittest

from openderm.motion.security import (
    MotionSecurityError,
    bearer_token_matches,
    bridge_auth_line_matches,
    is_loopback_host,
    require_secure_bind,
)


class MotionSecurityTests(unittest.TestCase):
    def test_loopback_bind_needs_no_token(self) -> None:
        for host in ("127.0.0.1", "::1", "localhost"):
            with self.subTest(host=host):
                self.assertTrue(is_loopback_host(host))
                require_secure_bind(host, None)

    def test_remote_bind_requires_token(self) -> None:
        for host in ("0.0.0.0", "::", "192.168.1.10", "gantry.local"):
            with self.subTest(host=host):
                with self.assertRaises(MotionSecurityError):
                    require_secure_bind(host, None)
                require_secure_bind(host, "shared-secret")

    def test_bearer_auth_requires_exact_token(self) -> None:
        self.assertTrue(bearer_token_matches("Bearer shared-secret", "shared-secret"))
        self.assertFalse(bearer_token_matches("Bearer wrong-secret", "shared-secret"))
        self.assertFalse(bearer_token_matches(None, "shared-secret"))

    def test_bridge_auth_requires_exact_single_line(self) -> None:
        self.assertTrue(bridge_auth_line_matches(b"AUTH shared-secret\n", "shared-secret"))
        self.assertFalse(bridge_auth_line_matches(b"AUTH wrong-secret\n", "shared-secret"))
        self.assertFalse(
            bridge_auth_line_matches(
                b"AUTH shared-secret\nMOVE Y 100\n",
                "shared-secret",
            )
        )


if __name__ == "__main__":
    unittest.main()
