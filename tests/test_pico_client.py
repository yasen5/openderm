from __future__ import annotations

import importlib
import sys
from types import ModuleType
import unittest
from unittest import mock

from openderm.motion import security


class PicoClientImportTests(unittest.TestCase):
    def test_client_imports_security_from_parent_motion_package(self) -> None:
        fake_serial = ModuleType("serial")
        fake_serial.SerialException = OSError
        module_name = "openderm.motion.pico.client"
        original_module = sys.modules.pop(module_name, None)

        try:
            with mock.patch.dict(sys.modules, {"serial": fake_serial}):
                client = importlib.import_module(module_name)
            self.assertIs(
                client.control_token_from_env,
                security.control_token_from_env,
            )
        finally:
            sys.modules.pop(module_name, None)
            if original_module is not None:
                sys.modules[module_name] = original_module
