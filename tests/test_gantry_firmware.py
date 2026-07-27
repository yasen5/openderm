"""Runs the Pico gantry-firmware simulator in a subprocess.

The simulator stubs MicroPython modules to import pico/gantry_firmware.py off
hardware; running it in its own process keeps those stubs from contaminating the rest of the
suite. It asserts the MOVEC cruise (look-ahead) behaviour and that every safety clamp still
holds; here we just assert it exits 0 with the success sentinel.
"""

import subprocess
import sys
import unittest
from pathlib import Path

SIM = Path(__file__).resolve().parent / "_gantry_firmware_sim.py"


class GantryFirmwareTests(unittest.TestCase):
    def test_cruise_sim_passes(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(SIM)], capture_output=True, text=True, timeout=120
        )
        self.assertEqual(
            proc.returncode, 0, f"sim failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
        self.assertIn("ALL OK", proc.stdout)


if __name__ == "__main__":
    unittest.main()
