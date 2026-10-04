from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

import build_collision_envelope as builder
from openderm.motion.collision_guard import CollisionGuard


class CollisionEnvelopeTests(unittest.TestCase):
    def test_derived_envelope_loads_in_runtime_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            grid = root / "clearance.npz"
            envelope = root / "envelope.npz"
            axes = np.array([0.0, 1.0])
            clearance = np.full((2, 2, 2, 2), 30.0, dtype=np.float32)
            np.savez(
                grid,
                rx=axes,
                z=axes,
                x=axes,
                y=axes,
                clearance=clearance,
            )

            old_grid, old_envelope = builder.GRID, builder.ENV
            builder.GRID, builder.ENV = str(grid), str(envelope)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    builder.derive(margin=20.0, backlash_deg=2.5)
            finally:
                builder.GRID, builder.ENV = old_grid, old_envelope

            guard = CollisionGuard.load(str(envelope))
            self.assertTrue(guard.is_safe(0.5, 0.5, 0.5, 0.5))
            self.assertFalse(guard.is_safe(2.0, 0.5, 0.5, 0.5))
            self.assertEqual(guard.margin, 20.0)
            self.assertEqual(guard.backlash_deg, 2.5)


if __name__ == "__main__":
    unittest.main()
