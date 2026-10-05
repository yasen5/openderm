from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover - depends on the optional vision extra
    raise unittest.SkipTest(
        'vision dependencies are not installed; use pip install -e ".[vision]"'
    ) from exc

from processing.tex_anchor import coverage_mask, load_gauge
from processing.track_moles import render_mole_change_overlay


def _write_placements(registration_dir: Path, *, include_texture: bool = True) -> None:
    payload = {
        "rig_model": {
            "base_R": np.eye(3).tolist(),
            "base_t": [1.0, 2.0, 3.0],
            "fx_fullres_px": 39237.0,
        }
    }
    if include_texture:
        payload["texture"] = {
            "umin_mm": 10.0,
            "vmin_mm": 20.0,
            "umax_mm": 14.0,
            "vmax_mm": 23.0,
            "ppmm": 2.0,
            "W": 8,
            "H": 6,
        }
    (registration_dir / "placements3d.json").write_text(json.dumps(payload))


class TextureGaugeTests(unittest.TestCase):
    def test_load_gauge_uses_required_reconstruction_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registration_dir = Path(tmp)
            _write_placements(registration_dir)
            gauge = load_gauge(registration_dir)

        self.assertEqual((gauge.umin, gauge.vmin), (10.0, 20.0))
        self.assertEqual((gauge.umax, gauge.vmax), (14.0, 23.0))
        self.assertEqual((gauge.ppmm, gauge.W, gauge.H), (2.0, 8, 6))
        self.assertEqual(gauge.fx_fullres_px, 39237.0)

    def test_load_gauge_rejects_incomplete_reconstruction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registration_dir = Path(tmp)
            _write_placements(registration_dir, include_texture=False)
            with self.assertRaisesRegex(ValueError, "openderm-process"):
                load_gauge(registration_dir)

    def test_coverage_mask_requires_lossless_processing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registration_dir = Path(tmp)
            with self.assertRaisesRegex(FileNotFoundError, "openderm-process"):
                coverage_mask(registration_dir)

            source = np.array([[0, 127, 128, 255]], dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(registration_dir / "coverage.png"), source))
            np.testing.assert_array_equal(
                coverage_mask(registration_dir),
                np.array([[0, 0, 255, 255]], dtype=np.uint8),
            )


class ChangeOverlayTests(unittest.TestCase):
    def test_new_lesion_overlay_supports_gantry_only_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            canvas = np.full((80, 120, 3), 100, dtype=np.uint8)
            render_mole_change_overlay(
                tmp,
                {"texA": canvas},
                np.ones(canvas.shape[:2], dtype=np.uint8),
                changes=[],
                molesA=[],
                molesB=[{"x": 30, "y": 40, "radius_mm": 1.0}],
                pairs=[],
                newJ=[0],
                disJ=[],
                T=None,
                ppmm=2.0,
            )
            self.assertTrue((Path(tmp) / "change_overlay.png").is_file())


if __name__ == "__main__":
    unittest.main()
