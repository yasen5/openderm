from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from openderm.processing import build_parser, commands, registration_flags


def make_args(**overrides) -> argparse.Namespace:
    values = dict(
        capture_dir=Path("captures/example"),
        quality="preview",
        fx_full=39237.0,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class ProcessingRecipeTests(unittest.TestCase):
    def test_reference_focal_length_is_the_default(self) -> None:
        args = build_parser().parse_args(["captures/example"])
        self.assertEqual(args.fx_full, 39237.0)

    def test_processor_has_no_anatomy_mode(self) -> None:
        parser = build_parser()
        self.assertFalse(any(action.dest == "mode" for action in parser._actions))

    def test_scan_recipe_uses_contour_surface(self) -> None:
        flags = registration_flags(make_args())
        self.assertIn("--contour", flags)
        self.assertIn("--group-by-row", flags)
        self.assertIn("--group-feather-mm", flags)
        self.assertIn("--blend", flags)
        self.assertIn("--focus-weight", flags)

    def test_stage_two_uses_same_scans_rig_fit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(capture_dir=Path(tmp))
            stage_one, stage_two, checks = commands(args)
        self.assertNotIn("--rig-from", stage_one)
        rig_index = stage_two.index("--rig-from")
        self.assertTrue(
            stage_two[rig_index + 1].endswith("registration3d-rigfit/placements3d.json")
        )
        self.assertEqual(len(checks), 2)

    def test_full_quality_is_78_pixels_per_mm(self) -> None:
        flags = registration_flags(make_args(quality="full"))
        self.assertEqual(flags[flags.index("--downscale") + 1], "1")
        self.assertEqual(flags[flags.index("--texture-ppmm") + 1], "78")


if __name__ == "__main__":
    unittest.main()
