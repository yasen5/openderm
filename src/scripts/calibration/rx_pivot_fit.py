#!/usr/bin/env python3
"""Fit the rx-pivot arc from a set of poses that all view the same fixed point.

Feed it a ``captures/rx_pivot_poses_*.jsonl`` file collected with
``src/scripts/calibration/rx_pivot_capture.py``. Each recorded pose must view the
same surface point at a constant standoff, at a different RX angle. The fit
recovers the camera arc about RX and writes the model used by
``openderm-scan``.

It prints the per-axis fit residual and a leave-one-out cross-validation error so
you can see how well an unseen rx angle would be predicted. Want both small (sub-
mm) and an rx span wide enough to cover the angles you intend to use -- the model
interpolates well inside the calibrated span and degrades if you extrapolate far
beyond it.

Run with a numpy-capable interpreter, e.g. the calibration venv::

    python src/scripts/calibration/rx_pivot_fit.py captures/rx_pivot_poses_<timestamp>.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from openderm.motion.rx_pivot import RxPivotModel, load_jsonl_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "records",
        help="captures/rx_pivot_poses_*.jsonl file of same-point poses.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Where to write the model JSON (default: captures/rx_pivot_model.json).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    records = load_jsonl_records(args.records)
    if len(records) < 3:
        print(f"Only {len(records)} usable poses in {args.records}; need >= 3.", file=sys.stderr)
        return 1

    try:
        model, diag = RxPivotModel.fit_records(records, source=str(args.records))
    except ValueError as exc:
        print(f"Fit failed: {exc}", file=sys.stderr)
        return 1

    md = model.metadata
    print(f"poses: {md['n_poses']}   rx span: {md['rx_min_deg']:.1f} .. {md['rx_max_deg']:.1f} deg")
    print(f"camera-centre arc radius: {diag['arc_radius_mm']:.1f} mm")
    print("\nper-axis   fit_rms   LOO_mean   LOO_max   (mm)")
    for axis in ("x", "y", "z"):
        print(
            f"   {axis}       {diag['fit_residual_rms_mm'][axis]:7.3f}   "
            f"{diag['loo_mean_err_mm'][axis]:8.3f}  {diag['loo_max_err_mm'][axis]:7.3f}"
        )

    worst_xy = max(diag["loo_max_err_mm"]["x"], diag["loo_max_err_mm"]["y"])
    verdict = (
        "GOOD -- x/y held to sub-mm across the span; usable."
        if worst_xy < 2.0
        else "MARGINAL -- widen the rx span and re-collect with the point well centred."
    )
    print(f"\nverdict: {verdict}")

    out = Path(args.out) if args.out else REPO_ROOT / "captures" / "rx_pivot_model.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    print(f"\nwrote model -> {out}")
    print("The model is ready for openderm-scan:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
