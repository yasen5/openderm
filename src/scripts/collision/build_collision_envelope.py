r"""Precompute the self-collision-free envelope for the gantry.

  sweep : python src/scripts/collision/build_collision_envelope.py sweep \
              --grid config/cad/collision_clearance.npz --out config/cad/collision_envelope.npz
  derive: python src/scripts/collision/build_collision_envelope.py derive \
              --grid config/cad/collision_clearance.npz --out config/cad/collision_envelope.npz

`sweep` computes the raw min-clearance grid over (rx,z,x,y) in REAL controller
units (mm for x/y/z, rad for rx) via the FCL model, saves it, and derives the
safe envelope. `derive` re-thresholds a saved grid without repeating the sweep.

Persistent settings come from config/scripts.json. Travel ranges come from
config/cad/frame_calibration.json. RX is interpolated to a fine grid so small
backlash is honored despite the coarse sweep.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from openderm.script_config import CollisionConfig, load_script_config

CALIBRATION_PATH = REPO_ROOT / "config" / "cad" / "frame_calibration.json"


_CM = None
_X = None
_Y = None


def _worker_init(x, y):
    # One CollisionModel per worker process (FCL objects are not picklable).
    global _CM, _X, _Y
    from collision_model import CollisionModel

    _CM = CollisionModel()
    _X, _Y = x, y


def _sweep_pair(job):
    i, j, rx, z = job
    out = np.empty((len(_X), len(_Y)), np.float32)
    for k, x in enumerate(_X):
        for l, y in enumerate(_Y):
            out[k, l] = _CM.clearance(x, y, z, rx)
    return i, j, out


def sweep(grid_path: Path, settings: CollisionConfig):
    # Embarrassingly parallel over (rx,z) pairs; each worker owns a model.
    import multiprocessing as mp

    travel = json.loads(CALIBRATION_PATH.read_text())["real_travel"]
    RX = np.linspace(*travel["rx_rad"], settings.rx_points)
    # Dense Z cells limit the runtime guard's conservative interpolation refusal.
    Z = np.linspace(*travel["z_mm"], settings.z_points)
    X = np.linspace(*travel["x_mm"], settings.x_points)
    Y = np.linspace(*travel["y_mm"], settings.y_points)
    jobs = max(1, min(12, (os.cpu_count() or 4) // 4))
    pairs = [(i, j, rx, z) for i, rx in enumerate(RX) for j, z in enumerate(Z)]
    C = np.empty((len(RX), len(Z), len(X), len(Y)), np.float32)
    t0 = time.time()
    done = 0
    with mp.Pool(jobs, initializer=_worker_init, initargs=(X, Y)) as pool:
        for i, j, block in pool.imap_unordered(_sweep_pair, pairs):
            C[i, j] = block
            done += 1
            el = time.time() - t0
            print(
                f"[{done:3d}/{len(pairs)}] rx={RX[i]:.2f} z={Z[j]:.0f}  "
                f"elapsed {el:5.0f}s eta {el / done * (len(pairs) - done):5.0f}s",
                flush=True,
            )
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(grid_path, rx=RX, z=Z, x=X, y=Y, clearance=C)
    print("saved", grid_path, C.shape)


def derive(grid_path: Path, out_path: Path, settings: CollisionConfig):
    from scipy.interpolate import interp1d

    margin, backlash_deg = settings.margin_mm, settings.backlash_deg
    with np.load(grid_path) as d:
        RXg, Zg, Xg, Yg, C = d["rx"], d["z"], d["x"], d["y"], d["clearance"]
    # interpolate rx to a fine grid so a small backlash is representable
    RXf = np.linspace(RXg[0], RXg[-1], settings.fine_rx_points)
    Cf = interp1d(RXg, C, axis=0)(RXf).astype(np.float32)
    drx = RXf[1] - RXf[0]
    span = max(1, int(round(np.radians(backlash_deg) / drx)))
    robust = np.empty_like(Cf)
    for i in range(len(RXf)):
        lo, hi = max(0, i - span), min(len(RXf), i + span + 1)
        robust[i] = Cf[lo:hi].min(axis=0)
    safe = robust >= margin
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        rx=RXf,
        z=Zg,
        x=Xg,
        y=Yg,
        safe=safe,
        clearance_robust=robust,
        margin=margin,
        backlash_deg=backlash_deg,
    )
    print(
        f"saved {out_path}  margin={margin}mm backlash=+/-{backlash_deg}deg "
        f"(rx-span={span} cells ~ +/-{np.degrees(span * drx):.1f}deg)\n"
    )
    # answer "does the x-limit depend on z?": max reachable x (any safe y) per (rx,z)
    print("max safe x (mm, any safe y) per (rx,z) -- shows x-limit vs z height:")
    print("   z:    " + "  ".join(f"{z:4.0f}" for z in Zg))
    for gi in range(0, len(RXf), 11):  # sample approximately the coarse RX rows
        row = []
        for j in range(len(Zg)):
            xs = Xg[safe[gi, j].any(axis=1)]
            row.append(f"{xs.max():4.0f}" if len(xs) else "none")
        print(f" rx={RXf[gi]:4.2f} " + " ".join(row))
    return RXf, Zg, Xg, Yg, robust, safe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a collision envelope using settings from config/scripts.json."
    )
    modes = parser.add_subparsers(dest="mode", required=True)
    sweep_parser = modes.add_parser("sweep", help="Compute a clearance grid and safe envelope.")
    derive_parser = modes.add_parser("derive", help="Derive an envelope from a saved grid.")
    for mode in (sweep_parser, derive_parser):
        mode.add_argument(
            "--grid",
            required=True,
            type=Path,
            help="Clearance grid NPZ (output for sweep, input for derive).",
        )
        mode.add_argument("--out", required=True, type=Path, help="Safe envelope NPZ to write.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        settings = CollisionConfig(**load_script_config("collision"))
    except ValueError as exc:
        parser.error(str(exc))
    if args.grid.resolve() == args.out.resolve():
        parser.error("--grid and --out must be different files")
    if args.mode == "sweep":
        sweep(args.grid, settings)
    derive(args.grid, args.out, settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
