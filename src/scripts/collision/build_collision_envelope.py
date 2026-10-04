"""Precompute the self-collision-free envelope for the gantry.

  sweep : python src/scripts/collision/build_collision_envelope.py sweep
  derive: python src/scripts/collision/build_collision_envelope.py derive <margin_mm> <backlash_deg>

`sweep` computes the raw min-clearance grid over (rx,z,x,y) in REAL controller units
(mm for x/y/z, rad for rx) via the FCL collision model and saves it (slow, ~10 min).
`derive` is instant: it re-thresholds that saved grid at a chosen margin + rx-backlash
into the safe (x,y) masks. rx is interpolated to a fine grid so a small backlash is
honored despite the coarse sweep.

Writes: config/cad/collision_clearance.npz (grid), config/cad/collision_envelope.npz (safe masks).
"""

import os
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CAD = str(Path(__file__).resolve().parents[3] / "config" / "cad")
GRID = os.path.join(CAD, "collision_clearance.npz")
ENV = os.path.join(CAD, "collision_envelope.npz")

RX = np.linspace(0.0, 1.92, 9)
# z MUST be dense: the runtime guard takes the conservative MIN over the grid
# cell around a query, and clearance is ~1-Lipschitz in z (mm per mm), so the
# cell size bounds worst-case over-refusal. Thirty points produce ~14mm cells,
# keeping interpolation conservatism below the 20mm margin.
Z = np.linspace(0.0, 406.0, 30)
X = np.linspace(0.0, 815.0, 30)
Y = np.linspace(0.0, 680.0, 30)


_CM = None


def _worker_init():
    # One CollisionModel per worker process (FCL objects are not picklable).
    global _CM
    from collision_model import CollisionModel

    _CM = CollisionModel()


def _sweep_pair(job):
    i, j, rx, z = job
    out = np.empty((len(X), len(Y)), np.float32)
    for k, x in enumerate(X):
        for l, y in enumerate(Y):
            out[k, l] = _CM.clearance(x, y, z, rx)
    return i, j, out


def sweep(jobs=None):
    # Embarrassingly parallel over (rx,z) pairs; each worker owns a model.
    import multiprocessing as mp

    jobs = int(jobs) if jobs else max(1, min(12, (os.cpu_count() or 4) // 4))
    pairs = [(i, j, rx, z) for i, rx in enumerate(RX) for j, z in enumerate(Z)]
    C = np.empty((len(RX), len(Z), len(X), len(Y)), np.float32)
    t0 = time.time()
    done = 0
    with mp.Pool(jobs, initializer=_worker_init) as pool:
        for i, j, block in pool.imap_unordered(_sweep_pair, pairs):
            C[i, j] = block
            done += 1
            el = time.time() - t0
            print(
                f"[{done:3d}/{len(pairs)}] rx={RX[i]:.2f} z={Z[j]:.0f}  "
                f"elapsed {el:5.0f}s eta {el / done * (len(pairs) - done):5.0f}s",
                flush=True,
            )
    np.savez(GRID, rx=RX, z=Z, x=X, y=Y, clearance=C)
    print("saved", GRID, C.shape)


def derive(margin=20.0, backlash_deg=2.5):
    from scipy.interpolate import interp1d

    d = np.load(GRID)
    RXg, Zg, Xg, Yg, C = d["rx"], d["z"], d["x"], d["y"], d["clearance"]
    # interpolate rx to a fine grid so a small backlash is representable
    RXf = np.linspace(RXg[0], RXg[-1], 89)
    Cf = interp1d(RXg, C, axis=0)(RXf).astype(np.float32)
    drx = RXf[1] - RXf[0]
    span = max(1, int(round(np.radians(backlash_deg) / drx)))
    robust = np.empty_like(Cf)
    for i in range(len(RXf)):
        lo, hi = max(0, i - span), min(len(RXf), i + span + 1)
        robust[i] = Cf[lo:hi].min(axis=0)
    safe = robust >= margin
    np.savez(
        ENV,
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
        f"saved {ENV}  margin={margin}mm backlash=+/-{backlash_deg}deg "
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


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if mode == "sweep":
        sweep(jobs=sys.argv[2] if len(sys.argv) > 2 else None)
        derive()
    else:
        m = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
        b = float(sys.argv[3]) if len(sys.argv) > 3 else 2.5
        derive(m, b)
