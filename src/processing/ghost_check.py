#!/usr/bin/env python3
"""Ghost / duplicated-feature validator for one 3D skin reconstruction.

The renderer can paint the SAME real skin feature at two slightly-offset texel
positions (residual per-frame misalignment + per-texel HF winner-take-all), so a
single mole/red-dot shows up TWICE in the ortho-texture -- a false "new lesion"
for cancer screening. This tool cross-checks the composite against the raw source
frames, which are ground truth (a raw photo shows each feature exactly once):

  for every candidate feature in texture.jpg:
    - re-detect blobs LOCALLY in the composite crop (catches close doubles the
      global detector's 2.5mm suppression merges);
    - if >=2 composite blobs, find every source frame that imaged that surface
      point, project the composite blobs into each, and detect the real blobs;
    - two composite blobs are the SAME feature (a ghost) unless a MAJORITY of the
      covering source frames resolve them as TWO separate blobs.
  ghost_excess = (#composite blobs with source support) - (#distinct real features).

total_ghosts (summed over regions) is a scalar to MINIMISE across render settings
-- the automatic acceptance test: a correct render -> 0.

Install OpenDerm with the `vision` extra, then run:
  openderm-check-ghosts captures/<scan>/registration3d-canonical
"""

from __future__ import annotations
import argparse, json, os
import numpy as np
import cv2
from scipy.ndimage import gaussian_laplace, maximum_filter
from scipy.interpolate import LinearNDInterpolator

from processing.tex_anchor import load_gauge, coverage_mask, _parse_obj_v_vt
from processing.lesions import (
    detect_angiomas,
    detect_moles,
    hemoglobin_flat,
    mel_threshold,
    melanin_flat,
)
from processing.register_scan_3d import project


# --------------------------------------------------------------------------- #
def local_peaks(
    chan, ppmm, mask=None, diam_mm=(0.3, 4.0), k_sigma=4.0, min_contrast=0.05, suppress_mm=0.8
):
    """Positive-bump detector (multiscale LoG) with a SMALL NMS radius so two
    features ~1-2mm apart stay separate. Returns [(x, y, r_mm, val)]."""
    if mask is None:
        mask = np.ones(chan.shape, np.uint8)
    sig_lo = diam_mm[0] / 2 * ppmm / np.sqrt(2)
    sig_hi = diam_mm[1] / 2 * ppmm / np.sqrt(2)
    sigmas = np.geomspace(sig_lo, sig_hi, 8)
    resp = np.stack([-gaussian_laplace(chan, s) * s**2 for s in sigmas])
    peak = resp.max(0)
    scl = resp.argmax(0)
    thr = mel_threshold(chan, mask, min_contrast, k_sigma)
    win = int(sig_lo * 2) | 1
    loc = (peak == maximum_filter(peak, size=max(3, win))) & (mask > 0) & (peak > 0)
    ys, xs = np.where(loc)
    out, taken = [], np.zeros(chan.shape, np.uint8)
    sup_px = int(max(suppress_mm * ppmm, 1))
    for val, x, y in sorted(zip(peak[ys, xs], xs, ys), reverse=True):
        if float(chan[y, x]) < thr or taken[y, x]:
            continue
        r_px = sigmas[scl[y, x]] * np.sqrt(2)
        if not (diam_mm[0] <= 2 * r_px / ppmm <= diam_mm[1]):
            continue
        cv2.circle(taken, (int(x), int(y)), sup_px, 1, -1)
        out.append((int(x), int(y), float(r_px / ppmm), float(val)))
    return out


def detect_on(bgr, ppmm, channel, mask=None):
    """Real lesion detections (calibrated + red-hue gate for angiomas), the SAME
    definition of 'a feature' used everywhere. Returns [(x, y, r_mm)]."""
    if mask is None:
        mask = np.full(bgr.shape[:2], 255, np.uint8)
    mel = melanin_flat(bgr, mask, ppmm)
    if channel == "hem":
        hem = hemoglobin_flat(bgr, mask, ppmm)
        feats = detect_angiomas(bgr, hem, mel, mask, ppmm)
    else:
        feats = detect_moles(mel, mask, ppmm)
    return [(f["x"], f["y"], f["radius_mm"]) for f in feats]


def build_uv_to_world(reg_dir, g):
    """Interpolator (u_mm, v_mm) -> world (x,y,z) from the exported mesh UVs."""
    V, VT = _parse_obj_v_vt(os.path.join(reg_dir, "surface_mesh.obj"))
    u = g.umin + VT[:, 0] * (g.umax - g.umin)  # invert build_mesh's VT
    v = g.vmin + (1.0 - VT[:, 1]) * (g.vmax - g.vmin)
    return LinearNDInterpolator(np.column_stack([u, v]), V)


def load_rig(reg_dir):
    pl = json.load(open(os.path.join(reg_dir, "placements3d.json")))
    r, ds = pl["rig_model"], pl["downscale"]
    K = dict(
        fxf=float(r["fx_fullres_px"]),
        k1=float(r["k1"]),
        cxf=float(r["cx"]) * ds,
        cyf=float(r["cy"]) * ds,
    )
    K["Wf"], K["Hf"] = K["cxf"] * 2, K["cyf"] * 2
    cap = pl.get("capture_dir") or os.path.dirname(reg_dir.rstrip("/"))
    frames = [
        dict(
            idx=f["idx"],
            station=f["station"],
            row=f["row"],
            path=os.path.join(cap, os.path.basename(f["image"])),
            R=np.array(f["R_cam2world"]),
            C=np.array(f["C_mm"]),
            standoff=f["gantry"]["standoff_mm"],
        )
        for f in pl["frames"]
    ]
    return K, frames


def project_pt(P, f, K):
    uv, z = project(np.atleast_2d(P), f["R"], f["C"], K["fxf"], K["k1"], K["cxf"], K["cyf"])
    return float(uv[0][0]), float(uv[0][1]), float(z[0])


_IMG: dict = {}


def load_src(path, ds):
    key = (path, ds)
    if key not in _IMG:
        im = cv2.imread(path)
        if im is not None and ds != 1:
            im = cv2.resize(im, None, fx=1 / ds, fy=1 / ds, interpolation=cv2.INTER_AREA)
        _IMG[key] = im
    return _IMG[key]


# --------------------------------------------------------------------------- #
def analyse(reg_dir, args):
    import time

    t0 = time.time()

    def log(m):
        print(f"  [{time.time() - t0:5.1f}s] {m}", flush=True)

    g = load_gauge(reg_dir)
    ppmm = g.ppmm
    tex = cv2.imread(os.path.join(reg_dir, "texture.jpg"))
    cov = coverage_mask(reg_dir)
    K, frames = load_rig(reg_dir)
    log(f"loaded texture {tex.shape[1]}x{tex.shape[0]} @ {ppmm:.0f}px/mm")
    uv2w = build_uv_to_world(reg_dir, g)
    log("built (u,v)->world interpolator")

    # global candidate hunt on a DOWNSCALED texture (locations only; precise
    # blob geometry is redone per-crop at full res). Full-res multiscale LoG over
    # the whole 48Mpx map is what made this slow.
    gds = args.global_ds
    tg = cv2.resize(tex, None, fx=1 / gds, fy=1 / gds, interpolation=cv2.INTER_AREA)
    cg = cv2.resize(cov, (tg.shape[1], tg.shape[0]), interpolation=cv2.INTER_NEAREST)
    dm_g = cv2.erode(
        cg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(3 * ppmm / gds) | 1,) * 2)
    )
    ppg = ppmm / gds
    mel_g = melanin_flat(tg, cg, ppg)
    hem_g = hemoglobin_flat(tg, cg, ppg)
    cands = [
        dict(x=m["x"] * gds, y=m["y"] * gds, r=m["radius_mm"], channel="mel")
        for m in detect_moles(mel_g, dm_g, ppg)
    ] + [
        dict(x=a["x"] * gds, y=a["y"] * gds, r=a["radius_mm"], channel="hem")
        for a in detect_angiomas(tg, hem_g, mel_g, dm_g, ppg)
    ]
    cands.sort(key=lambda c: c["channel"] != "hem")  # prefer red on overlap
    kept = []
    for c in cands:
        if all((c["x"] - k["x"]) ** 2 + (c["y"] - k["y"]) ** 2 > (2.5 * ppmm) ** 2 for k in kept):
            u_mm, v_mm = g.px_to_uv(c["x"], c["y"])
            P = uv2w(float(u_mm), float(v_mm))
            if P is None or np.any(np.isnan(P)):
                continue
            c["u"], c["v"], c["P"] = float(u_mm), float(v_mm), np.asarray(P).ravel()
            kept.append(c)
    log(f"{len(kept)} candidate features ({sum(c['channel'] == 'hem' for c in kept)} real red)")

    # group candidates that sit within merge_mm on the surface -- only a group of
    # >=2 can be a doubling. A ghost = members the source frames resolve as fewer.
    n = len(kept)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if (
                kept[i]["channel"] == kept[j]["channel"]
                and np.linalg.norm(kept[i]["P"] - kept[j]["P"]) < args.merge_mm
            ):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    records, total_ghosts, phantoms = [], 0, 0
    for gi, mem in enumerate(groups.values()):
        ms = [kept[i] for i in mem]
        ch = ms[0]["channel"]
        cworld = [m["P"] for m in ms]
        cblobs = [(m["x"], m["y"], m["r"], 0.0) for m in ms]
        cx, cy = int(np.mean([m["x"] for m in ms])), int(np.mean([m["y"] for m in ms]))
        rec = dict(
            id=gi,
            channel=ch,
            tex_px=[cx, cy],
            n_comp_blobs=len(ms),
            source_count=len(ms),
            n_phantom=0,
            ghost_excess=0,
            n_cover=0,
            cblobs=cblobs,
            per_frame=[],
        )
        if len(ms) < 2:  # lone feature -> no double
            records.append(rec)
            continue

        Pc = np.mean(cworld, axis=0)
        cover = []
        for f in frames:
            px, py, z = project_pt(Pc, f, K)
            if z > 10 and 10 <= px < K["Wf"] - 10 and 10 <= py < K["Hf"] - 10:
                cover.append((f, np.hypot(px - K["cxf"], py - K["cyf"])))
        cover.sort(key=lambda t: t[1])
        cover = [f for f, _ in cover[: args.max_frames]]

        nb = len(ms)
        sep = np.zeros((nb, nb))
        seen = np.zeros((nb, nb))
        support = np.zeros(nb)
        per_frame = []
        for f in cover:
            # downscale each source frame to ~src_ppmm (matches the composite's
            # scale): at ~20px/mm the pore/texture noise that wrecks the adaptive
            # threshold at full res is averaged away, so the calibrated detectors
            # behave as they do on the composite.
            sds = max(1, round(K["fxf"] / f["standoff"] / args.src_ppmm))
            im = load_src(f["path"], sds)
            if im is None:
                continue
            ppmm_s = K["fxf"] / f["standoff"] / sds
            preds = [project_pt(w, f, K) for w in cworld]
            xs = [p[0] / sds for p in preds]
            ys = [p[1] / sds for p in preds]
            mrg = int(args.src_pad_mm * ppmm_s)
            sx0 = int(max(0, min(xs) - mrg))
            sy0 = int(max(0, min(ys) - mrg))
            sx1 = int(min(im.shape[1], max(xs) + mrg))
            sy1 = int(min(im.shape[0], max(ys) + mrg))
            if sx1 - sx0 < 5 or sy1 - sy0 < 5:
                continue
            sblobs = detect_on(im[sy0:sy1, sx0:sx1], ppmm_s, ch)
            tol = args.match_mm * ppmm_s
            assign = []
            for px, py, z in preds:
                lx, ly = px / sds - sx0, py / sds - sy0
                best, bd = -1, tol
                for si, (sx, sy, sr) in enumerate(sblobs):
                    d = np.hypot(sx - lx, sy - ly)
                    if d < bd:
                        bd, best = d, si
                assign.append(best)
            for i in range(nb):
                if assign[i] >= 0:
                    support[i] += 1
                for j in range(i + 1, nb):
                    if assign[i] >= 0 and assign[j] >= 0:
                        seen[i, j] += 1
                        if assign[i] != assign[j]:
                            sep[i, j] += 1
            per_frame.append(
                dict(
                    idx=f["idx"],
                    station=f["station"],
                    sds=sds,
                    crop=[sx0, sy0, sx1, sy1],
                    sblobs=sblobs,
                    ppmm_s=ppmm_s,
                    assign=assign,
                    preds=[[p[0] / sds, p[1] / sds] for p in preds],
                )
            )

        par2 = list(range(nb))

        def find2(a):
            while par2[a] != a:
                par2[a] = par2[par2[a]]
                a = par2[a]
            return a

        for i in range(nb):
            for j in range(i + 1, nb):
                if seen[i, j] > 0 and sep[i, j] <= seen[i, j] / 2.0:
                    par2[find2(i)] = find2(j)
        n_support = int(np.sum(support > 0))
        source_count = len({find2(i) for i in range(nb) if support[i] > 0})
        n_phantom = int(np.sum(support == 0))
        ghost_excess = max(0, n_support - source_count)
        total_ghosts += ghost_excess
        phantoms += n_phantom
        rec.update(
            source_count=source_count,
            n_phantom=n_phantom,
            ghost_excess=ghost_excess,
            n_cover=len(cover),
            per_frame=per_frame,
        )
        records.append(rec)
        if ghost_excess > 0 or n_phantom > 0:
            tag = "GHOST" if ghost_excess > 0 else "phantom"
            log(
                f"  [{tag}] group {gi} ({ch}) @ (u={ms[0]['u']:.0f},v={ms[0]['v']:.0f})"
                f"mm: {nb} composite features, {source_count} real in {len(cover)} "
                f"source frames -> +{ghost_excess} ghost"
            )
    return dict(
        tex=tex,
        g=g,
        records=records,
        total_ghosts=total_ghosts,
        phantoms=phantoms,
        n_candidates=len(kept),
        K=K,
    )


# --------------------------------------------------------------------------- #
def write_outputs(res, out_dir):
    os.makedirs(os.path.join(out_dir, "montage"), exist_ok=True)
    tex, g = res["tex"], res["g"]
    ov = tex.copy()
    for r in res["records"]:
        gh = r["ghost_excess"] > 0
        col = (0, 0, 255) if gh else ((0, 165, 255) if r["n_phantom"] else (0, 200, 0))
        pts = [(bx, by) for bx, by, rr, v in r["cblobs"]]
        for bx, by in pts:
            cv2.circle(ov, (bx, by), int(12), col, 3)
        if gh:
            for a in range(len(pts)):
                for b in range(a + 1, len(pts)):
                    cv2.line(ov, pts[a], pts[b], col, 2)
    cv2.putText(
        ov,
        f"ghosts={res['total_ghosts']} phantoms={res['phantoms']} features={res['n_candidates']}",
        (20, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3,
    )
    cv2.imwrite(os.path.join(out_dir, "ghost_overlay.png"), ov)

    for r in res["records"]:
        if r["ghost_excess"] <= 0 and r["n_phantom"] <= 0:
            continue
        cx, cy = r["tex_px"]
        wp = int(6 * g.ppmm)
        cc = tex[max(0, cy - wp) : cy + wp, max(0, cx - wp) : cx + wp].copy()
        for bx, by, rr, v in r["cblobs"]:
            cv2.circle(
                cc,
                (bx - max(0, cx - wp), by - max(0, cy - wp)),
                int(rr * g.ppmm + 8),
                (0, 0, 255),
                2,
            )
        tiles = [_label(cv2.resize(cc, (280, 280)), "COMPOSITE")]
        for pf in r["per_frame"][:4]:
            im = load_src(_path_for(res, pf["idx"]), pf["sds"])
            if im is None:
                continue
            sx0, sy0, sx1, sy1 = pf["crop"]
            crop = im[sy0:sy1, sx0:sx1].copy()
            for sx, sy, sr in pf["sblobs"]:
                cv2.circle(crop, (int(sx), int(sy)), int(sr * pf["ppmm_s"] + 6), (0, 255, 0), 2)
            for px, py in pf["preds"]:
                cv2.drawMarker(
                    crop, (int(px - sx0), int(py - sy0)), (0, 0, 255), cv2.MARKER_CROSS, 18, 2
                )
            if crop.size:
                tiles.append(_label(cv2.resize(crop, (280, 280)), f"src #{pf['station']}"))
        if tiles:
            H = 280
            row = np.hstack([cv2.resize(t, (280, H)) for t in tiles])
            cv2.imwrite(os.path.join(out_dir, "montage", f"ghost_{r['id']}.png"), row)

    slim = dict(
        reg_dir=out_dir,
        total_ghosts=res["total_ghosts"],
        phantoms=res["phantoms"],
        n_candidates=res["n_candidates"],
        ghosts=[
            {
                "id": r["id"],
                "channel": r["channel"],
                "tex_px": r["tex_px"],
                "n_comp_blobs": r["n_comp_blobs"],
                "source_count": r["source_count"],
                "n_phantom": r["n_phantom"],
                "ghost_excess": r["ghost_excess"],
                "n_cover": r["n_cover"],
            }
            for r in res["records"]
            if r["ghost_excess"] > 0 or r["n_phantom"] > 0
        ],
    )
    json.dump(slim, open(os.path.join(out_dir, "ghost_report.json"), "w"), indent=1)


def _label(img, txt):
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


_PATHS: dict = {}


def _path_for(res, idx):
    if not _PATHS:
        _PATHS.update({f: p for f, p in res.get("_paths", [])})
    return _PATHS.get(idx)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("reg_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--window-mm", type=float, default=8.0)
    ap.add_argument(
        "--merge-mm",
        type=float,
        default=8.0,
        help="max surface distance to consider two features a possible "
        "doubling of one (only groups of >=2 are checked)",
    )
    ap.add_argument("--match-mm", type=float, default=2.0)
    ap.add_argument("--suppress-mm", type=float, default=0.8)
    ap.add_argument("--k-sigma", type=float, default=4.0)
    ap.add_argument("--max-frames", type=int, default=6)
    ap.add_argument(
        "--src-ppmm",
        type=float,
        default=20.0,
        help="downscale each source frame to ~this px/mm before "
        "detection (matches the composite; blurs away pore noise)",
    )
    ap.add_argument(
        "--src-pad-mm",
        type=float,
        default=16.0,
        help="half-size of the source-frame crop (gives the background flatten enough context)",
    )
    ap.add_argument(
        "--global-ds",
        type=int,
        default=3,
        help="downscale for the global candidate hunt (locations only)",
    )
    args = ap.parse_args()
    reg_dir = args.reg_dir.rstrip("/")
    out_dir = args.out or os.path.join(reg_dir, "ghost-check")
    _, frames = load_rig(reg_dir)
    res = analyse(reg_dir, args)
    res["_paths"] = [(f["idx"], f["path"]) for f in frames]
    write_outputs(res, out_dir)
    doublings = sum(1 for r in res["records"] if r["n_comp_blobs"] >= 2)
    print(
        f"DOUBLINGS: {doublings} region(s) where the composite shows >=2 "
        f"features at one spot  |  confirmed ghosts: {res['total_ghosts']}, "
        f"unconfirmed: {res['phantoms']}  over {res['n_candidates']} features"
    )
    print(f"  -> {out_dir}")


if __name__ == "__main__":
    main()
