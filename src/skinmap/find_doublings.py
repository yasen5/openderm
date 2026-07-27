#!/usr/bin/env python3
"""Find rendered feature DOUBLINGS in a skin ortho-texture (source-free).

A rendering doubling paints the SAME skin (a mole/dot/scar + its surrounding
pores) at two nearby texture locations, so the two patches are near-IDENTICAL.
Two genuinely-distinct lesions are never pixel-identical in their surroundings.
So: for every detected feature, template-match its patch against the local
neighbourhood; a strong secondary correlation peak (besides itself) is a
duplicate. Robust, and needs no source frames.

total_doublings is the scalar to drive to ZERO across render settings.

  openderm-check-doublings captures/<scan>/registration3d-canonical
"""

from __future__ import annotations
import argparse, os
import numpy as np
import cv2
from scipy.ndimage import maximum_filter

from skinmap.tex_anchor import load_gauge, coverage_mask
from skinmap.lesions import (
    detect_angiomas,
    detect_moles,
    hemoglobin_flat,
    melanin_flat,
)
from skinmap.ghost_check import load_rig, build_uv_to_world, project_pt, load_src


def matched_z(im, px, py, ppmm_s, channel):
    """Feature-vs-skin z-score at a KNOWN pixel: how much darker (mole) or redder
    (angioma) the spot is than its own local skin ring, in robust-sigma units.
    Robust to raw-frame pore noise because it's a local contrast at a fixed
    location -- no blind peak detection / adaptive global threshold."""
    R = im[:, :, 2].astype(np.float32)
    G = im[:, :, 1].astype(np.float32)
    ch = (
        (np.log(np.maximum(R, 1)) - np.log(np.maximum(G, 1)))
        if channel == "hem"
        else -np.log(np.maximum(R, 1))
    )  # mole = darker = higher
    r0 = max(2, int(0.7 * ppmm_s))
    rout = int(4.5 * ppmm_s)
    rin = int(2.0 * ppmm_s)
    x, y = int(round(px)), int(round(py))
    y0, y1 = max(0, y - rout), min(ch.shape[0], y + rout)
    x0, x1 = max(0, x - rout), min(ch.shape[1], x + rout)
    sub = ch[y0:y1, x0:x1]
    if sub.size < 25:
        return -99.0
    yy, xx = np.ogrid[: sub.shape[0], : sub.shape[1]]
    d = np.hypot(xx - (x - x0), yy - (y - y0))
    disk = sub[d <= r0]
    ring = sub[(d >= rin) & (d <= rout)]
    if disk.size < 3 or ring.size < 20:
        return -99.0
    rmed = float(np.median(ring))
    mad = 1.4826 * float(np.median(np.abs(ring - rmed))) + 1e-6
    return (float(np.median(disk)) - rmed) / mad


def seeds(tex, cov, ppmm, k_sigma):
    er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(3 * ppmm) | 1,) * 2)
    dm = cv2.erode(cov, er)
    mel = melanin_flat(tex, cov, ppmm)
    hem = hemoglobin_flat(tex, cov, ppmm)
    s = [
        dict(x=m["x"], y=m["y"], r=m["radius_mm"], ch="mel")
        for m in detect_moles(mel, dm, ppmm, k_sigma=k_sigma)
    ] + [
        dict(x=a["x"], y=a["y"], r=a["radius_mm"], ch="hem")
        for a in detect_angiomas(tex, hem, mel, dm, ppmm, k_sigma=k_sigma)
    ]
    return s, dm


def find(reg_dir, args):
    g = load_gauge(reg_dir)
    ppmm = g.ppmm
    tex = cv2.imread(os.path.join(reg_dir, "texture.jpg"))
    cov = coverage_mask(reg_dir)
    gray = cv2.cvtColor(tex, cv2.COLOR_BGR2GRAY).astype(np.float32)
    cand, dm = seeds(tex, cov, ppmm, args.k_sigma)
    covm = dm > 0
    print(f"  {len(cand)} seed features @ {ppmm:.0f}px/mm", flush=True)

    pr = int(args.patch_mm * ppmm)  # half patch
    sr = int(args.search_mm * ppmm)  # half search window
    excl = int(args.min_offset_mm * ppmm)
    H, W = gray.shape
    pairs = []
    for s in cand:
        x, y = s["x"], s["y"]
        if x - pr < 0 or y - pr < 0 or x + pr >= W or y + pr >= H:
            continue
        patch = gray[y - pr : y + pr, x - pr : x + pr]
        if patch.std() < args.min_std:  # featureless skin -> skip
            continue
        wx0, wy0 = max(0, x - sr), max(0, y - sr)
        win = gray[wy0 : min(H, y + sr), wx0 : min(W, x + sr)]
        if win.shape[0] <= patch.shape[0] or win.shape[1] <= patch.shape[1]:
            continue
        res = cv2.matchTemplate(win, patch, cv2.TM_CCOEFF_NORMED)
        # peak coords are top-left of the matched patch; centre offset:
        pk = (res == maximum_filter(res, size=int(2 * excl) | 1)) & (res > args.ncc)
        ys, xs = np.where(pk)
        for row_idx, col_idx in zip(ys, xs):
            dx = (col_idx + pr) - (x - wx0)  # dup_centre - seed_centre (px)
            dy = (row_idx + pr) - (y - wy0)
            off = np.hypot(dx, dy) / ppmm
            if off < args.min_offset_mm or off > args.search_mm:
                continue
            dupx, dupy = x + dx, y + dy
            if not (0 <= dupx < W and 0 <= dupy < H and covm[dupy, dupx]):
                continue
            pairs.append(
                (float(res[row_idx, col_idx]), x, y, int(dupx), int(dupy), float(off), s["ch"])
            )
    # dedupe symmetric / overlapping pairs: canonical unordered key at ~3mm grid
    pairs.sort(reverse=True)
    seen, uniq = set(), []
    q = max(1, int(3 * ppmm))
    for p in pairs:
        ncc, x, y, dx, dy, off, ch = p
        a = (min(x // q, dx // q), min(y // q, dy // q))
        b = (max(x // q, dx // q), max(y // q, dy // q))
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return dict(g=g, tex=tex, ppmm=ppmm, pairs=uniq, n_seeds=len(cand))


def write(res, out_dir):
    os.makedirs(os.path.join(out_dir, "pairs"), exist_ok=True)
    tex, ppmm = res["tex"], res["ppmm"]
    ov = tex.copy()
    for k, (ncc, x, y, dx, dy, off, ch) in enumerate(res["pairs"]):
        cv2.circle(ov, (x, y), 16, (0, 0, 255), 3)
        cv2.circle(ov, (dx, dy), 16, (0, 0, 255), 3)
        cv2.line(ov, (x, y), (dx, dy), (0, 0, 255), 2)
        cv2.putText(
            ov,
            f"{k}:{ncc:.2f}",
            (min(x, dx), min(y, dy) - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        w = int(6 * ppmm)
        a = tex[max(0, y - w) : y + w, max(0, x - w) : x + w]
        b = tex[max(0, dy - w) : dy + w, max(0, dx - w) : dx + w]
        if a.size and b.size:
            a = cv2.resize(a, (240, 240))
            b = cv2.resize(b, (240, 240))
            cv2.putText(a, "copy A", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(b, "copy B", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(
                os.path.join(out_dir, "pairs", f"pair_{k}_ncc{ncc:.2f}.png"), np.hstack([a, b])
            )
    cv2.putText(
        ov,
        f"doublings={len(res['pairs'])}",
        (20, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3,
    )
    cv2.imwrite(os.path.join(out_dir, "doublings_overlay.png"), ov)


def confirm(res, reg_dir, args):
    """For each NCC candidate pair A<->B, use the SOURCE frames as ground truth:
    if it's one real feature doubled, NO single frame can show a strong feature
    at BOTH placements; two genuinely-distinct lesions appear at both in most
    covering frames. Returns pairs with a verdict + evidence."""
    g = res["g"]
    K, frames = load_rig(reg_dir)
    uv2w = build_uv_to_world(reg_dir, g)
    zt = args.z_thresh
    out = []
    for ncc, x, y, dx, dy, off, ch in res["pairs"]:
        uA = g.px_to_uv(x, y)
        uB = g.px_to_uv(dx, dy)
        PA = uv2w(float(uA[0]), float(uA[1]))
        PB = uv2w(float(uB[0]), float(uB[1]))
        if PA is None or PB is None or np.any(np.isnan(PA)) or np.any(np.isnan(PB)):
            out.append((ncc, x, y, dx, dy, off, ch, "unknown", 0, 0, 0))
            continue
        PA = np.asarray(PA).ravel()
        PB = np.asarray(PB).ravel()
        both = a_only = b_only = ncov = 0
        for f in frames:
            aX, aY, aZ = project_pt(PA, f, K)
            bX, bY, bZ = project_pt(PB, f, K)
            inb = lambda px, py, z: z > 10 and 12 <= px < K["Wf"] - 12 and 12 <= py < K["Hf"] - 12
            if not (inb(aX, aY, aZ) and inb(bX, bY, bZ)):
                continue
            sds = max(1, round(K["fxf"] / f["standoff"] / args.src_ppmm))
            im = load_src(f["path"], sds)
            if im is None:
                continue
            pp = K["fxf"] / f["standoff"] / sds
            zA = matched_z(im, aX / sds, aY / sds, pp, ch)
            zB = matched_z(im, bX / sds, bY / sds, pp, ch)
            ncov += 1
            if zA > zt and zB > zt:
                both += 1
            elif zA > zt:
                a_only += 1
            elif zB > zt:
                b_only += 1
        if ncov < args.min_cover:
            verdict = "unknown"
        elif both >= max(2, 0.35 * ncov):
            verdict = "distinct"  # both real in many frames
        elif a_only + b_only >= max(2, 0.35 * ncov) and both <= 0.15 * ncov:
            verdict = "DOUBLING"  # each seen alone, never together
        else:
            verdict = "weak"  # not clearly a feature
        out.append((ncc, x, y, dx, dy, off, ch, verdict, both, a_only + b_only, ncov))
    res["confirmed"] = out
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("reg_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--patch-mm", type=float, default=5.0)
    ap.add_argument("--search-mm", type=float, default=22.0)
    ap.add_argument("--min-offset-mm", type=float, default=3.0)
    ap.add_argument("--ncc", type=float, default=0.5)
    ap.add_argument("--k-sigma", type=float, default=3.5)
    ap.add_argument("--min-std", type=float, default=4.0)
    ap.add_argument(
        "--no-confirm", action="store_true", help="skip the source-frame ground-truth confirmation"
    )
    ap.add_argument(
        "--z-thresh",
        type=float,
        default=3.5,
        help="matched-filter z for 'a feature is present here'",
    )
    ap.add_argument("--src-ppmm", type=float, default=20.0)
    ap.add_argument("--min-cover", type=int, default=3)
    args = ap.parse_args()
    reg = args.reg_dir.rstrip("/")
    out = args.out or os.path.join(reg, "doublings")
    res = find(reg, args)
    print(f"  {len(res['pairs'])} NCC candidate pairs; confirming vs source...", flush=True)
    if args.no_confirm:
        write(res, out)
        print(f"DOUBLINGS(NCC only): {len(res['pairs'])}  -> {out}", flush=True)
        return
    conf = confirm(res, reg, args)
    write(res, out)
    order = {"DOUBLING": 0, "distinct": 1, "weak": 2, "unknown": 3}
    conf.sort(key=lambda c: (order[c[7]], -c[0]))
    ndbl = 0
    for ncc, x, y, dx, dy, off, ch, verdict, both, alone, ncov in conf:
        u, v = g_uv(res["g"], x, y)
        if verdict == "DOUBLING":
            ndbl += 1
        if verdict in ("DOUBLING", "distinct"):
            print(
                f"    [{verdict:8}] {ch} off={off:4.1f}mm ncc={ncc:.2f} @ "
                f"(u={u:.0f},v={v:.0f})mm  frames both={both} alone={alone}/{ncov}",
                flush=True,
            )
    print(
        f"CONFIRMED DOUBLINGS: {ndbl}   (NCC candidates {len(res['pairs'])}, "
        f"seeds {res['n_seeds']})  -> {out}",
        flush=True,
    )


def g_uv(g, x, y):
    u, v = g.px_to_uv(x, y)
    return float(u), float(v)


if __name__ == "__main__":
    main()
