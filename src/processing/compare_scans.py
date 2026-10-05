#!/usr/bin/env python
"""Cross-scan lesion-change tracking.

Two scans of the same site, each registered by register_scan_3d.py into the
SAME gantry-anchored (u,v)-mm ortho-texture, are compared to flag moles that
appeared / disappeared and to report the residual cross-scan alignment quality.

Pipeline (see docs/skin-registration.md "Cross-time tracking"):
  0. Place both textures on one common (u,v)-mm canvas (tex_anchor gauges) and
     intersect their coverage masks -> the region imaged in BOTH scans.
  1. Melanin proxy M = -log(R) + large-sigma illumination flatten (no WB exists).
  2. Residual global alignment: SIFT+FLANN+RANSAC similarity (the maps are
     already ~aligned by the shared gantry gauge; this removes day-to-day
     repositioning). Hard abstain gates on inliers / residual.
  4. Multiscale-LoG mole detection on each flattened melanin map (physical
     scales in mm via the known 20 px/mm).
  5. Match moles across scans (NN under the registration-residual prior, with a
     constellation cross-check that degrades to "uninformative" when moles are
     few). Detected moles are an INDEPENDENT check on step 2, not its driver.
  6. Measure matched lesions in each scan's unwarped image and report
     size/shape changes with registration and detector uncertainty.

Color change is NOT reported: no white-balance / exposure / flash record exists
anywhere in the capture (an in-frame gray card is the capture-side prerequisite).

Run from an environment installed with the `vision` extra:
    openderm-compare baseline-scan follow-up-scan
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from .alignment import (
    IDENT,
    apply_T,
    common_canvas,
    constellation_check,
    global_align,
    match_moles,
    select_alignment,
)
from .lesions import (
    detect_moles,
    mel_threshold,
    melanin_flat,
    present_in_other,
)
from .tex_anchor import load_gauge
from . import track_moles as tm


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("scan_a")
    ap.add_argument("scan_b")
    ap.add_argument("--captures", default="captures")
    ap.add_argument("--reg-dir", default="registration3d-canonical")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ppmm", type=float, default=20.0)
    ap.add_argument(
        "--bg-mm",
        type=float,
        default=12.0,
        help="illumination-flatten background sigma (>=2x largest blob)",
    )
    ap.add_argument("--mole-diam-mm", default="0.8:6.0")
    ap.add_argument(
        "--min-contrast",
        type=float,
        default=0.08,
        help="absolute floor on melanin contrast M=-log(R) above skin",
    )
    ap.add_argument(
        "--k-sigma",
        type=float,
        default=6.0,
        help="per-scan adaptive detection: keep blobs k-sigma above "
        "that scan's own skin-noise floor (handles exposure diffs)",
    )
    ap.add_argument(
        "--prior-mm",
        type=float,
        default=None,
        help="match search radius (default: 3mm + global residual)",
    )
    ap.add_argument(
        "--min-inliers", type=int, default=12, help="abstain below this many RANSAC inliers"
    )
    ap.add_argument(
        "--max-resid-mm",
        type=float,
        default=3.0,
        help="abstain above this global alignment residual",
    )
    args = ap.parse_args()

    rdA = os.path.join(args.captures, args.scan_a, args.reg_dir)
    rdB = os.path.join(args.captures, args.scan_b, args.reg_dir)
    out = args.out or os.path.join(args.captures, args.scan_a, f"compare-{args.scan_b}")
    os.makedirs(out, exist_ok=True)
    diam = tuple(float(x) for x in args.mole_diam_mm.split(":"))

    gA, gB = load_gauge(rdA), load_gauge(rdB)
    # gauge-consistency assert: the (u,v) frame is only shared if the rig gauge is
    gauge_ok = (
        np.allclose(gA.base_R, gB.base_R, atol=1e-6)
        and np.allclose(gA.base_t, gB.base_t, atol=1e-3)
        and abs(gA.fx_fullres_px - gB.fx_fullres_px) < 1.0
    )
    print(f"[0] shared_gauge={gauge_ok}")
    if not gauge_ok:
        print(
            "    WARNING: rig gauges differ -- (u,v) frames are not directly "
            "comparable; alignment leans entirely on features."
        )

    cc = common_canvas(rdA, gA, rdB, gB, args.ppmm)
    mutual = (cc["covA"] & cc["covB"]).astype(np.uint8)
    print(
        f"[0] common canvas {cc['W']}x{cc['H']} @ {args.ppmm}px/mm  "
        f"mutual coverage {100 * mutual.mean():.1f}%  "
        f"({mutual.sum() / args.ppmm**2 / 100:.0f} cm^2)"
    )

    melA = melanin_flat(cc["texA"], cc["covA"], args.ppmm, args.bg_mm)
    melB = melanin_flat(cc["texB"], cc["covB"], args.ppmm, args.bg_mm)
    # erode coverage for detection so coverage-edge wedges aren't flagged as moles
    er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(3 * args.ppmm) | 1,) * 2)
    detA = cv2.erode(cc["covA"], er)
    detB = cv2.erode(cc["covB"], er)

    _sift, ainfo = global_align(cc["texA"], cc["texB"], cc["covA"], cc["covB"], args.ppmm)
    if ainfo["T_B_to_A"] is not None:
        print(
            f"[2] alignment: {ainfo['matches']} matches, {ainfo['inliers']} inliers, "
            f"resid {ainfo['median_resid_mm']:.2f}mm | scale {ainfo['scale']:.4f} "
            f"rot {ainfo['rotation_deg']:.2f}deg shift "
            f"({ainfo['tx_mm']:.1f},{ainfo['ty_mm']:.1f})mm "
            f"spread {ainfo['inlier_spread_mm']:.0f}mm"
        )
    else:
        print(f"[2] alignment FAILED: {ainfo['matches']} matches, no transform")

    # moles are detected BEFORE the alignment decision -- they are the fiducial
    # fallback when SIFT can't latch onto the (often near-featureless) skin.
    molesA = detect_moles(melA, detA, args.ppmm, diam, args.min_contrast, args.k_sigma)
    molesB = detect_moles(melB, detB, args.ppmm, diam, args.min_contrast, args.k_sigma)
    print(f"[4] moles detected: A={len(molesA)}  B={len(molesB)}")

    # pick the alignment by mole corroboration + cross-method agreement (shared)
    al = select_alignment(ainfo, molesA, molesB, args.ppmm, max_resid_mm=args.max_resid_mm)
    T, align_mode, confidence, n_corrob = (al["T"], al["mode"], al["confidence"], al["n_corrob"])
    mole_inl = al["mole_inl"]
    methods_agree, sift_overwhelming = al["methods_agree"], al["sift_overwhelming"]
    print("[3] candidates by moles-aligned: " + ", ".join(f"{nm}={n}" for nm, n in al["scored"]))
    why = []
    if n_corrob >= 5:
        why.append(f"{n_corrob} moles aligned")
    if methods_agree:
        why.append("SIFT+mole transforms agree")
    if sift_overwhelming:
        why.append(f"{ainfo['inliers']} SIFT inliers over {ainfo['inlier_spread_mm']:.0f}mm")
    prior_mm = args.prior_mm or 4.0
    print(
        f"[3] alignment: {align_mode} ({n_corrob} moles corroborate"
        + (f"; {', '.join(why)}" if why else "")
        + f")  ->  "
        f"confidence={confidence.upper()}"
    )

    pairs, onlyA, onlyB = match_moles(molesA, molesB, T, args.ppmm, prior_mm)
    constel = constellation_check(molesA, molesB, pairs, args.ppmm)

    # new/disappeared: gate to mutual coverage AND verify the spot isn't merely
    # sub-threshold in the other scan (exposure differs across scans -> a faint
    # mole below the other threshold must NOT be reported as appeared/vanished).
    Tinv = cv2.invertAffineTransform(T) if T is not None else IDENT  # A-frame->B
    # verify against the same lightly-smoothed melanin the detector uses
    melA_s = gaussian_filter(melA, 0.35 * args.ppmm)
    melB_s = gaussian_filter(melB, 0.35 * args.ppmm)
    thrB = mel_threshold(melB_s, detB, args.min_contrast, args.k_sigma)
    thrA = mel_threshold(melA_s, detA, args.min_contrast, args.k_sigma)
    disappeared = []
    for i in onlyA:
        xy = apply_T(Tinv, [[molesA[i]["x"], molesA[i]["y"]]])[0]  # into B frame
        xi, yi = int(round(xy[0])), int(round(xy[1]))
        if not (
            0 <= yi < cc["covB"].shape[0]
            and 0 <= xi < cc["covB"].shape[1]
            and cc["covB"][yi, xi] > 0
        ):
            continue  # outside mutual coverage
        if present_in_other(melB_s, xy, args.ppmm, thrB):
            continue  # sub-threshold in B, not gone
        disappeared.append(i)
    new = []
    for j in onlyB:
        xy = apply_T(T, [[molesB[j]["x"], molesB[j]["y"]]])[0]  # into A frame
        xi, yi = int(round(xy[0])), int(round(xy[1]))
        if not (
            0 <= yi < cc["covA"].shape[0]
            and 0 <= xi < cc["covA"].shape[1]
            and cc["covA"][yi, xi] > 0
        ):
            continue
        if present_in_other(melA_s, xy, args.ppmm, thrA):
            continue
        new.append(j)
    print(
        f"[5] matched {len(pairs)} moles (prior {prior_mm:.1f}mm)  "
        f"constellation: {constel['status']}"
    )

    # M2: per matched pair, measure size/shape change in each scan's OWN un-warped
    # frame (the transform is used only to MATCH, never to resample lesion pixels)
    changes, calib = tm.measure_changes(
        cc, melA, melB, detA, detB, molesA, molesB, pairs, al, args.ppmm
    )
    n_grew = sum(1 for c in changes if c["class"] == "grew")
    n_shrank = sum(1 for c in changes if c["class"] == "shrank")
    n_stable = sum(1 for c in changes if c["class"] == "stable")
    print(
        f"[6] size change: {n_grew} grew, {n_shrank} shrank, {n_stable} stable"
        + ("" if not calib.get("uncalibrated") else "  (uncalibrated floors)")
    )

    verdict = {
        "high": "RELIABLE",
        "medium": "PROVISIONAL",
        "low": "LOW CONFIDENCE -- registration unreliable",
    }[confidence]
    print(f"\n=== {verdict} ===")
    if confidence == "low":
        print(
            "  Too few shared fiducials to recover the subject repositioning "
            f"between scans ({args.scan_a} vs {args.scan_b})."
        )
        print(
            "  Reporting detected moles + aligned maps for visual review; "
            "new/disappeared and size-change calls below are PROVISIONAL."
        )
    print(f"  matched (candidate stable): {len(pairs)}")
    print(f"  NEW   (in {args.scan_b} only, within overlap): {len(new)}")
    print(f"  DISAPPEARED (in {args.scan_a} only, within overlap): {len(disappeared)}")
    for c in sorted(changes, key=lambda d: -abs(d["z_diam"])):
        if c["significant"]:
            print(
                f"  {c['class'].upper()}: {c['id']} @ {c['uv_a_mm']}mm  "
                f"{c['d_eq_a_mm']}->{c['d_eq_b_mm']}mm  "
                f"(d{c['delta_diam_mm']:+.2f}+-{c['sigma_diam_mm']}mm, z={c['z_diam']})"
            )

    # the recovered subject repositioning (B->A), in gantry mm
    if T is not None:
        Tn = np.asarray(T)
        reposition = dict(
            scale=float(np.hypot(Tn[0, 0], Tn[0, 1])),
            rotation_deg=float(np.degrees(np.arctan2(Tn[1, 0], Tn[0, 0]))),
            shift_u_mm=float(Tn[0, 2] / args.ppmm),
            shift_v_mm=float(Tn[1, 2] / args.ppmm),
        )
        print(
            f"  repositioning {args.scan_b}->{args.scan_a}: scale "
            f"{reposition['scale']:.3f}, rot {reposition['rotation_deg']:.2f}deg, "
            f"shift ({reposition['shift_u_mm']:.1f},{reposition['shift_v_mm']:.1f})mm"
        )
    else:
        reposition = None

    # aligned green/magenta overlay (single dark spot = aligned mole)
    Bw = cv2.warpAffine(
        cc["texB"],
        np.asarray(T) if T is not None else IDENT,
        (cc["W"], cc["H"]),
        flags=cv2.INTER_LINEAR,
        borderValue=(40, 40, 40),
    )
    mix = cc["texA"].copy()
    mix[:, :, 0] = Bw[:, :, 0]
    mix[:, :, 2] = Bw[:, :, 2]
    cv2.putText(
        mix,
        f"green={args.scan_a}  magenta={args.scan_b} aligned  (single dark spot = aligned mole)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(os.path.join(out, "alignment_overlay.png"), mix)

    # ----- outputs -----
    report = dict(
        scan_a=args.scan_a,
        scan_b=args.scan_b,
        reg_dir=args.reg_dir,
        verdict=verdict,
        confidence=confidence,
        align_mode=align_mode,
        corroboration=dict(
            moles_aligned=int(n_corrob),
            methods_agree=bool(methods_agree),
            sift_overwhelming=bool(sift_overwhelming),
        ),
        repositioning_b_to_a=reposition,
        gauge=dict(shared_gauge=bool(gauge_ok)),
        canvas=dict(
            umin=cc["umin"],
            vmin=cc["vmin"],
            ppmm=cc["ppmm"],
            W=cc["W"],
            H=cc["H"],
            mutual_coverage_cm2=float(mutual.sum() / args.ppmm**2 / 100),
        ),
        alignment=ainfo,
        mole_inliers=int(mole_inl),
        prior_mm=prior_mm,
        constellation=constel,
        color_note="color change NOT assessed: no white-balance/exposure/flash "
        "reference recorded (add an in-frame gray card to enable)",
        size_change=dict(
            sig_k=calib.get("k"),
            null_floor_diam_mm=calib.get("floor_diam"),
            calibrated=not calib.get("uncalibrated", True),
            measured="each scan's own un-warped frame (warp never touches lesion pixels)",
            n_grew=n_grew,
            n_shrank=n_shrank,
            n_stable=n_stable,
        ),
        matched=changes,
        new=[dict(uv_mm=list(cc_uv(cc, molesB[j])), diam_mm=molesB[j]["diam_mm"]) for j in new],
        disappeared=[
            dict(uv_mm=list(cc_uv(cc, molesA[i])), diam_mm=molesA[i]["diam_mm"])
            for i in disappeared
        ],
    )
    with open(os.path.join(out, "change_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    tm.write_change_csv(
        os.path.join(out, "change_report.csv"), cc, changes, molesB, molesA, new, disappeared
    )
    tm.write_change_overlay(
        out,
        cc,
        mutual,
        changes,
        molesA,
        molesB,
        pairs,
        new,
        disappeared,
        T,
        args.ppmm,
        labels=(args.scan_a, args.scan_b),
    )
    tm.write_change_montage(
        out, cc, changes, molesA, molesB, pairs, T, args.ppmm, labels=(args.scan_a, args.scan_b)
    )
    cv2.imwrite(os.path.join(out, "common_A.jpg"), cc["texA"], [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(os.path.join(out, "common_B.jpg"), cc["texB"], [cv2.IMWRITE_JPEG_QUALITY, 90])
    _write_detections(os.path.join(out, "detections_A.jpg"), cc["texA"], molesA, cc["ppmm"])
    _write_detections(os.path.join(out, "detections_B.jpg"), cc["texB"], molesB, cc["ppmm"])
    print(
        f"\nwrote {out}/  (change_report.json/.csv, change_overlay.png, "
        f"alignment_overlay.png, detections_A/B.jpg, common_A/B.jpg)"
    )


def cc_uv(cc, m):
    return (round(cc["umin"] + m["x"] / cc["ppmm"], 1), round(cc["vmin"] + m["y"] / cc["ppmm"], 1))


def _write_csv(path, cc, molesA, molesB, pairs, new, disappeared):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["type", "u_mm", "v_mm", "diam_a_mm", "diam_b_mm", "resid_mm"])
        for p in pairs:
            u, v = cc_uv(cc, molesA[p["iA"]])
            w.writerow(
                [
                    "matched",
                    u,
                    v,
                    round(molesA[p["iA"]]["diam_mm"], 2),
                    round(molesB[p["iB"]]["diam_mm"], 2),
                    round(p["resid_mm"], 2),
                ]
            )
        for j in new:
            u, v = cc_uv(cc, molesB[j])
            w.writerow(["new", u, v, "", round(molesB[j]["diam_mm"], 2), ""])
        for i in disappeared:
            u, v = cc_uv(cc, molesA[i])
            w.writerow(["disappeared", u, v, round(molesA[i]["diam_mm"], 2), "", ""])


def _write_detections(path, tex, moles, ppmm):
    canvas = tex.copy()
    for m in moles:
        cv2.circle(canvas, (m["x"], m["y"]), int(m["radius_mm"] * ppmm) + 3, (0, 255, 255), 2)
    cv2.putText(
        canvas,
        f"{len(moles)} detections",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])


def _write_overlay(out, cc, mutual, molesA, molesB, pairs, new, disappeared, T):
    base = cc["texA"].copy()
    dim = (base.astype(np.float32) * 0.45).astype(np.uint8)
    m3 = mutual[..., None].astype(bool)
    canvas = np.where(m3, base, dim)
    for p in pairs:  # stable = green
        m = molesA[p["iA"]]
        cv2.circle(canvas, (m["x"], m["y"]), int(m["radius_mm"] * cc["ppmm"] + 6), (0, 200, 0), 2)
    for j in new:  # new = red (B coords -> A)
        b = molesB[j]
        xy = apply_T(T, [[b["x"], b["y"]]])[0]
        cv2.circle(
            canvas, (int(xy[0]), int(xy[1])), int(b["radius_mm"] * cc["ppmm"] + 6), (0, 0, 255), 3
        )
    for i in disappeared:  # disappeared = blue
        a = molesA[i]
        cv2.circle(canvas, (a["x"], a["y"]), int(a["radius_mm"] * cc["ppmm"] + 6), (255, 80, 0), 3)
    cv2.putText(
        canvas,
        "green=stable  red=new  blue=disappeared  (dim=outside mutual coverage)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(os.path.join(out, "change_overlay.png"), canvas)


if __name__ == "__main__":
    main()
