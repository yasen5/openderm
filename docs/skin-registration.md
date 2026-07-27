# Skin scan registration — goal & method

## Goal

Build 2D skin maps for **(a)** reviewing fine surface features and **(b)** researching how those features can be tracked across scans taken on different days. The output is not a diagnosis and requires clinician review. The subject can move or **breathe** during the minutes-long scan, so the reconstruction must tolerate small between-pass changes.

The diagnostic signal is the **2D skin texture**, not 3D shape. So the 3D surface is only a **scaffold** to unwrap onto — it must be **smooth, stable, and reproducible**, not an accurate instantaneous reconstruction. This makes breathing a *nuisance to reject*, not motion to reconstruct.

## Key Components

1. **Lock fx.** The narrow ~8° FOV makes focal length and depth nearly degenerate. Calibrate the camera intrinsics independently with the exact capture configuration, then pass the full-resolution focal length in pixels through `--fx-full`. OpenDerm defaults to `39237 px` for the reference camera.
2. **Lock manual focus at capture.** Autofocus changes magnification ~1%/frame at macro distance (focus position *is* effective focal length).
3. **Use each measurement for what it observes.** Images provide sub-pixel *in-plane* precision while the recorded robot poses and surface landmarks provide the reconstruction scaffold.
4. **Fit the subject contour** (`--contour on`). The landmark surface follows the captured anatomy without relying on a camera-frame laser calibration.
5. **Deformable in-plane alignment, keeping the scaffold stable** (`--deformable`). A smooth per-frame affine warp *in the 2D map* aligns overlapping frames' features, removing the residual breathing offset so stitch seams don't cross moles. The 3D surface is left untouched — only texture content is aligned.
6. **Gantry-anchored unwrap.** Parameterize the map in the gantry (proprioception) frame so the same skin lands at the same map coordinate every scan — the precondition for cross-time comparison. Subject repositioning between days is handled later, by feature matching across scans.
7. **Two-band blend.** HF detail is winner-take-all per texel (sharp, single-source — no ghosting); LF illumination is feather-blended (smooth — no per-frame tiling); a small floor on the centre weight guarantees no coverage gaps. `--blend two-band --focus-weight 4`.

## Calibration

- **fx = 39237 px by default.** This value applies to the reference camera. Calibrate your own camera externally and override it with `--fx-full`. Recalibrate if the camera, lens, focus, or image dimensions change.
- **Surface model.** Reconstruction uses the per-scan landmark contour, avoiding camera-frame geometry that could become invalid after a mechanical change.

## Recommended commands

```bash
openderm-process captures/<scan>
```

The command first fits the current scan's rig model without `--rig-from`, then uses that scan's own stage-one `placements3d.json` for the canonical run. Preview quality uses `--downscale 3 --texture-ppmm 20`; add `--quality full` for `--downscale 1 --texture-ppmm 78`. The wrapper also runs the doubling and ghost checks.

## Cross-time tracking

Run both scans as above and they land in the **same gantry-anchored frame**. Compare by detecting moles in each map and matching them (the moles are their own fiducials); the residual after aligning the stable moles is the change signal.

**Tool: `openderm-compare SCAN_A SCAN_B`.** Reads each scan's `registration3d-canonical/` and:

1. **Common canvas** — recovers each map's `(u,v)`-mm gauge (`tex_anchor.py`) and resamples both onto one common `(u,v)` canvas at 20 px/mm; intersects the coverage masks to the region imaged in **both** scans. The texture is linear in `(u_mm, v_arclen_mm)`, so only each scan's `(umin,vmin)` offset is needed.
2. **Melanin + flatten** — `M = -log(R)` high-passed by a ≥12 mm background estimated from covered skin only, making the detector robust to residual session-wide exposure differences.
3. **Tiered alignment** — SIFT+RANSAC similarity, **accepted only if near-identity** (the shared gantry gauge guarantees the residual transform is ~identity; a scale-0/180°/clumped fit is rejected). Falls back to a **mole-constellation** fit (moles as fiducials), then to **gantry-only** (low confidence).
4. **Mole detection** — multiscale-LoG on the melanin map over physical scales, kept by a melanin-contrast threshold; NMS by blob radius.
5. **Match + verdict** — NN under the alignment-residual prior with a constellation cross-check; NEW / DISAPPEARED **hard-gated to mutual coverage**. Verdict tier: RELIABLE / PROVISIONAL / LOW-CONFIDENCE.

Outputs to `captures/<A>/compare-<B>/`: `change_report.json` + `.csv`, `change_overlay.png` (green=stable, red=new, blue=disappeared, dim=outside mutual coverage), `detections_{A,B}.jpg`, `common_{A,B}.jpg`.

Each reconstruction records the exact texture gauge in the `texture` block of `placements3d.json` and writes a lossless `coverage.png`. Comparison requires both artifacts and will ask the user to rebuild an incomplete reconstruction.

Use same-scan null tests to verify that all lesions match with no false new or disappeared detections. Cross-visit comparisons require generous overlap and enough stable fiducials to estimate repositioning; otherwise the tool should report low confidence and abstain.

### Lesion size-change and uncertainty

`src/skinmap/track_moles.py` (wired into `compare_scans`, stage `[6]`). For every matched mole pair it reports Δdiameter, Δarea, Δshape with a per-mole error bar and a significance flag (grew / shrank / stable).

- **Exposure-invariant size.** `melanin_flat` is a log-ratio (`log(R_skin/R)`), so a multiplicative exposure/WB gain cancels — size is measured at the **half-max iso-contour of each mole's own peak** (a fraction-of-peak boundary), not an absolute threshold. Saturated or badly exposed images are rejected during capture review because destroyed image information cannot be recovered.
- **Warp-independent measurement.** Every size number is measured in each scan's **own un-warped frame**; the transform is used only to MATCH moles, never to resample lesion pixels, so alignment cannot erase an apparent size change.
- **Uncertainty** = registration σ (local residual of the chosen transform's inliers, interpolated to the mole — RBF for dense SIFT, IDW for sparse moles) ⊕ detector σ (std over jittered re-detections, using the scan's own skin noise) ⊕ a null floor, in quadrature. **Significant** iff |z|≥k AND |Δ|≥floor. Conservative defaults are k=2.5 and a 0.30 mm floor. Use same-scan null comparisons to characterize the complete system before interpreting results.
- **Outputs:** `change_report.json/.csv` (per mole: d_eq_a/b, Δ±σ, z, class, significant), `change_overlay.png` (moles colored stable/grew/shrank/new/gone), `change_montage.png` (per-mole baseline|follow-up crops with the measured contour).

## Pitfalls already fixed (don't reintroduce)

- **Coverage gaps** from a too-peaked centre weight → small floor on the centre weight in the renderer.
- **Illumination tiling** from blending the LF band with the peaked weight → blend the LF with the gentle feather-only weight; keep HF winner-take-all.
- **Holes over featureless skin** from gating the surface support on SIFT-feature density → the support is filled (the surface is laser-driven, not feature-gated).
