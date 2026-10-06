# Simulated capture from freehand photos (sim-capture)

`sim-capture` replaces `src/capture` when the gantry is unavailable. It takes a
folder of photographs taken by hand, lets COLMAP estimate where every photo was
taken and the shape of the surface, and writes a capture folder that
`src/processing` reconstructs the same way it reconstructs a gantry scan.

It does not touch `src/capture`, and nothing in `src/capture` is needed to run it.

## What it is and is not

- **Poses are estimated, not measured.** The gantry reports camera position to
  about a millimetre from its encoders. Here the poses come from the images, so
  they are exactly as good as the images allow. Processing still refines them
  with its own bundle adjustment.
- **Scale is not observable from images.** You tell sim-capture the approximate
  mean camera-to-surface distance with `--standoff-mm`, and everything in
  millimetres follows from that number. A 5% error in the standoff is a 5% error
  in every length, including lesion sizes in `openderm-compare`. Do not use
  sim-capture output for clinical measurement.
- **One camera, one zoom.** All images must share a size and focal length. The
  focal length and radial distortion are estimated by COLMAP, not read from a
  calibration.
- **One-sided capture.** Processing models the subject as a heightfield
  `z = f(x, y)` seen from above. Photos should look at the same side of the
  subject with neighbouring views overlapping by about 60%. A full orbit around
  an arm is reported as a warning and will not reconstruct well.
- **No breathing-group handling.** The gantry path registers scan rows
  separately to cancel breathing; freehand captures are registered as one rigid
  group.

## Install

```bash
python -m pip install -e ".[vision,sim]"
```

`pycolmap` provides COLMAP. The CPU build is sufficient; COLMAP's own
`colmap` command is not required.

## Run

```bash
openderm-sim-capture photos/ --out captures/arm-sim-001 --standoff-mm 250
openderm-sim-capture photos/ --out captures/arm-sim-001 --standoff-mm 250 --force --process
```

The first command stops after writing the capture folder and prints the
processing command. `--process` runs it. COLMAP and processing memory grows with
the number and resolution of the images (about 1.5 GB for 24 images at 640x480
during COLMAP in testing), so cap or monitor memory on large sets. COLMAP's SIFT needs about 1 GB per thread at
1440x2560, so `--threads` defaults to 4 (16 threads ran out of memory at 10 GB).

Registration statistics, warnings and the scale factor are in
`captures/arm-sim-001/sim/report.json`. Read it before trusting the result:

| Field | What to look for |
|---|---|
| `images_registered` / `images_unregistered` | Most images should register. Below 50% sim-capture refuses to continue (`--min-registered-frac`). |
| `mean_reprojection_error_px` | Well under 1 px for a good run. |
| `other_models_sizes` | Non-empty means COLMAP found disconnected groups of photos. Only the largest is used. |
| `warnings` | Captures that wrap around the subject, strongly three-dimensional subjects, or very oblique views. |

## Output

```
captures/arm-sim-001/
  <image>.jpg, <image>.json   one pair per registered image (processing's input)
  sim/poses.json              COLMAP poses + intrinsics, read by --poses-from
  sim/report.json             statistics, scale, warnings
  sim/sparse.ply              sparse surface in millimetres
  registration3d/             processing outputs (after --process)
```

The `.json` sidecars follow the gantry schema so the loader accepts them. Their
position fields hold the camera centre and the rotation and distance-sensor fields
hold the surface distance; processing ignores them for pose when `--poses-from`
is given. Every sidecar carries `"sim": true`.

`sim-capture` refuses to write into a folder containing JSON it did not create,
so it cannot mix with a real scan. `--force` replaces only an earlier sim-capture
run.

## Restricting to skin (masks)

Real surroundings (a table, a floor) pull COLMAP's poses, the fitted surface and the
texture toward whatever has the most texture. To use only the subject:

```bash
# 1. one mask per image (needs torch + transformers; Sapiens2 body-part model)
python -m sim_capture.tools.skin_masks photos/ --out masks/
# 2. scale and axes from the masked points; processing restricted to the masks
openderm-sim-capture photos/ --out captures/arm-sim-001 --standoff-mm 150 --skin-masks masks/ --process
```

`skin_masks` merges every body-part class into one skin class, fills small holes,
drops small islands and trims the silhouette edge. It prints the skin share per frame
and flags frames far from the median. Any tool that writes `<image stem>.png`
(255 = subject) works instead.

With `--skin-masks`:

- COLMAP still sees the whole image, because skin alone often has too few features
  and the rigid background constrains the poses well.
- `--standoff-mm` now means camera-to-**skin** distance, and the surface axes come
  from the sparse points that land on skin. sim-capture stops if fewer than 20 do.
- The masks are copied to `sim/masks/` and processing gets `--mask-dir`: keypoints
  and landmarks, so the fitted surface, come from skin only, and the ortho-texture
  paints only skin. This forces the CPU texture renderer.

Check `points_on_mask` in `sim/report.json`; a few hundred is typical.

## Using video

```bash
python -m sim_capture.tools.frames_from_video clip.mp4 --out photos --max-frames 90 --max-width 1920
```

samples the video, then keeps the sharpest frame in each time bin so motion blur
does not reach COLMAP. Use footage with a slow, continuous camera move, no cuts or
zoom, and no electronic stabilization crop.

## What changed in src/processing

`register_scan_3d --poses-from sim/poses.json` is a new opt-in mode. When it is
not given, processing behaves as before.

- Camera poses and intrinsics come from the file instead of the gantry rig model.
  The rig pre-fit and the consecutive/cross-row matching are skipped.
- `--mask-dir DIR` (optional) restricts keypoints and texture to per-frame masks
  `<image stem>.png`. Every frame needs one.
- Candidate image pairs still come from predicted footprint overlap. Matches are
  verified against the epipolar geometry of the supplied poses
  (`verify_pair_with_known_poses`), not with the similarity-transform RANSAC,
  which assumes nearly frontal views and little perspective change.
- The gantry-specific bundle-adjustment heuristics (the narrow-field valley
  re-split and its fixed sigmas) are bypassed. `--sigma-t`, `--sigma-r` and
  `--fx-full 0` still apply.
- `placements3d.json` gains `pose_source` (`"gantry"` or `"external"`). With
  `"external"`, the rig fields `lever_mm`, `Rm`, `rx_sign` and `dz0_mm` are
  placeholders and each frame's `R_cam2world` and `C_mm` are authoritative.
