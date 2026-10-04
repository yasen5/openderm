# Capture and processing procedure

## 1. Prepare the system

Confirm all of the following with an inert target first:

- The emergency stop removes motion power and is reachable by the operator.
- X, Y, Z, and RX limit switches report correctly.
- X travel is limited to 0–800 mm, Y to 0–665 mm, and Z to 0–392 mm.
- Klipper owns X only. The Pico exclusively owns Y and Z.
- The collision model is enabled (`OPENDERM_COLLISION_MODE=enforce`).
- Both Pis have the same untracked `OPENDERM_CONTROL_TOKEN`, and motion ports
  are restricted to the private control network.
- The Canon lens is manually focused at the 110 mm working distance, and image stabilization is off.

Start Pi #1:

```bash
openderm-gantry-server --host 0.0.0.0 --port 8090
openderm-pico-bridge --host 0.0.0.0 --port 8095
```

Start the RX service on Pi #2:

```bash
openderm-rx-axis-server --axis rx --host 127.0.0.1 --port 8091
```

## 2. Calibration prerequisites

Calibrate your camera intrinsics independently using the exact camera, lens, manual-focus setting, and full-resolution image dimensions used for capture. OpenDerm does not provide a camera-calibration utility. Pass the calibrated full-resolution focal length in pixels to `openderm-process` with `--fx-full`. The default is `39237 px`, which applies to the reference camera configuration. Recalibrate after changing the camera, lens, focus, or image dimensions.

`openderm-process` fits the rig model for every scan and then uses that scan's own stage-one placements. Do not reuse placements from another scan or from a different mechanical configuration.

The RX pivot model used during capture must also match the current lever arm. With an inert target, record at least six same-point poses across the intended RX range, then fit the model:

```bash
python src/scripts/calibration/rx_pivot_capture.py 0 --degrees
python src/scripts/calibration/rx_pivot_fit.py captures/rx_pivot_poses_<timestamp>.jsonl
```

The capture utility regulates Z while the arrow keys make small X/Y alignment jogs. Press Enter to record a pose and enter the next RX angle. The fit writes `captures/rx_pivot_model.json`, which is the scanner default.

## 3. Home and position

Home one axis at a time while watching the motion direction:

```bash
openderm --axis x home
openderm --axis y home
openderm --axis z home
openderm --axis rx home
```

Move to a conservative pose above the inert target, then use explicit `move-to` commands to approach the planned start point. Never copy a start pose from another physical build without checking clearances.

```bash
openderm --axis x move-to <x-mm>
openderm --axis y move-to <y-mm>
openderm --axis z move-to <z-mm>
openderm --axis rx move-to <rx-rad> --speed 0.2
```

Keep Y inside the scan window 50–570 mm. The subject surface must already be within sensor range at the start point.

## 4. Verify standoff regulation

With an inert target, run:

```bash
openderm-regulate --debug
```

The average sensor distance should converge to 110 mm and `sensor1-sensor2` should converge toward zero.

The regulator defaults are:

```text
--target-mm 110 --rx-gain-rad-per-mm -0.005 --rx-max-step-rad 0.004
--rx-filter-alpha 0.2 --samples 1 --rx-speed-rad-s 0.02
--rx-deadband-mm 0.5
```

Press Ctrl-C after stable regulation. Verify the final position before capture.

## 5. Capture a scan

Place X along the primary scan direction. The scanner follows the subject surface in Y and tilts RX around its contour. Run a short `--no-camera` test first:

```bash
openderm-scan captures/test-scan-motion --no-camera --x-travel-mm 40 --debug
```

Confirm that edge recovery tilts toward the body, all pivots remain clear, and the scan parks safely. Then capture:

```bash
openderm-scan captures/<subject-id>-<site>-<session> --debug
```

The default scan uses 300 mm X travel, 20 mm X spacing, 15 mm Y spacing, continuous Pico motion, 110 mm standoff, Z 5–392 mm, floor rejection at 507±3 mm, edge recovery, RX velocity control, and motion tracing. Run `--show-config` to see every expanded setting.

For a different rig, floor, or field of view, treat those numbers as a starting profile—not universal geometry. Re-measure floor depth before human imaging.

## 6. Inspect the raw capture

Confirm that every entry in `poses.jsonl` has its expected JPEG and CR3 image. 

## 7. Reconstruct

Build a preview reconstruction:

```bash
openderm-process captures/<scan>
```

The command runs two stages. Stage one fits the current scan's rig model without `--rig-from` and writes `registration3d-rigfit/placements3d.json`. Stage two uses exactly that file and writes `registration3d-canonical/`. It then runs the doubling and projection-ghost artifact checks.

Drive confirmed doubling and ghost counts to zero. At full 78 px/mm, small freckle self-similarity can create false positives; inspect every reported crop. After a clean preview, run:

```bash
openderm-process captures/<scan> --quality full
```

## 8. Compare visits

Both scans must cover the same site with an in-frame fiducial and generous overlap. Register them independently with `openderm-process`, then run:

```bash
openderm-compare <baseline-scan> <follow-up-scan> --captures captures
```

Review mutual coverage, alignment confidence, residuals, and uncertainty before reviewing any lesion-change classification. Low-confidence or non-overlapping results are abstentions, not evidence of no change.
