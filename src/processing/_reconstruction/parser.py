"""Register a skin-scan capture folder in full 3D.

The rig sweeps a macro camera over skin at ~110mm standoff, saving each photo
with a JSON sidecar of gantry proprioception (x_mm, y_mm, z_mm, rx_rad) and two
standoff distance sensors. The reconstruction models subject curvature directly:

  1. rig-model pre-fit       -- a small nonlinear fit over consecutive-pair
     feature transforms estimates the camera lever arm, mount rotation, rx sign,
     and an initial focal length. The OpenDerm processing command replaces that
     focal-length seed with the calibrated --fx-full value before adjustment.
  2. feature matching        -- CLAHE-boosted SIFT + ratio test + RANSAC over
     all overlapping pairs (predicted by projecting footprints through the rig
     model), with prior gating against repetitive-texture false locks.
  3. 3D bundle adjustment    -- feature tracks are triangulated into 3D skin
     landmarks; alternating intersection/resection refines per-frame 6-DOF
     poses anchored to the proprioception prior. Expert runs that explicitly
     set --fx-full 0 can also refine shared fx and k1.
  4. surface + ortho-texture -- a smoothness-regularised heightfield z(x,y) is
     fit through the landmarks; every photo is projected onto it and blended
     (border feather x incidence weight) into a curvature-corrected texture
     parameterised by arc length, so distances on the texture are true mm.

Outputs (under <capture_dir>/registration3d/):
  texture.jpg / texture_index.jpg   ortho-mosaic on the surface (mm-true)
  surface_mesh.obj/.mtl             textured mesh, mm units (Blender/MeshLab)
  landmarks.ply                     3D landmark cloud coloured by reproj error
  viewer.html                       self-contained interactive 3D viewer
  overview.png                      static matplotlib 3D overview
  placements3d.json                 per-frame camera poses + rig model
  report.txt                        accuracy stats + capture improvement notes

Run from an environment installed with the `vision` extra:
  openderm-register captures/<scan>
"""

from __future__ import annotations

import argparse


def parse_scan_cli_arguments():
    """Parse registration CLI arguments without running the processing pipeline."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("capture_dir")
    ap.add_argument("--downscale", type=int, default=3)
    ap.add_argument("--nfeatures", type=int, default=6000)
    ap.add_argument("--ratio", type=float, default=0.8)
    ap.add_argument("--min-inliers", type=int, default=25)
    ap.add_argument("--overlap-frac", type=float, default=0.10)
    ap.add_argument("--max-partners", type=int, default=14)
    ap.add_argument("--max-corr-per-pair", type=int, default=250)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument(
        "--fx-full",
        type=float,
        default=39237.0,
        help="full-resolution focal length in pixels from an "
        "external camera calibration (default: 39237 for the "
        "reference camera); set to 0 only for experimental "
        "per-run refinement",
    )
    ap.add_argument(
        "--fit-k1",
        action="store_true",
        help="also fit radial distortion k1 (default off: Canon "
        "JPGs are lens-corrected and k1 absorbs IS artifacts)",
    )
    ap.add_argument("--sigma-px", type=float, default=1.5)
    ap.add_argument(
        "--sigma-t",
        type=float,
        default=4.0,
        help="pose translation prior, mm. Must accommodate the lens "
        "IS unit: a per-shot IS ray rotation acts about a point "
        "far from the entrance pupil, appearing as a coupled "
        "rotation+translation of the camera (observed ~10mm).",
    )
    ap.add_argument(
        "--sigma-r", type=float, default=1.5, help="pose rotation prior, deg (IS correction range)"
    )
    ap.add_argument("--surface-pitch", type=float, default=2.0)
    ap.add_argument("--surface-smooth", type=float, default=3.0)
    ap.add_argument("--mesh-pitch", type=float, default=1.0)
    ap.add_argument("--texture-ppmm", type=float, default=20.0)
    ap.add_argument(
        "--blend-sharpness",
        type=float,
        default=100.0,
        help="centre-weight decay; higher = closer to per-texel "
        "winner-take-all (crisper, harder seams)",
    )
    ap.add_argument(
        "--blend",
        choices=("soft", "two-band"),
        default="soft",
        help="soft: weighted average. two-band: high "
        "frequencies winner-take-all per texel (never "
        "averaged), low-pass illumination soft-blended -- "
        "single-image detail with seam-free photometry",
    )
    ap.add_argument(
        "--focus-weight",
        type=float,
        default=0.0,
        help="exponent on measured per-texel sharpness (local "
        "band-passed RMS) in the blend; 0 = off. With it on, "
        "the in-focus band of each frame wins overlaps instead "
        "of the geometric frame centre -- lower "
        "--blend-sharpness (e.g. 20) so focus can outvote "
        "centre distance. Try 4.",
    )
    ap.add_argument(
        "--hf-coherence-mm",
        type=float,
        default=3.0,
        help="two-band mode: decide the per-texel HF winner on a "
        "weight field smoothed by this radius (mm), so ONE frame "
        "owns a whole feature-neighbourhood and a single "
        "mole/dot can't be split into a doubled ghost across the "
        "winner boundary. The deposited HF stays full-res sharp. "
        "0 disables smoothing. ~3mm (a bit above the "
        "residual misalignment) is a good default.",
    )
    ap.add_argument(
        "--mesh-smooth",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("ALONG_MM", "ACROSS_MM"),
        help="smooth the DISPLAY mesh height field (viewer + "
        "surface_mesh.obj) with an anisotropic Gaussian: "
        "sigma along the U axis and sigma across V. On a "
        "breathing subject the placement-true render surface "
        "follows between-pass motion into ~8mm valleys that "
        "are not anatomy; '50 10' flattens them while keeping "
        "the measured cross-section. Texture placement is "
        "unaffected. 0 0 = off.",
    )
    ap.add_argument(
        "--max-incidence-deg",
        type=float,
        default=0.0,
        help="render: zero a frame's texture weight where its view "
        "ray hits the surface more obliquely than this (ramping "
        "from 15deg before). On a strongly curved surface a frame "
        "can see skin through the contour at high incidence; placement "
        "error scales with tan(incidence) so those deposits "
        "land 10-20mm off and ghost large features. 65 is a "
        "good reference value; 0 = off.",
    )
    ap.add_argument(
        "--group-feather-mm",
        type=float,
        default=0.0,
        help="two-band + --group-by-row: gate the LF (illumination/"
        "blush) band by a group-ownership field feathered over "
        "this radius (mm). The hard group gate only covers the "
        "HF winner; ungated LF lets every row-group deposit a "
        "breathing-displaced copy of any >2mm feature (a red "
        "lesion's blush) -- the classic 2-3x offset-blob ghost. "
        "With the feathered gate one group owns each region's LF "
        "(single placement) and illumination still cross-fades "
        "at boundaries. ~4mm is a good value; 0 disables "
        "the gate.",
    )
    ap.add_argument(
        "--hf-cross-group",
        action="store_true",
        help="two-band + --group-by-row: let the HF (detail) winner "
        "compete ACROSS row-groups instead of gating it to the "
        "group that owns each texel. The group gate prevents "
        "cross-row breathing ghosts but DOUBLES a feature sitting "
        "on a row seam (each group paints its own offset copy). "
        "With an accurate rig (small breathing residual) the "
        "coherence-smoothed global winner avoids both. LF/surface "
        "stay per-group.",
    )
    ap.add_argument(
        "--lf-gain",
        choices=("on", "field", "off"),
        default="off",
        help="per-frame photometric gain compensation fitted from "
        "BA track colors (estimate_camera_frame_texture_gains). The lamp travels "
        "with the camera, so frames render the same skin up "
        "to ~40 gray levels apart; the LF cross-fade then "
        "shows blocky tone steps at feather/group-gate "
        "boundaries. 'on' scales each frame's deposit by a "
        "fitted BGR gain (geometric mean 1, so overall "
        "exposure is kept); 'field' fits an affine gain "
        "field per frame (also removes the within-frame "
        "shading/vignetting gradient a scalar cannot). "
        "off disables gain compensation.",
    )
    ap.add_argument(
        "--deformable-order",
        type=int,
        choices=(1, 2),
        default=1,
        help="deformable warp basis per frame: 1 = affine "
        "(can only shift/scale/shear a whole frame), "
        "2 = quadratic (also bends within the footprint -- "
        "removes the residual local breathing misalignment "
        "that cuts hairs at HF-winner seams)",
    )
    ap.add_argument("--limit-rows", type=int, default=0, help="debug: first N rows only")
    ap.add_argument(
        "--rows", default=None, help="debug: only rows A:B inclusive (e.g. 3:4), or a single row N"
    )
    ap.add_argument(
        "--stations", default=None, help="debug: only stations A:B inclusive (e.g. 1:2)"
    )
    ap.add_argument("--cols", default=None, help="debug: only columns A:B inclusive (e.g. 1:3)")
    ap.add_argument(
        "--group-by-row",
        action="store_true",
        help="breathing-robust mode: register each scan row rigidly "
        "on its own tracks, then align rows by a per-row 3D "
        "translation from cross-row matches, and composite the "
        "texture per-row-group so cross-row features render once "
        "(no breathing ghosts). Use with --rig-from.",
    )
    ap.add_argument(
        "--group-align",
        choices=("none", "translation"),
        default="translation",
        help="inter-group correction model for --group-by-row",
    )
    ap.add_argument(
        "--deformable",
        action="store_true",
        help="deformable alignment: a smooth per-frame affine warp in "
        "the unwrapped map that aligns overlapping frames' "
        "features, removing residual breathing misalignment so "
        "stitch seams don't cross moles. 3D surface stays smooth.",
    )
    ap.add_argument(
        "--deformable-reg",
        type=float,
        default=2.0,
        help="regularisation for --deformable (higher = smaller/smoother warp)",
    )
    ap.add_argument(
        "--rig-from",
        default=None,
        help="load the rig model (lever/mount/sign/dz0/fx) from a "
        "stage-one placements3d.json instead of the "
        "self-calibrating pre-fit. Needed for tiny subsets "
        "(< ~2 rows) where the pre-fit is underdetermined.",
    )
    ap.add_argument(
        "--contour",
        choices=("auto", "on", "off"),
        default="auto",
        help="follow the measured subject contour when constructing "
        "the landmark surface. 'auto' enables contour following "
        "when the RX span and surface residual indicate pronounced "
        "curvature; 'on'/'off' force the setting.",
    )
    ap.add_argument(
        "--contour-rx-thresh-deg",
        type=float,
        default=30.0,
        help="auto contour mode: minimum per-scan RX tilt spread in degrees",
    )
    ap.add_argument(
        "--contour-rms-thresh",
        type=float,
        default=18.0,
        help="auto contour mode: minimum landmark-to-surface RMS in mm",
    )
    ap.add_argument(
        "--contour-smooth",
        type=float,
        default=120.0,
        help="surface smoothing used by contour following; larger values "
        "suppress depth noise and breathing-scale variation while "
        "preserving the subject's broad curvature",
    )
    ap.add_argument(
        "--reject-pose-mm",
        type=float,
        default=80.0,
        help="reject frames whose bundle-adjusted camera position is "
        "more than this many mm from the gantry proprioception "
        "prior. The rig encoders locate the camera to ~mm, so a "
        "large recovered translation is a false-match artifact, "
        "not real motion (legitimate breathing/IS is <~40mm). "
        "Such frames are snapped back to the trusted gantry pose "
        "and their observations dropped from the surface fit so a "
        "few bad frames cannot bend the surface into a spike. "
        "0 disables.",
    )
    ap.add_argument(
        "--reject-rot-deg",
        type=float,
        default=25.0,
        help="companion to --reject-pose-mm: also reject frames whose "
        "recovered orientation is more than this many degrees from "
        "the proprioception prior (normal |dr| is small; the "
        "rig's rx tilt is smooth so a large rotation is an "
        "artifact). 0 disables the rotation gate.",
    )
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument(
        "--group-ba",
        choices=["group", "global"],
        default="group",
        help="with --group-by-row: 'group' solves each breathing-"
        "coherent group's poses on its own tracks (best when "
        "groups have dense intra-matches); 'global' bundles "
        "all frames together but KEEPS the group-owned render "
        "compositing -- use when intra-group matches are too "
        "sparse to constrain per-group solves (e.g. a band-"
        "order scan whose full-res matching is weak: per-band "
        "chains drift and the surface/canvas blows up).",
    )
    ap.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="texture-render compute device. auto: CUDA via torch "
        "when available (the per-frame image pipeline runs on "
        "the GPU, ~10-30x faster at high --texture-ppmm), "
        "else CPU. The registration itself is unaffected.",
    )
    ap.add_argument("--out", default=None)
    return ap.parse_args()
