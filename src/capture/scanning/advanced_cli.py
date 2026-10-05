"""Detailed command-line options for OpenDerm contour scanning."""

from __future__ import annotations

import argparse
from capture.config import default_pico_port

from .config import *  # noqa: F401,F403 (DEFAULT_* used as argparse defaults)


def _build_base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Contour-following scan: sweep X across the subject at a fixed surface "
            "orientation, then follow the surface outward in Y toward both flanks. "
            "Regulates z-height (sensor average) and RX angle (sensor difference) at "
            "each station, tilts RX up to 20 deg to tell a steep surface from the "
            "body edge, and captures a Canon EOS R7 photo (or --no-camera to "
            "simulate) at each station."
        )
    )
    parser.add_argument(
        "--gantry-server-url",
        default=DEFAULT_SERVER_URL,
        help=(
            "Base URL of the gantry server on Pi #1 (default from GANTRY_SERVER_URL "
            "or %(default)s)."
        ),
    )
    parser.add_argument(
        "--pico-port",
        default=default_pico_port(),
        help=(
            "Pico serial port shared by Y and Z. A "
            "local device (/dev/ttyACM0) OR a pyserial URL if the Pico is on another host "
            "bridged over the network, e.g. socket://<pi1-ip>:8095 (see "
            "openderm-pico-bridge). Defaults to PICO_PORT or the bridge on the "
            "GANTRY_SERVER_URL host."
        ),
    )
    parser.add_argument(
        "--y-pico-vmax-mm-s",
        type=float,
        default=DEFAULT_PICO_VMAX_MM_S,
        help="Y max speed (mm/s) for the Pico controller (default %(default)s).",
    )
    parser.add_argument(
        "--y-pico-acc-mm-s2",
        type=float,
        default=None,
        help="Y acceleration (mm/s^2) for the Pico controller (default: firmware ACC).",
    )
    parser.add_argument(
        "--z-pico-vmax-mm-s",
        type=float,
        default=DEFAULT_PICO_VMAX_MM_S,
        help="Z max speed (mm/s) for the Pico controller (default %(default)s).",
    )
    parser.add_argument(
        "--z-pico-acc-mm-s2",
        type=float,
        default=None,
        help="Z acceleration (mm/s^2) for the Pico controller (default: firmware ACC).",
    )
    parser.add_argument(
        "--home-y",
        action="store_true",
        help="Home the Y axis on the Pico before scanning.",
    )
    parser.add_argument(
        "--home-z",
        action="store_true",
        help="Home the Z axis on the Pico before scanning.",
    )
    parser.add_argument(
        "--target-mm",
        type=float,
        default=DEFAULT_TARGET_MM,
        help="Target sensor-average distance in mm (default %(default)s).",
    )

    return parser


def _add_x_options(parser: argparse.ArgumentParser) -> None:
    # --- X scanning ---
    parser.add_argument(
        "--x-step-mm",
        type=float,
        default=DEFAULT_X_STEP_MM,
        help="Distance to advance x between stations in mm (default %(default)s).",
    )
    parser.add_argument(
        "--x-travel-mm",
        type=float,
        default=DEFAULT_X_TRAVEL_MM,
        help="Total x travel distance to cover in mm (default %(default)s).",
    )
    parser.add_argument(
        "--band-step-mm",
        type=float,
        default=0.0,
        help=(
            "Focus-y spacing between sweeps in mm. <=0 (default) falls back to "
            "--y-step-mm. Larger values "
            "trade flank resolution for speed (fewer bands = fewer RX steps); the "
            "band-to-band transition is walked in <=--y-step-mm sub-steps, so a "
            "large value remains contour-tracked between capture rows."
        ),
    )
    parser.add_argument(
        "--band-empty-stop",
        type=int,
        default=1,
        help=(
            "End a Y march after this many consecutive "
            "fully-empty bands (nothing captured at any x). Default %(default)s -- a "
            "single empty band ends the march, which is efficient because each x is "
            "PRUNED once it hits its own edge (see the per-x edge memory), so a band "
            "only comes up empty once EVERY remaining x has already flanked out. "
            "Raise it only for debounce against a full-width interior dropout (a "
            "specular/shadowed stripe off at every still-active x at once), at the "
            "cost of extra off-body bands past the flank. Applies to both marches "
            "(descend and ascend from the start row); the start row is assumed on-body."
        ),
    )
    parser.add_argument(
        "--band-edge-recovery",
        action="store_true",
        help=(
            "Run the RX edge-recovery probe at each station instead of skipping an "
            "out-of-range X position immediately. "
            "RX tilts about the surface's primary axis, so it follows the cross-surface "
            "(y) flank -- when a band edge is a side flank, the probe re-acquires the "
            "skin and captures down the curve, and rx hitting its axis limit is the "
            "edge. OFF by default because the fast axis is along the primary scan "
            "direction: most in-sweep dropouts are X edges or tapers, which "
            "recede in a direction RX cannot follow (lateral tilt is not available "
            "on this 4-DOF machine), so probing there is a wasted -- and near the bed, risky "
            "-- swing. The probe self-disambiguates (it only recovers what RX can "
            "physically reach), and the per-band empty-stop still bounds the y march. "
            "Run with --floor-depth-mm/--floor-model on so a probe cannot false-"
            "recover onto the bed."
        ),
    )
    parser.add_argument(
        "--band-miss-stop-frac",
        type=float,
        default=None,
        help=(
            "End a Y march as soon as one sweep's MISS "
            "fraction -- the share of its attempted stations that produced NO image "
            "(off-body / reach-limited / edge / refused move) -- is at or above this "
            "value (e.g. 0.30 = 30%%) AND the band's leveled tilt shows the march "
            "has actually reached its flank (see --band-miss-stop-rx-hi/-lo): a "
            "descend (-y) stops only if some capture in the band LEVELED at rx >= "
            "the hi regime, an ascend (+y) only at rx <= the lo regime. Misses at a "
            "mid-range leveled tilt are along-x dropouts (a surface edge under the "
            "band), not the y flank, and do not end the march. Off by "
            "default, so only --band-empty-stop (which waits for FULLY empty bands) "
            "bounds the march. Measured against the band's still-active x's; "
            "applies to both marches. Must be in (0, 1]. A band with NO captures "
            "has no tilt evidence and is left to --band-empty-stop."
        ),
    )
    parser.add_argument(
        "--band-miss-stop-rx-hi",
        type=float,
        default=DEFAULT_BAND_MISS_STOP_RX_HI,
        help=(
            "Leveled-tilt threshold (rad) marking the -y "
            "flank regime for --band-miss-stop-frac -- a DESCEND march may only "
            "miss-stop once a capture in the offending band leveled at rx >= this "
            "(default %(default)s; the camera has wrapped around toward the -y "
            "flank)."
        ),
    )
    parser.add_argument(
        "--band-miss-stop-rx-lo",
        type=float,
        default=DEFAULT_BAND_MISS_STOP_RX_LO,
        help=(
            "Leveled-tilt threshold (rad) marking the +y "
            "flank regime for --band-miss-stop-frac -- an ASCEND march may only "
            "miss-stop once a capture in the offending band leveled at rx <= this "
            "(default %(default)s)."
        ),
    )
    parser.add_argument(
        "--band-walk-speed-mm-s",
        type=float,
        default=DEFAULT_BAND_WALK_SPEED_MM_S,
        help=(
            "Focus-row cruise speed of the regulated "
            "band-transition walk (mm/s, default %(default)s). The walk advances a "
            "continuously-moving target by speed*dt per regulation iteration and "
            "dispatches without waiting, so the axes flow through the band change "
            "as one motion (streamed Pico batches cruise) instead of stop-at-target "
            "hops. Lower for gentler transitions; the effective ceiling is the z "
            "servo's tracking authority (--gain-mm-per-mm x --max-step-mm per "
            "iteration) on steep flanks -- the walk holds its advance when the "
            "standoff error lags."
        ),
    )
    parser.add_argument(
        "--x-tolerance-mm",
        type=float,
        default=DEFAULT_X_TOLERANCE_MM,
        help=("Position tolerance in mm for an x step to count as arrived (default %(default)s)."),
    )
    parser.add_argument(
        "--x-move-timeout-s",
        type=float,
        default=DEFAULT_X_MOVE_TIMEOUT_S,
        help="Max time to wait for an x step to arrive in stream mode (default %(default)s).",
    )
    parser.add_argument(
        "--record-pause-s",
        type=float,
        default=DEFAULT_RECORD_PAUSE_S,
        help="Pause at each station to simulate a photo capture (default %(default)s).",
    )
    parser.add_argument(
        "--capture-gate-mm",
        type=float,
        default=None,
        help=(
            "Breath gate: after settling, hold the head still and fire the "
            "shutter only when the live sensor average is within this of "
            "--target-mm (a live subject's surface breathes +/-1-2mm; without "
            "the gate the shutter fires ~1-2s after the pose was verified, "
            "mid-drift). On a gate timeout the photo is taken anyway and "
            "flagged in the metadata (capture_gated=false). Suggested: the "
            "z deadband or slightly above it. Default: off."
        ),
    )
    parser.add_argument(
        "--capture-gate-timeout-s",
        type=float,
        default=4.0,
        help=(
            "Max time to wait for the capture gate -- roughly one breath cycle "
            "(default %(default)s)."
        ),
    )
    parser.add_argument(
        "--settle-iters",
        type=int,
        default=DEFAULT_SETTLE_ITERS,
        help=(
            "Consecutive on-target iterations (z and RX both within deadband) "
            "required before recording (default %(default)s)."
        ),
    )
    parser.add_argument(
        "--station-timeout-s",
        type=float,
        default=DEFAULT_STATION_TIMEOUT_S,
        help="Max time to spend settling a station before recording anyway (default %(default)s).",
    )


def _add_y_options(parser: argparse.ArgumentParser) -> None:
    # --- Y scanning (auto-detected extent; step + safety caps only) ---
    parser.add_argument(
        "--y-step-mm",
        type=float,
        default=DEFAULT_Y_STEP_MM,
        help=("Distance between contour-following Y stations (default %(default)s)."),
    )
    parser.add_argument(
        "--y-max-travel-mm",
        type=float,
        default=DEFAULT_Y_MAX_TRAVEL_MM,
        help=(
            "Safety cap: maximum Y travel in mm per outward march before giving up "
            "on finding an edge. Default "
            "%(default)s."
        ),
    )
    parser.add_argument(
        "--y-max-rows",
        type=int,
        default=DEFAULT_Y_MAX_ROWS,
        help=(
            "Safety cap: maximum number of y steps to probe per sweep (default "
            "%(default)s), measured from that sweep's start. Whichever of this and "
            "--y-max-travel-mm is hit first ends the sweep."
        ),
    )
    parser.add_argument(
        "--y-tolerance-mm",
        type=float,
        default=DEFAULT_Y_TOLERANCE_MM,
        help=(
            "Position tolerance in mm for a y row step to count as arrived (default %(default)s)."
        ),
    )
    parser.add_argument(
        "--y-move-timeout-s",
        type=float,
        default=DEFAULT_Y_MOVE_TIMEOUT_S,
        help="Max time to wait for a y row step to arrive in stream mode (default %(default)s).",
    )


def _add_edge_options(parser: argparse.ArgumentParser) -> None:
    # --- Edge detection / RX recovery ---
    parser.add_argument(
        "--edge-tilt-max-deg",
        type=float,
        default=DEFAULT_EDGE_TILT_MAX_DEG,
        help=(
            "Hard cap on the RX recovery tilt in degrees when the sensors drop out "
            "of range (must be <= 20). If both sensors do not come back in range "
            "within this tilt, the station is the body edge (default %(default)s)."
        ),
    )
    parser.add_argument(
        "--edge-tilt-step-deg",
        type=float,
        default=DEFAULT_EDGE_TILT_STEP_DEG,
        help=(
            "Increment in degrees for the RX recovery tilt ramp; smaller is finer "
            "but slower (default %(default)s)."
        ),
    )
    parser.add_argument(
        "--edge-tilt-sign",
        type=float,
        default=DEFAULT_EDGE_TILT_SIGN,
        help=(
            "Sign of the recovery tilt: the RX delta is edge-tilt-sign * phase_dir "
            "* tilt (phase_dir is +1 in the +y phase, -1 in the -y phase). The "
            "default (%(default)s) tilts the head back toward the start y -- back "
            "onto the body that is falling away at the edge. FLIP this if recovery "
            "tilts the wrong way on your rig (this rig's rx->y geometry is sign-"
            "dependent; see the --rx-gain-rad-per-mm note)."
        ),
    )
    parser.add_argument(
        "--edge-oor-iters",
        type=int,
        default=DEFAULT_EDGE_OOR_ITERS,
        help=(
            "Consecutive not-both-in-range reads before the edge probe fires; "
            "rejects single-sample dropouts (default %(default)s)."
        ),
    )
    parser.add_argument(
        "--floor-model",
        default=None,
        help=(
            "Path to a src/scripts/calibration/floor_depth_tare.py JSON "
            "(captures/floor_depth.json). Supplies the bed-rejection threshold; "
            "with an --rx-sweep fit inside, the threshold follows the LIVE rx "
            "tilt (evaluated at the last-read rx, clamped into the fitted range "
            "-- steeper-than-fitted tilts read the bed even deeper, so clamping "
            "errs toward rejection). Mutually exclusive with --floor-depth-mm; "
            "--floor-margin-mm still applies."
        ),
    )
    parser.add_argument(
        "--floor-depth-mm",
        type=float,
        default=DEFAULT_FLOOR_DEPTH_MM,
        help=(
            "Absolute bed/floor rejection (default: off). The bed is at a fixed "
            "machine height, so gantry z + sensor distance is ~constant for any "
            "reading OF THE BED and always larger than for skin (skin sits above "
            "the bed). An in-range reading with z + distance >= this minus "
            "--floor-margin-mm is reclassified out-of-range ('floor') before any "
            "controller sees it: the rx leveler can no longer normal itself to "
            "the bed when both spots land past a body edge, the standoff loop "
            "cannot descend onto it, and a probe tilt onto the bed is never a "
            "'recovery'. Measure once: park the beam on bare bed at a "
            "scanning-like tilt and add gantry z + reading (e.g. 378 + 127 -> "
            "505). Distance is along the beam, so re-measure if the working tilt "
            "changes a lot."
        ),
    )
    parser.add_argument(
        "--floor-margin-mm",
        type=float,
        default=DEFAULT_FLOOR_MARGIN_MM,
        help=(
            "How far above --floor-depth-mm still counts as the floor (default "
            "%(default)s). Keep it BELOW the body's thickness-above-bed at the "
            "edges you care about; a too-large margin classifies the last on-skin "
            "stations as floor and ends a sweep early (safe, just conservative)."
        ),
    )
    parser.add_argument(
        "--edge-recover-window-mm",
        type=float,
        default=DEFAULT_EDGE_RECOVER_WINDOW_MM,
        help=(
            "Edge-probe recovery acceptance window (default %(default)s): a tilt "
            "counts as a recovery only if both sensors are in range AND their "
            "average is within this of --target-mm. Stops the bed/floor under the "
            "body edge -- which can sit inside the sensor window -- from faking a "
            "recovery and pulling z down onto it. The probe holds z, so the bed is "
            "only distinguishable when the body is thicker along the beam than this "
            "window; tighten for thin parts, and pair with --z-min/max-mm -- a bed that "
            "enters the window at a later station's ENTRY bypasses the probe and is "
            "instead caught as a reach-limited (z) edge by the z travel window. Set "
            "very large to disable."
        ),
    )
    parser.add_argument(
        "--edge-settle-iters",
        type=int,
        default=DEFAULT_EDGE_SETTLE_ITERS,
        help=(
            "Max z-only iterations to settle the standoff after a successful "
            "recovery (holding the recovered RX tilt) before capturing (default "
            "%(default)s)."
        ),
    )


def _add_camera_options(parser: argparse.ArgumentParser) -> None:
    # --- Camera capture (Canon EOS R7 over USB / EDSDK) ---
    parser.add_argument(
        "--no-camera",
        dest="capture",
        action="store_false",
        help="Skip real capture; just hold --record-pause-s and print RECORD (simulate).",
    )
    parser.set_defaults(capture=True)
    parser.add_argument(
        "--edsdk-lib",
        default=None,
        help="Path to libEDSDK.so (default: auto-detect via the EDSDK search paths).",
    )
    parser.add_argument(
        "--capture-dir",
        default=DEFAULT_CAPTURE_DIR,
        help="Directory to save captured JPEGs (default %(default)s).",
    )
    parser.add_argument(
        "--capture-basename",
        default=DEFAULT_CAPTURE_BASENAME,
        help="Filename prefix for captured images (default %(default)s).",
    )
    parser.add_argument(
        "--capture-timeout-s",
        type=float,
        default=DEFAULT_CAPTURE_TIMEOUT_S,
        help="Max time to wait for an image to transfer from the camera (default %(default)s).",
    )
    parser.add_argument(
        "--capture-dwell-s",
        type=float,
        default=0.35,
        help=(
            "How long the head stays parked after the shutter fires: covers "
            "trigger latency + the exposure, NOT the camera's image "
            "processing (its file-ready event lands ~0.5s+ after the shutter, "
            "and the transfer runs under the next traverse regardless). "
            "(default %(default)s)"
        ),
    )


def _add_trace_options(parser: argparse.ArgumentParser) -> None:
    # --- Motion trace (post-hoc timing analysis) ---
    parser.add_argument(
        "--record",
        action="store_true",
        help=(
            "Record EVERY visited pose (x/y/z/rx) with a timestamp to a JSONL trace, "
            "in addition to the normal poses.jsonl / sidecar output, for offline "
            "analysis of how the scan spends its time. One line per position read and "
            "per motion command, each tagged with the current activity "
            "(traverse/settle/edge_recover/settle_z/capture/park) and "
            "the station/phase/col. Off by default; passive (no extra hardware I/O)."
        ),
    )
    parser.add_argument(
        "--record-path",
        default=None,
        help=(
            "Where to write the --record trace (default: <capture-dir>/trace.jsonl). "
            "Ignored unless --record is set."
        ),
    )


def _add_rx_options(parser: argparse.ArgumentParser) -> None:
    # --- RX (tilt) regulation ---
    parser.add_argument(
        "--rx-server-url",
        default=DEFAULT_RX_SERVER_URL,
        help=(
            "Base URL of the RX-axis server controlling the rx axis (default from "
            "RX_AXIS_SERVER_URL or %(default)s)."
        ),
    )
    parser.add_argument(
        "--rx-gain-rad-per-mm",
        type=float,
        default=DEFAULT_RX_GAIN_RAD_PER_MM,
        help=(
            "Proportional RX velocity gain: commanded rate = gain * filtered "
            "(sensor1 - sensor2). The reference rig requires a negative gain; "
            "flip the sign if RX diverges (default %(default)s)."
        ),
    )
    parser.add_argument(
        "--rx-axis-min-rad",
        type=float,
        default=DEFAULT_RX_AXIS_MIN_RAD,
        help=(
            "Lower bound of the rx AXIS hardware range in rad -- the CAN-controlled RX-axis motor's "
            "command window (default %(default)s). Commands are clamped to it, and "
            "an X position that needs more tilt to follow the surface is marked as "
            "a reach-limited (rx) edge instead of erroring on the server's reject."
        ),
    )
    parser.add_argument(
        "--rx-axis-max-rad",
        type=float,
        default=DEFAULT_RX_AXIS_MAX_RAD,
        help=(
            "Upper bound of the rx AXIS hardware range in rad (the RX-axis motor command "
            "window; default %(default)s). See --rx-axis-min-rad."
        ),
    )
    parser.add_argument(
        "--rx-deadband-mm",
        type=float,
        default=DEFAULT_RX_DEADBAND_MM,
        help="sensor1-sensor2 difference magnitude (mm) treated as level (default %(default)s).",
    )
    parser.add_argument(
        "--rx-filter-alpha",
        type=float,
        default=DEFAULT_RX_FILTER_ALPHA,
        help=(
            "EMA weight (0-1] on the RX error to smooth jitter: filtered = "
            "alpha*raw + (1-alpha)*prev. Lower is smoother but laggier; 1.0 "
            "disables filtering (default %(default)s)."
        ),
    )
    parser.add_argument(
        "--rx-speed-rad-s",
        type=float,
        default=DEFAULT_RX_SPEED_RAD_S,
        help=(
            "Speed in rad/s for RX moves (default %(default)s). With the larger "
            "default step size, slower speeds leave the RX-axis motor mid-move when the "
            "next iteration reads the sensors."
        ),
    )
    parser.add_argument(
        "--simultaneous-sensors",
        action="store_true",
        help=(
            "Keep both sensor lasers on between captures and read them without the "
            "exclusive enable-settle delay. This increases the control rate, but "
            "the spots may optically cross-bias the analog readings. Validate "
            "d1/d2 against exclusive sampling on your optical layout. "
            "Lasers are still switched off for every photo."
        ),
    )
    parser.add_argument(
        "--rx-velocity-ki-rad-s-per-mm-s",
        type=float,
        default=DEFAULT_RX_VELOCITY_KI_RAD_S_PER_MM_S,
        help=(
            "Velocity-servo INTEGRAL gain (SIGNED like the P gain; default "
            "%(default)s): a persisting tilt error winds the commanded rate up, "
            "riding through curved-skin treadmill crawls. 0 disables."
        ),
    )
    parser.add_argument(
        "--rx-velocity-min-rad-s",
        type=float,
        default=DEFAULT_RX_VELOCITY_MIN_RAD_S,
        help=(
            "Minimum commanded rate (magnitude) while the tilt error is outside "
            "the deadband (default %(default)s): the RX-axis motor's speed loop stalls "
            "below ~0.033 rad/s, so sub-floor commands freeze the axis with the "
            "error unresolved."
        ),
    )
    parser.add_argument(
        "--rx-velocity-slew-rad-s2",
        type=float,
        default=DEFAULT_RX_VELOCITY_SLEW_RAD_S2,
        help=(
            "Max rate-of-change of the commanded rx rate in velocity-servo mode "
            "(rad/s^2; 0 disables; default %(default)s). Softens rate reversals "
            "through the coupling backlash instead of slamming them."
        ),
    )
    parser.add_argument(
        "--rx-accel-rad-s2",
        type=float,
        default=DEFAULT_RX_ACCEL_RAD_S2,
        help=(
            "Max angular acceleration in rad/s^2 for RX moves (default: the RX-axis "
            "server's own default_accel_rad_s2). Lower it to ramp the tilt more "
            "gently -- less head jerk shaking the standoff/leveling reads mid-swing; "
            "raise it for snappier tilts."
        ),
    )
    parser.add_argument(
        "--rx-min-rad",
        type=float,
        default=DEFAULT_RX_MIN_RAD,
        help=(
            "Safety bound: minimum ABSOLUTE RX angle in radians. The regulator "
            "never commands RX below this; if the leveling controller wants to, it "
            "holds at the bound and lets the station settle there. Default: unset "
            "(rely on the RX-axis server's own soft limits)."
        ),
    )
    parser.add_argument(
        "--rx-max-rad",
        type=float,
        default=DEFAULT_RX_MAX_RAD,
        help=(
            "Safety bound: maximum ABSOLUTE RX angle in radians. The regulator "
            "never commands RX above this; if the leveling controller wants to, it "
            "holds at the bound and lets the station settle there. Default: unset "
            "(rely on the RX-axis server's own soft limits)."
        ),
    )
    parser.add_argument(
        "--rx-station-budget-rad",
        type=float,
        default=None,
        help=(
            "Limit each station's rx leveling to (the station's starting rx "
            "+/- this budget). On curved skin d1-d2 reflects LOCAL surface "
            "shape, not rig tilt, and each rx step swings the sensor spots "
            "across the skin via the ~460mm pivot arm -- unbounded leveling "
            "limit-cycles. With a budget the station settles at the bound when "
            "the surface wants more tilt. Suggested for skin: 0.02. Default: "
            "unset (flat-surface behaviour)."
        ),
    )


def _add_pivot_options(parser: argparse.ArgumentParser) -> None:
    # --- RX-pivot compensation (hold viewed x/y fixed while RX tilts) ---
    parser.add_argument(
        "--rx-pivot-model",
        default=DEFAULT_RX_PIVOT_MODEL,
        help=(
            "rx-pivot model JSON (from src/scripts/calibration/rx_pivot_fit.py) used to compensate "
            "RX moves and keep the viewed x/y point fixed (default %(default)s)."
        ),
    )
    parser.add_argument(
        "--traverse-max-jump-mm",
        type=float,
        default=DEFAULT_TRAVERSE_MAX_JUMP_MM,
        help=(
            "HARD cap on the lateral pivot jump a grid traverse may execute after "
            "a carried-rx swing (a failed edge probe moves rx without the gantry; "
            "the pivot compensation then converts the swing into a single lateral "
            "move -- observed at -116mm at bed z and +215mm over the subject "
            "even WITH the arc-z rise applied, because the focus orbit says "
            "nothing about the surface between the head and the target). A jump "
            "above the cap is refused; the sweep then RETREATS ALONG THE WALKED "
            "PATH (breadcrumb) to the last good pose and retries from there "
            "(default %(default)s)."
        ),
    )


def _add_z_options(parser: argparse.ArgumentParser) -> None:
    # --- z regulation ---
    parser.add_argument(
        "--gain-mm-per-mm",
        type=float,
        default=DEFAULT_GAIN_MM_PER_MM,
        help=(
            "Proportional gain: z step = gain * (reading - target), clamped to "
            "±max-step (default %(default)s)."
        ),
    )
    parser.add_argument(
        "--max-step-mm",
        type=float,
        default=DEFAULT_MAX_STEP_MM,
        help="Maximum per-iteration z move in mm while in range (default %(default)s).",
    )
    parser.add_argument(
        "--z-min-mm",
        type=float,
        default=DEFAULT_Z_MIN_MM,
        help=(
            "Lower bound of the usable z travel in mm. z commands are clamped to "
            "it, and a station whose standoff stays unmet while z is pinned here is "
            "marked as a reach-limited (z) edge; it is also an absolute z safety "
            "floor. Default: unset (no z-saturation detection, warned at startup)."
        ),
    )
    parser.add_argument(
        "--z-max-mm",
        type=float,
        default=DEFAULT_Z_MAX_MM,
        help=(
            "Upper bound of the usable z travel in mm (e.g. this rig's ~392mm z "
            "travel). z commands are clamped to it, and a station pinned here is "
            "marked as a reach-limited (z) edge. Default: unset (no z-saturation "
            "detection, warned at startup)."
        ),
    )
    parser.add_argument(
        "--z-stall-iters",
        type=int,
        default=DEFAULT_Z_STALL_ITERS,
        help=(
            "Consecutive iterations the standoff must stay unmet while z is pinned "
            "at the --z-min/max-mm travel window before the station is called a "
            "reach-limited (z) edge (default %(default)s)."
        ),
    )
    parser.add_argument(
        "--deadband-mm",
        type=float,
        default=DEFAULT_DEADBAND_MM,
        help="Average-error magnitude (mm) treated as on-target (default %(default)s).",
    )
    parser.add_argument(
        "--period-s",
        type=float,
        default=DEFAULT_PERIOD_S,
        help="Control loop period in seconds (default %(default)s).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="ADC samples to average per reading (default %(default)s).",
    )
    parser.add_argument(
        "--feed-mm-min",
        type=float,
        default=DEFAULT_FEED_MM_MIN,
        help=(
            "Feed rate in mm/min for the X stream worker and trajectory budgets "
            "(default: server default)."
        ),
    )
    parser.add_argument(
        "--report-interval-s",
        type=float,
        default=DEFAULT_REPORT_INTERVAL_S,
        help="How often to print the measured control loop frequency (default %(default)s).",
    )
    parser.add_argument(
        "--stream-tick-s",
        type=float,
        default=0.05,
        help="X motion worker tick period in seconds (default %(default)s).",
    )
    parser.add_argument(
        "--stream-min-step-mm",
        type=float,
        default=0.01,
        help="Minimum setpoint change that triggers a new stream move (default %(default)s).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-loop controller decisions (reading, error, z/rx step).",
    )
    return None


def build_parser() -> argparse.ArgumentParser:
    """Build the contour CLI from cohesive option groups."""
    parser = _build_base_parser()
    for add_options in (
        _add_x_options,
        _add_y_options,
        _add_edge_options,
        _add_camera_options,
        _add_trace_options,
        _add_rx_options,
        _add_pivot_options,
        _add_z_options,
    ):
        add_options(parser)
    return parser
