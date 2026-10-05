"""Tunable defaults and status constants for contour scanning."""

from __future__ import annotations

import os

from capture.config import DEFAULT_GANTRY_SERVER_URL

DEFAULT_SERVER_URL = os.getenv("GANTRY_SERVER_URL", DEFAULT_GANTRY_SERVER_URL)
DEFAULT_RX_SERVER_URL = os.getenv("RX_AXIS_SERVER_URL", "http://127.0.0.1:8091")
DEFAULT_TARGET_MM = 110.0
DEFAULT_GAIN_MM_PER_MM = 0.5
DEFAULT_MAX_STEP_MM = 2.0
DEFAULT_DEADBAND_MM = 0.5
DEFAULT_PERIOD_S = 0.1
DEFAULT_SAMPLES = 1
DEFAULT_FEED_MM_MIN: float | None = None
DEFAULT_REPORT_INTERVAL_S = 1.0

# RX regulation. On the reference rig, d(d1-d2)/d(rx) is positive, so the
# feedback gain must be negative.
DEFAULT_RX_GAIN_RAD_PER_MM = -0.015
DEFAULT_RX_DEADBAND_MM = 0.5
DEFAULT_RX_SPEED_RAD_S: float | None = 0.1
# Max angular acceleration (rad/s^2) for RX moves. None = the RX-axis server's own
# default_accel_rad_s2 (the value it uses when the request omits it); set it to
# cap how hard the tilt ramps (a gentler accel reduces the head jerk that shakes
# the standoff/leveling readings mid-swing).
DEFAULT_RX_ACCEL_RAD_S2: float | None = None
# Default max speed (mm/s) for a Pico-backed Y or Z axis (applied as the firmware
# VMAX at startup; tune live nowhere -- this is the scan's fixed setting).
DEFAULT_PICO_VMAX_MM_S = 35.0
DEFAULT_RX_FILTER_ALPHA = 0.3
# RX uses a continuous velocity servo. The loop streams a signed angular
# velocity = --rx-gain-rad-per-mm * filtered d1-d2, clamped to
# --rx-speed-rad-s and zero inside the deadband.
# Velocity-servo INTEGRAL gain (rad/s per mm*s, signed like the P gain) and
# its dedicated error filter. The integral rides through the curved-skin
# treadmill: on the knee the filtered error saturates ~1mm while the local
# normal recedes ahead of the leveling motion, so a P-only rate can stick at
# gain*1mm (~0.06 rad/s); a persisting
# error instead winds the rate up until the error actually closes. Anti-
# windup: clamped to the speed limit, zeroed at any bound, bled inside the
# deadband. The filter is LIGHTER than the position-mode --rx-filter-alpha:
# the motor integrates the rate command, so noise becomes rate ripple rather
# than position jumps -- and the 0.4 alpha's 1.5-sample lag (~135ms) was most
# of the loop delay that can cause entry ringing.
DEFAULT_RX_VELOCITY_KI_RAD_S_PER_MM_S = -0.06
# Integrator anti-windup: the integral LEAKS toward zero at this rate (per
# second), is clamped to this fraction of the speed limit, and freezes while
# the total command is saturated. A pure integrator winds to full speed during
# a long treadmill chase and unwinds through a huge overshoot after the zero
# crossing (sim: a 0.5+ rad excursion past the normal); the leak bounds its
# steady-state authority to ki*err/leak (~2.5x the P rate at 1mm error) and
# unwinds it in ~2s regardless of the error history.
RX_VELOCITY_INTEGRATOR_LEAK_PER_S = 0.5
RX_VELOCITY_INTEGRATOR_CLAMP_FRACTION = 0.5
# Minimum commanded rate while OUTSIDE the deadband. The AK45's speed loop has
# a real stall floor: the measured moving-speed distribution bottoms out at
# 0.033 rad/s (~160 ERPM), with nothing sustained below it. Sub-floor commands
# do not turn the motor at all. A P-only controller can freeze until the station
# timeout whenever the
# error sat under ~1.1mm because every command it produced was sub-floor.
# 0.07 is ~2x the stall floor; the overshoot cost of deciding at this rate is
# ~0.3mm per iteration, inside the deadband width. Inside the deadband the
# command is a hard zero (sub-floor "bleed" rates were physically fictional).
DEFAULT_RX_VELOCITY_MIN_RAD_S = 0.07
# Rate-command slew limit (rad/s^2; 0 disables). An instant rate flip slams
# the ~5deg coupling backlash (the residual velocity-mode roughness); ramping
# the COMMAND at a bounded rate turns filter/noise flips into gentle
# decel-stop-reverse profiles. 2.0 reaches the 0.5 rad/s cruise in 0.25s, so
# transitions barely slow while reversals soften.
DEFAULT_RX_VELOCITY_SLEW_RAD_S2 = 2.0
# Deadband hysteresis for the velocity servo: once at rest inside the
# deadband, motion re-arms only when the filtered error exceeds deadband x
# this factor. One floor-speed decision moves ~0.25-0.4 deg (coarser than the
# deadband at the exclusive-read loop rate), so acting AT the deadband edge
# bang-bangs in and out of it every iteration. Size the threshold above the
# coupling backlash's error authority: the camera can sit anywhere within
# ~+/-2.5 deg of the rotor at rest, which at the 30-45mm/rad plant slope
# fakes up to ~1.5mm of d1-d2 drift with no commanded motion -- a re-arm
# threshold below that lets the slack itself re-arm the servo. 2.2 x the
# 0.7mm operating deadband = 1.54mm, just over the fake-drift ceiling.
RX_VELOCITY_REARM_FACTOR = 2.2
# Derating margin on the pivot-follow rate budget (see _pivot_rate_budget):
# the gantry should cruise the arc at <= this fraction of an axis's vmax so
# accel phases and dispatch latency don't accumulate lag.
PIVOT_FOLLOW_MARGIN = 0.8
# Collision gate on the rx VELOCITY servo (rx_vel_collision_blocked): probe the
# pivot-arc pose this many seconds of rotation ahead of the commanded rate
# (capped) before refreshing set_velocity. set_velocity bypasses
# move_regulated_pose's pre-dispatch collision check, so without the gate the
# RX-axis motor keeps tilting while the gantry's arc-follow dispatches are being
# refused -- the head ends stranded in a sub-margin pose that no dispatch (not
# even a hold-in-place) can leave. The
# lookahead must cover the rate-slew decel distance (~w/slew seconds).
RX_VEL_GUARD_LOOKAHEAD_S = 0.3
RX_VEL_GUARD_MAX_LOOK_RAD = 0.12
# Oscillation-adaptive gain reduction (velocity servo). On curved skin the
# d1-d2 plant slope inflates several-fold (each rx mrad also translates the
# spots ~0.46mm across the surface), so the fixed loop gain crosses its
# stability margin and the servo limit-cycles at full amplitude until the
# station timeout. The swing passes straight through the d1-d2 null twice per
# cycle -- it overshoots because the crossing speed times the loop lag
# (filter + dispatch + ~5deg coupling backlash) exceeds the deadband. Detect
# REPEAT REVERSALS of the filtered error's sign (deadband-hysteresis-gated,
# so noise inside the band never counts) and HALVE the gain and rate cap;
# each halving slows the crossing until the deadband disarm can catch it at
# the null. A cycle that survives the minimum scale freezes leveling for the
# station and settles best-effort -- a capture a few degrees off-normal beats
# burning the 20s timeout.
RX_OSC_REVERSALS_FIRST_CUT = 2  # error sign reversals before the first halving
RX_OSC_REVERSALS_PER_CUT = 1  # further reversals per additional halving
RX_OSC_MIN_GAIN_SCALE = 0.125  # floor: three halvings
RX_OSC_FREEZE_REVERSALS = 10  # still cycling at the floor: freeze leveling
# Backswing fast-track: reversal-paced cuts give slow oscillations prompt relief.
# Once one reversal has happened, a swing
# that grows back past this many deadbands is confirmed oscillation NOW --
# count it as a reversal immediately (latched once per half-swing) instead of
# waiting for it to cross zero again.
RX_OSC_BACKSWING_DEADBANDS = 3.0
# The cut scale carries across stations, recovering toward 1.0 by this factor
# per station. An unstable patch is a region, so the next station starts
# near the scale its neighbour learned; quiet stations double back to full
# gain within 1-3 stations.
RX_OSC_RECOVERY_FACTOR = 2.0
# End-of-scan park: raise the head this far off the surface (z DECREASES away
# from the body on this rig) and set this neutral tilt -- IN PLACE, instead of
# driving back across the body to the last capture pose. The tilt matches the
# rx homing final position, so the next scan/homing starts from a familiar
# attitude. The rx swing fires only after the FULL rise landed.
PARK_RISE_MM = 50.0
PARK_RX_RAD = 0.95
# Absolute RX tilt safety bounds (rad). None = unset (rely on the RX-axis server's
# own soft limits); when set, the regulator never commands RX outside the window.
DEFAULT_RX_MIN_RAD: float | None = None
DEFAULT_RX_MAX_RAD: float | None = None

# The RX-axis server's hard command window (rx_axis_server
# min/max_command_position_rad): it 400-rejects rx targets outside this. The
# contour scan clamps every RX command to it and treats an X position that needs
# more tilt as a body edge, so the scan continues instead of aborting on the
# server's 400 response.
# These mirror the server window exactly [0.07, 1.76] rad so the script floor
# isn't below the server's (which would make the bottom of the range
# unreachable).
DEFAULT_RX_AXIS_MIN_RAD = 0.07
DEFAULT_RX_AXIS_MAX_RAD = 1.76

# RX-pivot compensation (hold the viewed x/y point fixed while RX tilts).
DEFAULT_RX_PIVOT_MODEL = "captures/rx_pivot_model.json"
# Cap on the lateral pivot jump a grid traverse may run when its matching arc-z
# rise could not be applied (see --traverse-max-jump-mm). Sized above the offset
# deltas of normal carried-rx drift (a few mm/station) and below an unsafe
# cross-surface swing.
DEFAULT_TRAVERSE_MAX_JUMP_MM = 50.0
# Accepted sensor readings whose z+d falls within this band UNDER the floor-reject
# line are counted as FLOOR-SUSPECT (likely the bed leaking through a static
# --floor-depth-mm threshold whose true apparent depth moved with the tilt).
FLOOR_SUSPECT_BAND_MM = 5.0
# A traverse only rises for arc-z debts beyond this (mm): real probe swings owe
# tens of mm, while small commanded-vs-measured RX jitter is absorbed by the
# station's standoff loop.
TRAVERSE_ARC_RISE_MIN_MM = 2.0
# --- band-transition walk ---
# Consecutive hops the walk may take with the sensors dark (surface out of
# range). One fast-axis step past the last in-range station lets the next
# station's edge probe judge the surface. Beyond that the head would translate blind;
# the walk retreats along the breadcrumb trail and refuses instead. Rising
# away from the surface is never an option: dark sensors are un-actionable
# (no controller in this codebase approaches on below_range), so a head
# parked out of range deadlocks the scan.
BAND_WALK_MAX_DARK_HOPS = 1
# Leveled-tilt REGIMES that mark the two y flanks for the miss-fraction march
# stop (--band-miss-stop-frac). A band's misses only end a march when the
# band's own CAPTURES leveled into the flank regime the march walks toward:
# descend (-y) stops when some capture leveled at rx >= HI (the camera has
# wrapped toward the -y flank, ~1.5-1.76 on this rig); ascend (+y) when some
# capture leveled at rx <= LO (~0.07-0.45). Mid-range leveled tilts mean the
# misses are along-X dropouts (the end of the subject under the band, sensor dropouts),
# NOT the y flank. Otherwise, a run of along-x dropouts can incorrectly stop
# the y march before it reaches the flank.
DEFAULT_BAND_MISS_STOP_RX_HI = 1.45
DEFAULT_BAND_MISS_STOP_RX_LO = 0.45
# Default focus-row cruise speed for the band-transition walk (mm/s). The
# walk advances a continuously-moving target by speed*dt each regulation
# iteration and dispatches without waiting, so the
# axes flow through the transition as one motion instead of stop-at-target
# hops with sensor pauses between them. Sized so the per-iteration advance
# (~1-2.5mm at the 8-15Hz
# loop) stays within the z servo's --max-step-mm tracking authority on
# steep flanks and above PICO_CRUISE_MIN_STEP_MM so streamed batches cruise.
DEFAULT_BAND_WALK_SPEED_MM_S = 20.0
# Minimum per-axis setpoint delta (mm) for a streamed Pico batch to CRUISE
# (MOVEC / MOVEM C) instead of stop-at-target -- see _cruise_batch. At the
# Pico accel used for scans (>=100mm/s2) a move this size is still in flight
# one regulation period (~90-150ms) later, so a fresh waypoint always arrives
# before the cruise target is reached (no firmware EVT UNDERRUN hard-stop).
PICO_CRUISE_MIN_STEP_MM = 1.0
# Breadcrumb trail: the scan records the pose path it walks and, on an edge or
# floor outcome, retreats along that path in reverse to the last captured
# pose. The walked path tracked the contour under sensor guard, so replaying it
# is collision-safe by construction; any computed shortcut (straight line, orbit
# chord) can cross the subject. Waypoints are decimated to this spacing; the buffer
# halves its resolution when full (keeping the seed and the newest point exact).
BREADCRUMB_SPACING_MM = 4.0
BREADCRUMB_SPACING_RAD = 0.02
BREADCRUMB_MAX = 800
# X scanning.
DEFAULT_X_STEP_MM = 10.0
DEFAULT_X_TRAVEL_MM = 100.0
DEFAULT_X_TOLERANCE_MM = 0.5
DEFAULT_X_MOVE_TIMEOUT_S = 30.0
DEFAULT_RECORD_PAUSE_S = 1.0
DEFAULT_SETTLE_ITERS = 3
DEFAULT_STATION_TIMEOUT_S = 20.0

# Y scanning. The safety bounds cap how far the scan can probe before giving
# up on an edge.
DEFAULT_Y_STEP_MM = 10.0
DEFAULT_Y_TOLERANCE_MM = 0.5
DEFAULT_Y_MOVE_TIMEOUT_S = 30.0
# Hard caps per sweep so a path whose sensors never drop out of range (e.g. an
# object that extends past the body, or a stuck in-range read) cannot march y
# forever. Whichever is hit first ends the sweep (measured from the sweep start).
DEFAULT_Y_MAX_TRAVEL_MM = 200.0
DEFAULT_Y_MAX_ROWS = 60
# Edge detection / RX recovery. When the sensors drop out of range the station
# tilts RX in fixed increments (--edge-tilt-step-deg) up to a hard cap
# (--edge-tilt-max-deg, 20 deg) trying to bring BOTH back in range. The tilt is
# applied toward the body -- back toward the start y -- via edge_tilt_sign *
# phase_dir; flip --edge-tilt-sign if the recovery tilts the wrong way on this
# rig (see the --rx-gain sign note). --edge-oor-iters consecutive
# not-both-in-range reads trigger the probe (rejects single-sample dropouts).
DEFAULT_EDGE_TILT_MAX_DEG = 20.0
DEFAULT_EDGE_TILT_STEP_DEG = 2.0
DEFAULT_EDGE_TILT_SIGN = -1.0
DEFAULT_EDGE_OOR_ITERS = 3
# Recovery acceptance window (mm). The edge probe accepts a tilt as a RECOVERY
# only when both sensors are in range AND their average is within this of
# --target-mm. Bare in-range is NOT recovery: the bed/floor under the body edge
# can sit inside the HG-C window too. A floor return can otherwise be mistaken
# for a recovery and drive the standoff loop toward the floor. The surface the
# station lost was at ~target, so a genuine recovery reads near it; the bed reads
# deeper. Set very large to use in-range-only acceptance. NOTE: the
# probe holds z, so the bed is only distinguishable when the body is thicker
# (along the beam) than this window -- tighten it for thin parts (ankle).
DEFAULT_EDGE_RECOVER_WINDOW_MM = 15.0
# Absolute floor/bed rejection (--floor-depth-mm; None = off). The bed is at a
# FIXED machine height, so gantry z + sensor distance is ~constant for any reading
# OF THE BED (as z descends the reading shrinks equally -- the sum cannot be
# fooled), while skin always sits ABOVE the bed (smaller z+d). Readings with
# z + d >= floor_depth - floor_margin are reclassified signal_status='floor' /
# out-of-range at the read chokepoint, BEFORE any controller sees them. So: one
# spot on the bed -> not-both-in-range -> the edge probe runs; BOTH spots on the
# bed -> same, and the rx leveler can never normal itself to the bed; a probe tilt that lands on the
# bed reads 'floor', never 'recovered'. Measure once: park the beam on bare bed at
# a scanning-like tilt and add gantry z + reading.
# The distance is measured along the BEAM, so the bed's z+d drifts if the working
# tilt moves far from the tare tilt -- keep the margin below the body's
# thickness-above-bed at the edges, but generous.
DEFAULT_FLOOR_DEPTH_MM: float | None = None
DEFAULT_FLOOR_MARGIN_MM = 10.0
# Brief z-only settle after a successful recovery so the in-range sensors are
# near target before the shutter, without re-leveling RX (which would undo the
# deliberate recovery tilt). Bounded; never blind-searches z.
DEFAULT_EDGE_SETTLE_ITERS = 20

# Z-travel saturation -> reach-limited edge. On a body flank the surface recedes
# faster than the z axis can follow; once z is pinned at its travel limit it can no
# longer reach the standoff, so the camera can't be placed normal to the skin at the
# target distance -- that y is past the scannable edge. --z-min/max-mm are the usable
# z travel window: commands are clamped to it, and a station whose standoff stays
# unmet while z is pinned at the window (for --z-stall-iters reads) is a z-limit
# edge. Without the window, z saturation is NOT detected (warned at startup) -- the
# body edge is then caught by the floor / rx-axis edges instead. (Clamp-only is the
# only breathing/arc/glitch-robust signal; an error-trend heuristic false-fires on a
# breathing live surface.)
DEFAULT_Z_MIN_MM: float | None = None
DEFAULT_Z_MAX_MM: float | None = None
DEFAULT_Z_STALL_ITERS = 8

# Per-station regulation outcomes (returned by regulate_station).
STATION_SETTLED = "settled"  # both controllers on target -> capture
STATION_RECOVERED = (
    "recovered"  # went out of range, an RX tilt <=20 deg recovered it -> capture at the tilt
)
STATION_EDGE = "edge"  # out of range even at 20 deg of tilt: the body edge -> do not capture
STATION_TIMEOUT = "timeout"  # did not settle but stayed in range -> capture anyway
STATION_Z_LIMIT = "z-limit"  # z saturated at its travel limit -> can't reach standoff; reach-limited edge, do NOT capture
STATION_RX_LIMIT = "rx-limit"  # rx hit its axis range trying to follow the surface normal -> reach-limited edge, do NOT capture
STATION_Y_LIMIT = "y-limit"  # pivot-compensated gantry y would leave the soft travel limit -> reach-limited edge, do NOT capture (NOT an abort)

# Camera capture (Canon EOS R7 over USB via EDSDK).
DEFAULT_CAPTURE_DIR = "captures"
DEFAULT_CAPTURE_BASENAME = "eos-r7"
DEFAULT_CAPTURE_TIMEOUT_S = 15.0


__all__ = [
    "DEFAULT_SERVER_URL",
    "DEFAULT_RX_SERVER_URL",
    "DEFAULT_TARGET_MM",
    "DEFAULT_GAIN_MM_PER_MM",
    "DEFAULT_MAX_STEP_MM",
    "DEFAULT_DEADBAND_MM",
    "DEFAULT_PERIOD_S",
    "DEFAULT_SAMPLES",
    "DEFAULT_FEED_MM_MIN",
    "DEFAULT_REPORT_INTERVAL_S",
    "DEFAULT_RX_GAIN_RAD_PER_MM",
    "DEFAULT_RX_DEADBAND_MM",
    "DEFAULT_RX_SPEED_RAD_S",
    "DEFAULT_RX_ACCEL_RAD_S2",
    "DEFAULT_PICO_VMAX_MM_S",
    "DEFAULT_RX_FILTER_ALPHA",
    "DEFAULT_RX_VELOCITY_KI_RAD_S_PER_MM_S",
    "RX_VELOCITY_INTEGRATOR_LEAK_PER_S",
    "RX_VELOCITY_INTEGRATOR_CLAMP_FRACTION",
    "DEFAULT_RX_VELOCITY_MIN_RAD_S",
    "DEFAULT_RX_VELOCITY_SLEW_RAD_S2",
    "RX_VELOCITY_REARM_FACTOR",
    "PIVOT_FOLLOW_MARGIN",
    "RX_VEL_GUARD_LOOKAHEAD_S",
    "RX_VEL_GUARD_MAX_LOOK_RAD",
    "RX_OSC_REVERSALS_FIRST_CUT",
    "RX_OSC_REVERSALS_PER_CUT",
    "RX_OSC_MIN_GAIN_SCALE",
    "RX_OSC_FREEZE_REVERSALS",
    "RX_OSC_BACKSWING_DEADBANDS",
    "RX_OSC_RECOVERY_FACTOR",
    "PARK_RISE_MM",
    "PARK_RX_RAD",
    "DEFAULT_RX_MIN_RAD",
    "DEFAULT_RX_MAX_RAD",
    "DEFAULT_RX_AXIS_MIN_RAD",
    "DEFAULT_RX_AXIS_MAX_RAD",
    "DEFAULT_RX_PIVOT_MODEL",
    "DEFAULT_TRAVERSE_MAX_JUMP_MM",
    "FLOOR_SUSPECT_BAND_MM",
    "TRAVERSE_ARC_RISE_MIN_MM",
    "BAND_WALK_MAX_DARK_HOPS",
    "DEFAULT_BAND_MISS_STOP_RX_HI",
    "DEFAULT_BAND_MISS_STOP_RX_LO",
    "DEFAULT_BAND_WALK_SPEED_MM_S",
    "PICO_CRUISE_MIN_STEP_MM",
    "BREADCRUMB_SPACING_MM",
    "BREADCRUMB_SPACING_RAD",
    "BREADCRUMB_MAX",
    "DEFAULT_X_STEP_MM",
    "DEFAULT_X_TRAVEL_MM",
    "DEFAULT_X_TOLERANCE_MM",
    "DEFAULT_X_MOVE_TIMEOUT_S",
    "DEFAULT_RECORD_PAUSE_S",
    "DEFAULT_SETTLE_ITERS",
    "DEFAULT_STATION_TIMEOUT_S",
    "DEFAULT_Y_STEP_MM",
    "DEFAULT_Y_TOLERANCE_MM",
    "DEFAULT_Y_MOVE_TIMEOUT_S",
    "DEFAULT_Y_MAX_TRAVEL_MM",
    "DEFAULT_Y_MAX_ROWS",
    "DEFAULT_EDGE_TILT_MAX_DEG",
    "DEFAULT_EDGE_TILT_STEP_DEG",
    "DEFAULT_EDGE_TILT_SIGN",
    "DEFAULT_EDGE_OOR_ITERS",
    "DEFAULT_EDGE_RECOVER_WINDOW_MM",
    "DEFAULT_FLOOR_DEPTH_MM",
    "DEFAULT_FLOOR_MARGIN_MM",
    "DEFAULT_EDGE_SETTLE_ITERS",
    "DEFAULT_Z_MIN_MM",
    "DEFAULT_Z_MAX_MM",
    "DEFAULT_Z_STALL_ITERS",
    "STATION_SETTLED",
    "STATION_RECOVERED",
    "STATION_EDGE",
    "STATION_TIMEOUT",
    "STATION_Z_LIMIT",
    "STATION_RX_LIMIT",
    "STATION_Y_LIMIT",
    "DEFAULT_CAPTURE_DIR",
    "DEFAULT_CAPTURE_BASENAME",
    "DEFAULT_CAPTURE_TIMEOUT_S",
]
