# Self-collision guard — design

Config-dependent self-collision avoidance for the 4-DOF gantry (x, y, z, rx). A safe `(x, y)` envelope that varies with `(rx, z)` is derived from the CAD model, allowing the machine to use the portion of its travel that is collision-free.

## Pipeline (offline → runtime)

1. **CAD → URDF** (`config/cad/robot.urdf`, generated with `onshape-to-robot`).
2. **Collision engine** (`src/scripts/collision/collision_model.py`; install with `pip install -e ".[collision]"`): clearance between every moving link and the static frame or other non-adjacent links, using the real→CAD calibration (`config/cad/frame_calibration.json`). ACM = adjacent (carriage-on-rail) + structural overlaps from known-safe poses.
3. **Envelope sweep** (`src/scripts/collision/build_collision_envelope.py`): raw min-clearance grid over `(rx, z, x, y)` in real controller units → `config/cad/collision_clearance.npz`. `derive` re-thresholds it at a margin + rx-backlash (rx interpolated fine) → safe masks + robust-clearance grid in `config/cad/collision_envelope.npz`. Current: **20 mm margin, ±2.5°**.
4. **Runtime guard** (`src/openderm/motion/collision_guard.py`, numpy-only, control host): loads the envelope, answers `is_safe / min_clearance / check_pose / check_path`. **Never calls FCL at runtime** — pure table lookup.



## The architectural constraint

X (Klipper/Moonraker through the gantry server on port 8090), Y/Z (Pico through the serial bridge on port 8095), and RX (RX-axis server on port 8091) are **separate controllers with no shared state or atomic full-pose command**. The complete `(x, y, z, rx)` target exists **in the scan control loop**, where `move_regulated_pose()` computes the x/y pivot arc, finalizes z, and holds `rx_target` before dispatch. The **primary, authoritative check lives in the loop**, using the *commanded target* pose.

## Decisions

- **On violation: reject discrete moves, hold streamed moves.** A discrete `move_xyz` that would enter an unsafe pose raises `GantryCollisionError`; a streamed regulation setpoint that would cross the boundary holds at the last safe setpoint (does not advance).
- The scan controller enforces the complete four-axis pose. Manual motion must still be performed conservatively because separate axis services do not share an atomic four-axis state.
- The collision envelope governs frame clearance. The physical axis travel limits enforced by the X server and Pico firmware remain independent safety boundaries.



## Runtime guard API (`collision_guard.py`, numpy-only)

- `min_clearance(x, y, z, rx) -> float` — **conservative** lookup: min robust-clearance over the grid hypercube surrounding the query (fail-safe against the coarse grid); queries outside the swept ranges return `-inf` (fail closed). rx is already ±2.5° backlash-padded in the envelope, so the commanded rx is used directly.
- `is_safe(x, y, z, rx) -> bool` — `min_clearance >= margin`.
- `check_pose(x, y, z, rx)` — raises `GantryCollisionError` if unsafe (discrete-move use).
- `check_path(a, b, n)` — samples n interpolated poses A→B; also samples the **rx-leads-xyz** transient corner (rx is dispatched fire-and-forget before x/y/z), so a path whose endpoints are safe but whose transit isn't is rejected.
- Margin is a **runtime config** applied to the robust-clearance grid at load, so it is retunable without regenerating the envelope. Regenerate and independently validate the envelope after any CAD, calibration, or mechanical change.



## Enforcement

In `openderm.scanning.contour`:

- **Startup gate**: perform a full-pose safety check on the current parked pose; abort the scan if already unsafe.
- **Per-move**: call `check_pose(x, y, z, rx_target)` after assembling the full target and before dispatch. On `GantryCollisionError`, skip the move and flag the station edge as collision-limited.
- **Path**: `check_path(previous_commanded_pose, target)` so transit is safe too.
- **Streamed mode**: hold the setpoint at the last safe pose instead of advancing.



## Rollout / validation

**Enforced by default.** `OPENDERM_COLLISION_MODE=off` only *disables* the guard (e.g. bench work without hardware) — there is deliberately **no "shadow"/log-only mode**: a safety limit that prints but doesn't stop gives a false sense of protection and cannot be relied upon. Commission the physical limits and collision envelope with an independent engineering safety procedure before human use. Margin is retunable via `build_collision_envelope.py derive` without a new sweep. An over-conservative margin may stop a scan early; do not reduce it solely because the software model reports clearance.

## Caveats

- Use **commanded** targets, not live `/state` reads, for the loop check (gantry `/state` is extrapolated mid-move).
- rx readback is firmware-belief, not true camera angle (backlash downstream of the encoder). The ±2.5° envelope padding + commanded rx cover this; widen if backlash grows.
- Don't mask silent move failures (blocking `/move` can return 200 with `status='failed'`).
- Envelope fidelity == CAD + calibration fidelity; regenerate on any CAD/calibration change.

