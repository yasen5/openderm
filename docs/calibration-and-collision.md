# Calibration and collision scripts

Routine operation uses the installed `openderm-*` commands documented in the root README. Utilities under `src/scripts/` create rig-specific calibration files and validate the collision envelope.

All utility arguments are required run-specific inputs: angles, modes, input files, and output locations. Persistent settings are loaded from [`config/scripts.json`](../config/scripts.json), resolved relative to the checkout regardless of the working directory. There are no CLI defaults or tuning overrides. Missing, unknown, or invalid settings fail before hardware connections or collision generation start.

Configure the file once for the rig:

- `calibration`: gantry/RX URLs, shared Pico connection, standoff, Z regulation, ADC averaging, and Z feed rate. These settings apply to both capture and floor tare. Connection values in this file replace the scripts' former URL/port environment defaults; the control token still comes from `OPENDERM_CONTROL_TOKEN`.
- `rx_pivot_capture`: angle units (`degrees: false` means radians), initial jog size, display cadence, axis motion settings, and startup homing policy. Jog size can still be adjusted interactively with `[` and `]`.
- `floor_depth_tare`: startup homing policy, bounded search travel/time, settling and measurement counts, RX sweep motion settings, and debugging. `max_travel_mm` limits the calibration search from its starting position; controller travel limits remain enforced independently.
- `collision`: margin, measured RX backlash, worker count, and grid resolution. `workers: null` selects the worker count from available CPUs. Travel ranges come from `config/cad/frame_calibration.json`; keep Z resolution dense enough for the runtime guard's conservative lookup (the reference grid uses 30 points).

`null` for a motion setting delegates to the existing controller setting. The supplied homing settings are `false`, so axes must already be homed unless you change that policy. The former `--tolerance-mm` option was removed because the Pico Z adapter did not use it.

## Hardware calibration

Run these tools from `src/scripts/calibration/`.

- `rx_pivot_capture.py` records same-point poses across the RX scan range.
- `rx_pivot_fit.py` fits those poses into the RX rotation model used by `openderm-scan`.
- `floor_depth_tare.py` measures the bed rejection threshold.

From the repository root:

```bash
python src/scripts/calibration/rx_pivot_capture.py 0.95 \
    --record-file captures/rx_pivot_poses.jsonl
python src/scripts/calibration/rx_pivot_fit.py captures/rx_pivot_poses.jsonl \
    --out captures/rx_pivot_model.json
python src/scripts/calibration/floor_depth_tare.py point \
    --out captures/floor_depth.json
python src/scripts/calibration/floor_depth_tare.py sweep 0.35:1.5:6 \
    --out captures/floor_depth_sweep.json
```

Capture appends to the explicitly selected record file; fit and tare write to the explicitly selected output file. Floor tare's mode must be selected: `point` records the current tilt without moving RX, while `sweep LO:HI:N` samples N angles in radians and fits the RX dependence. Standoff comes from `calibration.target_mm` for both utilities.

These calibration files are physical measurements, not universal defaults. Recreate them after changes to the sensor mount or lever arm. Camera intrinsics must be calibrated independently as described in the root README.

## Collision tools

Run these tools from `src/scripts/collision/`.

- Install the offline collision stack with `pip install -e ".[collision]"`.
- `collision_model.py` evaluates CAD clearances with FCL.
- `build_collision_envelope.py` generates the runtime collision envelope from `config/cad/robot.urdf`.

```bash
python src/scripts/collision/build_collision_envelope.py sweep \
    --grid config/cad/collision_clearance.npz --out config/cad/collision_envelope.npz
python src/scripts/collision/build_collision_envelope.py derive \
    --grid config/cad/collision_clearance.npz --out config/cad/collision_envelope.npz
```

`sweep` writes the raw grid and derives the envelope. After changing margin or backlash in `config/scripts.json`, `derive` reads the selected grid and writes the selected envelope without repeating the sweep. Both paths are required and must be different files; use `.npz` filenames.

Run every calibration on an inert target and at reduced speed before a person enters the workspace. Treat a generated collision envelope as unvalidated until the physical machine has completed an independent engineering safety review.
