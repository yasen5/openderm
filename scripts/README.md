# Calibration and collision scripts

Routine operation uses the installed `openderm-*` commands documented in the root README. The scripts retained here create rig-specific calibration files and validate the collision envelope.

## Hardware calibration

- `rx_pivot_capture.py` records same-point poses across the RX scan range.
- `rx_pivot_fit.py` fits those poses into the RX rotation model used by `openderm-scan`.
- `floor_depth_tare.py` measures the bed rejection threshold.

These calibration files are physical measurements, not universal defaults. Recreate them after changes to the sensor mount or lever arm. Camera intrinsics must be calibrated independently as described in the root README.

## Collision tools

- Install the offline collision stack with `pip install -e ".[collision]"`.
- `collision_model.py` evaluates CAD clearances with FCL.
- `build_collision_envelope.py` generates the runtime collision envelope from `cad/robot.urdf`.

Run every calibration on an inert target and at reduced speed before a person enters the workspace. Treat a generated collision envelope as unvalidated until the physical machine has completed an independent engineering safety review.
