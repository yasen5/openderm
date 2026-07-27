#!/usr/bin/env python3
"""Contour-following total-body photography.

The scanner sweeps X at each focus-Y position, then walks toward both Y flanks.
At every station it regulates camera standoff with Z, levels the camera with RX,
applies RX-pivot compensation, enforces floor and collision limits, and captures
an image.

X is controlled through ``openderm-gantry-server``. Y and Z are controlled by
the Pico bridge, and RX is controlled by the RX-axis server. All four axes
must be homed before a scan. The two distance sensors are sampled sequentially
to prevent optical interference.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from openderm.motion.gantry.server import (
    GantryServerClient,
    GantryServerError,
)
from openderm.motion.rx_axis.server import (
    RxAxisServerClient,
    RxAxisServerError,
)
from openderm.motion.collision_guard import CollisionGuard
from openderm.motion.rx_pivot import RxPivotModel
from openderm.sensors.hg_c import build_sensor_controller
from openderm.camera.canon import (
    CanonCamera,
    CanonCaptureConfig,
    CanonEdsdk,
    CanonError,
)

# Keep the historical constants and math helpers importable from this module;
# downstream calibration scripts and tests use ``contour`` as their facade.
from .config import *  # noqa: F401,F403
from .helpers import (
    _cruise_batch,  # noqa: F401
    _edge_side_from_rx,  # noqa: F401
    _load_floor_model,
    _pivot_rate_budget,  # noqa: F401
    _read_startup_collision_pose,  # noqa: F401
)
from .advanced_cli import build_parser
from ._contour import execution as contour_execution
from ._contour import runtime as contour_runtime


def run(args: argparse.Namespace) -> int:
    if not 0.0 < args.rx_filter_alpha <= 1.0:
        print("--rx-filter-alpha must be in (0, 1].", file=sys.stderr)
        return 1
    if args.x_step_mm <= 0:
        print("--x-step-mm must be positive.", file=sys.stderr)
        return 1
    if args.x_travel_mm < 0:
        print("--x-travel-mm must be non-negative.", file=sys.stderr)
        return 1
    if args.y_step_mm <= 0:
        print("--y-step-mm must be positive.", file=sys.stderr)
        return 1
    if args.y_max_travel_mm <= 0:
        print("--y-max-travel-mm must be positive.", file=sys.stderr)
        return 1
    if args.y_max_rows < 1:
        print("--y-max-rows must be at least 1.", file=sys.stderr)
        return 1
    if args.band_miss_stop_frac is not None and not 0.0 < args.band_miss_stop_frac <= 1.0:
        # <=0 would end a march after the very first band (0 misses clears the
        # threshold); >1 is unreachable (a band's miss fraction maxes out at 1.0).
        print("--band-miss-stop-frac must be in (0, 1].", file=sys.stderr)
        return 1
    if not 0.0 < args.edge_tilt_max_deg <= 20.0:
        # The 20 deg cap is the spec's hard limit on the recovery tilt.
        print("--edge-tilt-max-deg must be in (0, 20].", file=sys.stderr)
        return 1
    if args.edge_tilt_step_deg <= 0:
        print("--edge-tilt-step-deg must be positive.", file=sys.stderr)
        return 1
    if args.edge_oor_iters < 1:
        print("--edge-oor-iters must be at least 1.", file=sys.stderr)
        return 1
    if args.edge_recover_window_mm <= 0:
        print("--edge-recover-window-mm must be positive.", file=sys.stderr)
        return 1
    if args.floor_depth_mm is not None and args.floor_depth_mm <= 0:
        print("--floor-depth-mm must be positive.", file=sys.stderr)
        return 1
    if args.floor_margin_mm < 0:
        print("--floor-margin-mm must be non-negative.", file=sys.stderr)
        return 1
    if args.traverse_max_jump_mm <= 0:
        print("--traverse-max-jump-mm must be positive.", file=sys.stderr)
        return 1
    if args.floor_model and args.floor_depth_mm is not None:
        print(
            "--floor-model and --floor-depth-mm are mutually exclusive (the model "
            "file carries its own depth).",
            file=sys.stderr,
        )
        return 1
    floor_model: dict | None = None
    if args.floor_model:
        try:
            floor_model = _load_floor_model(Path(args.floor_model))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"failed to load --floor-model {args.floor_model}: {exc}", file=sys.stderr)
            return 1
        kind = (
            f"rx fit over [{floor_model['rx_min']:+.2f}, {floor_model['rx_max']:+.2f}]rad"
            if floor_model["coeffs"] is not None
            else "constant"
        )
        print(
            f"floor model: {args.floor_model} ({kind}, depth {floor_model['const']:.0f}mm)",
            file=sys.stderr,
        )
    if args.z_min_mm is not None and args.z_max_mm is not None and args.z_min_mm >= args.z_max_mm:
        print("--z-min-mm must be less than --z-max-mm.", file=sys.stderr)
        return 1
    if args.z_stall_iters < 1:
        print("--z-stall-iters must be at least 1.", file=sys.stderr)
        return 1
    if (
        args.rx_min_rad is not None
        and args.rx_max_rad is not None
        and args.rx_min_rad >= args.rx_max_rad
    ):
        print("--rx-min-rad must be less than --rx-max-rad.", file=sys.stderr)
        return 1
    if args.rx_axis_min_rad >= args.rx_axis_max_rad:
        print("--rx-axis-min-rad must be less than --rx-axis-max-rad.", file=sys.stderr)
        return 1
    if args.z_max_mm is None and args.z_min_mm is None:
        print(
            "note: --z-min-mm/--z-max-mm are unset. z-travel saturation will NOT be "
            "detected (a flank that out-reaches z won't end as a z-limit edge), and z "
            "commands are unbounded. Set them to the usable z travel (e.g. "
            "--z-max-mm 392) for the z-limit edge and an absolute z safety floor.",
            file=sys.stderr,
        )

    # X is controlled by Klipper through the gantry server. Y and Z are controlled
    # by the Pico and are read independently below.
    client_x = GantryServerClient(args.gantry_server_url, axis="x", timeout_s=30.0)

    # Y and Z share one real-time Pico connection (pico/gantry_firmware.py).
    pico_axes = {"y", "z"}
    pico_port = args.pico_port
    pico_link = None
    pico_clients: dict[str, object] = {}
    pico_multi = None  # batched Y+Z read/stream over the shared link (set below when 2+ axes)
    # Keep the Pico firmware's physical-travel backstop enabled for both axes.
    _pico_cfg = {
        "y": dict(vmax=args.y_pico_vmax_mm_s, acc=args.y_pico_acc_mm_s2, home=args.home_y),
        "z": dict(vmax=args.z_pico_vmax_mm_s, acc=args.z_pico_acc_mm_s2, home=args.home_z),
    }
    try:
        from openderm.motion.pico.adapter import (
            PicoAxisClient,
            PicoMultiAxis,
            open_pico_link,
        )

        print(f"connecting to Pico (Y+Z) on {pico_port} ...", file=sys.stderr)
        pico_link = open_pico_link(pico_port, timeout_s=30.0)
        for axis in ("y", "z"):
            c = _pico_cfg[axis]
            cl = PicoAxisClient(
                pico_link,
                axis,
                enforce_limits=True,
                vmax_mm_s=c["vmax"],
                acc_mm_s2=c["acc"],
                timeout_s=30.0,
            )
            if c["home"]:
                print(f"homing {axis.upper()} via Pico on {pico_port} ...", file=sys.stderr)
                cl.home()
            pico_clients[axis] = cl
        pico_multi = PicoMultiAxis(pico_clients)
        print(f"Pico link up (Y+Z) on {pico_port}.", file=sys.stderr)
    except Exception as exc:  # ImportError (pyserial) / serial / firmware not responding
        print(f"could not start the Y/Z Pico on {pico_port}: {exc}", file=sys.stderr)
        return 1

    client_y = pico_clients["y"]
    client_z = pico_clients["z"]

    try:
        state = client_x.status()
    except GantryServerError as exc:
        print(
            f"Gantry server not reachable at {args.gantry_server_url}: {exc}",
            file=sys.stderr,
        )
        return 1
    homed_axes = state.get("homed_axes", [])
    if "x" not in homed_axes:
        print(
            "x-axis is not homed. Run `openderm --axis x home` first "
            "(pointing at the same gantry server).",
            file=sys.stderr,
        )
        return 1
    # Ask the Pico directly for Y/Z state. This is
    # the first contact with the Pico's status, so handle an unreachable port cleanly
    # (mirrors the gantry reachability check above).
    for axis in ("y", "z"):
        cl = client_y if axis == "y" else client_z
        try:
            pst = cl.status()
        except GantryServerError as exc:
            print(f"Pico {axis.upper()} not reachable on {pico_port}: {exc}", file=sys.stderr)
            return 1
        if axis not in (pst.get("homed_axes") or []):
            extra = " (the contour scan probes y to find the body edges)" if axis == "y" else ""
            print(
                f"{axis}-axis (Pico) is not homed{extra}. Re-run with --home-{axis}, or "
                "home it first via pico/gantry_client.py.",
                file=sys.stderr,
            )
            return 1

    # Self-collision guard: the config-dependent (rx,z) -> safe (x,y) envelope that replaces
    # the old static y soft-limit. ENFORCED -- any move that would bring a moving part within
    # the margin of the frame is refused. Set env OPENDERM_COLLISION_MODE=off to disable
    # the guard entirely (e.g. bench work without the envelope).
    collision_enabled = os.environ.get("OPENDERM_COLLISION_MODE", "enforce").lower() != "off"
    collision_guard: CollisionGuard | None = None
    if collision_enabled:
        try:
            collision_guard = CollisionGuard.load()
        except Exception as exc:  # missing/unreadable envelope -> fail closed
            print(
                f"Aborting contour scan: cannot load self-collision envelope: {exc}",
                file=sys.stderr,
            )
            return 1
        print(
            f"self-collision guard: ENFORCE (margin {collision_guard.margin:.0f}mm, "
            f"backlash +/-{collision_guard.backlash_deg:.1f}deg)",
            file=sys.stderr,
        )

    rx_client = RxAxisServerClient(args.rx_server_url, timeout_s=30.0)
    try:
        # /state returns 409 until the RX axis is homed.
        rx_client.status()
    except RxAxisServerError as exc:
        print(
            f"RX-axis server not reachable or rx axis not homed at "
            f"{args.rx_server_url}: {exc}\n"
            "Home it with `openderm --axis rx home`.",
            file=sys.stderr,
        )
        return 1
    try:
        # Zero velocity safely verifies the servo endpoint before any motion.
        rx_client.set_velocity(0.0)
    except RxAxisServerError as exc:
        print(
            f"The RX-axis server at {args.rx_server_url} does not provide the "
            f"required velocity-control endpoint: {exc}",
            file=sys.stderr,
        )
        return 1

    # (The self-collision guard startup gate runs below, AFTER read_position /
    # read_rx_rad are defined -- calling them here would hit an unbound-local error
    # and silently skip the check.)

    # Load the rx-pivot model used to hold the viewed x/y point fixed while RX
    # tilts. Active only when RX is being regulated (it compensates RX moves).
    model_path = Path(args.rx_pivot_model)
    if not model_path.exists():
        print(
            f"rx-pivot model {model_path} not found. Capture and fit one with "
            "scripts/calibration/rx_pivot_capture.py and "
            "scripts/calibration/rx_pivot_fit.py.",
            file=sys.stderr,
        )
        return 1
    try:
        pivot_model = RxPivotModel.load(model_path)
    except (OSError, ValueError, KeyError) as exc:
        print(f"failed to load rx-pivot model {model_path}: {exc}", file=sys.stderr)
        return 1
    pivot_homed = set(homed_axes) | pico_axes
    missing = [axis for axis in ("x", "y", "z") if axis not in pivot_homed]
    if missing:
        print(
            f"rx-pivot compensation needs {'/'.join(missing)} homed "
            "(it moves the gantry along the RX arc).",
            file=sys.stderr,
        )
        return 1

    controller = build_sensor_controller()
    runtime = contour_runtime.build_runtime(
        args=args,
        floor_model=floor_model,
        client_x=client_x,
        client_y=client_y,
        client_z=client_z,
        pico_axes=pico_axes,
        pico_link=pico_link,
        pico_multi=pico_multi,
        collision_guard=collision_guard,
        rx_client=rx_client,
        pivot_model=pivot_model,
        controller=controller,
        camera_types=(
            CanonCamera,
            CanonCaptureConfig,
            CanonEdsdk,
            CanonError,
        ),
    )
    if runtime is None:
        return 1
    return contour_execution.execute_scan(runtime)


def main() -> int:
    args = build_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
