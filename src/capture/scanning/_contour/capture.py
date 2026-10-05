"""Camera capture finalization and per-station metadata recording."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime

from ..config import STATION_RECOVERED, STATION_SETTLED


def finalize_pending_capture(
    ctx,
    state,
    timeout_s: float | None = None,
) -> bool:
    """Collect an in-flight photo and persist its metadata."""
    pending = state.camera_pending["info"]
    if pending is None or state.camera is None:
        return True
    state.camera_pending["info"] = None
    try:
        photo = state.camera.wait_capture_result(timeout_s=timeout_s)
    except ctx.CanonError as exc:
        print(
            f"Aborting scan: capture failed at station {pending['station_no']} "
            f"x={pending['x_report']:.2f}mm: {exc}",
            file=sys.stderr,
        )
        ctx.stop_state["requested"] = True
        return False

    capture_state = pending["state"]
    capture_state["image"] = str(photo)
    try:
        photo.with_suffix(".json").write_text(json.dumps(capture_state, indent=2))
    except OSError as exc:
        print(f"  warning: failed to write sidecar metadata: {exc}", file=sys.stderr)
    print(
        f"RECORD  station={pending['station_no']} {pending['phase_label']} "
        f"x={pending['x_report']:.2f}mm y={pending['y_now']:.2f}mm "
        f"({pending['status_str']}) -> {photo}"
    )
    _append_pose(ctx, state, capture_state)
    return True


def _append_pose(ctx, state, capture_state: dict) -> None:
    try:
        with state.pose_log_path.open("a") as log:
            log.write(json.dumps(capture_state) + "\n")
    except OSError as exc:
        print(f"  warning: failed to write pose log: {exc}", file=sys.stderr)


def _capture_snapshot(ctx, state, station_no: int, result: dict) -> dict:
    """Read the held pose and sensors after the capture gate settles."""
    args = ctx.args
    time.sleep(args.record_pause_s)
    sensor1, sensor2, gated, gate_wait_s = ctx.snapshot_for_capture()
    if args.simultaneous_sensors:
        ctx.controller.set_enabled_for_selection("all", False)
        ctx.lasers_both_on["on"] = False
    position = ctx.read_position()
    return {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "station": station_no,
        "station_status": result["status"],
        "recovered": result["status"] == STATION_RECOVERED,
        "settled": result["status"] == STATION_SETTLED,
        "target_mm": args.target_mm,
        "x_mm": position.get("x"),
        "y_mm": position.get("y"),
        "z_mm": position.get("z"),
        "rx_rad": ctx.read_rx_rad(),
        "sensor1_mm": sensor1.distance_mm,
        "sensor2_mm": sensor2.distance_mm,
        "sensor1_in_range": sensor1.in_range,
        "sensor2_in_range": sensor2.in_range,
        "capture_gated": gated,
        "capture_gate_mm": args.capture_gate_mm,
        "capture_gate_wait_s": round(gate_wait_s, 3),
        "image": None,
    }


def _remember_capture_pose(ctx, state, capture_state: dict) -> None:
    pose_keys = ("x_mm", "y_mm", "z_mm", "rx_rad")
    if not all(capture_state[key] is not None for key in pose_keys):
        return
    state.last_capture_pose.update(
        x=capture_state["x_mm"],
        y=capture_state["y_mm"],
        z=capture_state["z_mm"],
        rx=capture_state["rx_rad"],
    )
    ctx.breadcrumb_reseed(
        capture_state["x_mm"],
        capture_state["y_mm"],
        capture_state["z_mm"],
        capture_state["rx_rad"],
    )


def _trigger_camera(
    ctx,
    state,
    capture_state: dict,
    station_no: int,
    phase_label: str,
    x_report: float,
    y_now: float,
    status_str: str,
) -> bool:
    try:
        state.camera.trigger_capture(index=state.global_k)
    except ctx.CanonError as exc:
        print(
            f"Aborting scan: capture failed at station {station_no} x={x_report:.2f}mm: {exc}",
            file=sys.stderr,
        )
        ctx.stop_state["requested"] = True
        return False
    state.camera_pending["info"] = {
        "state": capture_state,
        "station_no": station_no,
        "phase_label": phase_label,
        "x_report": x_report,
        "y_now": y_now,
        "status_str": status_str,
    }
    return True


def capture_station(
    ctx,
    state,
    x_now: float,
    y_now: float,
    phase_label: str,
    row_idx: int,
    col_idx: int,
    result: dict,
) -> bool:
    """Hold, snapshot, capture, and log one station."""
    station_no = state.global_k + 1
    ctx.set_activity("capture")
    ctx.rec_ctx["station"] = station_no
    if not finalize_pending_capture(ctx, state):
        return False

    capture_state = _capture_snapshot(ctx, state, station_no, result)
    capture_state.update(
        phase=phase_label,
        row=row_idx,
        col=col_idx + 1,
        cols=state.cols,
    )
    _remember_capture_pose(ctx, state, capture_state)

    x_report = x_now if capture_state["x_mm"] is None else capture_state["x_mm"]
    status_str = result["status"]
    gated = capture_state["capture_gated"]
    gate_wait_s = capture_state["capture_gate_wait_s"]
    if gated is not None:
        status_str += f", gate {'ok' if gated else 'TIMEOUT'} {gate_wait_s:.1f}s"

    if state.camera is not None:
        if not _trigger_camera(
            ctx,
            state,
            capture_state,
            station_no,
            phase_label,
            x_report,
            y_now,
            status_str,
        ):
            return False
    else:
        capture_state["simulated"] = True
        print(
            f"RECORD  station={station_no} {phase_label} x={x_report:.2f}mm "
            f"y={y_now:.2f}mm ({status_str}) [simulated]"
        )
        _append_pose(ctx, state, capture_state)

    ctx.record_sample(
        "capture",
        status=result["status"],
        image=capture_state.get("image"),
    )
    state.global_k += 1
    return True
