"""Check the CLI/config boundary without connecting to hardware."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import build_collision_envelope as builder
import floor_depth_tare
import rx_pivot_capture
import rx_pivot_fit
from openderm import script_config
from openderm.motion.collision_guard import CollisionGuard


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    document = json.loads(script_config.CONFIG_PATH.read_text())
    path = tmp_path / "scripts.json"
    path.write_text(json.dumps(document))
    monkeypatch.setattr(script_config, "CONFIG_PATH", path)
    return path, document


@pytest.mark.parametrize(
    "module,inputs",
    [
        (rx_pivot_capture, ["0.95", "--record-file", "poses.jsonl"]),
        (rx_pivot_fit, ["poses.jsonl", "--out", "model.json"]),
        (floor_depth_tare, ["point", "--out", "floor.json"]),
        (floor_depth_tare, ["sweep", "0.35:1.5:6", "--out", "floor.json"]),
        (builder, ["sweep", "--grid", "grid.npz", "--out", "envelope.npz"]),
        (builder, ["derive", "--grid", "grid.npz", "--out", "envelope.npz"]),
    ],
)
def test_all_run_arguments_are_required(module, inputs):
    parser = module.build_parser()
    parser.parse_args(inputs)

    def check_required(current):
        for action in current._actions:
            if isinstance(action, argparse._HelpAction):
                continue
            assert action.required, action.dest
            if isinstance(action, argparse._SubParsersAction):
                for child in action.choices.values():
                    check_required(child)

    check_required(parser)
    output_flag = "--record-file" if module is rx_pivot_capture else "--out"
    without_output = inputs[:]
    index = without_output.index(output_flag)
    del without_output[index : index + 2]
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(without_output)
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "module,inputs",
    [
        (rx_pivot_capture, ["0.95", "--record-file", "poses.jsonl", "--gain-mm-per-mm", "1"]),
        (floor_depth_tare, ["point", "--out", "floor.json", "--target-mm", "120"]),
        (builder, ["derive", "--grid", "grid.npz", "--out", "envelope.npz", "20", "2.5"]),
    ],
)
def test_persistent_settings_cannot_be_overridden_on_cli(module, inputs):
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(inputs)


def test_config_path_does_not_depend_on_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = script_config.load_script_config("floor_depth_tare")
    assert settings["pico_port"] == "socket://openderm-gantry.local:8095"
    assert settings["target_mm"] == 110.0
    assert settings["pico_vmax_mm_s"] is None


def test_capture_uses_config_and_explicit_output(config_file):
    path, document = config_file
    document["calibration"].update(target_mm=120, pico_port="socket://rig:8095")
    document["rx_pivot_capture"].update(degrees=True, home_y=True, jog_step_mm=0.25)
    path.write_text(json.dumps(document))
    args = rx_pivot_capture.build_parser().parse_args(["45", "--record-file", "run/poses.jsonl"])
    with (
        mock.patch.object(rx_pivot_capture.sys.stdin, "isatty", return_value=True),
        mock.patch.object(rx_pivot_capture, "_connect_session") as connect,
        mock.patch.object(rx_pivot_capture, "_PivotCaptureWorkflow") as workflow,
    ):
        workflow.return_value.run.return_value = 0
        assert rx_pivot_capture.run(args) == 0
    options = connect.call_args.args[0]
    assert options.target_mm == 120
    assert options.pico_port == "socket://rig:8095"
    assert options.degrees is True
    assert options.home_y is True
    assert options.jog_step_mm == 0.25
    assert options.target_rx == 45
    assert options.record_file == Path("run/poses.jsonl")
    connect.return_value.close.assert_called_once()


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("calibration", "samples", 0),
        ("calibration", "samples", True),
        ("calibration", "period_s", -1),
        ("calibration", "gain_mm_per_mm", float("nan")),
        ("calibration", "pico_port", ""),
        ("floor_depth_tare", "home_z", "false"),
        ("floor_depth_tare", "tare_samples", 2),
        ("floor_depth_tare", "max_travel_mm", 0),
        ("floor_depth_tare", "pico_vmax_mm_s", -1),
        ("floor_depth_tare", "unknown_setting", 1),
    ],
)
def test_invalid_config_fails_before_hardware(config_file, section, key, value):
    path, document = config_file
    document[section][key] = value
    path.write_text(json.dumps(document))
    args = floor_depth_tare.build_parser().parse_args(["point", "--out", "floor.json"])
    with mock.patch.object(floor_depth_tare, "_connect_session") as connect:
        assert floor_depth_tare.run(args) == 1
        connect.assert_not_called()


@pytest.mark.parametrize("problem", ["missing_file", "malformed_json", "missing_setting"])
def test_config_has_no_silent_fallback(config_file, problem):
    path, document = config_file
    if problem == "missing_file":
        path.unlink()
    elif problem == "malformed_json":
        path.write_text("{")
    else:
        del document["calibration"]["target_mm"]
        path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        script_config.load_script_config("floor_depth_tare")


def test_collision_cli_uses_configured_margin_and_resolution(config_file, tmp_path, monkeypatch):
    path, document = config_file
    document["collision"].update(margin_mm=35, backlash_deg=4, fine_rx_points=5)
    path.write_text(json.dumps(document))
    grid = tmp_path / "grid.npz"
    out = tmp_path / "output" / "envelope.npz"
    axes = np.array([0.0, 1.0])
    np.savez(grid, rx=axes, z=axes, x=axes, y=axes, clearance=np.full((2, 2, 2, 2), 30.0))
    monkeypatch.setattr("sys.argv", ["builder", "derive", "--grid", str(grid), "--out", str(out)])
    assert builder.main() == 0
    guard = CollisionGuard.load(str(out))
    assert not guard.is_safe(0.5, 0.5, 0.5, 0.5)
    assert guard.margin == 35
    assert guard.backlash_deg == 4
    with np.load(out) as envelope:
        assert len(envelope["rx"]) == 5


def test_sweep_reads_travel_from_cad_config(tmp_path, monkeypatch):
    calibration = tmp_path / "frame_calibration.json"
    travel = {"rx_rad": [0.1, 1.7], "z_mm": [2, 300], "x_mm": [3, 700], "y_mm": [4, 600]}
    calibration.write_text(json.dumps({"real_travel": travel}))
    monkeypatch.setattr(builder, "CALIBRATION_PATH", calibration)
    settings = script_config.CollisionConfig(
        workers=1,
        margin_mm=20,
        backlash_deg=2.5,
        rx_points=2,
        z_points=2,
        x_points=3,
        y_points=4,
        fine_rx_points=5,
    )
    pool = mock.MagicMock()
    pool.__enter__.return_value.imap_unordered.side_effect = lambda fn, pairs: (
        (i, j, np.full((3, 4), 30)) for i, j, rx, z in pairs
    )
    grid = tmp_path / "output" / "grid.npz"
    with mock.patch("multiprocessing.Pool", return_value=pool) as factory:
        builder.sweep(grid, settings)
    assert factory.call_args.args == (1,)
    x, y = factory.call_args.kwargs["initargs"]
    np.testing.assert_allclose(x, np.linspace(3, 700, 3))
    np.testing.assert_allclose(y, np.linspace(4, 600, 4))
    with np.load(grid) as result:
        for axis, key in [("rx", "rx_rad"), ("z", "z_mm"), ("x", "x_mm"), ("y", "y_mm")]:
            np.testing.assert_allclose(result[axis][[0, -1]], travel[key])
        assert result["clearance"].shape == (2, 2, 3, 4)
