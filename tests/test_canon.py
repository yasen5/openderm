from __future__ import annotations

from pathlib import Path
import tempfile
import time
import unittest


from openderm.camera.canon import (
    CanonCamera,
    CanonCaptureConfig,
    CanonError,
    K_EDS_OBJECT_EVENT_DIR_ITEM_REQUEST_TRANSFER,
    K_EDS_PROP_ID_AE_MODE,
    K_EDS_PROP_ID_AV,
    K_EDS_PROP_ID_ISO_SPEED,
    K_EDS_PROP_ID_LENS_NAME,
    K_EDS_PROP_ID_TV,
    build_parser,
    config_from_args,
    keep_camera_awake,
    release_camera,
)


class FakeSdk:
    def __init__(self) -> None:
        self.object_callback_type = lambda func: func
        self.state_callback_type = lambda func: func
        self.object_handler = None
        self.state_handler = None
        self.downloaded_paths: list[Path] = []
        self.released: list[object] = []
        self.retained = []
        self.commands: list[str] = []
        self.property_values: dict[int, object] = {
            K_EDS_PROP_ID_AE_MODE: 3,
            K_EDS_PROP_ID_ISO_SPEED: 0x48,
            K_EDS_PROP_ID_AV: 0x30,
            K_EDS_PROP_ID_TV: 0x68,
            K_EDS_PROP_ID_LENS_NAME: "RF100mm F2.8 L Macro IS USM",
        }
        self.property_descs: dict[int, tuple[int, ...]] = {
            K_EDS_PROP_ID_ISO_SPEED: (0x48, 0x50, 0x58),
            K_EDS_PROP_ID_AV: (0x28, 0x30, 0x38),
            K_EDS_PROP_ID_TV: (0x5B, 0x63, 0x68),
        }

    def initialize_sdk(self) -> None:
        self.commands.append("initialize")

    def terminate_sdk(self) -> None:
        self.commands.append("terminate")

    def get_camera_list(self) -> object:
        return "camera-list"

    def get_child_count(self, ref: object) -> int:
        self.commands.append(f"count:{ref}")
        return 1

    def get_child_at_index(self, ref: object, index: int) -> object:
        self.commands.append(f"child:{ref}:{index}")
        return "camera-0"

    def open_session(self, camera: object) -> None:
        self.commands.append(f"open:{camera}")

    def close_session(self, camera: object) -> None:
        self.commands.append(f"close:{camera}")

    def retain(self, ref: object) -> None:
        self.retained.append(ref)

    def release(self, ref: object) -> None:
        self.released.append(ref)

    def set_object_handler(self, camera: object, callback: object) -> None:
        self.object_handler = callback

    def set_state_handler(self, camera: object, callback: object) -> None:
        self.state_handler = callback

    def set_save_to_host(self, camera: object) -> None:
        self.commands.append(f"save_to_host:{camera}")

    def set_capacity(self, camera: object) -> None:
        self.commands.append(f"capacity:{camera}")

    def send_take_picture(self, camera: object) -> None:
        self.commands.append(f"take:{camera}")
        assert self.object_handler is not None
        self.object_handler(K_EDS_OBJECT_EVENT_DIR_ITEM_REQUEST_TRANSFER, "dir-item-1", None)

    def extend_shutdown_timer(self, camera: object) -> None:
        self.commands.append(f"extend:{camera}")

    def get_event(self) -> None:
        self.commands.append("event")

    def get_directory_item_info(self, directory_item: object):
        self.commands.append(f"info:{directory_item}")
        return type("Info", (), {"size_bytes": 12, "file_name": "IMG_0001.JPG"})()

    def get_property_uint32(self, camera: object, property_id: int, *, parameter: int = 0) -> int:
        del camera, parameter
        value = self.property_values[property_id]
        assert isinstance(value, int)
        return value

    def get_property_string(self, camera: object, property_id: int, *, parameter: int = 0) -> str:
        del camera, parameter
        value = self.property_values[property_id]
        assert isinstance(value, str)
        return value

    def set_property_uint32(
        self,
        camera: object,
        property_id: int,
        value: int,
        *,
        parameter: int = 0,
        operation_name: str | None = None,
    ) -> None:
        del camera, parameter, operation_name
        self.property_values[property_id] = value
        self.commands.append(f"set:{property_id:08X}:{value:08X}")

    def get_property_desc(self, camera: object, property_id: int) -> tuple[int, ...]:
        del camera
        return self.property_descs[property_id]

    def create_file_stream(self, path: Path) -> object:
        self.downloaded_paths.append(path)
        return path

    def download(self, directory_item: object, size_bytes: int, stream: object) -> None:
        Path(stream).write_bytes(b"fake-jpeg")

    def download_complete(self, directory_item: object) -> None:
        self.commands.append(f"complete:{directory_item}")


class ConfigTests(unittest.TestCase):
    def test_parser_builds_expected_config(self) -> None:
        args = build_parser().parse_args(
            [
                "--output-dir",
                "/tmp/captures",
                "--basename",
                "skin-scan",
                "--count",
                "3",
                "--interval-s",
                "0.5",
                "--timeout-s",
                "30",
                "--edsdk-lib",
                "/opt/canon/libEDSDK.so",
            ]
        )
        config = config_from_args(args)
        self.assertEqual(config.output_dir, Path("/tmp/captures"))
        self.assertEqual(config.basename, "skin-scan")
        self.assertEqual(config.count, 3)
        self.assertEqual(config.interval_s, 0.5)
        self.assertEqual(config.timeout_s, 30.0)
        self.assertEqual(config.edsdk_lib, "/opt/canon/libEDSDK.so")

    def test_invalid_count_is_rejected(self) -> None:
        args = build_parser().parse_args(["--count", "0"])
        with self.assertRaises(CanonError):
            config_from_args(args)


class KeepAwakeTests(unittest.TestCase):
    def test_opens_session_and_never_closes_it(self) -> None:
        # --keep-awake parks the camera in PC-remote mode (the no-sleep state a
        # Ctrl-C'd scan leaves behind): the session MUST be left open -- a
        # close_session or terminate_sdk would put the camera back in its
        # normal UI where the auto-power-off timer runs.
        fake = FakeSdk()
        keep_camera_awake(sdk=fake)  # type: ignore[arg-type]
        self.assertIn("initialize", fake.commands)
        self.assertIn("open:camera-0", fake.commands)
        self.assertFalse(
            [c for c in fake.commands if c.startswith("close:")],
            f"keep-awake closed the session: {fake.commands}",
        )
        self.assertNotIn("terminate", fake.commands)

    def test_no_camera_is_a_clear_error(self) -> None:
        class NoCameraSdk(FakeSdk):
            def get_child_count(self, ref: object) -> int:
                return 0

        with self.assertRaises(CanonError):
            keep_camera_awake(sdk=NoCameraSdk())  # type: ignore[arg-type]

    def test_parser_accepts_keep_awake(self) -> None:
        args = build_parser().parse_args(["--keep-awake"])
        self.assertTrue(args.keep_awake)

    def test_release_opens_then_cleanly_closes(self) -> None:
        # --release is the inverse: a clean open+close drops the camera out of
        # PC-remote mode so its own auto-power-off timer sleeps it.
        fake = FakeSdk()
        release_camera(sdk=fake)  # type: ignore[arg-type]
        self.assertIn("open:camera-0", fake.commands)
        self.assertIn("close:camera-0", fake.commands)
        self.assertLess(
            fake.commands.index("open:camera-0"),
            fake.commands.index("close:camera-0"),
        )
        self.assertIn("terminate", fake.commands)

    def test_keep_awake_and_release_are_exclusive(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--keep-awake", "--release"])


class CameraTests(unittest.TestCase):
    def test_callback_failure_surfaces_instead_of_hanging(self) -> None:
        # EdsRetain returns a REFERENCE COUNT, not an EDS_ERR code -- the
        # wrapper mis-checking it raised inside the ctypes object callback,
        # where exceptions are SWALLOWED ("Exception ignored...") and the
        # capture can otherwise hang until its timeout. Any callback failure
        # must surface through the results queue as a CanonError/exception
        # from the capture call, with image_ready still fired.
        sdk = FakeSdk()

        def bad_retain(ref):
            raise CanonError("retain exploded")

        sdk.retain = bad_retain
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CanonCaptureConfig(
                output_dir=Path(tmpdir), basename="r7", count=1, timeout_s=1.0
            )
            camera = CanonCamera(sdk, config)
            with camera:
                with self.assertRaises(CanonError):
                    camera.capture_one()
            self.assertTrue(camera.image_ready.is_set())

    def test_capture_downloads_file(self) -> None:
        sdk = FakeSdk()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CanonCaptureConfig(
                output_dir=Path(tmpdir), basename="r7", count=1, timeout_s=1.0
            )
            with CanonCamera(sdk, config) as camera:
                output = camera.capture_one()
            self.assertTrue(output.exists())
            self.assertEqual(output.read_bytes(), b"fake-jpeg")
            self.assertEqual(output.suffix, ".jpg")
            self.assertIn("take:camera-0", sdk.commands)
            self.assertIn("dir-item-1", sdk.released)
            self.assertIn("complete:dir-item-1", sdk.commands)

    def test_camera_can_read_and_write_exposure_properties(self) -> None:
        sdk = FakeSdk()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CanonCaptureConfig(
                output_dir=Path(tmpdir), basename="r7", count=1, timeout_s=1.0
            )
            with CanonCamera(sdk, config) as camera:
                self.assertEqual(
                    camera.get_string_property(K_EDS_PROP_ID_LENS_NAME),
                    "RF100mm F2.8 L Macro IS USM",
                )
                self.assertEqual(camera.get_uint32_property(K_EDS_PROP_ID_ISO_SPEED), 0x48)
                self.assertEqual(
                    camera.get_supported_uint32_values(K_EDS_PROP_ID_TV), (0x5B, 0x63, 0x68)
                )
                camera.set_uint32_property(K_EDS_PROP_ID_ISO_SPEED, 0x58)
                self.assertEqual(camera.get_uint32_property(K_EDS_PROP_ID_ISO_SPEED), 0x58)
            self.assertIn("set:00000402:00000058", sdk.commands)

    def test_raw_quality_selector_prefers_raw_only_code(self) -> None:
        raw_only = 0x0064FF0F
        raw_plus_jpeg = 0x00641308
        selected = CanonCamera._select_raw_image_quality_code((raw_plus_jpeg, raw_only))
        self.assertEqual(selected, raw_only)

    def test_trigger_returns_at_dwell_and_transfer_rides_the_worker_pump(self) -> None:
        # The head is released at END OF EXPOSURE (the dwell), not at the
        # camera's file-ready event: on hardware that event lands ~0.5s+ after
        # the shutter. With the DirItem event delayed until later get_event()
        # pumps, trigger_capture must return promptly anyway, and the transfer
        # worker's own pump must still receive the event and complete the
        # download in the background.
        class DelayedEventSdk(FakeSdk):
            def __init__(self) -> None:
                super().__init__()
                self._pending_fire = 0

            def send_take_picture(self, camera: object) -> None:
                self.commands.append(f"take:{camera}")
                self._pending_fire = 3  # deliver on the 3rd pump

            def get_event(self) -> None:
                super().get_event()
                if self._pending_fire:
                    self._pending_fire -= 1
                    if self._pending_fire == 0 and self.object_handler is not None:
                        self.object_handler(
                            K_EDS_OBJECT_EVENT_DIR_ITEM_REQUEST_TRANSFER,
                            "dir-item-1",
                            None,
                        )

        sdk = DelayedEventSdk()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CanonCaptureConfig(
                output_dir=Path(tmpdir),
                basename="r7",
                timeout_s=5.0,
                exposure_dwell_s=0.15,
            )
            with CanonCamera(sdk, config) as camera:
                t0 = time.monotonic()
                camera.trigger_capture()
                took = time.monotonic() - t0
                self.assertLess(took, 1.0, "trigger_capture waited past the exposure dwell")
                output = camera.wait_capture_result(timeout_s=5.0)
            self.assertTrue(output.exists())
            self.assertEqual(output.read_bytes(), b"fake-jpeg")

    def test_capture_times_out_without_transfer_event(self) -> None:
        class NoTransferSdk(FakeSdk):
            def send_take_picture(self, camera: object) -> None:
                self.commands.append(f"take:{camera}")

        sdk = NoTransferSdk()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CanonCaptureConfig(output_dir=Path(tmpdir), timeout_s=0.1)
            with CanonCamera(sdk, config) as camera:
                with self.assertRaises(CanonError):
                    camera.capture_one()


if __name__ == "__main__":
    unittest.main()
