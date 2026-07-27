"""High-level Canon camera session and capture workflow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import queue
import threading
import time
from typing import cast

from .sdk import (
    EDS_ERR_OK,
    K_EDS_OBJECT_EVENT_DIR_ITEM_REQUEST_TRANSFER,
    K_EDS_PROP_ID_IMAGE_QUALITY,
    K_EDS_STATE_EVENT_WILL_SOON_SHUTDOWN,
    CanonEdsdk,
    CanonError,
)


@dataclass(frozen=True)
class CanonCaptureConfig:
    output_dir: Path = Path("captures")
    basename: str = "eos-r7"
    count: int = 1
    interval_s: float = 0.0
    timeout_s: float = 15.0
    edsdk_lib: str | None = None
    prefer_raw: bool = False
    # This covers shutter latency and exposure, not image processing or USB
    # transfer. The transfer worker continues after the head is free to move.
    exposure_dwell_s: float = 0.35


class CanonCamera:
    """Own an EDSDK session and coordinate asynchronous image transfers."""

    def __init__(
        self,
        sdk: CanonEdsdk,
        config: CanonCaptureConfig,
    ) -> None:
        self.sdk = sdk
        self.config = config
        self._camera_list: object | None = None
        self._camera: object | None = None
        self._started = False
        self._sdk_initialized = False
        self._capture_results: queue.Queue[Path | Exception] = queue.Queue()
        self._shutdown_event = threading.Event()
        # DirItem transfers run outside the event callback. All SDK calls are
        # serialized so downloads never overlap another EDSDK call.
        self._sdk_lock = threading.RLock()
        self._pending_items: queue.Queue = queue.Queue()
        self._transfer_thread: threading.Thread | None = None
        self._transfer_stop = threading.Event()
        # Fired when exposure is complete and an item is ready to transfer.
        self.image_ready = threading.Event()
        self._active_capture_index = 0
        self._object_callback = self._build_object_callback()
        self._state_callback = self._build_state_callback()

    def __enter__(self) -> "CanonCamera":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        tb: object,
    ) -> None:
        self.close()

    def start(self) -> None:
        if self._started:
            return
        self.sdk.initialize_sdk()
        self._sdk_initialized = True
        try:
            self._camera_list = self.sdk.get_camera_list()
            if self.sdk.get_child_count(self._camera_list) < 1:
                raise CanonError("No Canon camera detected through EDSDK.")
            self._camera = self.sdk.get_child_at_index(
                self._camera_list,
                0,
            )
            self.sdk.open_session(self._camera)
            self.sdk.set_object_handler(
                self._camera,
                self._object_callback,
            )
            self.sdk.set_state_handler(
                self._camera,
                self._state_callback,
            )
            self.sdk.set_save_to_host(self._camera)
            self.sdk.set_capacity(self._camera)
            if self.config.prefer_raw:
                self._set_raw_image_quality()
            self._started = True
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        # Stop the transfer worker first, then release retained items before
        # terminating the SDK.
        if self._transfer_thread is not None:
            self._transfer_stop.set()
            self._transfer_thread.join(timeout=10.0)
            self._transfer_thread = None
            while True:
                try:
                    item, _ = self._pending_items.get_nowait()
                except queue.Empty:
                    break
                try:
                    self.sdk.release(item)
                except CanonError:
                    pass
        if self._camera is not None:
            try:
                self.sdk.close_session(self._camera)
            except CanonError:
                pass
            try:
                self.sdk.release(self._camera)
            except CanonError:
                pass
            self._camera = None
        if self._camera_list is not None:
            try:
                self.sdk.release(self._camera_list)
            except CanonError:
                pass
            self._camera_list = None
        if self._sdk_initialized:
            try:
                self.sdk.terminate_sdk()
            except CanonError:
                pass
            self._sdk_initialized = False
            self._started = False

    def capture_many(self) -> list[Path]:
        captures: list[Path] = []
        for index in range(self.config.count):
            captures.append(self.capture_one(index=index))
            if index + 1 < self.config.count and self.config.interval_s > 0:
                time.sleep(self.config.interval_s)
        return captures

    def _ensure_transfer_worker(self) -> None:
        if self._transfer_thread is not None and self._transfer_thread.is_alive():
            return
        self._transfer_stop.clear()
        self._transfer_thread = threading.Thread(
            target=self._transfer_loop,
            name="canon-transfer",
            daemon=True,
        )
        self._transfer_thread.start()

    def _transfer_loop(self) -> None:
        # The worker also pumps EdsGetEvent so transfers and shutdown warnings
        # arrive without requiring the motion thread to poll.
        next_keepalive_at = time.monotonic() + 2.0
        while not self._transfer_stop.is_set():
            try:
                item, index = self._pending_items.get(timeout=0.05)
            except queue.Empty:
                try:
                    with self._sdk_lock:
                        camera = self._camera
                        if camera is None:
                            continue
                        self.sdk.get_event()
                        if self._shutdown_event.is_set():
                            self._shutdown_event.clear()
                            self.sdk.extend_shutdown_timer(camera)
                        if time.monotonic() >= next_keepalive_at:
                            self.sdk.extend_shutdown_timer(camera)
                            next_keepalive_at = time.monotonic() + 2.0
                except CanonError:
                    # A transient pump failure will surface on a real call.
                    pass
                continue
            try:
                with self._sdk_lock:
                    result: Path | Exception = self._download_directory_item(
                        item,
                        index=index,
                    )
            except Exception as exc:  # noqa: BLE001
                result = exc
            self._capture_results.put(result)

    def trigger_capture(self, *, index: int = 0) -> None:
        """Fire the shutter and return once exposure is complete.

        The transfer worker handles the later directory-item event and USB
        download, allowing the scanner to begin its next movement.
        """
        self.require_camera()
        self._ensure_transfer_worker()
        self._active_capture_index = index
        self._drain_capture_results()
        self.image_ready.clear()
        with self._sdk_lock:
            self.sdk.send_take_picture(self.require_camera())
        self.image_ready.wait(timeout=self.config.exposure_dwell_s)

    def wait_capture_result(
        self,
        timeout_s: float | None = None,
    ) -> Path:
        """Wait for the transfer worker to deliver the captured file."""
        self.require_camera()
        deadline = time.monotonic() + (
            timeout_s if timeout_s is not None else self.config.timeout_s
        )
        while True:
            result = self._get_pending_capture_result()
            if result is not None:
                return result
            if time.monotonic() >= deadline:
                raise CanonError("Timed out waiting for EOS R7 image transfer.")
            with self._sdk_lock:
                self.sdk.get_event()
            time.sleep(0.02)

    def capture_one(self, *, index: int = 0) -> Path:
        """Trigger a capture and wait for its transferred file."""
        self.trigger_capture(index=index)
        return self.wait_capture_result()

    def capture_one_with_config(
        self,
        config: CanonCaptureConfig,
        *,
        index: int = 0,
    ) -> Path:
        previous_config = self.config
        self.config = config
        try:
            if config.prefer_raw:
                self._set_raw_image_quality()
            return self.capture_one(index=index)
        finally:
            self.config = previous_config

    def require_camera(self) -> object:
        if not self._started or self._camera is None:
            raise CanonError("Camera session is not open.")
        return self._camera

    def get_uint32_property(self, property_id: int) -> int:
        return self.sdk.get_property_uint32(
            self.require_camera(),
            property_id,
        )

    def set_uint32_property(
        self,
        property_id: int,
        value: int,
        *,
        operation_name: str | None = None,
    ) -> None:
        self.sdk.set_property_uint32(
            self.require_camera(),
            property_id,
            value,
            operation_name=operation_name,
        )

    def get_string_property(self, property_id: int) -> str:
        return self.sdk.get_property_string(
            self.require_camera(),
            property_id,
        )

    def get_supported_uint32_values(
        self,
        property_id: int,
    ) -> tuple[int, ...]:
        return self.sdk.get_property_desc(
            self.require_camera(),
            property_id,
        )

    def _set_raw_image_quality(self) -> None:
        camera = self.require_camera()
        supported = self.sdk.get_property_desc(
            camera,
            K_EDS_PROP_ID_IMAGE_QUALITY,
        )
        raw_quality = self._select_raw_image_quality_code(supported)
        if raw_quality is None:
            raise CanonError("Camera did not report a RAW-only image quality option via EDSDK.")
        self.sdk.set_property_uint32(
            camera,
            K_EDS_PROP_ID_IMAGE_QUALITY,
            raw_quality,
            operation_name="kEdsPropID_ImageQuality",
        )

    @staticmethod
    def _select_raw_image_quality_code(
        codes: tuple[int, ...],
    ) -> int | None:
        def split(
            code: int,
        ) -> tuple[int, int, int, int, int, int]:
            return (
                (code >> 24) & 0xFF,
                (code >> 20) & 0x0F,
                (code >> 16) & 0x0F,
                (code >> 8) & 0x0F,
                (code >> 4) & 0x0F,
                code & 0x0F,
            )

        raw_only_candidates: list[int] = []
        fallback_candidates: list[int] = []
        for code in codes:
            (
                main_size,
                main_type,
                main_quality,
                secondary_size,
                secondary_type,
                secondary_quality,
            ) = split(code)
            del main_size
            has_raw_main = main_type in {0x4, 0x6}
            secondary_unknown = secondary_size == 0xF and secondary_quality == 0xF
            secondary_absent = secondary_unknown and secondary_type in {0x0, 0xF}
            if has_raw_main and secondary_absent:
                raw_only_candidates.append(code)
            elif has_raw_main:
                fallback_candidates.append(code)
        if raw_only_candidates:
            return sorted(raw_only_candidates)[0]
        if fallback_candidates:
            return sorted(fallback_candidates)[0]
        return None

    def _drain_capture_results(self) -> None:
        while True:
            try:
                self._capture_results.get_nowait()
            except queue.Empty:
                return

    def _get_pending_capture_result(self) -> Path | None:
        try:
            result = self._capture_results.get_nowait()
        except queue.Empty:
            return None
        if isinstance(result, Exception):
            raise result
        return cast(Path, result)

    def _download_directory_item(
        self,
        directory_item: object,
        *,
        index: int,
    ) -> Path:
        info = self.sdk.get_directory_item_info(directory_item)
        output_path = self._build_output_path(
            info.file_name,
            index=index,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.sdk.create_file_stream(output_path)
        try:
            self.sdk.download(
                directory_item,
                info.size_bytes,
                stream,
            )
            self.sdk.download_complete(directory_item)
        finally:
            try:
                self.sdk.release(stream)
            finally:
                self.sdk.release(directory_item)
        return output_path

    def _build_output_path(
        self,
        original_name: str,
        *,
        index: int,
    ) -> Path:
        source_path = Path(original_name or "")
        suffix = source_path.suffix.lower() or ".jpg"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        candidate = self.config.output_dir / (
            f"{self.config.basename}-{stamp}-{index + 1:03d}{suffix}"
        )
        collision_index = 1
        while candidate.exists():
            candidate = self.config.output_dir / (
                f"{self.config.basename}-{stamp}-{index + 1:03d}-{collision_index}{suffix}"
            )
            collision_index += 1
        return candidate

    def _build_object_callback(self) -> object:
        def callback(
            event: int,
            directory_item: object,
            context: object,
        ) -> int:
            del context
            if event == K_EDS_OBJECT_EVENT_DIR_ITEM_REQUEST_TRANSFER and directory_item:
                # Retain the borrowed reference for the deferred transfer.
                # ctypes callbacks swallow exceptions, so route failures to
                # the results queue and always release the waiting caller.
                try:
                    self.sdk.retain(directory_item)
                    self._pending_items.put(
                        (
                            directory_item,
                            self._active_capture_index,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    self._capture_results.put(exc)
                finally:
                    self.image_ready.set()
            return EDS_ERR_OK

        return self.sdk.object_callback_type(callback)

    def _build_state_callback(self) -> object:
        def callback(
            event: int,
            parameter: int,
            context: object,
        ) -> int:
            del parameter, context
            if event == K_EDS_STATE_EVENT_WILL_SOON_SHUTDOWN:
                self._shutdown_event.set()
            return EDS_ERR_OK

        return self.sdk.state_callback_type(callback)


def capture_images(config: CanonCaptureConfig) -> list[Path]:
    sdk = CanonEdsdk(config.edsdk_lib)
    with CanonCamera(sdk, config) as camera:
        captures = camera.capture_many()
        for path in captures:
            print(path)
        return captures


def keep_camera_awake(
    edsdk_lib: str | None = None,
    *,
    sdk: CanonEdsdk | None = None,
) -> None:
    """Park the camera in PC-remote mode and deliberately leave it there."""
    if sdk is None:
        sdk = CanonEdsdk(edsdk_lib)
    sdk.initialize_sdk()
    camera_list = sdk.get_camera_list()
    if sdk.get_child_count(camera_list) < 1:
        raise CanonError(
            "No Canon camera detected through EDSDK. If it auto-powered-off it "
            "is off the USB bus -- wake it physically (half-press the shutter), "
            "then rerun."
        )
    camera = sdk.get_child_at_index(camera_list, 0)
    sdk.open_session(camera)
    # No close_session/terminate_sdk by design: a clean teardown returns the
    # camera to its UI where the auto-power-off timer runs.


def release_camera(
    edsdk_lib: str | None = None,
    *,
    sdk: CanonEdsdk | None = None,
) -> None:
    """Return a parked camera to its normal UI and sleep-timer behavior."""
    if sdk is None:
        sdk = CanonEdsdk(edsdk_lib)
    sdk.initialize_sdk()
    try:
        camera_list = sdk.get_camera_list()
        if sdk.get_child_count(camera_list) < 1:
            raise CanonError(
                "No Canon camera detected through EDSDK (already asleep or "
                "unplugged -- nothing to release)."
            )
        camera = sdk.get_child_at_index(camera_list, 0)
        sdk.open_session(camera)
        sdk.close_session(camera)
        sdk.release(camera)
        sdk.release(camera_list)
    finally:
        try:
            sdk.terminate_sdk()
        except CanonError:
            pass
