"""Low-level ctypes bindings for Canon's EDSDK."""

from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass
from pathlib import Path


EDS_ERR_OK = 0x00000000
K_EDS_PROP_ID_SAVE_TO = 0x0000000B
K_EDS_PROP_ID_IMAGE_QUALITY = 0x00000100
K_EDS_PROP_ID_WHITE_BALANCE = 0x00000106
K_EDS_PROP_ID_AE_MODE = 0x00000400
K_EDS_PROP_ID_ISO_SPEED = 0x00000402
K_EDS_PROP_ID_AV = 0x00000405
K_EDS_PROP_ID_TV = 0x00000406
K_EDS_PROP_ID_LENS_NAME = 0x0000040D
K_EDS_SAVE_TO_HOST = 2
K_EDS_OBJECT_EVENT_ALL = 0x00000200
K_EDS_OBJECT_EVENT_DIR_ITEM_REQUEST_TRANSFER = 0x00000208
K_EDS_STATE_EVENT_ALL = 0x00000300
K_EDS_STATE_EVENT_WILL_SOON_SHUTDOWN = 0x00000303
K_EDS_CAMERA_COMMAND_TAKE_PICTURE = 0x00000000
K_EDS_CAMERA_COMMAND_EXTEND_SHUTDOWN_TIMER = 0x00000001
K_EDS_FILE_CREATE_DISPOSITION_CREATE_ALWAYS = 1
K_EDS_ACCESS_READ_WRITE = 2
DEFAULT_EDSDK_PATHS = (
    "/usr/local/lib/libEDSDK.so",
    "/usr/lib/libEDSDK.so",
    "/opt/canon/edsdk/libEDSDK.so",
    "/opt/canon/EDSDK/libEDSDK.so",
)


class CanonError(RuntimeError):
    """Raised when Canon camera control through EDSDK fails."""


@dataclass(frozen=True)
class DirectoryItemInfo:
    size_bytes: int
    file_name: str


class _EdsDirectoryItemInfo(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint64),
        ("isFolder", ctypes.c_int32),
        ("groupID", ctypes.c_uint32),
        ("option", ctypes.c_uint32),
        ("szFileName", ctypes.c_char * 256),
        ("format", ctypes.c_uint32),
        ("dateTime", ctypes.c_uint32),
    ]


class _EdsCapacity(ctypes.Structure):
    _fields_ = [
        ("numberOfFreeClusters", ctypes.c_int32),
        ("bytesPerSector", ctypes.c_uint32),
        ("reset", ctypes.c_int32),
    ]


class _EdsPropertyDesc(ctypes.Structure):
    _fields_ = [
        ("form", ctypes.c_int32),
        ("access", ctypes.c_int32),
        ("numElements", ctypes.c_int32),
        ("propDesc", ctypes.c_int32 * 128),
    ]


class CanonEdsdk:
    """Thin, typed wrapper around the native EDSDK library."""

    def __init__(self, library_path: str | None = None) -> None:
        self.library_path = self._resolve_library_path(library_path)
        self._lib = ctypes.CDLL(self.library_path)
        self._configure_signatures()

    @staticmethod
    def _resolve_library_path(explicit_path: str | None) -> str:
        candidates: list[str] = []
        if explicit_path:
            candidates.append(explicit_path)
        detected = ctypes.util.find_library("EDSDK")
        if detected:
            candidates.append(detected)
        candidates.extend(DEFAULT_EDSDK_PATHS)
        for candidate in candidates:
            if candidate and ("/" not in candidate or Path(candidate).exists()):
                return candidate
        checked = ", ".join(dict.fromkeys(candidates)) or "libEDSDK.so"
        raise CanonError(
            "Unable to locate Canon EDSDK. Install the Linux/Raspberry Pi OS SDK from Canon and "
            f"pass --edsdk-lib if needed. Checked: {checked}"
        )

    def _configure_signatures(self) -> None:
        void_p = ctypes.c_void_p
        uint32 = ctypes.c_uint32

        object_cb = ctypes.CFUNCTYPE(uint32, uint32, void_p, void_p)
        state_cb = ctypes.CFUNCTYPE(uint32, uint32, uint32, void_p)
        self.object_callback_type = object_cb
        self.state_callback_type = state_cb

        self._lib.EdsInitializeSDK.restype = uint32
        self._lib.EdsTerminateSDK.restype = uint32
        self._lib.EdsGetEvent.restype = uint32
        self._lib.EdsRelease.argtypes = [void_p]
        self._lib.EdsRelease.restype = uint32
        self._lib.EdsRetain.argtypes = [void_p]
        self._lib.EdsRetain.restype = uint32
        self._lib.EdsGetCameraList.argtypes = [ctypes.POINTER(void_p)]
        self._lib.EdsGetCameraList.restype = uint32
        self._lib.EdsGetChildCount.argtypes = [void_p, ctypes.POINTER(uint32)]
        self._lib.EdsGetChildCount.restype = uint32
        self._lib.EdsGetChildAtIndex.argtypes = [
            void_p,
            ctypes.c_int32,
            ctypes.POINTER(void_p),
        ]
        self._lib.EdsGetChildAtIndex.restype = uint32
        self._lib.EdsOpenSession.argtypes = [void_p]
        self._lib.EdsOpenSession.restype = uint32
        self._lib.EdsCloseSession.argtypes = [void_p]
        self._lib.EdsCloseSession.restype = uint32
        self._lib.EdsSetObjectEventHandler.argtypes = [
            void_p,
            uint32,
            object_cb,
            void_p,
        ]
        self._lib.EdsSetObjectEventHandler.restype = uint32
        self._lib.EdsSetCameraStateEventHandler.argtypes = [
            void_p,
            uint32,
            state_cb,
            void_p,
        ]
        self._lib.EdsSetCameraStateEventHandler.restype = uint32
        self._lib.EdsSetPropertyData.argtypes = [
            void_p,
            uint32,
            ctypes.c_int32,
            uint32,
            void_p,
        ]
        self._lib.EdsSetPropertyData.restype = uint32
        self._lib.EdsGetPropertySize.argtypes = [
            void_p,
            uint32,
            ctypes.c_int32,
            ctypes.POINTER(uint32),
            ctypes.POINTER(uint32),
        ]
        self._lib.EdsGetPropertySize.restype = uint32
        self._lib.EdsGetPropertyData.argtypes = [
            void_p,
            uint32,
            ctypes.c_int32,
            uint32,
            void_p,
        ]
        self._lib.EdsGetPropertyData.restype = uint32
        self._lib.EdsGetPropertyDesc.argtypes = [
            void_p,
            uint32,
            ctypes.POINTER(_EdsPropertyDesc),
        ]
        self._lib.EdsGetPropertyDesc.restype = uint32
        self._lib.EdsSetCapacity.argtypes = [void_p, _EdsCapacity]
        self._lib.EdsSetCapacity.restype = uint32
        self._lib.EdsSendCommand.argtypes = [void_p, uint32, ctypes.c_int32]
        self._lib.EdsSendCommand.restype = uint32
        self._lib.EdsGetDirectoryItemInfo.argtypes = [
            void_p,
            ctypes.POINTER(_EdsDirectoryItemInfo),
        ]
        self._lib.EdsGetDirectoryItemInfo.restype = uint32
        self._lib.EdsCreateFileStream.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(void_p),
        ]
        self._lib.EdsCreateFileStream.restype = uint32
        self._lib.EdsDownload.argtypes = [void_p, ctypes.c_uint64, void_p]
        self._lib.EdsDownload.restype = uint32
        self._lib.EdsDownloadComplete.argtypes = [void_p]
        self._lib.EdsDownloadComplete.restype = uint32

    def initialize_sdk(self) -> None:
        self._check(self._lib.EdsInitializeSDK(), "EdsInitializeSDK")

    def terminate_sdk(self) -> None:
        self._check(self._lib.EdsTerminateSDK(), "EdsTerminateSDK")

    def get_event(self) -> None:
        self._check(self._lib.EdsGetEvent(), "EdsGetEvent")

    def retain(self, ref: object) -> None:
        # EdsRetain/EdsRelease return the resulting reference count, not an
        # EDS_ERR code. A result of 0xFFFFFFFF is the failure sentinel.
        result = self._lib.EdsRetain(ctypes.c_void_p(ref))
        if result == 0xFFFFFFFF:
            raise CanonError("EdsRetain failed (object not retainable).")

    def release(self, ref: object) -> None:
        result = self._lib.EdsRelease(ctypes.c_void_p(ref))
        if result == 0xFFFFFFFF:
            raise CanonError("EdsRelease failed (object not releasable).")

    def get_camera_list(self) -> object:
        ref = ctypes.c_void_p()
        self._check(
            self._lib.EdsGetCameraList(ctypes.byref(ref)),
            "EdsGetCameraList",
        )
        return ref.value

    def get_child_count(self, ref: object) -> int:
        count = ctypes.c_uint32()
        self._check(
            self._lib.EdsGetChildCount(
                ctypes.c_void_p(ref),
                ctypes.byref(count),
            ),
            "EdsGetChildCount",
        )
        return int(count.value)

    def get_child_at_index(self, ref: object, index: int) -> object:
        child = ctypes.c_void_p()
        self._check(
            self._lib.EdsGetChildAtIndex(
                ctypes.c_void_p(ref),
                index,
                ctypes.byref(child),
            ),
            "EdsGetChildAtIndex",
        )
        return child.value

    def open_session(self, camera: object) -> None:
        self._check(
            self._lib.EdsOpenSession(ctypes.c_void_p(camera)),
            "EdsOpenSession",
        )

    def close_session(self, camera: object) -> None:
        self._check(
            self._lib.EdsCloseSession(ctypes.c_void_p(camera)),
            "EdsCloseSession",
        )

    def set_object_handler(self, camera: object, callback: object) -> None:
        self._check(
            self._lib.EdsSetObjectEventHandler(
                ctypes.c_void_p(camera),
                K_EDS_OBJECT_EVENT_ALL,
                callback,
                None,
            ),
            "EdsSetObjectEventHandler",
        )

    def set_state_handler(self, camera: object, callback: object) -> None:
        self._check(
            self._lib.EdsSetCameraStateEventHandler(
                ctypes.c_void_p(camera),
                K_EDS_STATE_EVENT_ALL,
                callback,
                None,
            ),
            "EdsSetCameraStateEventHandler",
        )

    def set_save_to_host(self, camera: object) -> None:
        value = ctypes.c_uint32(K_EDS_SAVE_TO_HOST)
        self.set_property_uint32(
            camera,
            K_EDS_PROP_ID_SAVE_TO,
            int(value.value),
            operation_name="kEdsPropID_SaveTo",
        )

    def get_property_size(
        self,
        camera: object,
        property_id: int,
        *,
        parameter: int = 0,
    ) -> tuple[int, int]:
        data_type = ctypes.c_uint32()
        size = ctypes.c_uint32()
        self._check(
            self._lib.EdsGetPropertySize(
                ctypes.c_void_p(camera),
                property_id,
                parameter,
                ctypes.byref(data_type),
                ctypes.byref(size),
            ),
            f"EdsGetPropertySize(0x{property_id:08X})",
        )
        return int(data_type.value), int(size.value)

    def get_property_uint32(
        self,
        camera: object,
        property_id: int,
        *,
        parameter: int = 0,
    ) -> int:
        value = ctypes.c_uint32()
        self._check(
            self._lib.EdsGetPropertyData(
                ctypes.c_void_p(camera),
                property_id,
                parameter,
                ctypes.sizeof(value),
                ctypes.byref(value),
            ),
            f"EdsGetPropertyData(0x{property_id:08X})",
        )
        return int(value.value)

    def get_property_string(
        self,
        camera: object,
        property_id: int,
        *,
        parameter: int = 0,
    ) -> str:
        _, size = self.get_property_size(
            camera,
            property_id,
            parameter=parameter,
        )
        buffer = ctypes.create_string_buffer(max(1, size))
        self._check(
            self._lib.EdsGetPropertyData(
                ctypes.c_void_p(camera),
                property_id,
                parameter,
                ctypes.sizeof(buffer),
                buffer,
            ),
            f"EdsGetPropertyData(0x{property_id:08X})",
        )
        return buffer.value.decode("utf-8", errors="ignore")

    def set_property_uint32(
        self,
        camera: object,
        property_id: int,
        value: int,
        *,
        parameter: int = 0,
        operation_name: str | None = None,
    ) -> None:
        raw_value = ctypes.c_uint32(value)
        self._check(
            self._lib.EdsSetPropertyData(
                ctypes.c_void_p(camera),
                property_id,
                parameter,
                ctypes.sizeof(raw_value),
                ctypes.byref(raw_value),
            ),
            f"EdsSetPropertyData({operation_name or f'0x{property_id:08X}'})",
        )

    def get_property_desc(
        self,
        camera: object,
        property_id: int,
    ) -> tuple[int, ...]:
        description = _EdsPropertyDesc()
        self._check(
            self._lib.EdsGetPropertyDesc(
                ctypes.c_void_p(camera),
                property_id,
                ctypes.byref(description),
            ),
            f"EdsGetPropertyDesc(0x{property_id:08X})",
        )
        count = max(
            0,
            min(
                int(description.numElements),
                len(description.propDesc),
            ),
        )
        return tuple(int(description.propDesc[index]) for index in range(count))

    def set_capacity(self, camera: object) -> None:
        capacity = _EdsCapacity(
            numberOfFreeClusters=0x7FFFFFFF,
            bytesPerSector=0x1000,
            reset=1,
        )
        self._check(
            self._lib.EdsSetCapacity(
                ctypes.c_void_p(camera),
                capacity,
            ),
            "EdsSetCapacity",
        )

    def send_take_picture(self, camera: object) -> None:
        self._check(
            self._lib.EdsSendCommand(
                ctypes.c_void_p(camera),
                K_EDS_CAMERA_COMMAND_TAKE_PICTURE,
                0,
            ),
            "EdsSendCommand(kEdsCameraCommand_TakePicture)",
        )

    def extend_shutdown_timer(self, camera: object) -> None:
        self._check(
            self._lib.EdsSendCommand(
                ctypes.c_void_p(camera),
                K_EDS_CAMERA_COMMAND_EXTEND_SHUTDOWN_TIMER,
                0,
            ),
            "EdsSendCommand(kEdsCameraCommand_ExtendShutDownTimer)",
        )

    def get_directory_item_info(
        self,
        directory_item: object,
    ) -> DirectoryItemInfo:
        info = _EdsDirectoryItemInfo()
        self._check(
            self._lib.EdsGetDirectoryItemInfo(
                ctypes.c_void_p(directory_item),
                ctypes.byref(info),
            ),
            "EdsGetDirectoryItemInfo",
        )
        file_name = (
            bytes(info.szFileName)
            .split(b"\x00", 1)[0]
            .decode(
                "utf-8",
                errors="ignore",
            )
        )
        return DirectoryItemInfo(
            size_bytes=int(info.size),
            file_name=file_name,
        )

    def create_file_stream(self, path: Path) -> object:
        stream = ctypes.c_void_p()
        self._check(
            self._lib.EdsCreateFileStream(
                str(path).encode("utf-8"),
                K_EDS_FILE_CREATE_DISPOSITION_CREATE_ALWAYS,
                K_EDS_ACCESS_READ_WRITE,
                ctypes.byref(stream),
            ),
            "EdsCreateFileStream",
        )
        return stream.value

    def download(
        self,
        directory_item: object,
        size_bytes: int,
        stream: object,
    ) -> None:
        self._check(
            self._lib.EdsDownload(
                ctypes.c_void_p(directory_item),
                ctypes.c_uint64(size_bytes),
                ctypes.c_void_p(stream),
            ),
            "EdsDownload",
        )

    def download_complete(self, directory_item: object) -> None:
        self._check(
            self._lib.EdsDownloadComplete(
                ctypes.c_void_p(directory_item),
            ),
            "EdsDownloadComplete",
        )

    @staticmethod
    def _check(error_code: int, operation: str) -> None:
        if error_code != EDS_ERR_OK:
            raise CanonError(f"{operation} failed with Canon error 0x{error_code:08X}.")
