"""Public façade for Canon EDSDK still-capture support."""

from __future__ import annotations

from ._canon.cli import build_parser, config_from_args, main
from ._canon.sdk import (
    DEFAULT_EDSDK_PATHS,
    EDS_ERR_OK,
    K_EDS_ACCESS_READ_WRITE,
    K_EDS_CAMERA_COMMAND_EXTEND_SHUTDOWN_TIMER,
    K_EDS_CAMERA_COMMAND_TAKE_PICTURE,
    K_EDS_FILE_CREATE_DISPOSITION_CREATE_ALWAYS,
    K_EDS_OBJECT_EVENT_ALL,
    K_EDS_OBJECT_EVENT_DIR_ITEM_REQUEST_TRANSFER,
    K_EDS_PROP_ID_AE_MODE,
    K_EDS_PROP_ID_AV,
    K_EDS_PROP_ID_IMAGE_QUALITY,
    K_EDS_PROP_ID_ISO_SPEED,
    K_EDS_PROP_ID_LENS_NAME,
    K_EDS_PROP_ID_SAVE_TO,
    K_EDS_PROP_ID_TV,
    K_EDS_PROP_ID_WHITE_BALANCE,
    K_EDS_SAVE_TO_HOST,
    K_EDS_STATE_EVENT_ALL,
    K_EDS_STATE_EVENT_WILL_SOON_SHUTDOWN,
    CanonEdsdk,
    CanonError,
    DirectoryItemInfo,
)
from ._canon.session import (
    CanonCamera,
    CanonCaptureConfig,
    capture_images,
    keep_camera_awake,
    release_camera,
)


__all__ = [
    "DEFAULT_EDSDK_PATHS",
    "EDS_ERR_OK",
    "K_EDS_ACCESS_READ_WRITE",
    "K_EDS_CAMERA_COMMAND_EXTEND_SHUTDOWN_TIMER",
    "K_EDS_CAMERA_COMMAND_TAKE_PICTURE",
    "K_EDS_FILE_CREATE_DISPOSITION_CREATE_ALWAYS",
    "K_EDS_OBJECT_EVENT_ALL",
    "K_EDS_OBJECT_EVENT_DIR_ITEM_REQUEST_TRANSFER",
    "K_EDS_PROP_ID_AE_MODE",
    "K_EDS_PROP_ID_AV",
    "K_EDS_PROP_ID_IMAGE_QUALITY",
    "K_EDS_PROP_ID_ISO_SPEED",
    "K_EDS_PROP_ID_LENS_NAME",
    "K_EDS_PROP_ID_SAVE_TO",
    "K_EDS_PROP_ID_TV",
    "K_EDS_PROP_ID_WHITE_BALANCE",
    "K_EDS_SAVE_TO_HOST",
    "K_EDS_STATE_EVENT_ALL",
    "K_EDS_STATE_EVENT_WILL_SOON_SHUTDOWN",
    "CanonCamera",
    "CanonCaptureConfig",
    "CanonEdsdk",
    "CanonError",
    "DirectoryItemInfo",
    "build_parser",
    "capture_images",
    "config_from_args",
    "keep_camera_awake",
    "main",
    "release_camera",
]


if __name__ == "__main__":
    raise SystemExit(main())
