"""Per-image skin masks as seen by sim-capture.

Masks come from ``sim_capture.tools.skin_masks`` (or anything else that writes
``<image stem>.png``, 255 = region of interest). Here they decide which sparse
COLMAP points sit on the subject, so scale and surface axes refer to it and not to
whatever else is in view (a table, a floor).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from .model import SparseModel

MASK_SUBDIR = "masks"
MIN_OBSERVATION_FRACTION = 0.5  # a point is "on skin" when most of its observations are


class MaskError(ValueError):
    pass


def _mask_path(mask_dir: Path, image_name: str) -> Path:
    return mask_dir / f"{Path(image_name).stem}.png"


def check_masks(mask_dir: Path, image_names: list[str]) -> None:
    if not mask_dir.is_dir():
        raise MaskError(f"mask directory does not exist: {mask_dir}")
    missing = [name for name in image_names if not _mask_path(mask_dir, name).exists()]
    if missing:
        raise MaskError(
            f"{len(missing)} image(s) have no mask in {mask_dir} (expected <stem>.png), e.g. {', '.join(missing[:3])}"
        )


def skin_point_filter(model: SparseModel, mask_dir: Path) -> NDArray[np.bool_]:
    """(m,) bool: sparse points whose observations mostly land inside the masks."""
    if len(model.visible_xy) != len(model.image_names):
        raise MaskError("the reconstruction carries no 2D observations; cannot apply masks")
    inside = np.zeros(len(model.points), np.int64)
    seen = np.zeros(len(model.points), np.int64)
    for index, name in enumerate(model.image_names):
        mask = cv2.imread(str(_mask_path(mask_dir, name)), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise MaskError(f"cannot read mask {_mask_path(mask_dir, name)}")
        if mask.shape != (model.height, model.width):
            mask = cv2.resize(mask, (model.width, model.height), interpolation=cv2.INTER_NEAREST)
        xy = model.visible_xy[index]
        columns = np.clip(np.round(xy[:, 0]).astype(int), 0, model.width - 1)
        rows = np.clip(np.round(xy[:, 1]).astype(int), 0, model.height - 1)
        np.add.at(inside, model.visible_points[index], (mask[rows, columns] > 127).astype(np.int64))
        np.add.at(seen, model.visible_points[index], 1)
    return (seen > 0) & (inside >= MIN_OBSERVATION_FRACTION * np.maximum(seen, 1))


def install_masks(mask_dir: Path, dest_dir: Path, image_names: list[str]) -> None:
    """Copy the masks of the registered images next to the capture (processing's --mask-dir)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in image_names:
        shutil.copyfile(_mask_path(mask_dir, name), _mask_path(dest_dir, name))
