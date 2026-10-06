"""Write one skin mask per image with Sapiens2 body-part segmentation.

Every body-part class is merged into a single "skin" class (see
``sapiens_seg_preview.SKIN_CLASSES``); background, clothing, hair, shoes and teeth
are not skin. The mask is cleaned so a few mislabelled pixels inside an arm do not
leave holes, then eroded a little so the arm's silhouette edge, where the surface
jumps in depth, carries no weight.

    python -m sim_capture.tools.skin_masks photos/ --out masks/

Masks are 8-bit PNGs named ``<image stem>.png`` at each image's own resolution
(255 = skin). They feed ``openderm-sim-capture --skin-masks`` and, through it,
``register_scan_3d --mask-dir``. Needs torch + transformers (see
``sapiens_seg_preview``); sim-capture itself does not.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from .sapiens_seg_preview import SKIN_CLASSES, predict_classes

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MIN_COMPONENT_FRACTION = 0.01  # of the image: smaller islands are not the subject
OUTLIER_RATIO = 0.5  # a frame whose skin share differs from the median by more is flagged


def clean_skin_mask(labels: NDArray[np.int64], erode_fraction: float = 0.004) -> NDArray[np.uint8]:
    """Merged, hole-filled, island-free, slightly eroded skin mask (255 = skin)."""
    height, width = labels.shape
    skin = np.isin(labels, list(SKIN_CLASSES)).astype(np.uint8)

    close_px = max(3, int(round(0.01 * width))) | 1  # bridges the model's patch-grid speckle
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px)))

    count, component, stats, _ = cv2.connectedComponentsWithStats(skin, connectivity=8)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= MIN_COMPONENT_FRACTION * height * width
    skin = keep[component].astype(np.uint8)

    # fill enclosed holes (background components that do not touch the border)
    inverse = (1 - skin).astype(np.uint8)
    count, component, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=4)
    border = set(np.unique(np.concatenate([component[0], component[-1], component[:, 0], component[:, -1]])))
    enclosed = np.array([label not in border and label != 0 for label in range(count)])
    skin = np.where(enclosed[component], 1, skin).astype(np.uint8)

    erode_px = int(round(erode_fraction * width))
    if erode_px > 0:
        skin = cv2.erode(skin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1)))
    return np.asarray(skin * 255, dtype=np.uint8)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="directory for <stem>.png masks")
    parser.add_argument("--model", default="facebook/sapiens2-seg-0.4b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--erode-fraction", type=float, default=0.004, help="erosion radius as a fraction of image width")
    args = parser.parse_args(argv)

    paths = sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        print(f"no images in {args.image_dir}", file=sys.stderr)
        return 1
    images = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            print(f"cannot read {path}", file=sys.stderr)
            return 1
        images.append(image)

    args.out.mkdir(parents=True, exist_ok=True)
    shares: list[float] = []
    for path, labels in zip(paths, predict_classes(images, args.model, args.device), strict=True):
        mask = clean_skin_mask(labels, args.erode_fraction)
        cv2.imwrite(str(args.out / f"{path.stem}.png"), mask)
        shares.append(float((mask > 0).mean()))

    median = float(np.median(shares))
    flagged = [p.name for p, s in zip(paths, shares, strict=True) if abs(s - median) > OUTLIER_RATIO * max(median, 1e-6)]
    print(f"wrote {len(paths)} masks to {args.out}; skin share median {100 * median:.1f}% "
          f"(min {100 * min(shares):.1f}%, max {100 * max(shares):.1f}%)")
    if flagged:
        print(f"check these frames, their skin share is far from the median: {', '.join(flagged)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
