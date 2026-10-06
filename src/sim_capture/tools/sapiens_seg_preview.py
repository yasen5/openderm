"""Preview what Sapiens2 body-part segmentation treats as skin in a few images.

A quick look, not part of the pipeline: for each image it writes one panel with the
original, the full 29-class map, and the "skin" mask (body-part classes only) over
the photo, plus a per-class pixel share so you can judge whether the arm, hand and
background (here a metal mesh) are separated well enough to mask COLMAP/processing.

    python -m sim_capture.tools.sapiens_seg_preview IMG [IMG ...] --out DIR

Needs torch, transformers and opencv. The checkpoint is read from the local
Hugging Face cache (``--model``, default facebook/sapiens2-seg-0.4b).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

# Class ids from the Sapiens2 body-part segmentation guide (docs/SEG.md).
CLASS_NAMES = [
    "Background", "Apparel", "Eyeglass", "Face_Neck", "Hair",
    "Left_Foot", "Left_Hand", "Left_Lower_Arm", "Left_Lower_Leg", "Left_Shoe",
    "Left_Sock", "Left_Upper_Arm", "Left_Upper_Leg", "Lower_Clothing", "Right_Foot",
    "Right_Hand", "Right_Lower_Arm", "Right_Lower_Leg", "Right_Shoe", "Right_Sock",
    "Right_Upper_Arm", "Right_Upper_Leg", "Torso", "Upper_Clothing", "Lower_Lip",
    "Upper_Lip", "Lower_Teeth", "Upper_Teeth", "Tongue",
]  # fmt: skip

# Classes that are bare skin: limbs, hands, feet, face/neck, torso, lips.
SKIN_CLASSES = frozenset(
    i
    for i, name in enumerate(CLASS_NAMES)
    if name.split("_", 1)[-1] in {"Foot", "Hand", "Lower_Arm", "Lower_Leg", "Upper_Arm", "Upper_Leg"}
    or name in {"Face_Neck", "Torso", "Lower_Lip", "Upper_Lip"}
)


def class_palette(count: int) -> NDArray[np.uint8]:
    """Distinct BGR colours per class (background black)."""
    hues = (np.arange(count) * 47) % 180
    hsv = np.stack([hues, np.full(count, 220), np.full(count, 255)], axis=-1).astype(np.uint8)
    palette = cv2.cvtColor(hsv[None], cv2.COLOR_HSV2BGR)[0]
    palette[0] = 0
    return palette


def predict_classes(images: Sequence[NDArray[np.uint8]], model_id: str, device: str) -> Iterator[NDArray[np.int64]]:
    """Per-pixel class ids at each image's own resolution (BGR uint8 in)."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_id).to(device).eval()
    for bgr in images:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = processor(images=rgb, return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits  # (1, classes, h, w) at the head's resolution
        logits = torch.nn.functional.interpolate(
            logits.float(), size=bgr.shape[:2], mode="bilinear", align_corners=False
        )
        yield logits.argmax(1)[0].cpu().numpy()


def render_panel(bgr: NDArray[np.uint8], labels: NDArray[np.int64], height: int = 900) -> NDArray[np.uint8]:
    palette = class_palette(len(CLASS_NAMES))
    colored = palette[np.clip(labels, 0, len(CLASS_NAMES) - 1)]
    skin = np.isin(labels, list(SKIN_CLASSES))

    classes_overlay = cv2.addWeighted(bgr, 0.45, colored, 0.55, 0)
    skin_overlay = bgr.copy()
    skin_overlay[~skin] = (skin_overlay[~skin] * 0.25).astype(np.uint8)  # dim everything that is not skin
    contours, _ = cv2.findContours(skin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(skin_overlay, contours, -1, (0, 255, 0), max(2, bgr.shape[1] // 400))

    # legend of the classes present, largest first, drawn on the class panel
    total = labels.size
    shares = [(float((labels == i).sum()) / total, i) for i in np.unique(labels)]
    scale = bgr.shape[1] / 720
    for row, (share, index) in enumerate(sorted(shares, reverse=True)[:8]):
        y = int((28 + 30 * row) * scale)
        cv2.rectangle(classes_overlay, (int(8 * scale), y - int(18 * scale)), (int(30 * scale), y), palette[index].tolist(), -1)
        text = f"{index} {CLASS_NAMES[index]} {100 * share:.0f}%"
        cv2.putText(classes_overlay, text, (int(38 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (255, 255, 255), max(1, int(2 * scale)), cv2.LINE_AA)

    def fit(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        resized = cv2.resize(image, (int(image.shape[1] * height / image.shape[0]), height), interpolation=cv2.INTER_AREA)
        return np.asarray(resized, dtype=np.uint8)

    return np.hstack([fit(bgr), fit(np.asarray(classes_overlay, dtype=np.uint8)), fit(skin_overlay)])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="directory for the preview panels")
    parser.add_argument("--model", default="facebook/sapiens2-seg-0.4b")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    images = []
    for path in args.images:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"cannot read {path}", file=sys.stderr)
            return 1
        images.append(image)

    args.out.mkdir(parents=True, exist_ok=True)
    for path, image, labels in zip(args.images, images, predict_classes(images, args.model, args.device), strict=True):
        skin_share = float(np.isin(labels, list(SKIN_CLASSES)).mean())
        present = {CLASS_NAMES[i]: round(float((labels == i).mean()), 3) for i in np.unique(labels)}
        print(f"{path.name}: skin {100 * skin_share:.1f}% of pixels; classes {present}")
        cv2.imwrite(str(args.out / f"{path.stem}_seg.jpg"), render_panel(image, labels))
    print(f"wrote {len(images)} panels to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
