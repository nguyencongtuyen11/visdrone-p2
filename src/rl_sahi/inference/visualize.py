from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from rl_sahi.common.boxes import as_boxes
from rl_sahi.common.data import read_image


def draw_boxes(image: np.ndarray, boxes: np.ndarray, color: tuple[int, int, int], thickness: int = 1) -> None:
    for box in as_boxes(boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    sources: np.ndarray,
    full_color: tuple[int, int, int] = (0, 190, 0),
    slice_color: tuple[int, int, int] = (255, 120, 0),
) -> None:
    boxes = as_boxes(boxes)
    sources = np.asarray(sources, dtype=np.int32).reshape(-1)
    if len(boxes) == 0:
        return
    draw_boxes(image, boxes[sources == 0], full_color, thickness=1)
    draw_boxes(image, boxes[sources != 0], slice_color, thickness=1)


def save_inference_visual(
    image_path: Path,
    boxes: np.ndarray,
    sources: np.ndarray,
    accepted_rois: np.ndarray,
    rejected_rois: np.ndarray,
    out_path: Path,
) -> None:
    image = read_image(image_path)
    draw_detections(image, boxes, sources)
    draw_boxes(image, rejected_rois, (0, 165, 255), thickness=2)
    draw_boxes(image, accepted_rois, (0, 0, 255), thickness=2)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)
