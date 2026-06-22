from __future__ import annotations

import numpy as np

from rl_sahi.common.box_types import as_boxes


def xywhn_to_xyxy(xywhn: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    boxes = np.asarray(xywhn, dtype=np.float32).reshape(-1, 4)
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    h, w = image_shape
    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    out = np.stack(
        [
            (cx - bw / 2.0) * w,
            (cy - bh / 2.0) * h,
            (cx + bw / 2.0) * w,
            (cy + bh / 2.0) * h,
        ],
        axis=1,
    )
    return clip_boxes(out, image_shape)


def xyxy_to_xywhn(boxes: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    boxes = as_boxes(boxes)
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    h, w = image_shape
    bw = boxes[:, 2] - boxes[:, 0]
    bh = boxes[:, 3] - boxes[:, 1]
    cx = boxes[:, 0] + bw / 2.0
    cy = boxes[:, 1] + bh / 2.0
    return np.stack([cx / w, cy / h, bw / w, bh / h], axis=1).astype(np.float32)


def clip_boxes(boxes: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    boxes = as_boxes(boxes).copy()
    if boxes.size == 0:
        return boxes
    h, w = image_shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, max(0, w - 1))
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, max(0, h - 1))
    boxes[:, 2] = np.maximum(boxes[:, 2], boxes[:, 0] + 1.0)
    boxes[:, 3] = np.maximum(boxes[:, 3], boxes[:, 1] + 1.0)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return boxes.astype(np.float32)


def box_from_center(
    cx: float,
    cy: float,
    side: float,
    image_shape: tuple[int, int],
) -> np.ndarray:
    half = side / 2.0
    return clip_boxes(np.array([[cx - half, cy - half, cx + half, cy + half]], dtype=np.float32), image_shape)[0]


def translate_box(box: np.ndarray, dx: float, dy: float, image_shape: tuple[int, int]) -> np.ndarray:
    b = as_boxes(box).reshape(1, 4)[0].copy()
    width = b[2] - b[0]
    height = b[3] - b[1]
    h, w = image_shape
    b[[0, 2]] += dx
    b[[1, 3]] += dy
    if b[0] < 0:
        b[[0, 2]] -= b[0]
    if b[1] < 0:
        b[[1, 3]] -= b[1]
    if b[2] > w:
        b[[0, 2]] -= b[2] - w
    if b[3] > h:
        b[[1, 3]] -= b[3] - h
    b[2] = b[0] + width
    b[3] = b[1] + height
    return clip_boxes(b.reshape(1, 4), image_shape)[0]


def zoom_box(
    box: np.ndarray,
    factor: float,
    image_shape: tuple[int, int],
    min_side: float,
    max_side: float,
) -> np.ndarray:
    b = as_boxes(box).reshape(1, 4)[0]
    cx = float((b[0] + b[2]) / 2.0)
    cy = float((b[1] + b[3]) / 2.0)
    side = max(float(b[2] - b[0]), float(b[3] - b[1])) * factor
    side = float(np.clip(side, min_side, max_side))
    return box_from_center(cx, cy, side, image_shape)
