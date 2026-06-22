from __future__ import annotations

import numpy as np

from rl_sahi.common.box_types import EPS, as_boxes


def rasterize_boxes(
    boxes: np.ndarray,
    image_shape: tuple[int, int],
    grid_size: int,
    values: np.ndarray | None = None,
    fill_mode: str = "max",
) -> np.ndarray:
    boxes = as_boxes(boxes)
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    if len(boxes) == 0:
        return grid
    h, w = image_shape
    if values is None:
        values = np.ones((len(boxes),), dtype=np.float32)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    for box, value in zip(boxes, values):
        x1 = int(np.floor((box[0] / max(w, EPS)) * grid_size))
        y1 = int(np.floor((box[1] / max(h, EPS)) * grid_size))
        x2 = int(np.ceil((box[2] / max(w, EPS)) * grid_size))
        y2 = int(np.ceil((box[3] / max(h, EPS)) * grid_size))
        x1 = int(np.clip(x1, 0, grid_size - 1))
        y1 = int(np.clip(y1, 0, grid_size - 1))
        x2 = int(np.clip(x2, x1 + 1, grid_size))
        y2 = int(np.clip(y2, y1 + 1, grid_size))
        if fill_mode == "add":
            grid[y1:y2, x1:x2] += value
        else:
            grid[y1:y2, x1:x2] = np.maximum(grid[y1:y2, x1:x2], value)
    return np.clip(grid, 0.0, 1.0)
