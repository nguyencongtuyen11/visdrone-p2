from __future__ import annotations

import numpy as np


EPS = 1e-9


def as_boxes(boxes: np.ndarray) -> np.ndarray:
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return arr.reshape(-1, 4).astype(np.float32, copy=False)
