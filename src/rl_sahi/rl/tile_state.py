"""Trích state THẤP CHIỀU từ SliceEnv để dùng với tile coding.

Bài toán gốc (di chuyển ROI) không đổi; đây chỉ là cách BIỂU DIỄN state gọn lại
(deep-net đọc bản đồ 16x16 -> vài đặc trưng vô hướng) để hàm Q tuyến tính + tile
coding học được. Chỉ ĐỌC thuộc tính công khai của env, không sửa SliceEnv.
"""
from __future__ import annotations

import numpy as np


NUM_TILE_FEATURES = 7


def tile_features(env) -> np.ndarray:
    """Trả về vector 7 chiều trong [0,1]:
    [cx, cy, scale, objectness@ROI, proposal_density@ROI, step_frac, old_overlap].
    """
    h, w = env.image_shape
    x1, y1, x2, y2 = [float(v) for v in np.asarray(env.roi, dtype=np.float32).reshape(4)]
    cx = np.clip(((x1 + x2) * 0.5) / max(w, 1.0), 0.0, 1.0)
    cy = np.clip(((y1 + y2) * 0.5) / max(h, 1.0), 0.0, 1.0)
    side = max(x2 - x1, y2 - y1, 1.0)
    scale = np.clip(side / max(min(h, w), 1.0), 0.0, 1.0)

    obj_roi = float(np.clip(env._objectness_roi_score(), 0.0, 1.0))

    # mật độ proposal trung bình trong cửa sổ ROI (kênh 2 của detection_map)
    dens_roi = 0.0
    dmap = np.asarray(env.detection_map, dtype=np.float32)
    if dmap.ndim == 3 and dmap.shape[0] > 2:
        y1g, y2g, x1g, x2g = env._roi_grid_window()
        window = dmap[2][y1g:y2g, x1g:x2g]
        if window.size:
            cn = float(getattr(env.state_cfg, "count_norm", 100.0))
            dens_roi = float(np.clip(window.mean() * cn / 10.0, 0.0, 1.0))

    step_frac = float(np.clip(env.step_index / max(env.env_cfg.max_steps, 1), 0.0, 1.0))
    overlap = float(np.clip(env._old_slice_overlap(), 0.0, 1.0))

    return np.array([cx, cy, scale, obj_roi, dens_roi, step_frac, overlap], dtype=np.float64)
