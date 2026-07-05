from __future__ import annotations

import numpy as np

from rl_sahi.common.boxes import area, as_boxes, rasterize_boxes
from rl_sahi.rl.state_config import DETECTION_MAP_CHANNELS, StateConfig


# giải thích: Hàm tạo mặt nạ boolean lọc ra các hộp đề xuất nằm trong ngưỡng độ tin cậy được cấu hình
def proposal_mask(scores: np.ndarray, cfg: StateConfig) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    return (scores >= cfg.proposal_min_conf) & (scores < cfg.proposal_max_conf)


# giải thích: Hàm đánh giá chất lượng đề xuất bằng hàm tam giác (peak là 1.0 tại ngưỡng peak_conf, giảm dần về hai phía)
def proposal_quality(scores: np.ndarray, cfg: StateConfig) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    peak = float(np.clip(cfg.proposal_peak_conf, cfg.proposal_min_conf, cfg.proposal_max_conf))
    below = (scores - cfg.proposal_min_conf) / max(peak - cfg.proposal_min_conf, 1e-6)
    above = (cfg.proposal_max_conf - scores) / max(cfg.proposal_max_conf - peak, 1e-6)
    return np.clip(np.minimum(below, above), 0.0, 1.0).astype(np.float32)


# giải thích: Hàm tạo bản đồ phát hiện (detection map) gồm 4 kênh tương ứng với các nhóm vật thể khác nhau phục vụ đầu vào Spatial CNN
def build_detection_map(
    boxes: np.ndarray,
    scores: np.ndarray,
    image_shape: tuple[int, int],
    cfg: StateConfig,
) -> np.ndarray:
    boxes = as_boxes(boxes)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    valid_mask = scores >= cfg.proposal_min_conf
    boxes = boxes[valid_mask]
    scores = scores[valid_mask]
    if len(boxes) == 0:
        return np.zeros((DETECTION_MAP_CHANNELS, cfg.grid_size, cfg.grid_size), dtype=np.float32)

    image_area = float(image_shape[0] * image_shape[1])
    areas = area(boxes) / max(image_area, 1.0)
    
    # giải thích: Kênh 1: Rasterize các vật thể có độ tin cậy cao
    high_mask = scores >= cfg.proposal_max_conf
    high_map = rasterize_boxes(boxes[high_mask], image_shape, cfg.grid_size, values=np.clip(scores[high_mask], 0.0, 1.0))

    # giải thích: Kênh 2: Rasterize các hộp đề xuất (proposal) đi kèm chất lượng
    prop_mask = proposal_mask(scores, cfg)
    prop_values = proposal_quality(scores[prop_mask], cfg)
    proposal_map = rasterize_boxes(boxes[prop_mask], image_shape, cfg.grid_size, values=prop_values)
    
    # giải thích: Kênh 3: Tính toán mật độ các hộp đề xuất trên lưới (cộng dồn các hộp đè lên nhau)
    density_values = np.full((int(prop_mask.sum()),), 1.0 / max(cfg.count_norm, 1.0), dtype=np.float32)
    proposal_density_map = rasterize_boxes(
        boxes[prop_mask],
        image_shape,
        cfg.grid_size,
        values=density_values,
        fill_mode="add",
    )

    # giải thích: Kênh 4: Rasterize các vật thể có diện tích nhỏ
    small_mask = areas <= cfg.small_area_ratio
    small_values = np.maximum(np.clip(scores[small_mask], 0.0, 1.0), 0.25)
    small_map = rasterize_boxes(boxes[small_mask], image_shape, cfg.grid_size, values=small_values)

    return np.stack([high_map, proposal_map, proposal_density_map, small_map], axis=0).astype(np.float32)


# giải thích: Cập nhật bản đồ lịch sử ghi lại các vùng ROI mà tác nhân đã từng đi qua
def mark_history(history: np.ndarray, roi: np.ndarray, image_shape: tuple[int, int], grid_size: int) -> np.ndarray:
    history = np.asarray(history, dtype=np.float32).reshape(grid_size, grid_size).copy()
    roi_map = rasterize_boxes(np.asarray(roi, dtype=np.float32).reshape(1, 4), image_shape, grid_size)
    return np.maximum(history, roi_map).astype(np.float32)
