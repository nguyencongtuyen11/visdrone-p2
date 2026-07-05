from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# giải thích: Lớp dữ liệu EnvConfig định nghĩa các siêu tham số cấu hình cho môi trường cắt lát SliceEnv
@dataclass(slots=True)
class EnvConfig:
    # giải thích: Các tham số giới hạn số bước đi và số lát cắt tối đa trong một tập (episode)
    max_steps: int = 20
    max_slices: int = 8
    
    # giải thích: Cấu hình kích thước và tỷ lệ thay đổi (dịch chuyển, zoom) của vùng cắt (ROI)
    initial_slice_fraction: float = 0.28
    move_fraction: float = 0.30
    zoom_factor: float = 0.75
    min_slice_fraction: float = 0.10
    max_slice_fraction: float = 0.35
    max_roi_area_ratio: float = 0.12
    min_scale_gain: float = 2.5
    reward_imgsz: int = 320
    target_projected_size: float = 32.0
    min_projected_size: float = 12.0
    max_projected_size: float = 96.0
    context_margin: float = 0.08
    high_conf_threshold: float = 0.5
    old_slice_overlap_threshold: float = 0.5
    min_new_hits_to_accept: int = 1
    
    # giải thích: Cấu hình tính toán hộp bọc bằng GPU để tăng tốc độ
    use_gpu_box_ops: bool = True
    gpu_box_device: str = "cuda"
    
    # giải thích: Thiết lập các hệ số và trọng số phạt / thưởng trong hàm phần thưởng (reward function)
    use_simplified_reward: bool = True
    target_reward: float = 0.75
    efficiency_weight: float = 0.5
    constraint_weight: float = 3.0
    stop_bonus_weight: float = 0.5
    step_penalty: float = 0.03
    empty_slice_penalty: float = 0.35
    area_penalty: float = 0.35
    detected_overlap_penalty: float = 1.0
    attempted_overlap_penalty: float = 2.0
    new_hard_reward: float = 0.5
    hard_density_reward: float = 0.2
    compactness_reward: float = 0.2
    observable_target_reward: float = 0.1
    continue_target_penalty: float = 0.3
    max_steps_without_stop_penalty: float = 4.0
    stalled_without_stop_penalty: float = 4.0
    stop_target_reward: float = 0.4
    stop_observable_target_reward: float = 0.15
    stop_early_penalty: float = 0.8
    large_roi_penalty: float = 2.0
    low_scale_penalty: float = 1.0
    old_slice_overlap_penalty: float = 3.0


# giải thích: Lớp dữ liệu chứa kết quả phản hồi sau mỗi bước hành động (step) của agent trên môi trường
@dataclass(slots=True)
class StepResult:
    state: np.ndarray
    reward: float
    done: bool
    info: dict
