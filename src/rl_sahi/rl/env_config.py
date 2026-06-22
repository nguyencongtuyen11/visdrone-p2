from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class EnvConfig:
    max_steps: int = 20
    max_slices: int = 8
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
    use_gpu_box_ops: bool = True
    gpu_box_device: str = "cuda"
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


@dataclass(slots=True)
class StepResult:
    state: np.ndarray
    reward: float
    done: bool
    info: dict
