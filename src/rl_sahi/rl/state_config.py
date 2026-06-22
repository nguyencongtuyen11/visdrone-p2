from __future__ import annotations

from dataclasses import dataclass


SUMMARY_DIM = 28
DETECTION_MAP_CHANNELS = 4
BASE_MAP_CHANNELS = 1 + 1 + 1 + 1 + DETECTION_MAP_CHANNELS + 1


@dataclass(slots=True)
class StateConfig:
    grid_size: int = 16
    low_conf_threshold: float = 0.5
    proposal_min_conf: float = 0.01
    proposal_max_conf: float = 0.5
    proposal_peak_conf: float = 0.25
    small_area_ratio: float = 0.01
    count_norm: float = 100.0
    roi_count_norm: float = 50.0
    slice_count_norm: float = 10.0
    spatial_feature_channels: int = 4


@dataclass(slots=True)
class StateLayout:
    state_dim: int
    feature_dim: int
    grid_size: int
    map_channels: int
    summary_dim: int = SUMMARY_DIM
