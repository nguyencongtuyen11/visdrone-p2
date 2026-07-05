from __future__ import annotations

import numpy as np

from rl_sahi.rl.state_config import BASE_MAP_CHANNELS, SUMMARY_DIM, StateConfig, StateLayout


# giải thích: Hàm khởi tạo bố cục trạng thái (StateLayout), tính toán tổng kích thước vector trạng thái state_dim
def make_state_layout(feature_dim: int, spatial_map_channels: int, grid_size: int) -> StateLayout:
    map_channels = BASE_MAP_CHANNELS + int(spatial_map_channels)
    state_dim = int(feature_dim) + map_channels * int(grid_size) * int(grid_size) + SUMMARY_DIM
    return StateLayout(
        state_dim=state_dim,
        feature_dim=int(feature_dim),
        grid_size=int(grid_size),
        map_channels=int(map_channels),
        summary_dim=SUMMARY_DIM,
    )


# giải thích: Hàm trích xuất thông tin đặc trưng từ dữ liệu phát hiện và tạo đối tượng StateLayout
def state_layout_from_detection(detection, state_cfg: StateConfig) -> StateLayout:
    spatial = np.asarray(detection.spatial_feature_map, dtype=np.float32)
    spatial_channels = 0 if spatial.size == 0 else int(spatial.reshape(-1, state_cfg.grid_size, state_cfg.grid_size).shape[0])
    return make_state_layout(
        feature_dim=int(np.asarray(detection.feature, dtype=np.float32).reshape(-1).shape[0]),
        spatial_map_channels=spatial_channels,
        grid_size=state_cfg.grid_size,
    )
