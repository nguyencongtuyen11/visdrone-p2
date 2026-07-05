from __future__ import annotations

from dataclasses import dataclass


# giải thích: Chiều dài phần tóm tắt thông tin phát hiện trong vector trạng thái
SUMMARY_DIM = 28
# giải thích: Số lượng kênh bản đồ biểu diễn các thuộc tính phát hiện vật thể (tọa độ, confidence, class, density)
DETECTION_MAP_CHANNELS = 4
# giải thích: Tổng số kênh bản đồ cơ sở (kết hợp lịch sử, ROI hiện tại, slice maps, objectness map, các kênh phát hiện)
BASE_MAP_CHANNELS = 1 + 1 + 1 + 1 + DETECTION_MAP_CHANNELS + 1


# giải thích: Lớp dữ liệu StateConfig chứa các thông số cấu hình không gian trạng thái (State)
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


# giải thích: Lớp dữ liệu StateLayout lưu trữ các thông tin kích thước và bố cục của vector trạng thái để truyền vào mạng QNetwork
@dataclass(slots=True)
class StateLayout:
    state_dim: int
    feature_dim: int
    grid_size: int
    map_channels: int
    summary_dim: int = SUMMARY_DIM
