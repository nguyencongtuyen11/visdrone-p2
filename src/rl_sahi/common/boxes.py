# Cho phép import các annotation nâng cao từ tương lai
from __future__ import annotations

# Import toàn bộ các hàm tiện ích hình học và biến đổi hộp giới hạn từ các module con
from rl_sahi.common.box_geometry import area, center_inside, centers, intersection_matrix, ioa_matrix, iou_matrix, normalized_box
from rl_sahi.common.box_transforms import box_from_center, clip_boxes, translate_box, xywhn_to_xyxy, xyxy_to_xywhn, zoom_box
from rl_sahi.common.box_types import EPS, as_boxes
from rl_sahi.common.nms import nms_numpy
from rl_sahi.common.raster import rasterize_boxes

# Khai báo các hàm và biến công khai được xuất khẩu (exposed) khi sử dụng `from rl_sahi.common.boxes import *`
__all__ = [
    "EPS",
    "area",
    "as_boxes",
    "box_from_center",
    "center_inside",
    "centers",
    "clip_boxes",
    "intersection_matrix",
    "ioa_matrix",
    "iou_matrix",
    "nms_numpy",
    "normalized_box",
    "rasterize_boxes",
    "translate_box",
    "xywhn_to_xyxy",
    "xyxy_to_xywhn",
    "zoom_box",
]

