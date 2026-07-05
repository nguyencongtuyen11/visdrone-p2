# Cho phép import các annotation nâng cao từ tương lai
from __future__ import annotations

import numpy as np  # Thư viện NumPy xử lý mảng

# Import hàm tính IoU và chuẩn hóa hộp
from rl_sahi.common.box_geometry import iou_matrix
from rl_sahi.common.box_types import as_boxes


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """
    Thuật toán Non-Maximum Suppression (NMS - Triệt tiêu cực đại phụ) viết bằng NumPy.
    Dùng để loại bỏ các hộp phát hiện bị trùng lặp và giữ lại hộp tốt nhất.
    
    Tham số:
        boxes: Mảng tọa độ các hộp giới hạn dạng (N, 4).
        scores: Mảng điểm tự tin tương ứng dạng (N,).
        iou_threshold: Ngưỡng IoU tối đa cho phép chồng lấn. Nếu lớn hơn ngưỡng này, hộp có score thấp hơn sẽ bị lọc bỏ.
    """
    boxes = as_boxes(boxes)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
        
    # Sắp xếp các hộp theo điểm số giảm dần thu được danh sách các chỉ số (indices)
    order = scores.argsort()[::-1]
    keep: list[int] = [] # Danh sách các chỉ số hộp được giữ lại
    
    while order.size > 0:
        i = int(order[0]) # Chọn hộp có điểm số cao nhất trong các hộp còn lại
        keep.append(i)
        if order.size == 1:
            break
        # Tính IoU của hộp hiện tại với toàn bộ các hộp còn lại
        ious = iou_matrix(boxes[[i]], boxes[order[1:]])[0]
        # Chỉ giữ lại các hộp có mức độ trùng lặp (IoU) nhỏ hơn hoặc bằng ngưỡng cho phép
        order = order[1:][ious <= iou_threshold]
        
    return np.asarray(keep, dtype=np.int64)

