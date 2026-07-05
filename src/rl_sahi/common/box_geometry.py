# Cho phép import các annotation nâng cao từ tương lai
from __future__ import annotations

import numpy as np  # Thư viện tính toán số học và xử lý mảng (arrays) mạnh mẽ

# Import hằng số EPS (số cực nhỏ tránh lỗi chia cho 0) và hàm as_boxes để định dạng mảng về dạng chuẩn boxes
from rl_sahi.common.box_types import EPS, as_boxes


def area(boxes: np.ndarray) -> np.ndarray:
    """
    Tính diện tích của danh sách các hộp giới hạn (bounding boxes).
    Mỗi hộp có định dạng [x1, y1, x2, y2].
    """
    boxes = as_boxes(boxes)  # Chuẩn hóa đầu vào về dạng mảng 2D (N, 4)
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    # Công thức: width * height = (x2 - x1) * (y2 - y1). np.maximum dùng để tránh chiều âm.
    return np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])


def intersection_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Tính diện tích giao nhau (intersection area) giữa mọi cặp hộp trong tập a và tập b.
    Kết quả trả về ma trận kích thước (len(a), len(b)).
    """
    a = as_boxes(a)
    b = as_boxes(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
        
    # Tính tọa độ vùng giao nhau bằng cách so sánh từng cặp phần tử (broadcasting)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])  # Tọa độ x bên trái lớn nhất
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])  # Tọa độ y phía trên lớn nhất
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])  # Tọa độ x bên phải nhỏ nhất
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])  # Tọa độ y phía dưới nhỏ nhất
    
    # Diện tích giao nhau = (x2 - x1) * (y2 - y1), nếu không giao nhau thì diện tích bằng 0
    return (np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)).astype(np.float32)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Tính ma trận IoU (Intersection over Union - tỷ lệ giao trên diện tích hợp) giữa mọi cặp hộp trong tập a và tập b.
    """
    inter = intersection_matrix(a, b)  # Tính diện tích phần giao nhau
    if inter.size == 0:
        return inter
    area_a = area(a)[:, None]  # Thêm một chiều để thực hiện phép toán ma trận
    area_b = area(b)[None, :]
    
    # IoU = Inter / (Area_A + Area_B - Inter). Dùng np.maximum với EPS để tránh lỗi chia cho 0
    return (inter / np.maximum(area_a + area_b - inter, EPS)).astype(np.float32)


def ioa_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Tính ma trận IoA (Intersection over Area - tỷ lệ giao trên diện tích của hộp trong b).
    Rất hữu ích để kiểm tra xem một Vùng quan tâm (ROI) có bao trùm một vật thể b hay không.
    """
    inter = intersection_matrix(a, b)
    if inter.size == 0:
        return inter
    area_b = area(b)[None, :]  # Lấy diện tích của các hộp trong b
    
    # IoA = Inter / Area_B. Tránh lỗi chia cho 0 bằng EPS
    return (inter / np.maximum(area_b, EPS)).astype(np.float32)


def centers(boxes: np.ndarray) -> np.ndarray:
    """
    Tính tọa độ tâm [x_center, y_center] cho danh sách các hộp.
    """
    boxes = as_boxes(boxes)
    if boxes.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    # Tâm x = (x1 + x2)/2, Tâm y = (y1 + y2)/2
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0], axis=1)


def center_inside(roi: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Kiểm tra xem tọa độ tâm của các hộp (boxes) có nằm bên trong vùng ROI (Region of Interest) hay không.
    Trả về mảng boolean một chiều.
    """
    roi = as_boxes(roi).reshape(1, 4)[0]  # Chuẩn hóa ROI về dạng [x1, y1, x2, y2]
    pts = centers(boxes)                 # Lấy tâm của tất cả các boxes
    if len(pts) == 0:
        return np.zeros((0,), dtype=bool)
    # Kiểm tra điều kiện: tâm x nằm trong [roi_x1, roi_x2] và tâm y nằm trong [roi_y1, roi_y2]
    return (pts[:, 0] >= roi[0]) & (pts[:, 0] <= roi[2]) & (pts[:, 1] >= roi[1]) & (pts[:, 1] <= roi[3])


def normalized_box(box: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """
    Chuẩn hóa các tọa độ của hộp giới hạn (bounding box) chia cho kích thước ảnh [height, width].
    Đưa tọa độ về khoảng [0, 1].
    """
    h, w = image_shape
    b = as_boxes(box).reshape(1, 4)[0]  # Lấy tọa độ hộp dạng [x1, y1, x2, y2]
    return np.array([b[0] / w, b[1] / h, b[2] / w, b[3] / h], dtype=np.float32)

