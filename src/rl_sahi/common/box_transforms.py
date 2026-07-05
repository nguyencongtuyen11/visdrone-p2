# Cho phép import các annotation nâng cao từ tương lai
from __future__ import annotations

import numpy as np  # Thư viện xử lý mảng số học NumPy

# Import hàm chuẩn hóa as_boxes từ box_types
from rl_sahi.common.box_types import as_boxes


def xywhn_to_xyxy(xywhn: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """
    Chuyển đổi tọa độ hộp từ dạng [center_x, center_y, width, height] chuẩn hóa (trong khoảng [0, 1])
    sang dạng [x1, y1, x2, y2] tuyệt đối (pixel).
    """
    boxes = np.asarray(xywhn, dtype=np.float32).reshape(-1, 4)
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    h, w = image_shape
    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    
    # Tính tọa độ góc trái trên (x1, y1) và góc phải dưới (x2, y2) rồi nhân với kích thước ảnh
    out = np.stack(
        [
            (cx - bw / 2.0) * w,
            (cy - bh / 2.0) * h,
            (cx + bw / 2.0) * w,
            (cy + bh / 2.0) * h,
        ],
        axis=1,
    )
    # Giới hạn các hộp nằm trong biên của bức ảnh để tránh tọa độ bị tràn ra ngoài
    return clip_boxes(out, image_shape)


def xyxy_to_xywhn(boxes: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """
    Chuyển đổi tọa độ hộp từ dạng [x1, y1, x2, y2] tuyệt đối (pixel)
    sang dạng [center_x, center_y, width, height] chuẩn hóa (trong khoảng [0, 1]).
    """
    boxes = as_boxes(boxes)
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    h, w = image_shape
    bw = boxes[:, 2] - boxes[:, 0]  # Chiều rộng hộp tuyệt đối
    bh = boxes[:, 3] - boxes[:, 1]  # Chiều cao hộp tuyệt đối
    cx = boxes[:, 0] + bw / 2.0     # Tâm x tuyệt đối
    cy = boxes[:, 1] + bh / 2.0     # Tâm y tuyệt đối
    
    # Chia cho kích thước ảnh để chuẩn hóa về đoạn [0, 1]
    return np.stack([cx / w, cy / h, bw / w, bh / h], axis=1).astype(np.float32)


def clip_boxes(boxes: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """
    Giới hạn tọa độ của các hộp (bounding boxes) nằm hoàn toàn trong biên của ảnh.
    Tránh trường hợp tọa độ bị âm hoặc lớn hơn kích thước ảnh.
    """
    boxes = as_boxes(boxes).copy()
    if boxes.size == 0:
        return boxes
    h, w = image_shape
    
    # Cắt x1, x2 vào khoảng [0, w-1]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, max(0, w - 1))
    # Cắt y1, y2 vào khoảng [0, h-1]
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, max(0, h - 1))
    
    # Đảm bảo chiều rộng và chiều cao hộp tối thiểu là 1.0 pixel
    boxes[:, 2] = np.maximum(boxes[:, 2], boxes[:, 0] + 1.0)
    boxes[:, 3] = np.maximum(boxes[:, 3], boxes[:, 1] + 1.0)
    
    # Tiếp tục clip một lần nữa đảm bảo tuyệt đối không vượt quá w, h
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return boxes.astype(np.float32)


def box_from_center(
    cx: float,
    cy: float,
    side: float,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """
    Tạo ra một hộp giới hạn hình vuông dạng [x1, y1, x2, y2] từ tọa độ tâm (cx, cy) và độ dài cạnh (side).
    """
    half = side / 2.0
    return clip_boxes(np.array([[cx - half, cy - half, cx + half, cy + half]], dtype=np.float32), image_shape)[0]


def translate_box(box: np.ndarray, dx: float, dy: float, image_shape: tuple[int, int]) -> np.ndarray:
    """
    Tịnh tiến (dịch chuyển) hộp giới hạn đi một khoảng (dx, dy) và đảm bảo hộp vẫn nằm trong biên ảnh.
    Nếu hộp chạm biên, nó sẽ được đẩy lại để giữ nguyên kích thước gốc nếu có thể.
    """
    b = as_boxes(box).reshape(1, 4)[0].copy()
    width = b[2] - b[0]
    height = b[3] - b[1]
    h, w = image_shape
    
    # Cộng độ dời dx, dy vào tọa độ
    b[[0, 2]] += dx
    b[[1, 3]] += dy
    
    # Xử lý đẩy hộp ngược lại nếu nó bị vượt quá biên trái/phải/trên/dưới
    if b[0] < 0:
        b[[0, 2]] -= b[0]
    if b[1] < 0:
        b[[1, 3]] -= b[1]
    if b[2] > w:
        b[[0, 2]] -= b[2] - w
    if b[3] > h:
        b[[1, 3]] -= b[3] - h
        
    # Giữ nguyên kích thước chiều rộng và chiều cao nguyên bản
    b[2] = b[0] + width
    b[3] = b[1] + height
    return clip_boxes(b.reshape(1, 4), image_shape)[0]


def zoom_box(
    box: np.ndarray,
    factor: float,
    image_shape: tuple[int, int],
    min_side: float,
    max_side: float,
) -> np.ndarray:
    """
    Thu nhỏ hoặc phóng to hộp giới hạn quanh tâm của nó theo một hệ số (factor).
    Kết quả trả về một hộp vuông được clip trong biên ảnh.
    """
    b = as_boxes(box).reshape(1, 4)[0]
    cx = float((b[0] + b[2]) / 2.0)  # Tìm tọa độ tâm x
    cy = float((b[1] + b[3]) / 2.0)  # Tìm tọa độ tâm y
    
    # Xác định kích thước cạnh ban đầu (lấy max của width, height) rồi nhân với factor
    side = max(float(b[2] - b[0]), float(b[3] - b[1])) * factor
    # Giới hạn kích thước cạnh trong đoạn [min_side, max_side]
    side = float(np.clip(side, min_side, max_side))
    
    # Tạo hộp mới từ tâm và kích thước cạnh đã tính
    return box_from_center(cx, cy, side, image_shape)

