# Cho phép import các annotation nâng cao từ tương lai
from __future__ import annotations

import numpy as np  # Thư viện tính toán NumPy

# Import hằng số EPS và hàm as_boxes
from rl_sahi.common.box_types import EPS, as_boxes


def rasterize_boxes(
    boxes: np.ndarray,
    image_shape: tuple[int, int],
    grid_size: int,
    values: np.ndarray | None = None,
    fill_mode: str = "max",
) -> np.ndarray:
    """
    Chuyển đổi (rasterize) các hộp giới hạn (boxes) thành một bản đồ lưới không gian kích thước grid_size x grid_size.
    Phương pháp này ánh xạ vị trí các bounding boxes lên lưới và điền các giá trị (values) tương ứng,
    rất hữu ích để xây dựng biểu diễn trạng thái không gian (spatial state map) cho tác tử RL.
    
    Tham số:
        boxes: Mảng chứa các hộp giới hạn tuyệt đối.
        image_shape: Kích thước ảnh gốc (height, width).
        grid_size: Kích thước lưới đầu ra (ví dụ: 16 cho lưới 16x16).
        values: Mảng chứa giá trị gán cho mỗi hộp (ví dụ: điểm score). Mặc định điền giá trị 1.0 cho mọi hộp.
        fill_mode: Chế độ điền khi có các hộp chồng lấn:
                   - "add": Cộng dồn các giá trị.
                   - "max": Lấy giá trị lớn nhất (mặc định).
    """
    boxes = as_boxes(boxes)
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    if len(boxes) == 0:
        return grid
    h, w = image_shape
    if values is None:
        values = np.ones((len(boxes),), dtype=np.float32)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    
    for box, value in zip(boxes, values):
        # Tính toán chỉ số lưới tương ứng (làm tròn dưới cho góc trái trên và làm tròn trên cho góc phải dưới)
        x1 = int(np.floor((box[0] / max(w, EPS)) * grid_size))
        y1 = int(np.floor((box[1] / max(h, EPS)) * grid_size))
        x2 = int(np.ceil((box[2] / max(w, EPS)) * grid_size))
        y2 = int(np.ceil((box[3] / max(h, EPS)) * grid_size))
        
        # Giới hạn chỉ số nằm trong phạm vi kích thước lưới
        x1 = int(np.clip(x1, 0, grid_size - 1))
        y1 = int(np.clip(y1, 0, grid_size - 1))
        x2 = int(np.clip(x2, x1 + 1, grid_size))
        y2 = int(np.clip(y2, y1 + 1, grid_size))
        
        # Áp dụng điền giá trị vào lưới con tương ứng
        if fill_mode == "add":
            grid[y1:y2, x1:x2] += value
        else:
            grid[y1:y2, x1:x2] = np.maximum(grid[y1:y2, x1:x2], value)
            
    # Giới hạn giá trị của lưới trong khoảng [0.0, 1.0]
    return np.clip(grid, 0.0, 1.0)

