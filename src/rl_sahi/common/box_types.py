# Cho phép import các annotation nâng cao từ tương lai
from __future__ import annotations

import numpy as np  # Thư viện tính toán mảng NumPy

# Hằng số vô cùng nhỏ để tránh chia cho 0 trong các phép tính IoU/IoA
EPS = 1e-9


def as_boxes(boxes: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa dữ liệu đầu vào (mảng hoặc danh sách các hộp)
    về định dạng mảng NumPy 2D chuẩn (N, 4) chứa các giá trị float32.
    """
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.size == 0:
        # Nếu mảng rỗng, trả về mảng rỗng kích thước (0, 4)
        return np.zeros((0, 4), dtype=np.float32)
    # Định hình lại mảng về dạng (N, 4) và ép kiểu sang float32
    return arr.reshape(-1, 4).astype(np.float32, copy=False)

