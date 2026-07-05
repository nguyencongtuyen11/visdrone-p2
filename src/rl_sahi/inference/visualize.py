# Cho phép import các annotation nâng cao từ tương lai
from __future__ import annotations

from pathlib import Path  # Thư viện xử lý đường dẫn

import cv2               # Thư viện OpenCV vẽ hình và lưu tệp ảnh
import numpy as np       # Thư viện NumPy xử lý mảng

# Import các hàm tiện ích hình học và dữ liệu từ common
from rl_sahi.common.boxes import as_boxes
from rl_sahi.common.data import read_image


def draw_boxes(image: np.ndarray, boxes: np.ndarray, color: tuple[int, int, int], thickness: int = 1) -> None:
    """
    Vẽ các hộp giới hạn (boxes) lên hình ảnh bằng OpenCV.
    """
    for box in as_boxes(boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    sources: np.ndarray,
    full_color: tuple[int, int, int] = (0, 190, 0),
    slice_color: tuple[int, int, int] = (255, 120, 0),
) -> None:
    """
    Vẽ các hộp vật thể phát hiện được lên hình ảnh, phân biệt màu sắc dựa trên nguồn gốc (source).
    - Hộp phát hiện từ ảnh gốc (source == 0) vẽ bằng màu xanh lá (full_color).
    - Hộp phát hiện bổ sung từ các lát cắt (source != 0) vẽ bằng màu cam (slice_color).
    """
    boxes = as_boxes(boxes)
    sources = np.asarray(sources, dtype=np.int32).reshape(-1)
    if len(boxes) == 0:
        return
    # Vẽ các hộp phát hiện trên ảnh đầy đủ
    draw_boxes(image, boxes[sources == 0], full_color, thickness=1)
    # Vẽ các hộp phát hiện trên các lát cắt
    draw_boxes(image, boxes[sources != 0], slice_color, thickness=1)


def save_inference_visual(
    image_path: Path,
    boxes: np.ndarray,
    sources: np.ndarray,
    accepted_rois: np.ndarray,
    rejected_rois: np.ndarray,
    out_path: Path,
) -> None:
    """
    Vẽ kết quả suy luận thích ứng đầy đủ lên ảnh gốc và lưu file đầu ra.
    - Vẽ các bounding boxes phát hiện được (xanh lá/cam).
    - Vẽ viền các lát cắt được chấp nhận (accepted_rois - màu đỏ).
    - Vẽ viền các lát cắt bị từ chối (rejected_rois - màu cam/vàng).
    """
    image = read_image(image_path)
    draw_detections(image, boxes, sources)
    # Vẽ các lát cắt bị từ chối (màu cam nhạt)
    draw_boxes(image, rejected_rois, (0, 165, 255), thickness=2)
    # Vẽ các lát cắt được chấp nhận (màu đỏ)
    draw_boxes(image, accepted_rois, (0, 0, 255), thickness=2)
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Ghi ảnh kết quả ra ổ đĩa
    cv2.imwrite(str(out_path), image)

