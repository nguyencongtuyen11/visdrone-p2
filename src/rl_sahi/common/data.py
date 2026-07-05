# Cho phép import các annotation nâng cao từ tương lai
from __future__ import annotations

from pathlib import Path  # Thư viện xử lý đường dẫn tệp tin

import cv2               # Thư viện OpenCV xử lý ảnh máy tính
import numpy as np       # Thư viện xử lý mảng số học NumPy

# Import hàm phụ trợ chuyển đổi từ xywh chuẩn hóa sang xyxy tuyệt đối
from .boxes import xywhn_to_xyxy


# Các phần mở rộng tệp tin được coi là ảnh hợp lệ
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(image_root: Path, split: str | None = None, limit: int | None = None) -> list[Path]:
    """
    Quét và liệt kê tất cả các tệp tin ảnh trong thư mục image_root (và thư mục con split nếu có).
    Sắp xếp theo thứ tự bảng chữ cái và hỗ trợ giới hạn số lượng ảnh (limit).
    """
    root = Path(image_root)
    search_root = root / split if split else root
    images = sorted(p for p in search_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if limit is not None:
        images = images[:limit]
    return images


def image_id(image_path: Path) -> str:
    """
    Lấy chuỗi định danh duy nhất của ảnh (chính là tên tệp không kèm phần mở rộng).
    Ví dụ: /data/raw/images/train/0000001.jpg -> 0000001
    """
    return Path(image_path).stem


def image_to_label_path(image_path: Path, image_root: Path, label_root: Path) -> Path:
    """
    Tìm đường dẫn file nhãn tương ứng (.txt) từ đường dẫn file ảnh.
    Giữ nguyên cấu trúc thư mục con tương đối.
    """
    image_path = Path(image_path)
    image_root = Path(image_root)
    label_root = Path(label_root)
    # Lấy đường dẫn tương đối của ảnh so với thư mục gốc ảnh
    rel = image_path.relative_to(image_root)
    # Gán vào thư mục gốc nhãn và đổi phần mở rộng sang .txt
    return (label_root / rel).with_suffix(".txt")


def read_image(image_path: Path) -> np.ndarray:
    """
    Đọc ảnh từ đường dẫn bằng thư viện OpenCV dưới dạng mảng màu BGR.
    Ném lỗi FileNotFoundError nếu không thể đọc được ảnh.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return image


def read_image_shape(image_path: Path) -> tuple[int, int]:
    """
    Đọc nhanh kích thước ảnh (height, width) từ đường dẫn mà không cần lưu giữ lại bức ảnh trong bộ nhớ lâu.
    """
    image = read_image(image_path)
    h, w = image.shape[:2]
    return int(h), int(w)


def read_yolo_labels(label_path: Path, image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Đọc nhãn định dạng YOLO (class_id x_center y_center width height) từ file txt.
    Tự động chuyển đổi tọa độ bounding boxes từ chuẩn hóa sang tọa độ pixel tuyệt đối [x1, y1, x2, y2].
    
    Trả về:
        classes: Mảng 1D chứa ID của các class.
        boxes: Mảng 2D chứa tọa độ tuyệt đối của các hộp.
    """
    label_path = Path(label_path)
    if not label_path.exists():
        # Nếu file nhãn không tồn tại (ảnh không có vật thể), trả về mảng rỗng
        return np.zeros((0,), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
    rows: list[list[float]] = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            rows.append([float(x) for x in parts[:5]])
    if not rows:
        return np.zeros((0,), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
        
    arr = np.asarray(rows, dtype=np.float32)
    classes = arr[:, 0]  # Cột đầu tiên là ID của class đối tượng
    
    # Chuyển đổi cột 1:5 (xywhn) sang định dạng xyxy tuyệt đối dựa trên kích thước ảnh
    boxes = xywhn_to_xyxy(arr[:, 1:5], image_shape)
    return classes, boxes


def ensure_dir(path: Path) -> Path:
    """
    Đảm bảo thư mục chỉ định tồn tại. Nếu chưa có, tự động tạo nó và các thư mục cha.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

