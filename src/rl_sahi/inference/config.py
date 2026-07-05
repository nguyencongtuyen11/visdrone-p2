# Cho phép import các annotation nâng cao từ tương lai
from __future__ import annotations

from dataclasses import dataclass, field  # Hỗ trợ định nghĩa lớp dữ liệu ngắn gọn

from rl_sahi.common.class_mapping import ClassMapping  # Quản lý ánh xạ class đối tượng


@dataclass(slots=True)
class InferenceConfig:
    """
    Cấu hình các tham số phục vụ cho quá trình suy luận thích ứng (Adaptive SAHI).
    """
    full_imgsz: int = 640                  # Kích thước ảnh đầy đủ để YOLO chạy lần đầu
    slice_imgsz: int = 640                 # Kích thước của lát cắt (slice/crop)
    full_conf: float = 0.01                # Confidence threshold khi YOLO chạy trên ảnh gốc
    output_conf: float = 0.3               # Confidence threshold lọc đầu ra cuối cùng sau khi gộp
    iou: float = 0.7                       # IoU threshold cho NMS của YOLO
    merge_iou: float = 0.5                 # IoU threshold để thực hiện gộp (merge) các hộp từ nhiều lát cắt
    max_det: int = 3000                    # Số lượng vật thể tối đa phát hiện được
    device: str | None = None              # Thiết bị chạy suy luận (CPU/GPU)
    feature_layers: tuple[int, ...] = (10,) # Các layer trích xuất đặc trưng backbone
    min_slice_detections: int = 1          # Số lượng phát hiện tối thiểu trên lát cắt để xem là hữu dụng
    min_slice_utility: float = 0.5         # Lợi ích tối thiểu để tác tử RL chấp nhận một lát cắt
    duplicate_iou: float = 0.5             # Ngưỡng trùng lặp giữa các lát cắt đã quét
    max_slice_attempts: int = 0            # Số lượng lát cắt quét tối đa (nếu bằng 0 là không giới hạn)
    target_classes: tuple[int, ...] = (0, 2, 3, 5, 8, 9) # Các class đối tượng mục tiêu cần phát hiện
    require_stop_for_acceptance: bool = True # Ràng buộc tác tử RL phải chọn hành động dừng (stop) để chấp nhận kết quả
    class_mapping: ClassMapping = field(default_factory=ClassMapping) # Ánh xạ nhãn lớp tương ứng

