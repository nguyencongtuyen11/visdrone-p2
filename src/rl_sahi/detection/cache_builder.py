# Cho phép import các annotation nâng cao từ tương lai
from __future__ import annotations

from pathlib import Path  # Thư viện xử lý đường dẫn tệp tin

# Import các hàm liên quan đến cache phát hiện đối tượng từ module common.cache
from rl_sahi.common.cache import (
    detection_cache_is_current,
    detection_cache_metadata,
    detection_cache_path,
    save_detection_cache,
)
# Import hàm duyệt quét ảnh từ module common.data
from rl_sahi.common.data import iter_images
# Import các hàm tải YOLO và suy luận từ module detection.yolo
from rl_sahi.detection.yolo import DEFAULT_AUX_GRID_SIZE, DEFAULT_SPATIAL_FEATURE_CHANNELS, detect_one_image, load_yolo


def cache_detections_for_split(
    weights: Path,
    image_root: Path,
    cache_root: Path,
    split: str,
    imgsz: int = 640,
    conf: float = 0.01,
    iou: float = 0.7,
    max_det: int = 3000,
    device: str | None = None,
    feature_layers: tuple[int, ...] = (10,),
    aux_grid_size: int = DEFAULT_AUX_GRID_SIZE,
    spatial_feature_channels: int = DEFAULT_SPATIAL_FEATURE_CHANNELS,
    limit: int | None = None,
    overwrite: bool = False,
) -> int:
    """
    Thực hiện chạy YOLO trên toàn bộ tập dữ liệu (split) và lưu kết quả phát hiện cùng đặc trưng backbone vào cache.
    Hàm này giúp tăng tốc quá trình huấn luyện Reinforcement Learning (DQN) bằng cách tránh phải suy luận lại YOLO nhiều lần.
    
    Trả về:
        written: Số lượng file cache được ghi mới hoặc ghi đè thành công.
    """
    # Khởi tạo mô hình YOLO
    model = load_yolo(weights, device=device)
    # Lấy danh sách toàn bộ ảnh trong phân vùng split
    images = iter_images(image_root, split=split, limit=limit)
    # Xây dựng metadata mong đợi để so khớp với cache cũ
    expected_metadata = detection_cache_metadata(
        weights=weights,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        feature_layers=feature_layers,
        aux_grid_size=aux_grid_size,
        spatial_feature_channels=spatial_feature_channels,
    )
    written = 0
    # Duyệt qua từng ảnh trong danh sách
    for index, image_path in enumerate(images, start=1):
        out_path = detection_cache_path(cache_root, split, image_path)
        # Nếu file cache đã tồn tại, không yêu cầu ghi đè và cache vẫn hợp lệ (không đổi weights/config), thì bỏ qua
        if out_path.exists() and not overwrite and detection_cache_is_current(out_path, expected_metadata):
            continue
        # Chạy YOLO trên ảnh hiện tại để lấy boxes và feature maps
        cache = detect_one_image(
            model=model,
            image_path=image_path,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
            feature_layers=feature_layers,
            aux_grid_size=aux_grid_size,
            spatial_feature_channels=spatial_feature_channels,
        )
        # Gán metadata và lưu cache xuống ổ đĩa
        cache.metadata = expected_metadata
        save_detection_cache(out_path, cache)
        written += 1
        # In thông tin tiến độ sau mỗi 50 ảnh
        if index == 1 or index % 50 == 0:
            print(f"[detect] {split}: {index}/{len(images)} cached -> {out_path}")
    return written

