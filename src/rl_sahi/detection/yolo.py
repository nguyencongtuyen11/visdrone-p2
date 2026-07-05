# Cho phép import các annotation nâng cao từ tương lai
from __future__ import annotations

from pathlib import Path  # Thư viện xử lý đường dẫn tệp tin

import numpy as np        # Thư viện tính toán mảng số học NumPy
from ultralytics import YOLO  # Thư viện mô hình YOLO của Ultralytics

# Import các thành phần tiện ích liên quan từ module common
from rl_sahi.common.cache import DetectionCache
from rl_sahi.common.data import read_image_shape
from rl_sahi.common.device import DeviceLike, configure_torch_runtime, configure_ultralytics_for_device
# Import các collector thu thập đặc trưng và thông tin phụ từ module detection.features
from rl_sahi.detection.features import DetectAuxCollector, FeatureCollector

# Các giá trị mặc định cho kích thước lưới trạng thái không gian và số kênh đặc trưng không gian
DEFAULT_AUX_GRID_SIZE = 16
DEFAULT_SPATIAL_FEATURE_CHANNELS = 4


def load_yolo(weights: Path, device: DeviceLike = None) -> YOLO:
    """
    Tải mô hình YOLO với trọng số chỉ định và chuyển mô hình sang thiết bị tính toán thích hợp (CPU/GPU).
    Tự động áp dụng các tối ưu hóa và các bản vá (patches) DirectML nếu cần.
    """
    model = YOLO(str(weights))
    resolved_device = configure_torch_runtime(device)  # Giải quyết thiết bị tính toán và tối ưu hóa runtime
    configure_ultralytics_for_device(resolved_device)  # Áp dụng bản vá lỗi DirectML cho Ultralytics
    model.to(resolved_device)                         # Chuyển mô hình sang GPU hoặc CPU
    return model


def detect_one_image(
    model: YOLO,
    image_path: Path,
    imgsz: int = 640,
    conf: float = 0.01,
    iou: float = 0.7,
    max_det: int = 3000,
    device: DeviceLike = None,
    feature_layers: tuple[int, ...] = (10,),
    aux_grid_size: int = DEFAULT_AUX_GRID_SIZE,
    spatial_feature_channels: int = DEFAULT_SPATIAL_FEATURE_CHANNELS,
) -> DetectionCache:
    """
    Chạy dự báo YOLO trên một bức ảnh duy nhất.
    Đồng thời đăng ký các hooks để trích xuất đặc trưng trung gian và bản đồ không gian từ mô hình,
    sau đó trả về một đối tượng DetectionCache hoàn chỉnh.
    """
    image_shape = read_image_shape(image_path)  # Đọc kích thước ảnh gốc
    resolved_device = configure_torch_runtime(device)
    configure_ultralytics_for_device(resolved_device)
    
    # Sử dụng context manager của collector để đăng ký hooks trước khi dự báo và tự động gỡ hooks sau đó
    with FeatureCollector(model, feature_layers) as collector, DetectAuxCollector(model) as aux_collector:
        collector.clear()
        aux_collector.clear()
        
        # Chạy suy luận (predict) YOLO trên ảnh gốc
        results = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=resolved_device,
            half=resolved_device.type == "cuda",  # Bật chế độ bán chính xác FP16 để tăng tốc nếu chạy trên card CUDA Nvidia
            verbose=False,
        )
        # Rút trích các vector đặc trưng backbone
        feature = collector.vector()
        # Rút trích bản đồ objectness và bản đồ đặc trưng không gian phục vụ DQN state biểu diễn trạng thái
        objectness_map, spatial_feature_map = aux_collector.maps(
            grid_size=aux_grid_size,
            spatial_feature_channels=spatial_feature_channels,
        )
        
    result = results[0]
    # Phân tích kết quả hộp giới hạn phát hiện được từ YOLO
    if result.boxes is None or len(result.boxes) == 0:
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
        classes = np.zeros((0,), dtype=np.float32)
    else:
        boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)   # Tọa độ hộp dạng tuyệt đối xyxy
        scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)  # Độ tự tin của dự đoán
        classes = result.boxes.cls.detach().cpu().numpy().astype(np.float32)  # ID nhãn lớp đối tượng
        
    return DetectionCache(
        image_path=str(image_path),
        image_shape=image_shape,
        boxes=boxes,
        scores=scores,
        classes=classes,
        feature=feature,
        feature_layers=feature_layers,
        objectness_map=objectness_map,
        spatial_feature_map=spatial_feature_map,
    )

