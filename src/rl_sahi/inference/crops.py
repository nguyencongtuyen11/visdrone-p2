# Cho phép import các annotation nâng cao từ tương lai
from __future__ import annotations

from pathlib import Path  # Thư viện xử lý đường dẫn tệp tin

import cv2               # Thư viện xử lý ảnh OpenCV
import numpy as np       # Thư viện tính toán mảng số học NumPy
from ultralytics import YOLO  # Thư viện mô hình YOLO

# Import cấu hình thiết bị chạy từ common
from rl_sahi.common.device import DeviceLike, configure_torch_runtime, configure_ultralytics_for_device


def crop_roi(image_path: Path, roi: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Cắt một Vùng quan tâm (ROI) ra khỏi ảnh gốc dựa trên tọa độ hộp giới hạn roi [x1, y1, x2, y2].
    Đồng thời trả về tọa độ dời offset (x1, y1) để phục vụ cho việc khôi phục tọa độ về ảnh gốc sau này.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
        
    # Làm tròn tọa độ hộp về số nguyên
    x1, y1, x2, y2 = [int(round(v)) for v in roi]
    
    # Cắt biên để đảm bảo tọa độ nằm hoàn toàn bên trong ảnh gốc
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, image.shape[1])
    y2 = min(y2, image.shape[0])
    
    return image[y1:y2, x1:x2].copy(), (x1, y1)


def run_yolo_on_crop(
    model: YOLO,
    image_path: Path,
    roi: np.ndarray,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: DeviceLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chạy YOLO trên một lát cắt (crop) duy nhất.
    """
    return run_yolo_on_crops(
        model,
        [image_path],
        [roi],
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=device,
    )[0]


def run_yolo_on_crops(
    model: YOLO,
    image_paths: list[Path],
    rois: list[np.ndarray],
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: DeviceLike,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Chạy YOLO dự báo theo batch song song trên nhiều lát cắt (crops) khác nhau.
    Tự động cộng tọa độ offset của từng lát cắt để đưa tọa độ phát hiện cuối cùng về hệ trục tọa độ của ảnh gốc.
    
    Trả về:
        outputs: Danh sách các tuple dạng (boxes, scores, classes) tương ứng với từng lát cắt đầu vào.
    """
    if len(image_paths) != len(rois):
        raise ValueError("image_paths and rois must have the same length")
        
    # Giá trị mặc định khi không phát hiện ra gì hoặc lát cắt bị lỗi/rỗng
    empty = (
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
    )
    outputs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = [empty for _ in image_paths]
    
    crops: list[np.ndarray] = []
    offsets: list[tuple[int, int]] = []
    output_indices: list[int] = []
    
    # Thực hiện cắt ảnh và thu thập các lát cắt hợp lệ
    for index, (image_path, roi) in enumerate(zip(image_paths, rois)):
        crop, offset = crop_roi(Path(image_path), roi)
        if crop.size == 0:
            continue
        crops.append(crop)
        offsets.append(offset)
        output_indices.append(index)

    if not crops:
        return outputs

    # Cấu hình thiết bị chạy PyTorch/YOLO
    resolved_device = configure_torch_runtime(device)
    configure_ultralytics_for_device(resolved_device)
    
    # Chạy dự đoán YOLO trên toàn bộ các lát cắt theo batch
    results = model.predict(
        crops,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        batch=len(crops),
        device=resolved_device,
        half=resolved_device.type == "cuda",
        verbose=False,
    )
    
    # Duyệt qua từng kết quả dự đoán và khôi phục tọa độ tương đối của lát cắt về tọa độ tuyệt đối ảnh gốc
    from rl_sahi.common.data import read_image_shape
    _shape_cache: dict[str, tuple[int, int]] = {}
    EDGE_MARGIN = 2.0  # px: box cham bien TRONG cua crop trong khoang nay => coi la bi cat cut
    for output_index, offset, result in zip(output_indices, offsets, results):
        if result.boxes is None or len(result.boxes) == 0:
            continue
        boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        # Cộng thêm offset của lát cắt để khôi phục về tọa độ ảnh gốc
        boxes[:, [0, 2]] += offset[0]  # Tọa độ x
        boxes[:, [1, 3]] += offset[1]  # Tọa độ y

        scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
        classes = result.boxes.cls.detach().cpu().numpy().astype(np.float32)

        # --- FIX (audit MAJOR merge.py:207): loc box CAT CUT o bien TRONG cua lat cat ---
        # Vat the bi crop cat ngang tao box cut o bien lat; box nay khong khop box day du
        # (khac hinh) nen lot qua NMS -> thoi phong FP. Chi loc bien KHONG phai bien anh
        # (bien anh thi vat that su ket thuc o do). Vat o bien trong da duoc lat chong lan /
        # full-image bat tron ven -> bo manh cut la an toan (chuan SAHI).
        try:
            ch, cw = int(result.orig_shape[0]), int(result.orig_shape[1])
            ip = str(image_paths[output_index])
            if ip not in _shape_cache:
                _shape_cache[ip] = read_image_shape(Path(ip))
            H, W = _shape_cache[ip]
            x1c, y1c = float(offset[0]), float(offset[1])
            x2c, y2c = x1c + cw, y1c + ch
            truncated = (
                ((x1c > 0.5) & (boxes[:, 0] <= x1c + EDGE_MARGIN)) |
                ((y1c > 0.5) & (boxes[:, 1] <= y1c + EDGE_MARGIN)) |
                ((x2c < W - 0.5) & (boxes[:, 2] >= x2c - EDGE_MARGIN)) |
                ((y2c < H - 0.5) & (boxes[:, 3] >= y2c - EDGE_MARGIN))
            )
            keep = ~truncated
            boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
        except Exception:
            pass  # bat ky loi hinh hoc/shape nao -> giu nguyen (an toan, khong lam hong pipeline)

        outputs[output_index] = (boxes, scores, classes)

    return outputs

