# Cho phép import các annotation nâng cao từ tương lai
from __future__ import annotations

from pathlib import Path  # Thư viện xử lý đường dẫn

import numpy as np        # Thư viện tính toán mảng NumPy

# Import hàm tính diện tích và IoU ma trận từ common.boxes
from rl_sahi.common.boxes import area, iou_matrix
# Import lớp cache vùng khó
from rl_sahi.common.cache import HardRegionCache
# Import lớp quản lý ánh xạ class đối tượng
from rl_sahi.common.class_mapping import ClassMapping
# Import các hàm đọc nhãn gốc
from rl_sahi.common.data import image_to_label_path, read_yolo_labels


def build_hard_region_cache(
    image_path: Path,
    image_root: Path,
    label_root: Path,
    detection_boxes: np.ndarray,
    detection_scores: np.ndarray,
    image_shape: tuple[int, int],
    detection_classes: np.ndarray | None = None,
    small_area_ratio: float = 0.01,
    match_iou: float = 0.4,
    min_detect_score: float = 0.5,
    target_classes: tuple[int, ...] = (),
    class_mapping: ClassMapping | None = None,
) -> HardRegionCache:
    """
    Xây dựng cache vùng khó (HardRegionCache) cho một bức ảnh duy nhất bằng cách đối chiếu
    kết quả phát hiện của YOLO với nhãn gốc (ground truth).
    
    Quy trình lọc vùng khó:
      1. Đọc toàn bộ nhãn ground truth (GT) của ảnh.
      2. Lọc ra các vật thể GT "nhỏ" (có diện tích tương đối so với ảnh nhỏ hơn hoặc bằng small_area_ratio).
      3. Tìm hộp YOLO phát hiện khớp tốt nhất với mỗi vật thể GT nhỏ dựa trên IoU và trùng lớp (class).
      4. Định nghĩa một vật thể GT nhỏ là "vùng khó" (hard box) nếu:
         - Không có hộp YOLO nào phát hiện đè lên nó đủ tốt (IoU cực đại < match_iou), HOẶC
         - Hộp YOLO phát hiện đè lên nó có độ tự tin quá thấp (score < min_detect_score).
    """
    # Đọc nhãn ground truth từ ổ đĩa
    gt_classes, gt_boxes = read_yolo_labels(image_to_label_path(image_path, image_root, label_root), image_shape)
    class_mapping = class_mapping or ClassMapping()
    
    # Ánh xạ class ID của ground truth và của các phát hiện YOLO về hệ quy chiếu so sánh chung
    gt_classes = class_mapping.map_label_classes(gt_classes)
    det_classes = None if detection_classes is None else class_mapping.map_model_classes(detection_classes)
    
    # Lọc chỉ giữ lại các class mục tiêu của GT
    if target_classes:
        gt_mask = np.isin(gt_classes.astype(np.int64), np.asarray(target_classes, dtype=np.int64))
        gt_classes = gt_classes[gt_mask]
        gt_boxes = gt_boxes[gt_mask]
        
    # Lọc chỉ giữ lại các class mục tiêu của phát hiện YOLO
    if det_classes is not None and target_classes:
        det_mask = np.isin(det_classes.astype(np.int64), np.asarray(target_classes, dtype=np.int64))
        detection_boxes = detection_boxes[det_mask]
        detection_scores = detection_scores[det_mask]
        det_classes = det_classes[det_mask]
        
    image_area = float(image_shape[0] * image_shape[1])
    
    # Trường hợp đặc biệt 1: Ảnh không có nhãn ground truth nào
    if len(gt_boxes) == 0:
        return HardRegionCache(
            image_path=str(image_path),
            image_shape=image_shape,
            hard_boxes=np.zeros((0, 4), dtype=np.float32),
            small_gt_boxes=np.zeros((0, 4), dtype=np.float32),
            gt_boxes=gt_boxes,
            matched_iou=np.zeros((0,), dtype=np.float32),
            matched_score=np.zeros((0,), dtype=np.float32),
        )

    # Bước 2: Tạo mask lọc ra các vật thể ground truth kích thước nhỏ
    small_mask = (area(gt_boxes) / max(image_area, 1.0)) <= small_area_ratio
    small_gt_boxes = gt_boxes[small_mask]
    small_gt_classes = gt_classes[small_mask]
    
    # Trường hợp đặc biệt 2: Ảnh không chứa bất kỳ vật thể nhỏ nào
    if len(small_gt_boxes) == 0:
        return HardRegionCache(
            image_path=str(image_path),
            image_shape=image_shape,
            hard_boxes=np.zeros((0, 4), dtype=np.float32),
            small_gt_boxes=small_gt_boxes,
            gt_boxes=gt_boxes,
            matched_iou=np.zeros((0,), dtype=np.float32),
            matched_score=np.zeros((0,), dtype=np.float32),
        )

    # Trường hợp đặc biệt 3: YOLO hoàn toàn không phát hiện được hộp nào trên ảnh gốc
    if len(detection_boxes) == 0:
        # Toàn bộ vật thể nhỏ đều có IoU khớp = 0.0 và score khớp = 0.0
        matched_iou = np.zeros((len(small_gt_boxes),), dtype=np.float32)
        matched_score = np.zeros((len(small_gt_boxes),), dtype=np.float32)
    else:
        # Bước 3: Tính toán ma trận IoU giữa các hộp GT nhỏ và các phát hiện YOLO
        ious = iou_matrix(small_gt_boxes, detection_boxes)
        if det_classes is not None:
            # Ràng buộc trùng lớp: chỉ khớp nếu nhãn lớp của GT và YOLO giống nhau
            same_class = small_gt_classes.astype(np.int64)[:, None] == det_classes.astype(np.int64)[None, :]
            # Nếu khác lớp đối tượng, ép IoU về 0.0
            ious = np.where(same_class, ious, 0.0)
            
        # Tìm phát hiện YOLO khớp tốt nhất (IoU lớn nhất) cho từng hộp GT nhỏ
        best_idx = ious.argmax(axis=1)
        matched_iou = ious[np.arange(len(small_gt_boxes)), best_idx].astype(np.float32)
        matched_score = detection_scores[best_idx].astype(np.float32)

    # Bước 4: Định nghĩa vùng khó (hard box) dựa trên ngưỡng IoU và score
    hard_mask = (matched_iou < match_iou) | (matched_score < min_detect_score)
    hard_boxes = small_gt_boxes[hard_mask]
    
    return HardRegionCache(
        image_path=str(image_path),
        image_shape=image_shape,
        hard_boxes=hard_boxes.astype(np.float32),
        small_gt_boxes=small_gt_boxes.astype(np.float32),
        gt_boxes=gt_boxes.astype(np.float32),
        matched_iou=matched_iou.astype(np.float32),
        matched_score=matched_score.astype(np.float32),
    )

