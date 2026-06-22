from __future__ import annotations

from pathlib import Path

import numpy as np

from rl_sahi.common.boxes import area, iou_matrix
from rl_sahi.common.cache import HardRegionCache
from rl_sahi.common.class_mapping import ClassMapping
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
    gt_classes, gt_boxes = read_yolo_labels(image_to_label_path(image_path, image_root, label_root), image_shape)
    class_mapping = class_mapping or ClassMapping()
    gt_classes = class_mapping.map_label_classes(gt_classes)
    det_classes = None if detection_classes is None else class_mapping.map_model_classes(detection_classes)
    if target_classes:
        gt_mask = np.isin(gt_classes.astype(np.int64), np.asarray(target_classes, dtype=np.int64))
        gt_classes = gt_classes[gt_mask]
        gt_boxes = gt_boxes[gt_mask]
    if det_classes is not None and target_classes:
        det_mask = np.isin(det_classes.astype(np.int64), np.asarray(target_classes, dtype=np.int64))
        detection_boxes = detection_boxes[det_mask]
        detection_scores = detection_scores[det_mask]
        det_classes = det_classes[det_mask]
    image_area = float(image_shape[0] * image_shape[1])
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

    small_mask = (area(gt_boxes) / max(image_area, 1.0)) <= small_area_ratio
    small_gt_boxes = gt_boxes[small_mask]
    small_gt_classes = gt_classes[small_mask]
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

    if len(detection_boxes) == 0:
        matched_iou = np.zeros((len(small_gt_boxes),), dtype=np.float32)
        matched_score = np.zeros((len(small_gt_boxes),), dtype=np.float32)
    else:
        ious = iou_matrix(small_gt_boxes, detection_boxes)
        if det_classes is not None:
            same_class = small_gt_classes.astype(np.int64)[:, None] == det_classes.astype(np.int64)[None, :]
            ious = np.where(same_class, ious, 0.0)
        best_idx = ious.argmax(axis=1)
        matched_iou = ious[np.arange(len(small_gt_boxes)), best_idx].astype(np.float32)
        matched_score = detection_scores[best_idx].astype(np.float32)

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
