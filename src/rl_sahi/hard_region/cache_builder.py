from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rl_sahi.common.cache import (
    detection_cache_is_current,
    detection_cache_path,
    hard_region_cache_path,
    load_detection_cache,
    save_hard_region_cache,
)
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.boxes import area
from rl_sahi.common.data import image_to_label_path, iter_images, read_image_shape, read_yolo_labels
from rl_sahi.hard_region.regions import build_hard_region_cache


def _area_threshold_from_percentile(
    images: list[Path],
    image_root: Path,
    label_root: Path,
    percentile: float,
    target_classes: tuple[int, ...],
    class_mapping: ClassMapping,
) -> float:
    ratios: list[np.ndarray] = []
    target = np.asarray(target_classes, dtype=np.int64)
    for image_path in images:
        image_shape = read_image_shape(image_path)
        classes, boxes = read_yolo_labels(image_to_label_path(image_path, image_root, label_root), image_shape)
        classes = class_mapping.map_label_classes(classes)
        if target_classes:
            boxes = boxes[np.isin(classes.astype(np.int64), target)]
        if len(boxes) == 0:
            continue
        image_area = max(float(image_shape[0] * image_shape[1]), 1.0)
        ratios.append(area(boxes) / image_area)
    if not ratios:
        return 0.0
    return float(np.percentile(np.concatenate(ratios, axis=0), percentile))


def cache_hard_regions_for_split(
    image_root: Path,
    label_root: Path,
    cache_root: Path,
    split: str,
    small_area_ratio: float = 0.01,
    small_area_percentile: float | None = None,
    match_iou: float = 0.4,
    min_detect_score: float = 0.5,
    target_classes: tuple[int, ...] = (),
    class_mapping: ClassMapping | None = None,
    detection_metadata: dict[str, Any] | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> int:
    images = iter_images(image_root, split=split, limit=limit)
    class_mapping = class_mapping or ClassMapping()
    if small_area_percentile is not None:
        small_area_ratio = _area_threshold_from_percentile(
            images,
            image_root,
            label_root,
            small_area_percentile,
            target_classes,
            class_mapping,
        )
        print(
            f"[hard] {split}: small_area_ratio={small_area_ratio:.8f} "
            f"from p{small_area_percentile:g}"
        )
    written = 0
    for index, image_path in enumerate(images, start=1):
        det_path = detection_cache_path(cache_root, split, image_path)
        if not detection_cache_is_current(det_path, detection_metadata):
            raise FileNotFoundError(f"Missing current detection cache: {det_path}. Run scripts/detect.py first.")
        out_path = hard_region_cache_path(cache_root, split, image_path)
        if out_path.exists() and not overwrite:
            continue
        det = load_detection_cache(det_path)
        hard = build_hard_region_cache(
            image_path=image_path,
            image_root=image_root,
            label_root=label_root,
            detection_boxes=det.boxes,
            detection_scores=det.scores,
            image_shape=det.image_shape,
            detection_classes=det.classes,
            small_area_ratio=small_area_ratio,
            match_iou=match_iou,
            min_detect_score=min_detect_score,
            target_classes=target_classes,
            class_mapping=class_mapping,
        )
        save_hard_region_cache(out_path, hard)
        written += 1
        if index == 1 or index % 50 == 0:
            print(f"[hard] {split}: {index}/{len(images)} cached -> {out_path}")
    return written
