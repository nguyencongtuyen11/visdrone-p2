from __future__ import annotations

from pathlib import Path

import numpy as np

from rl_sahi.common.boxes import clip_boxes, iou_matrix, nms_numpy


def save_prediction_txt(
    path: Path,
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    sources: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for cls, score, box, source in zip(classes, scores, boxes, sources):
            x1, y1, x2, y2 = [float(v) for v in box]
            f.write(f"{int(cls)} {float(score):.6f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {int(source)}\n")


def class_aware_nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_threshold: float) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    keep_parts: list[np.ndarray] = []
    for cls in np.unique(classes.astype(np.int64)):
        idx = np.flatnonzero(classes.astype(np.int64) == cls)
        keep_local = nms_numpy(boxes[idx], scores[idx], iou_threshold)
        keep_parts.append(idx[keep_local])
    keep = np.concatenate(keep_parts, axis=0) if keep_parts else np.zeros((0,), dtype=np.int64)
    return keep[np.argsort(scores[keep])[::-1]].astype(np.int64)


def merge_predictions(
    image_shape: tuple[int, int],
    merge_iou: float,
    boxes_parts: list[np.ndarray],
    scores_parts: list[np.ndarray],
    classes_parts: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = np.concatenate(boxes_parts, axis=0) if boxes_parts else np.zeros((0, 4), dtype=np.float32)
    scores = np.concatenate(scores_parts, axis=0) if scores_parts else np.zeros((0,), dtype=np.float32)
    classes = np.concatenate(classes_parts, axis=0) if classes_parts else np.zeros((0,), dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    classes = np.asarray(classes, dtype=np.float32).reshape(-1)
    if len(boxes) == 0:
        return boxes, scores, classes
    boxes = clip_boxes(boxes, image_shape)
    keep = class_aware_nms(boxes, scores, classes, merge_iou)
    return boxes[keep], scores[keep], classes[keep]


def merge_predictions_with_sources(
    image_shape: tuple[int, int],
    merge_iou: float,
    boxes_parts: list[np.ndarray],
    scores_parts: list[np.ndarray],
    classes_parts: list[np.ndarray],
    sources_parts: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    boxes = np.concatenate(boxes_parts, axis=0) if boxes_parts else np.zeros((0, 4), dtype=np.float32)
    scores = np.concatenate(scores_parts, axis=0) if scores_parts else np.zeros((0,), dtype=np.float32)
    classes = np.concatenate(classes_parts, axis=0) if classes_parts else np.zeros((0,), dtype=np.float32)
    sources = np.concatenate(sources_parts, axis=0) if sources_parts else np.zeros((0,), dtype=np.int32)
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    classes = np.asarray(classes, dtype=np.float32).reshape(-1)
    sources = np.asarray(sources, dtype=np.int32).reshape(-1)
    if len(boxes) == 0:
        return boxes, scores, classes, sources
    boxes = clip_boxes(boxes, image_shape)
    keep = class_aware_nms(boxes, scores, classes, merge_iou)
    return boxes[keep], scores[keep], classes[keep], sources[keep]


def source_counts_after_merge(
    full_boxes: np.ndarray,
    full_scores: np.ndarray,
    full_classes: np.ndarray,
    slice_boxes_parts: list[np.ndarray],
    slice_scores_parts: list[np.ndarray],
    slice_classes_parts: list[np.ndarray],
    image_shape: tuple[int, int],
    merge_iou: float,
) -> tuple[int, int]:
    sources_parts = [np.zeros((len(full_boxes),), dtype=np.int32)] + [
        np.full((len(boxes),), index + 1, dtype=np.int32)
        for index, boxes in enumerate(slice_boxes_parts)
    ]
    _boxes, _scores, _classes, sources = merge_predictions_with_sources(
        image_shape,
        merge_iou,
        [full_boxes, *slice_boxes_parts],
        [full_scores, *slice_scores_parts],
        [full_classes, *slice_classes_parts],
        sources_parts,
    )
    return int((sources == 0).sum()), int((sources > 0).sum())


def _novel_candidate_detections_after_merge(
    image_shape: tuple[int, int],
    merge_iou: float,
    previous_boxes_parts: list[np.ndarray],
    previous_scores_parts: list[np.ndarray],
    previous_classes_parts: list[np.ndarray],
    candidate_boxes: np.ndarray,
    candidate_scores: np.ndarray,
    candidate_classes: np.ndarray,
    duplicate_iou: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    before_boxes, _before_scores, before_classes = merge_predictions(
        image_shape,
        merge_iou,
        previous_boxes_parts,
        previous_scores_parts,
        previous_classes_parts,
    )
    candidate_boxes = np.asarray(candidate_boxes, dtype=np.float32).reshape(-1, 4)
    candidate_scores = np.asarray(candidate_scores, dtype=np.float32).reshape(-1)
    candidate_classes = np.asarray(candidate_classes, dtype=np.float32).reshape(-1)
    if len(candidate_boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    sources_parts = [
        np.zeros((sum(len(boxes) for boxes in previous_boxes_parts),), dtype=np.int32),
        np.ones((len(candidate_boxes),), dtype=np.int32),
    ]
    after_boxes, after_scores, after_classes, after_sources = merge_predictions_with_sources(
        image_shape,
        merge_iou,
        [*previous_boxes_parts, candidate_boxes],
        [*previous_scores_parts, candidate_scores],
        [*previous_classes_parts, candidate_classes],
        sources_parts,
    )
    candidate_mask = after_sources == 1
    if not candidate_mask.any():
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    if len(before_boxes) == 0:
        return after_boxes[candidate_mask], after_scores[candidate_mask], after_classes[candidate_mask]

    duplicate_threshold = float(merge_iou) if duplicate_iou is None else float(duplicate_iou)
    duplicate_threshold = float(np.clip(duplicate_threshold, 0.0, 1.0))
    novel = np.ones((int(candidate_mask.sum()),), dtype=bool)
    for idx, (box, cls) in enumerate(zip(after_boxes[candidate_mask], after_classes[candidate_mask])):
        same_cls = before_classes.astype(np.int64) == int(cls)
        if not same_cls.any():
            continue
        ious = iou_matrix(box.reshape(1, 4), before_boxes[same_cls])[0]
        if len(ious) > 0 and float(ious.max()) >= duplicate_threshold:
            novel[idx] = False
    return (
        after_boxes[candidate_mask][novel],
        after_scores[candidate_mask][novel],
        after_classes[candidate_mask][novel],
    )


def new_detection_gain_after_merge(
    image_shape: tuple[int, int],
    merge_iou: float,
    previous_boxes_parts: list[np.ndarray],
    previous_scores_parts: list[np.ndarray],
    previous_classes_parts: list[np.ndarray],
    candidate_boxes: np.ndarray,
    candidate_scores: np.ndarray,
    candidate_classes: np.ndarray,
    duplicate_iou: float | None = None,
) -> int:
    boxes, _scores, _classes = _novel_candidate_detections_after_merge(
        image_shape,
        merge_iou,
        previous_boxes_parts,
        previous_scores_parts,
        previous_classes_parts,
        candidate_boxes,
        candidate_scores,
        candidate_classes,
        duplicate_iou=duplicate_iou,
    )
    return int(len(boxes))


def new_detection_utility_after_merge(
    image_shape: tuple[int, int],
    merge_iou: float,
    previous_boxes_parts: list[np.ndarray],
    previous_scores_parts: list[np.ndarray],
    previous_classes_parts: list[np.ndarray],
    candidate_boxes: np.ndarray,
    candidate_scores: np.ndarray,
    candidate_classes: np.ndarray,
    duplicate_iou: float | None = None,
) -> float:
    _boxes, scores, _classes = _novel_candidate_detections_after_merge(
        image_shape,
        merge_iou,
        previous_boxes_parts,
        previous_scores_parts,
        previous_classes_parts,
        candidate_boxes,
        candidate_scores,
        candidate_classes,
        duplicate_iou=duplicate_iou,
    )
    if len(scores) == 0:
        return 0.0
    return float(np.clip(scores, 0.0, 1.0).sum())
