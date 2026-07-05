from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from rl_sahi.common.boxes import iou_matrix
from rl_sahi.common.cache import DetectionCache
from rl_sahi.common.data import image_id, image_to_label_path, read_yolo_labels
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.inference.crops import run_yolo_on_crops
from rl_sahi.inference.merge import (
    merge_predictions,
    new_detection_gain_after_merge,
    new_detection_utility_after_merge,
    source_counts_after_merge,
)


# giải thích: Lớp dữ liệu CropOutcome chứa kết quả đánh giá của một vùng cắt (crop)
@dataclass(slots=True)
class CropOutcome:
    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray
    new_detection_gain: int
    new_detection_utility: float
    accepted_new_count_after: int
    tp_gain: int
    fp_gain: int
    reward: float
    accepted: bool

    # giải thích: Trả về thông tin đánh giá dưới dạng một từ điển để ghi log hoặc phân tích
    def info(self) -> dict[str, int | float | bool]:
        return {
            "crop_new_detection_gain": int(self.new_detection_gain),
            "crop_new_detection_utility": float(self.new_detection_utility),
            "crop_accepted_new_count_after": int(self.accepted_new_count_after),
            "crop_tp_gain": int(self.tp_gain),
            "crop_fp_gain": int(self.fp_gain),
            "crop_outcome_reward": float(self.reward),
            "crop_outcome_accepted": bool(self.accepted),
        }


# giải thích: Lớp CropOutcomeEvaluator đánh giá chất lượng phát hiện vật thể trên vùng cắt lát so với ảnh gốc và nhãn đúng (ground truth)
class CropOutcomeEvaluator:
    def __init__(
        self,
        model: YOLO,
        image_root: Path,
        label_root: Path | None,
        cache_root: Path,
        split: str,
        infer_cfg: InferenceConfig,
        weights: Path | None = None,
        iou_threshold: float = 0.5,
        use_cache: bool = True,
        detection_reward: float = 0.5,
        tp_reward: float = 3.0,
        fp_penalty: float = 1.5,
        empty_penalty: float = 1.2,
        no_gain_penalty: float = 1.2,
    ) -> None:
        self.model = model
        self.image_root = Path(image_root)
        self.label_root = None if label_root is None else Path(label_root)
        self.cache_root = Path(cache_root)
        self.split = str(split)
        self.cfg = infer_cfg
        self.weights = None if weights is None else Path(weights)
        self.iou_threshold = float(iou_threshold)
        self.use_cache = bool(use_cache)
        self.detection_reward = float(detection_reward)
        self.tp_reward = float(tp_reward)
        self.fp_penalty = float(fp_penalty)
        self.empty_penalty = float(empty_penalty)
        self.no_gain_penalty = float(no_gain_penalty)

    # giải thích: Lấy các dự đoán trên toàn bộ ảnh gốc có độ tin cậy thỏa mãn ngưỡng
    def full_predictions(self, det: DetectionCache) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask = np.asarray(det.scores, dtype=np.float32).reshape(-1) >= float(self.cfg.output_conf)
        boxes = np.asarray(det.boxes, dtype=np.float32).reshape(-1, 4)[mask]
        scores = np.asarray(det.scores, dtype=np.float32).reshape(-1)[mask]
        classes = self.cfg.class_mapping.map_model_classes(det.classes[mask])
        return self._filter_classes(boxes, scores, classes)

    # giải thích: Tính toán số lượng phát hiện mới ban đầu sau khi thực hiện việc hợp nhất các dự đoán
    def initial_new_count(
        self,
        full_boxes: np.ndarray,
        full_scores: np.ndarray,
        full_classes: np.ndarray,
        image_shape: tuple[int, int],
    ) -> int:
        _full_count, slice_count = _merged_source_counts(
            full_boxes,
            full_scores,
            full_classes,
            [],
            [],
            [],
            image_shape,
            self.cfg.merge_iou,
        )
        return int(slice_count)

    # giải thích: Xác định xem có nên bỏ qua bước đánh giá cuối cùng dựa trên thông tin trạng thái môi trường không
    def should_skip_terminal(self, info: dict) -> bool:
        if info.get("stop_due_to_old_overlap", False):
            return True
        if info.get("stop_due_to_attempted_overlap", False):
            return True
        if self.cfg.require_stop_for_acceptance and info.get("stop_due_to_max_steps", False):
            return True
        if self.cfg.require_stop_for_acceptance and info.get("stop_due_to_stalled_roi", False):
            return True
        return False

    # giải thích: Thực hiện đánh giá vùng cắt lát bằng cách chạy YOLO trực tiếp trên vùng cắt đó
    def evaluate(
        self,
        image_path: Path | str,
        det: DetectionCache,
        roi: np.ndarray,
        full_boxes: np.ndarray,
        full_scores: np.ndarray,
        full_classes: np.ndarray,
        slice_boxes_parts: list[np.ndarray],
        slice_scores_parts: list[np.ndarray],
        slice_classes_parts: list[np.ndarray],
        accepted_new_count: int,
    ) -> CropOutcome:
        image_path = self._resolve_image_path(image_path)
        raw_boxes, raw_scores, raw_classes = self._crop_predictions(image_path, roi)
        return self.evaluate_from_predictions(
            image_path=image_path,
            det=det,
            full_boxes=full_boxes,
            full_scores=full_scores,
            full_classes=full_classes,
            slice_boxes_parts=slice_boxes_parts,
            slice_scores_parts=slice_scores_parts,
            slice_classes_parts=slice_classes_parts,
            accepted_new_count=accepted_new_count,
            raw_boxes=raw_boxes,
            raw_scores=raw_scores,
            raw_classes=raw_classes,
        )

    # giải thích: Đánh giá chất lượng và tính toán phần thưởng từ các dự đoán YOLO đã có sẵn trên vùng cắt
    def evaluate_from_predictions(
        self,
        image_path: Path | str,
        det: DetectionCache,
        full_boxes: np.ndarray,
        full_scores: np.ndarray,
        full_classes: np.ndarray,
        slice_boxes_parts: list[np.ndarray],
        slice_scores_parts: list[np.ndarray],
        slice_classes_parts: list[np.ndarray],
        accepted_new_count: int,
        raw_boxes: np.ndarray,
        raw_scores: np.ndarray,
        raw_classes: np.ndarray,
    ) -> CropOutcome:
        image_path = self._resolve_image_path(image_path)
        classes = self.cfg.class_mapping.map_model_classes(raw_classes)
        boxes, scores, classes = self._filter_classes(raw_boxes, raw_scores, classes)
        # giải thích: Tính toán lượng thông tin mới phát hiện (new_detection_gain và new_detection_utility)
        new_detection_gain = new_detection_gain_after_merge(
            det.image_shape,
            self.cfg.merge_iou,
            [full_boxes, *slice_boxes_parts],
            [full_scores, *slice_scores_parts],
            [full_classes, *slice_classes_parts],
            boxes,
            scores,
            classes,
            duplicate_iou=self.cfg.duplicate_iou,
        )
        new_detection_utility = new_detection_utility_after_merge(
            det.image_shape,
            self.cfg.merge_iou,
            [full_boxes, *slice_boxes_parts],
            [full_scores, *slice_scores_parts],
            [full_classes, *slice_classes_parts],
            boxes,
            scores,
            classes,
            duplicate_iou=self.cfg.duplicate_iou,
        )
        # giải thích: Tính toán độ tăng trưởng True Positive (tp_gain) và False Positive (fp_gain) so với Ground Truth
        tp_gain, fp_gain = self._tp_fp_gain(
            image_path,
            det.image_shape,
            full_boxes,
            full_scores,
            full_classes,
            slice_boxes_parts,
            slice_scores_parts,
            slice_classes_parts,
            boxes,
            scores,
            classes,
        )
        # giải thích: Công thức tính toán phần thưởng (reward) kết hợp độ tiện ích phát hiện, TP và FP
        reward = self.detection_reward * max(float(new_detection_utility), 0.0)
        reward += self.tp_reward * max(float(tp_gain), 0.0)
        reward -= self.fp_penalty * max(float(fp_gain), 0.0)
        if len(boxes) == 0:
            reward -= self.empty_penalty
        elif new_detection_gain <= 0:
            reward -= self.no_gain_penalty
        # giải thích: Kiểm tra vùng cắt có được chấp nhận hay không dựa trên các ngưỡng cấu hình
        accepted = (
            new_detection_gain >= int(self.cfg.min_slice_detections)
            and new_detection_utility >= float(self.cfg.min_slice_utility)
        )
        return CropOutcome(
            boxes=boxes,
            scores=scores,
            classes=classes,
            new_detection_gain=new_detection_gain,
            new_detection_utility=float(new_detection_utility),
            accepted_new_count_after=int(accepted_new_count) + int(new_detection_gain),
            tp_gain=int(tp_gain),
            fp_gain=int(fp_gain),
            reward=float(reward),
            accepted=bool(accepted),
        )

    # giải thích: Lọc các hộp dự đoán thuộc các lớp mục tiêu được quan tâm
    def _filter_classes(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        classes = np.asarray(classes, dtype=np.float32).reshape(-1)
        if not self.cfg.target_classes:
            return boxes, scores, classes
        target = np.asarray(self.cfg.target_classes, dtype=np.int64)
        mask = np.isin(classes.astype(np.int64), target)
        return boxes[mask], scores[mask], classes[mask]

    # giải thích: Giải quyết đường dẫn tuyệt đối của ảnh, kiểm tra tính tồn tại của ảnh trong thư mục gốc
    def _resolve_image_path(self, image_path: Path | str) -> Path:
        path = Path(image_path)
        if path.exists():
            return path
        candidate = self.image_root / self.split / path.name
        if candidate.exists():
            return candidate
        return path

    # giải thích: Lấy các dự đoán trên một vùng cắt duy nhất
    def _crop_predictions(self, image_path: Path, roi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.crop_predictions_many([image_path], [roi])[0]

    # giải thích: Thực hiện suy luận mô hình YOLO song song trên danh sách nhiều vùng cắt (crops), sử dụng bộ đệm (cache) để tăng tốc độ nếu có sẵn
    def crop_predictions_many(
        self,
        image_paths: list[Path | str],
        rois: list[np.ndarray],
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if len(image_paths) != len(rois):
            raise ValueError("image_paths and rois must have the same length")
        outputs: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = [None] * len(image_paths)
        missing_indices: list[int] = []
        missing_paths: list[Path] = []
        missing_rois: list[np.ndarray] = []
        missing_cache_paths: list[Path] = []
        missing_metadata: list[dict[str, Any]] = []
        
        # giải thích: Duyệt qua các vùng cắt và kiểm tra xem có tệp cache kết quả suy luận tương ứng không
        for index, (image_path, roi) in enumerate(zip(image_paths, rois)):
            resolved_path = self._resolve_image_path(image_path)
            metadata = self._metadata(resolved_path, roi)
            path = self._cache_path(resolved_path, metadata)
            if self.use_cache and path.exists():
                loaded = self._load_cache(path, metadata)
                if loaded is not None:
                    outputs[index] = loaded
                    continue
            missing_indices.append(index)
            missing_paths.append(resolved_path)
            missing_rois.append(np.asarray(roi, dtype=np.float32).reshape(4))
            missing_cache_paths.append(path)
            missing_metadata.append(metadata)

        # giải thích: Nếu thiếu cache, tiến hành chạy YOLO trên các vùng cắt đó rồi lưu lại vào bộ nhớ đệm
        if missing_indices:
            predictions = run_yolo_on_crops(
                self.model,
                missing_paths,
                missing_rois,
                imgsz=self.cfg.slice_imgsz,
                conf=self.cfg.output_conf,
                iou=self.cfg.iou,
                max_det=self.cfg.max_det,
                device=self.cfg.device,
            )
            for index, path, metadata, prediction in zip(
                missing_indices,
                missing_cache_paths,
                missing_metadata,
                predictions,
            ):
                boxes, scores, classes = prediction
                if self.use_cache:
                    self._save_cache(path, metadata, boxes, scores, classes)
                outputs[index] = prediction

        empty = (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
        return [item if item is not None else empty for item in outputs]

    # giải thích: Tính toán độ tăng trưởng TP (True Positive) và FP (False Positive) so với nhãn đúng khi thêm vùng cắt mới vào ảnh
    def _tp_fp_gain(
        self,
        image_path: Path,
        image_shape: tuple[int, int],
        full_boxes: np.ndarray,
        full_scores: np.ndarray,
        full_classes: np.ndarray,
        slice_boxes_parts: list[np.ndarray],
        slice_scores_parts: list[np.ndarray],
        slice_classes_parts: list[np.ndarray],
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
    ) -> tuple[int, int]:
        if self.label_root is None:
            return 0, 0
        gt_boxes, gt_classes = self._ground_truth(image_path, image_shape)
        # giải thích: Hợp nhất các dự đoán trước khi có vùng cắt hiện tại
        before_boxes, before_scores, before_classes = _merge_predictions(
            image_shape,
            self.cfg.merge_iou,
            [full_boxes, *slice_boxes_parts],
            [full_scores, *slice_scores_parts],
            [full_classes, *slice_classes_parts],
        )
        # giải thích: Hợp nhất các dự đoán sau khi đã thêm vùng cắt hiện tại
        after_boxes, after_scores, after_classes = _merge_predictions(
            image_shape,
            self.cfg.merge_iou,
            [full_boxes, *slice_boxes_parts, boxes],
            [full_scores, *slice_scores_parts, scores],
            [full_classes, *slice_classes_parts, classes],
        )
        # giải thích: Tính toán số lượng TP và FP của các trạng thái trước và sau để lấy hiệu số độ lợi
        before_tp, before_fp = _match_counts(
            before_boxes,
            before_scores,
            before_classes,
            gt_boxes,
            gt_classes,
            self.iou_threshold,
        )
        after_tp, after_fp = _match_counts(
            after_boxes,
            after_scores,
            after_classes,
            gt_boxes,
            gt_classes,
            self.iou_threshold,
        )
        return int(after_tp - before_tp), int(after_fp - before_fp)

    # giải thích: Tải nhãn Ground Truth (độ phân giải thực tế) và áp dụng bộ lọc các lớp mục tiêu
    def _ground_truth(self, image_path: Path, image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        assert self.label_root is not None
        classes, boxes = read_yolo_labels(image_to_label_path(image_path, self.image_root, self.label_root), image_shape)
        classes = self.cfg.class_mapping.map_label_classes(classes)
        if self.cfg.target_classes:
            target = np.asarray(self.cfg.target_classes, dtype=np.int64)
            mask = np.isin(classes.astype(np.int64), target)
            boxes = boxes[mask]
            classes = classes[mask]
        return boxes.astype(np.float32), classes.astype(np.float32)

    # giải thích: Tạo siêu dữ liệu cấu hình đặc trưng cho vùng cắt và mô hình để làm khóa lưu trữ bộ đệm (metadata hash)
    def _metadata(self, image_path: Path, roi: np.ndarray) -> dict[str, Any]:
        weights = None
        if self.weights is not None:
            stat = self.weights.stat() if self.weights.exists() else None
            weights = {
                "path": str(self.weights.resolve()),
                "exists": self.weights.exists(),
                "size": int(stat.st_size) if stat is not None else None,
                "mtime_ns": int(stat.st_mtime_ns) if stat is not None else None,
            }
        return {
            "image": image_id(image_path),
            "roi": [round(float(x), 2) for x in np.asarray(roi, dtype=np.float32).reshape(4).tolist()],
            "weights": weights,
            "slice_imgsz": int(self.cfg.slice_imgsz),
            "conf": float(self.cfg.output_conf),
            "iou": float(self.cfg.iou),
            "max_det": int(self.cfg.max_det),
        }

    # giải thích: Trả về đường dẫn của tệp cache .npz dựa trên hàm băm SHA1 của metadata
    def _cache_path(self, image_path: Path, metadata: dict[str, Any]) -> Path:
        payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
        return self.cache_root / "crop_outcomes" / self.split / image_id(image_path) / f"{digest}.npz"

    # giải thích: Tải dữ liệu dự đoán từ tệp cache .npz nếu thông tin băm trùng khớp
    def _load_cache(
        self,
        path: Path,
        metadata: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        try:
            with np.load(path, allow_pickle=False) as data:
                actual = str(data["metadata_json"].item()) if "metadata_json" in data.files else ""
                expected = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
                if actual != expected:
                    return None
                return (
                    data["boxes"].astype(np.float32),
                    data["scores"].astype(np.float32),
                    data["classes"].astype(np.float32),
                )
        except Exception:
            return None

    # giải thích: Lưu các dự đoán trên vùng cắt vào bộ đệm cache .npz dưới dạng nén
    def _save_cache(
        self,
        path: Path,
        metadata: dict[str, Any],
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
            boxes=np.asarray(boxes, dtype=np.float32),
            scores=np.asarray(scores, dtype=np.float32),
            classes=np.asarray(classes, dtype=np.float32),
        )


def _merge_predictions(
    image_shape: tuple[int, int],
    merge_iou: float,
    boxes_parts: list[np.ndarray],
    scores_parts: list[np.ndarray],
    classes_parts: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return merge_predictions(image_shape, merge_iou, boxes_parts, scores_parts, classes_parts)


def _merged_source_counts(
    full_boxes: np.ndarray,
    full_scores: np.ndarray,
    full_classes: np.ndarray,
    slice_boxes_parts: list[np.ndarray],
    slice_scores_parts: list[np.ndarray],
    slice_classes_parts: list[np.ndarray],
    image_shape: tuple[int, int],
    merge_iou: float,
) -> tuple[int, int]:
    return source_counts_after_merge(
        full_boxes,
        full_scores,
        full_classes,
        slice_boxes_parts,
        slice_scores_parts,
        slice_classes_parts,
        image_shape,
        merge_iou,
    )


# giải thích: Khớp các dự đoán với nhãn đúng bằng IoU để đếm số lượng TP và FP một cách chính xác
def _match_counts(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    iou_threshold: float,
) -> tuple[int, int]:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    classes = np.asarray(classes, dtype=np.float32).reshape(-1)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)
    gt_classes = np.asarray(gt_classes, dtype=np.float32).reshape(-1)
    if len(boxes) == 0:
        return 0, 0
    # giải thích: Sắp xếp các dự đoán giảm dần theo độ tin cậy (confidence score)
    order = np.argsort(scores)[::-1]
    matched = np.zeros((len(gt_boxes),), dtype=bool)
    tp = 0
    fp = 0
    for idx in order:
        cls = int(classes[idx])
        gt_idx = np.flatnonzero(gt_classes.astype(np.int64) == cls)
        if len(gt_idx) == 0:
            fp += 1
            continue
        ious = iou_matrix(boxes[idx].reshape(1, 4), gt_boxes[gt_idx])[0]
        best_local = int(ious.argmax())
        best = int(gt_idx[best_local])
        # giải thích: Nếu IoU lớn hơn hoặc bằng ngưỡng và nhãn đúng chưa được khớp, tính là TP, ngược lại là FP
        if float(ious[best_local]) >= iou_threshold and not matched[best]:
            matched[best] = True
            tp += 1
        else:
            fp += 1
    return int(tp), int(fp)
