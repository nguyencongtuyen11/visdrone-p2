# Cho phép import các annotation nâng cao từ tương lai
from __future__ import annotations

import json                  # Thư viện JSON xử lý lưu siêu dữ liệu
from pathlib import Path     # Thư viện xử lý đường dẫn

import numpy as np           # Thư viện tính toán NumPy
import torch                 # Thư viện mạng nơ-ron PyTorch
from ultralytics import YOLO  # Thư viện YOLO Ultralytics

# Import các hàm tiện ích hình học từ common.boxes
from rl_sahi.common.boxes import area, clip_boxes, intersection_matrix
# Import các hàm tiện ích liên quan đến cache
from rl_sahi.common.cache import (
    DetectionCache,
    detection_cache_is_current,
    detection_cache_metadata,
    detection_cache_path,
    load_detection_cache,
    save_detection_cache,
)
# Import ánh xạ class và tải cấu hình mặc định
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.config import ProjectConfig, load_default_config
from rl_sahi.common.device import DeviceLike, resolve_torch_device
from rl_sahi.detection.yolo import detect_one_image, load_yolo
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.inference.crops import run_yolo_on_crop
# Import các thuật toán gộp và đo lường từ common.merge
from rl_sahi.inference.merge import (
    class_aware_nms,
    new_detection_gain_after_merge,
    new_detection_utility_after_merge,
    save_prediction_txt,
    source_counts_after_merge,
)
# Import các module bổ trợ RL
from rl_sahi.inference.rollout import rollout_one_slice
from rl_sahi.inference.visualize import save_inference_visual
from rl_sahi.rl.checkpoint import load_policy
from rl_sahi.rl.slice_env import SliceEnv
from rl_sahi.rl.state_config import StateConfig


def _class_mask(classes: np.ndarray, target_classes: tuple[int, ...]) -> np.ndarray:
    """
    Tạo mảng lọc boolean để xác định các nhãn lớp có thuộc lớp mục tiêu (target_classes) hay không.
    """
    classes = np.asarray(classes, dtype=np.float32).reshape(-1)
    if not target_classes:
        return np.ones((len(classes),), dtype=bool)
    return np.isin(classes.astype(np.int64), np.asarray(target_classes, dtype=np.int64))


def _filter_classes(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    target_classes: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Lọc các hộp giới hạn, điểm số và lớp đối tượng, chỉ giữ lại những lớp thuộc target_classes.
    """
    mask = _class_mask(classes, target_classes)
    return boxes[mask], scores[mask], classes[mask]


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
    """
    Tính số lượng phát hiện cuối cùng thuộc ảnh gốc và lát cắt.
    """
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


def _new_detection_gain(
    full_boxes: np.ndarray,
    full_scores: np.ndarray,
    full_classes: np.ndarray,
    slice_boxes_parts: list[np.ndarray],
    slice_scores_parts: list[np.ndarray],
    slice_classes_parts: list[np.ndarray],
    candidate_boxes: np.ndarray,
    candidate_scores: np.ndarray,
    candidate_classes: np.ndarray,
    image_shape: tuple[int, int],
    merge_iou: float,
    duplicate_iou: float | None = None,
) -> int:
    """
    Tính số lượng hộp mới phát hiện thành công qua lát cắt ứng viên mới.
    """
    return new_detection_gain_after_merge(
        image_shape,
        merge_iou,
        [full_boxes, *slice_boxes_parts],
        [full_scores, *slice_scores_parts],
        [full_classes, *slice_classes_parts],
        candidate_boxes,
        candidate_scores,
        candidate_classes,
        duplicate_iou=duplicate_iou,
    )


def _new_detection_utility(
    full_boxes: np.ndarray,
    full_scores: np.ndarray,
    full_classes: np.ndarray,
    slice_boxes_parts: list[np.ndarray],
    slice_scores_parts: list[np.ndarray],
    slice_classes_parts: list[np.ndarray],
    candidate_boxes: np.ndarray,
    candidate_scores: np.ndarray,
    candidate_classes: np.ndarray,
    image_shape: tuple[int, int],
    merge_iou: float,
    duplicate_iou: float | None = None,
) -> float:
    """
    Tính toán điểm tiện ích (utility) thu hoạch được từ các hộp mới phát hiện.
    """
    return new_detection_utility_after_merge(
        image_shape,
        merge_iou,
        [full_boxes, *slice_boxes_parts],
        [full_scores, *slice_scores_parts],
        [full_classes, *slice_classes_parts],
        candidate_boxes,
        candidate_scores,
        candidate_classes,
        duplicate_iou=duplicate_iou,
    )


def _attempt_overlap(roi: np.ndarray, attempted_rois: list[np.ndarray]) -> float:
    """
    Tính toán tỷ lệ chồng lấn lớn nhất của ROI ứng viên hiện tại với tất cả các ROI đã quét trước đó.
    Dùng để phát hiện xem tác tử RL có quét lặp lại một vùng cũ hay không.
    """
    if not attempted_rois:
        return 0.0
    previous = np.stack(attempted_rois).astype(np.float32)
    roi_arr = np.asarray(roi, dtype=np.float32).reshape(1, 4)
    inter = intersection_matrix(roi_arr, previous)[0]
    current_area = max(float(area(roi_arr)[0]), 1.0)
    return float(np.clip(inter.max() / current_area, 0.0, 1.0))


def _checkpoint_detection_mismatches(
    metadata: dict | None,
    cfg: InferenceConfig,
    state_cfg: StateConfig,
) -> list[str]:
    """
    Kiểm tra sự không khớp cấu hình giữa checkpoint mô hình RL và cấu hình suy luận hiện tại.
    Đảm bảo tính nhất quán của mạng nơ-ron DQN.
    """
    if not isinstance(metadata, dict) or not metadata:
        return []
    expected = {
        "imgsz": int(cfg.full_imgsz),
        "conf": float(cfg.full_conf),
        "iou": float(cfg.iou),
        "max_det": int(cfg.max_det),
        "feature_layers": tuple(int(x) for x in cfg.feature_layers),
        "aux_grid_size": int(state_cfg.grid_size),
        "spatial_feature_channels": int(state_cfg.spatial_feature_channels),
    }
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        if key not in metadata:
            continue
        actual_value = metadata[key]
        if key == "feature_layers":
            actual_value = tuple(int(x) for x in actual_value)
        elif isinstance(expected_value, int):
            actual_value = int(actual_value)
        elif isinstance(expected_value, float):
            actual_value = float(actual_value)
        if actual_value != expected_value:
            mismatches.append(f"{key}: checkpoint={actual_value!r}, inference={expected_value!r}")
    return mismatches


def get_initial_detection(
    model: YOLO,
    weights: Path | None,
    image_path: Path,
    weights_imgsz: int,
    full_conf: float,
    full_iou: float,
    max_det: int,
    device: DeviceLike,
    feature_layers: tuple[int, ...],
    aux_grid_size: int,
    spatial_feature_channels: int,
    cache_root: Path | str | None = None,
    split: str | None = None,
    use_cache: bool = True,
) -> DetectionCache:
    """
    Tải cache hoặc thực hiện chạy YOLO ảnh gốc lần đầu để lấy các boxes và feature maps khởi điểm.
    """
    expected_metadata = (
        detection_cache_metadata(
            weights=weights,
            imgsz=weights_imgsz,
            conf=full_conf,
            iou=full_iou,
            max_det=max_det,
            feature_layers=feature_layers,
            aux_grid_size=aux_grid_size,
            spatial_feature_channels=spatial_feature_channels,
        )
        if weights is not None
        else None
    )
    if cache_root is not None and split is not None:
        cache_path = detection_cache_path(cache_root, split, image_path)
        # Nếu cache hợp lệ và được cấu hình dùng cache, đọc trực tiếp từ cache để tăng tốc
        if use_cache and detection_cache_is_current(cache_path, expected_metadata):
            return load_detection_cache(cache_path)
        # Ngược lại chạy YOLO phát hiện và lưu cache mới
        det = detect_one_image(
            model=model,
            image_path=image_path,
            imgsz=weights_imgsz,
            conf=full_conf,
            iou=full_iou,
            max_det=max_det,
            device=device,
            feature_layers=feature_layers,
            aux_grid_size=aux_grid_size,
            spatial_feature_channels=spatial_feature_channels,
        )
        det.metadata = expected_metadata
        save_detection_cache(cache_path, det)
        return det
        
    # Trường hợp không dùng thư mục cache lưu trữ lâu dài
    det = detect_one_image(
        model=model,
        image_path=image_path,
        imgsz=weights_imgsz,
        conf=full_conf,
        iou=full_iou,
        max_det=max_det,
        device=device,
        feature_layers=feature_layers,
        aux_grid_size=aux_grid_size,
        spatial_feature_channels=spatial_feature_channels,
    )
    det.metadata = expected_metadata
    return det


class AdaptiveSahiInferencer:
    """
    Bộ suy luận thích ứng (Adaptive SAHI) kết hợp YOLO và tác tử học tăng cường (RL DQN).
    Tự động chọn các vùng cần cắt lát dựa trên trạng thái phát hiện để tối ưu mAP và giảm số lượng crop.
    """
    def __init__(self, weights: Path, checkpoint: Path, cfg: InferenceConfig) -> None:
        self.cfg = cfg
        self.device_t = resolve_torch_device(cfg.device)
        # Tải mô hình RL policy và thông số huấn luyện đi kèm
        self.policy, checkpoint_data = load_policy(checkpoint, self.device_t)
        self.env_cfg = checkpoint_data["env_cfg_obj"]
        self.state_cfg = checkpoint_data.get("state_cfg_obj", StateConfig())
        
        # Kiểm tra tính đồng bộ
        mismatches = _checkpoint_detection_mismatches(
            checkpoint_data.get("detection_metadata"),
            cfg,
            self.state_cfg,
        )
        if mismatches:
            raise ValueError(
                "Checkpoint detection metadata does not match inference config: "
                + "; ".join(mismatches)
            )
        self.weights = Path(weights)
        # Tải mô hình YOLO
        self.yolo = load_yolo(weights, device=self.device_t)

    def infer_image(
        self,
        image_path: Path,
        out_dir: Path,
        cache_root: Path | None = None,
        split: str | None = None,
        use_cache: bool = True,
    ) -> dict:
        """
        Khởi chạy suy luận thích ứng trên 1 file ảnh cụ thể.
        """
        cfg = self.cfg
        # Bước 1: Chạy YOLO trên ảnh gốc để lấy các phát hiện khởi điểm
        det = get_initial_detection(
            model=self.yolo,
            weights=self.weights,
            image_path=image_path,
            weights_imgsz=cfg.full_imgsz,
            full_conf=cfg.full_conf,
            full_iou=cfg.iou,
            max_det=cfg.max_det,
            device=cfg.device,
            feature_layers=cfg.feature_layers,
            aux_grid_size=self.state_cfg.grid_size,
            spatial_feature_channels=self.state_cfg.spatial_feature_channels,
            cache_root=cache_root,
            split=split,
            use_cache=use_cache,
        )

        # Bước 2: Bắt đầu tiến trình vòng lặp RL quyết định cắt lát ảnh và lưu đầu ra
        return _infer_with_loaded(
            image_path=image_path,
            out_dir=out_dir,
            yolo=self.yolo,
            policy=self.policy,
            device_t=self.device_t,
            env_cfg=self.env_cfg,
            state_cfg=self.state_cfg,
            det=det,
            cfg=cfg,
        )


def _infer_with_loaded(
    image_path: Path,
    out_dir: Path,
    yolo: YOLO,
    policy,
    device_t: torch.device,
    env_cfg,
    state_cfg: StateConfig,
    det: DetectionCache,
    cfg: InferenceConfig,
) -> dict:
    """
    Tiến hành vòng lặp RL Rollout để quyết định cắt lát ảnh và tổng hợp kết quả cuối cùng.
    """
    accepted_rois: list[np.ndarray] = []  # Các ROI (lát cắt) được chấp nhận do phát hiện thêm đối tượng mới
    rejected_rois: list[np.ndarray] = []  # Các ROI bị từ chối
    attempted_rois: list[np.ndarray] = [] # Toàn bộ các ROI đã được quét thử qua
    
    slice_boxes_all: list[np.ndarray] = []   # Lưu tọa độ hộp phát hiện từ các lát cắt được chấp nhận
    slice_scores_all: list[np.ndarray] = []  # Lưu score
    slice_classes_all: list[np.ndarray] = [] # Lưu nhãn lớp
    slice_meta: list[dict] = []              # Siêu dữ liệu từng lượt quét

    # Lọc các hộp phát hiện trên ảnh gốc bằng ngưỡng tin cậy đầu ra
    full_mask = det.scores >= cfg.output_conf
    full_boxes = det.boxes[full_mask]
    full_scores = det.scores[full_mask]
    full_classes = cfg.class_mapping.map_model_classes(det.classes[full_mask])
    
    # Chỉ giữ các lớp mục tiêu
    full_boxes, full_scores, full_classes = _filter_classes(
        full_boxes,
        full_scores,
        full_classes,
        cfg.target_classes,
    )
    
    # Xác định giới hạn tối đa số lượt chạy thử quét
    max_attempts = int(cfg.max_slice_attempts) if cfg.max_slice_attempts > 0 else int(env_cfg.max_slices * 2)
    
    # Vòng lặp chính: RL tác tử tương tác với môi trường để đưa ra vị trí lát cắt tiếp theo
    for attempt_idx in range(1, max_attempts + 1):
        if len(accepted_rois) >= env_cfg.max_slices:
            break
            
        history_arr = (
            np.stack(attempted_rois).astype(np.float32)
            if attempted_rois
            else np.zeros((0, 4), dtype=np.float32)
        )
        overlap_arr = (
            np.stack(accepted_rois).astype(np.float32)
            if accepted_rois
            else np.zeros((0, 4), dtype=np.float32)
        )
        
        # Khởi tạo môi trường mô phỏng lát cắt (SliceEnv) cho bước hiện tại
        env = SliceEnv(
            det,
            None,
            env_cfg=env_cfg,
            state_cfg=state_cfg,
            previous_rois=history_arr,
            overlap_rois=overlap_arr,
            target_classes=cfg.target_classes,
            class_mapping=cfg.class_mapping,
        )
        
        # Tác tử RL chạy suy luận chính sách để đưa ra hộp ROI mục tiêu
        roi, actions, info = rollout_one_slice(policy, env, device_t)
        
        # Xử lý các điều kiện dừng sớm hoặc từ chối do trùng lặp lát cắt cũ
        if info.get("stop_due_to_old_overlap", False):
            repeat_attempt_overlap = _attempt_overlap(roi, attempted_rois)
            attempted_rois.append(roi)
            rejected_rois.append(roi)
            slice_meta.append(
                {
                    "attempt_index": attempt_idx,
                    "slice_index": None,
                    "accepted": False,
                    "rejection_reason": "old_slice_overlap",
                    "roi": [float(x) for x in roi.tolist()],
                    "actions": actions,
                    "steps": len(actions),
                    "old_slice_overlap": float(info.get("old_slice_overlap", 0.0)),
                    "attempted_slice_overlap": float(info.get("attempted_slice_overlap", 0.0)),
                    "stop_due_to_stalled_roi": bool(info.get("stop_due_to_stalled_roi", False)),
                    "repeat_attempt_overlap": repeat_attempt_overlap,
                    "detections": 0,
                }
            )
            if repeat_attempt_overlap >= 0.95:
                break
            continue
            
        if info.get("stop_due_to_attempted_overlap", False):
            repeat_attempt_overlap = _attempt_overlap(roi, attempted_rois)
            attempted_rois.append(roi)
            rejected_rois.append(roi)
            slice_meta.append(
                {
                    "attempt_index": attempt_idx,
                    "slice_index": None,
                    "accepted": False,
                    "rejection_reason": "attempted_slice_overlap",
                    "roi": [float(x) for x in roi.tolist()],
                    "actions": actions,
                    "steps": len(actions),
                    "old_slice_overlap": float(info.get("old_slice_overlap", 0.0)),
                    "attempted_slice_overlap": float(info.get("attempted_slice_overlap", 0.0)),
                    "stop_due_to_stalled_roi": bool(info.get("stop_due_to_stalled_roi", False)),
                    "repeat_attempt_overlap": repeat_attempt_overlap,
                    "detections": 0,
                }
            )
            if repeat_attempt_overlap >= 0.95:
                break
            continue
            
        if cfg.require_stop_for_acceptance and info.get("stop_due_to_max_steps", False):
            repeat_attempt_overlap = _attempt_overlap(roi, attempted_rois)
            attempted_rois.append(roi)
            rejected_rois.append(roi)
            slice_meta.append(
                {
                    "attempt_index": attempt_idx,
                    "slice_index": None,
                    "accepted": False,
                    "rejection_reason": "max_steps_without_stop",
                    "roi": [float(x) for x in roi.tolist()],
                    "actions": actions,
                    "steps": len(actions),
                    "old_slice_overlap": float(info.get("old_slice_overlap", 0.0)),
                    "attempted_slice_overlap": float(info.get("attempted_slice_overlap", 0.0)),
                    "stop_due_to_stalled_roi": bool(info.get("stop_due_to_stalled_roi", False)),
                    "repeat_attempt_overlap": repeat_attempt_overlap,
                    "detections": 0,
                }
            )
            if repeat_attempt_overlap >= 0.95:
                break
            continue
            
        if cfg.require_stop_for_acceptance and info.get("stop_due_to_stalled_roi", False):
            repeat_attempt_overlap = _attempt_overlap(roi, attempted_rois)
            attempted_rois.append(roi)
            rejected_rois.append(roi)
            slice_meta.append(
                {
                    "attempt_index": attempt_idx,
                    "slice_index": None,
                    "accepted": False,
                    "rejection_reason": "stalled_without_stop",
                    "roi": [float(x) for x in roi.tolist()],
                    "actions": actions,
                    "steps": len(actions),
                    "old_slice_overlap": float(info.get("old_slice_overlap", 0.0)),
                    "attempted_slice_overlap": float(info.get("attempted_slice_overlap", 0.0)),
                    "stop_due_to_stalled_roi": bool(info.get("stop_due_to_stalled_roi", False)),
                    "repeat_attempt_overlap": repeat_attempt_overlap,
                    "detections": 0,
                }
            )
            if repeat_attempt_overlap >= 0.95:
                break
            continue

        # Nếu ROI được đề xuất hợp lệ, thực thi YOLO trên ROI lát cắt này
        boxes_i, scores_i, classes_i = run_yolo_on_crop(
            yolo,
            image_path,
            roi,
            imgsz=cfg.slice_imgsz,
            conf=cfg.output_conf,
            iou=cfg.iou,
            max_det=cfg.max_det,
            device=cfg.device,
        )
        classes_i = cfg.class_mapping.map_model_classes(classes_i)
        boxes_i, scores_i, classes_i = _filter_classes(boxes_i, scores_i, classes_i, cfg.target_classes)
        attempted_rois.append(roi)
        
        # Đo lường xem lát cắt này có cung cấp thêm phát hiện mới nào không
        new_detection_gain = _new_detection_gain(
            full_boxes,
            full_scores,
            full_classes,
            slice_boxes_all,
            slice_scores_all,
            slice_classes_all,
            boxes_i,
            scores_i,
            classes_i,
            det.image_shape,
            cfg.merge_iou,
            cfg.duplicate_iou,
        )
        new_detection_utility = _new_detection_utility(
            full_boxes,
            full_scores,
            full_classes,
            slice_boxes_all,
            slice_scores_all,
            slice_classes_all,
            boxes_i,
            scores_i,
            classes_i,
            det.image_shape,
            cfg.merge_iou,
            cfg.duplicate_iou,
        )
        
        # Chấp nhận lát cắt nếu nó phát hiện thêm ít nhất min_slice_detections và đạt mức tiện ích tối thiểu
        accepted = (
            new_detection_gain >= int(cfg.min_slice_detections)
            and new_detection_utility >= float(cfg.min_slice_utility)
        )
        
        if accepted:
            rejection_reason = None
        elif len(boxes_i) == 0:
            rejection_reason = "empty_slice"
        elif new_detection_gain <= 0:
            rejection_reason = "no_new_detection_after_nms"
        elif new_detection_utility < float(cfg.min_slice_utility):
            rejection_reason = "low_new_detection_utility"
        else:
            rejection_reason = "low_new_detection_count"
            
        slice_index = None
        if accepted:
            accepted_rois.append(roi)
            slice_index = len(accepted_rois)
            slice_boxes_all.append(boxes_i)
            slice_scores_all.append(scores_i)
            slice_classes_all.append(classes_i)
        else:
            rejected_rois.append(roi)
            
        slice_meta.append(
            {
                "attempt_index": attempt_idx,
                "slice_index": slice_index,
                "accepted": accepted,
                "rejection_reason": rejection_reason,
                "roi": [float(x) for x in roi.tolist()],
                "actions": actions,
                "steps": len(actions),
                "old_slice_overlap": float(info.get("old_slice_overlap", 0.0)),
                "attempted_slice_overlap": float(info.get("attempted_slice_overlap", 0.0)),
                "stop_due_to_stalled_roi": bool(info.get("stop_due_to_stalled_roi", False)),
                "detections": int(len(boxes_i)),
                "new_detections_after_nms": int(new_detection_gain),
                "new_detection_utility": float(new_detection_utility),
            }
        )

    # Tổng hợp các hộp từ ảnh gốc và từ toàn bộ lát cắt đã được chấp nhận
    boxes_parts = [full_boxes] + slice_boxes_all
    scores_parts = [full_scores] + slice_scores_all
    classes_parts = [full_classes] + slice_classes_all
    sources_parts = [np.zeros((len(full_boxes),), dtype=np.int32)] + [
        np.full((len(boxes_i),), index + 1, dtype=np.int32)
        for index, boxes_i in enumerate(slice_boxes_all)
    ]

    boxes = np.concatenate(boxes_parts, axis=0) if boxes_parts else np.zeros((0, 4), dtype=np.float32)
    scores = np.concatenate(scores_parts, axis=0) if scores_parts else np.zeros((0,), dtype=np.float32)
    classes = np.concatenate(classes_parts, axis=0) if classes_parts else np.zeros((0,), dtype=np.float32)
    sources = np.concatenate(sources_parts, axis=0) if sources_parts else np.zeros((0,), dtype=np.int32)

    # Clip các hộp trong ảnh và gộp NMS lần cuối để ra kết quả phát hiện duy nhất
    boxes = clip_boxes(boxes, det.image_shape)
    keep = class_aware_nms(boxes, scores, classes, cfg.merge_iou)
    boxes, scores, classes, sources = boxes[keep], scores[keep], classes[keep], sources[keep]

    # Thiết lập đường dẫn lưu kết quả đầu ra
    out_dir = Path(out_dir)
    pred_path = out_dir / "detections" / f"{image_path.stem}.txt"
    viz_path = out_dir / "visualizations" / f"{image_path.stem}.jpg"
    meta_path = out_dir / "metadata" / f"{image_path.stem}.json"
    
    accepted_rois_array = (
        np.stack(accepted_rois).astype(np.float32) if accepted_rois else np.zeros((0, 4), dtype=np.float32)
    )
    rejected_rois_array = (
        np.stack(rejected_rois).astype(np.float32) if rejected_rois else np.zeros((0, 4), dtype=np.float32)
    )
    
    # Lưu tệp txt, hình ảnh trực quan hóa các lát cắt và file json lưu siêu dữ liệu (metadata)
    save_prediction_txt(pred_path, boxes, scores, classes, sources)
    save_inference_visual(image_path, boxes, sources, accepted_rois_array, rejected_rois_array, viz_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    
    meta = {
        "image": str(image_path),
        "num_slices": len(accepted_rois),
        "num_attempts": len(slice_meta),
        "num_rejected_slices": len(rejected_rois),
        "slices": slice_meta,
        "detections": int(len(boxes)),
        "prediction_file": str(pred_path),
        "visualization_file": str(viz_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _resolve_project_path(path: Path | str, root: Path) -> Path:
    """Chuyển đổi đường dẫn tương đối thành tuyệt đối dựa trên thư mục gốc dự án."""
    value = Path(path).expanduser()
    return value if value.is_absolute() else root / value


def _config_path_or_override(cfg: ProjectConfig, key: str, value: Path | str | None) -> Path:
    if value is None:
        return cfg.path_value(key)
    return _resolve_project_path(value, cfg.root)


def _value_or_config(section: dict, key: str, value, cast):
    raw = section[key] if value is None else value
    return cast(raw)


def _feature_layers_or_config(cfg: ProjectConfig, value: tuple[int, ...] | list[int] | str | None) -> tuple[int, ...]:
    if value is None:
        return cfg.feature_layers("infer")
    if isinstance(value, str):
        return tuple(int(x.strip()) for x in value.split(",") if x.strip())
    return tuple(int(x) for x in value)


def _int_tuple_value(value: tuple[int, ...] | list[int] | str | None) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(int(x.strip()) for x in value.split(",") if x.strip())
    return tuple(int(x) for x in value)


def infer_one_image(
    image_path: Path | str,
    weights: Path | str | None = None,
    checkpoint: Path | str | None = None,
    out_dir: Path | str | None = None,
    cache_root: Path | None = None,
    split: str | None = None,
    use_cache: bool | None = None,
    full_imgsz: int | None = None,
    slice_imgsz: int | None = None,
    full_conf: float | None = None,
    output_conf: float | None = None,
    iou: float | None = None,
    merge_iou: float | None = None,
    max_det: int | None = None,
    device: str | None = None,
    feature_layers: tuple[int, ...] | list[int] | str | None = None,
    min_slice_detections: int | None = None,
    min_slice_utility: float | None = None,
    duplicate_iou: float | None = None,
    max_slice_attempts: int | None = None,
    target_classes: tuple[int, ...] | list[int] | str | None = None,
    require_stop_for_acceptance: bool | None = None,
    class_mapping: ClassMapping | None = None,
    config: ProjectConfig | Path | str | None = None,
) -> dict:
    """
    Hàm wrapper đầu ngoài cho phép suy luận RL-SAHI trực tiếp trên 1 ảnh bằng cách chỉ định các tham số ghi đè.
    """
    project_cfg = config if isinstance(config, ProjectConfig) else load_default_config(config)
    infer_cfg = project_cfg.section("infer")
    image_path = _resolve_project_path(image_path, project_cfg.root)
    weights = _config_path_or_override(project_cfg, "weights", weights)
    checkpoint = _config_path_or_override(project_cfg, "checkpoint", checkpoint)
    out_dir = _config_path_or_override(project_cfg, "infer_out_dir", out_dir)
    cache_root = _config_path_or_override(project_cfg, "cache_root", cache_root)
    use_cache = bool(infer_cfg.get("use_cache", True)) if use_cache is None else bool(use_cache)

    # Thiết lập cấu hình InferenceConfig hoàn chỉnh
    cfg = InferenceConfig(
        full_imgsz=_value_or_config(infer_cfg, "full_imgsz", full_imgsz, int),
        slice_imgsz=_value_or_config(infer_cfg, "slice_imgsz", slice_imgsz, int),
        full_conf=_value_or_config(infer_cfg, "full_conf", full_conf, float),
        output_conf=_value_or_config(infer_cfg, "output_conf", output_conf, float),
        iou=_value_or_config(infer_cfg, "iou", iou, float),
        merge_iou=_value_or_config(infer_cfg, "merge_iou", merge_iou, float),
        max_det=_value_or_config(infer_cfg, "max_det", max_det, int),
        device=device if device is not None else project_cfg.optional_str("infer", "device"),
        feature_layers=_feature_layers_or_config(project_cfg, feature_layers),
        min_slice_detections=_value_or_config(infer_cfg, "min_slice_detections", min_slice_detections, int),
        min_slice_utility=(
            float(infer_cfg.get("min_slice_utility", 0.5))
            if min_slice_utility is None
            else float(min_slice_utility)
        ),
        duplicate_iou=(
            float(infer_cfg.get("duplicate_iou", infer_cfg.get("merge_iou", 0.5)))
            if duplicate_iou is None
            else float(duplicate_iou)
        ),
        max_slice_attempts=_value_or_config(infer_cfg, "max_slice_attempts", max_slice_attempts, int),
        target_classes=_int_tuple_value(
            target_classes if target_classes is not None else infer_cfg.get("target_classes", (0, 2, 3, 5, 8, 9))
        ),
        require_stop_for_acceptance=(
            bool(infer_cfg.get("require_stop_for_acceptance", True))
            if require_stop_for_acceptance is None
            else bool(require_stop_for_acceptance)
        ),
        class_mapping=class_mapping or ClassMapping.from_config(project_cfg.section("classes")),
    )
    # Khởi tạo bộ suy luận và chạy
    inferencer = AdaptiveSahiInferencer(weights=weights, checkpoint=checkpoint, cfg=cfg)
    return inferencer.infer_image(
        image_path=image_path,
        out_dir=out_dir,
        cache_root=cache_root,
        split=split,
        use_cache=use_cache,
    )

