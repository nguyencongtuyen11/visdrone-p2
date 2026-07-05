"""Suy luận adaptive-slicing dùng tác nhân TILE-CODING (thay cho Deep QNetwork).

Mirror logic của eval.benchmark._predict_rl_sahi nhưng dùng TileQAgent + tile_features,
và ghi lại metadata từng lát (để vẽ bản đồ ROI). Dùng chung cho infer 1 ảnh & benchmark.
"""
from __future__ import annotations

import numpy as np

from rl_sahi.inference.config import InferenceConfig
from rl_sahi.inference.crops import run_yolo_on_crop
from rl_sahi.inference.merge import merge_predictions_with_sources
from rl_sahi.inference.pipeline import (
    _attempt_overlap,
    _filter_classes,
    _new_detection_gain,
    _new_detection_utility,
)
from rl_sahi.eval.benchmark import _full_predictions
from rl_sahi.rl.slice_env import SliceEnv
from rl_sahi.rl.tile_state import tile_features

_RNG = np.random.default_rng(0)


def build_infer_cfg(cfg, device, target_classes, class_mapping) -> InferenceConfig:
    """Dựng InferenceConfig từ ProjectConfig (dùng chung cho infer/benchmark tile)."""
    ic = cfg.section("infer")
    return InferenceConfig(
        full_imgsz=int(ic["full_imgsz"]), slice_imgsz=int(ic["slice_imgsz"]),
        full_conf=float(ic["full_conf"]), output_conf=float(ic["output_conf"]),
        iou=float(ic["iou"]), merge_iou=float(ic["merge_iou"]), max_det=int(ic["max_det"]),
        device=device, feature_layers=cfg.feature_layers("infer"),
        min_slice_detections=int(ic.get("min_slice_detections", 1)),
        min_slice_utility=float(ic.get("min_slice_utility", 0.5)),
        duplicate_iou=float(ic.get("duplicate_iou", ic.get("merge_iou", 0.5))),
        max_slice_attempts=int(ic.get("max_slice_attempts", 0)),
        target_classes=target_classes,
        require_stop_for_acceptance=bool(ic.get("require_stop_for_acceptance", True)),
        class_mapping=class_mapping,
    )


def _rollout_one_slice(agent, env):
    env.reset()
    f = tile_features(env)
    info = {}
    for _ in range(env.env_cfg.max_steps + 1):
        valid = env.valid_actions()
        a = agent.act(f, valid, 0.0, _RNG)  # greedy
        result = env.step(a)
        f = tile_features(env)
        info = result.info
        if result.done:
            break
    return env.roi.copy(), info


def predict_tile_sahi(agent, model, image_path, det, infer_cfg, env_cfg, state_cfg, record=False):
    """Trả (boxes, scores, classes, crop_count, slices_meta)."""
    full_boxes, full_scores, full_classes = _full_predictions(det, infer_cfg)
    slice_boxes, slice_scores, slice_classes = [], [], []
    accepted_rois, attempted_rois = [], []
    slices_meta = []
    max_attempts = int(infer_cfg.max_slice_attempts) if infer_cfg.max_slice_attempts > 0 else int(env_cfg.max_slices * 2)

    for _attempt in range(1, max_attempts + 1):
        if len(accepted_rois) >= env_cfg.max_slices:
            break
        env = SliceEnv(det, None, env_cfg=env_cfg, state_cfg=state_cfg,
                       previous_rois=np.stack(attempted_rois).astype(np.float32) if attempted_rois else np.zeros((0, 4), np.float32),
                       overlap_rois=np.stack(accepted_rois).astype(np.float32) if accepted_rois else np.zeros((0, 4), np.float32),
                       target_classes=infer_cfg.target_classes, class_mapping=infer_cfg.class_mapping)
        roi, info = _rollout_one_slice(agent, env)

        rejected_reason = None
        if info.get("stop_due_to_old_overlap") or info.get("stop_due_to_attempted_overlap"):
            rejected_reason = "overlap"
        elif infer_cfg.require_stop_for_acceptance and (info.get("stop_due_to_max_steps") or info.get("stop_due_to_stalled_roi")):
            rejected_reason = "no_stop"
        if rejected_reason is not None:
            repeat = _attempt_overlap(roi, attempted_rois)
            attempted_rois.append(roi)
            if record:
                slices_meta.append({"slice_index": len(accepted_rois), "accepted": False,
                                    "rejection_reason": rejected_reason, "roi": [float(v) for v in roi],
                                    "new_detections_after_nms": 0})
            if repeat >= 0.95:
                break
            continue

        boxes_i, scores_i, classes_i = run_yolo_on_crop(model, image_path, roi, imgsz=infer_cfg.slice_imgsz,
                                                        conf=infer_cfg.output_conf, iou=infer_cfg.iou,
                                                        max_det=infer_cfg.max_det, device=infer_cfg.device)
        attempted_rois.append(roi)
        classes_i = infer_cfg.class_mapping.map_model_classes(classes_i)
        boxes_i, scores_i, classes_i = _filter_classes(boxes_i, scores_i, classes_i, infer_cfg.target_classes)
        gain = _new_detection_gain(full_boxes, full_scores, full_classes, slice_boxes, slice_scores, slice_classes,
                                   boxes_i, scores_i, classes_i, det.image_shape, infer_cfg.merge_iou, infer_cfg.duplicate_iou)
        util = _new_detection_utility(full_boxes, full_scores, full_classes, slice_boxes, slice_scores, slice_classes,
                                      boxes_i, scores_i, classes_i, det.image_shape, infer_cfg.merge_iou, infer_cfg.duplicate_iou)
        if gain < int(infer_cfg.min_slice_detections) or util < float(infer_cfg.min_slice_utility):
            if record:
                slices_meta.append({"slice_index": len(accepted_rois), "accepted": False,
                                    "rejection_reason": "low_gain", "roi": [float(v) for v in roi],
                                    "new_detections_after_nms": int(len(boxes_i))})
            continue
        accepted_rois.append(roi)
        slice_boxes.append(boxes_i); slice_scores.append(scores_i); slice_classes.append(classes_i)
        if record:
            slices_meta.append({"slice_index": len(accepted_rois), "accepted": True, "rejection_reason": None,
                                "roi": [float(v) for v in roi], "new_detections_after_nms": int(len(boxes_i))})

    # merge tất cả + source
    parts_b = [full_boxes, *slice_boxes]
    parts_s = [full_scores, *slice_scores]
    parts_c = [full_classes, *slice_classes]
    src = [np.zeros((len(full_boxes),), np.int32)] + [np.full((len(b),), i + 1, np.int32) for i, b in enumerate(slice_boxes)]
    boxes, scores, classes, sources = merge_predictions_with_sources(det.image_shape, infer_cfg.merge_iou, parts_b, parts_s, parts_c, src)
    return boxes, scores, classes, len(accepted_rois), slices_meta, sources
