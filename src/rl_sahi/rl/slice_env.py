from __future__ import annotations

import numpy as np
import torch

from rl_sahi.common.actions import NUM_ACTIONS, Action
from rl_sahi.common.boxes import (
    area,
    as_boxes,
    box_from_center,
    centers,
    intersection_matrix,
    ioa_matrix,
    rasterize_boxes,
    translate_box,
    zoom_box,
)
from rl_sahi.common.cache import DetectionCache, HardRegionCache
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.rl.env_config import EnvConfig, StepResult
from rl_sahi.rl.state_config import StateConfig
from rl_sahi.rl.state_maps import build_detection_map, mark_history, proposal_mask, proposal_quality
from rl_sahi.rl.state_summary import detection_summary
from rl_sahi.rl.state_vector import build_state_vector, normalize_feature


class SliceEnv:
    def __init__(
        self,
        detection: DetectionCache,
        hard_regions: HardRegionCache | None,
        env_cfg: EnvConfig | None = None,
        state_cfg: StateConfig | None = None,
        previous_rois: np.ndarray | None = None,
        overlap_rois: np.ndarray | None = None,
        previous_covered: np.ndarray | None = None,
        target_classes: tuple[int, ...] = (),
        class_mapping: ClassMapping | None = None,
    ) -> None:
        self.detection = detection
        self.hard_regions = hard_regions
        self.env_cfg = env_cfg or EnvConfig()
        self.state_cfg = state_cfg or StateConfig()
        self.target_classes = tuple(int(x) for x in target_classes)
        self.class_mapping = class_mapping or ClassMapping()
        self.image_shape = detection.image_shape
        self.det_boxes, self.det_scores, self.det_classes = self._filtered_detections()
        self.detection_map = build_detection_map(self.det_boxes, self.det_scores, self.image_shape, self.state_cfg)
        self.feature_state = normalize_feature(self.detection.feature)
        self.objectness_state = np.nan_to_num(
            np.asarray(self.detection.objectness_map, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self.spatial_feature_state = np.nan_to_num(
            np.asarray(self.detection.spatial_feature_map, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self.hard_boxes = as_boxes(hard_regions.hard_boxes if hard_regions is not None else np.zeros((0, 4)))
        self.previous_rois = as_boxes(previous_rois if previous_rois is not None else np.zeros((0, 4), dtype=np.float32))
        self.overlap_rois = as_boxes(overlap_rois if overlap_rois is not None else self.previous_rois)
        self.box_device = self._resolve_box_device()
        self.hard_boxes_t = self._box_tensor(self.hard_boxes)
        self.previous_rois_t = self._box_tensor(self.previous_rois)
        self.overlap_rois_t = self._box_tensor(self.overlap_rois)
        self.high_conf_det_boxes_t = self._box_tensor(self.det_boxes[self.det_scores >= self.env_cfg.high_conf_threshold])
        self.attempted_slice_map = self._build_slice_map(self.previous_rois)
        self.accepted_slice_map = self._build_slice_map(self.overlap_rois)
        self.previous_slice_map = self.attempted_slice_map
        self.previous_covered = self._init_previous_covered(previous_covered)
        self.history = np.zeros((self.state_cfg.grid_size, self.state_cfg.grid_size), dtype=np.float32)
        self.covered = self.previous_covered.copy()
        self.roi = self._initial_roi()
        self.step_index = 0

    def _resolve_box_device(self) -> torch.device | None:
        if not bool(getattr(self.env_cfg, "use_gpu_box_ops", True)):
            return None
        name = str(getattr(self.env_cfg, "gpu_box_device", "cuda") or "cuda")
        if name.startswith("cuda") and not torch.cuda.is_available():
            return None
        try:
            device = torch.device(name)
            torch.empty((1,), device=device)
            return device
        except Exception:
            return None

    def _box_tensor(self, boxes: np.ndarray) -> torch.Tensor | None:
        if self.box_device is None:
            return None
        boxes = as_boxes(boxes)
        if len(boxes) == 0:
            return torch.zeros((0, 4), dtype=torch.float32, device=self.box_device)
        return torch.as_tensor(boxes, dtype=torch.float32, device=self.box_device)

    def _filtered_detections(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        boxes = as_boxes(self.detection.boxes)
        scores = np.asarray(self.detection.scores, dtype=np.float32).reshape(-1)
        classes = self.class_mapping.map_model_classes(self.detection.classes)
        if not self.target_classes:
            return boxes, scores, classes
        target = np.asarray(self.target_classes, dtype=np.int64)
        mask = np.isin(classes.astype(np.int64), target)
        return boxes[mask], scores[mask], classes[mask]

    def reset(self) -> np.ndarray:
        self.history.fill(0.0)
        self.covered = self.previous_covered.copy()
        self.roi = self._initial_roi()
        self.step_index = 0
        self.history = mark_history(self.history, self.roi, self.image_shape, self.state_cfg.grid_size)
        return self._state()

    def step(self, action: int | Action) -> StepResult:
        action = Action(int(action))
        done = action == Action.STOP
        stalled_roi = False
        previous_roi = self.roi.copy()
        if action != Action.STOP:
            self.roi = self._apply_action(action)
            stalled_roi = bool(np.allclose(previous_roi, self.roi, atol=1e-3))
            self.step_index += 1
            self.history = mark_history(self.history, self.roi, self.image_shape, self.state_cfg.grid_size)

        reward, info = self._reward(action, previous_roi)
        if stalled_roi:
            done = True
            reward = min(reward, -1e-6) - self.env_cfg.stalled_without_stop_penalty
        if info["old_slice_overlap"] >= self.env_cfg.old_slice_overlap_threshold:
            done = True
            info["stop_due_to_old_overlap"] = True
            reward = min(reward, -1e-6) - self.env_cfg.constraint_weight
        else:
            info["stop_due_to_old_overlap"] = False
        if action == Action.STOP and info["attempted_slice_overlap"] >= self.env_cfg.old_slice_overlap_threshold:
            info["stop_due_to_attempted_overlap"] = True
            reward = min(reward, -1e-6) - self.env_cfg.attempted_overlap_penalty
        else:
            info["stop_due_to_attempted_overlap"] = False
        if self.step_index >= self.env_cfg.max_steps:
            done = True
            info["stop_due_to_max_steps"] = action != Action.STOP
            if info["stop_due_to_max_steps"]:
                reward = min(reward, -1e-6) - self.env_cfg.max_steps_without_stop_penalty
        else:
            info["stop_due_to_max_steps"] = False
        info["stop_due_to_stalled_roi"] = stalled_roi
        info["roi"] = self.roi.copy()
        info["covered"] = int(self.covered.sum())
        info["hard_total"] = int(len(self.hard_boxes))
        return StepResult(self._state(), reward, done, info)

    def valid_actions(self) -> np.ndarray:
        valid = np.ones((NUM_ACTIONS,), dtype=bool)
        non_stop_actions = [action for action in Action if action != Action.STOP]
        next_rois = np.stack([self._apply_action(action) for action in non_stop_actions]).astype(np.float32)
        stalled = np.all(np.isclose(next_rois, self.roi.reshape(1, 4), atol=1e-3), axis=1)
        old_overlaps = self._old_slice_overlaps(next_rois)
        attempted_overlaps = self._attempted_slice_overlaps(next_rois)
        next_overlaps = np.maximum(old_overlaps, attempted_overlaps)
        for idx, action in enumerate(non_stop_actions):
            if stalled[idx]:
                valid[int(action)] = False
        if np.any((~stalled) & (next_overlaps < self.env_cfg.old_slice_overlap_threshold)):
            for idx, action in enumerate(non_stop_actions):
                if next_overlaps[idx] >= self.env_cfg.old_slice_overlap_threshold:
                    valid[int(action)] = False
        overlap = max(self._old_slice_overlap(), self._attempted_slice_overlap())
        non_stop_valid = bool(valid[[int(a) for a in non_stop_actions]].any())
        valid[int(Action.STOP)] = not (
            overlap >= self.env_cfg.old_slice_overlap_threshold and non_stop_valid
        )
        return valid

    def _valid_actions_legacy(self) -> np.ndarray:
        valid = np.ones((NUM_ACTIONS,), dtype=bool)
        non_stop_actions: list[Action] = []
        next_overlaps: dict[Action, float] = {}
        for action in Action:
            if action == Action.STOP:
                continue
            next_roi = self._apply_action(action)
            if np.allclose(next_roi, self.roi, atol=1e-3):
                valid[int(action)] = False
                continue
            non_stop_actions.append(action)
            next_overlaps[action] = max(self._old_slice_overlap(next_roi), self._attempted_slice_overlap(next_roi))
        if any(next_overlaps[action] < self.env_cfg.old_slice_overlap_threshold for action in non_stop_actions):
            for action in non_stop_actions:
                if next_overlaps[action] >= self.env_cfg.old_slice_overlap_threshold:
                    valid[int(action)] = False
        overlap = max(self._old_slice_overlap(), self._attempted_slice_overlap())
        non_stop_valid = bool(valid[[int(a) for a in Action if a != Action.STOP]].any())
        valid[int(Action.STOP)] = not (
            overlap >= self.env_cfg.old_slice_overlap_threshold and non_stop_valid
        )
        return valid

    def guided_action(self) -> Action:
        escape = self._overlap_escape_action()
        if escape is not None:
            return escape
        heatmap_target = self._heatmap_target()
        boxes = self.det_boxes
        scores = self.det_scores
        valid_mask = scores >= self.state_cfg.proposal_min_conf
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        if len(boxes) == 0:
            return self._action_toward_target(heatmap_target[0]) if heatmap_target is not None else Action.STOP

        image_area = max(float(self.image_shape[0] * self.image_shape[1]), 1.0)
        det_area_ratio = area(boxes) / image_area
        prop_mask = proposal_mask(scores, self.state_cfg)
        small_mask = det_area_ratio <= self.state_cfg.small_area_ratio
        target_mask = prop_mask | (small_mask & (scores < self.env_cfg.high_conf_threshold))
        if not target_mask.any():
            return self._action_toward_target(heatmap_target[0]) if heatmap_target is not None else Action.STOP

        candidate_boxes = boxes[target_mask]
        candidate_scores = scores[target_mask]
        candidate_centers = centers(candidate_boxes)
        if len(self.previous_rois) > 0:
            old_seen = self._points_in_previous_rois(candidate_centers)
        else:
            old_seen = np.zeros((len(candidate_boxes),), dtype=bool)

        roi_center = centers(self.roi.reshape(1, 4))[0]
        distances = np.linalg.norm(candidate_centers - roi_center[None, :], axis=1)
        quality = proposal_quality(candidate_scores, self.state_cfg)
        heat_support = self._objectness_values_at_points(candidate_centers)
        density_support = self._proposal_density_values_at_points(candidate_centers)
        high_seen = self._points_in_boxes(
            candidate_centers,
            boxes[scores >= self.env_cfg.high_conf_threshold],
        )
        priority = quality
        priority += small_mask[target_mask].astype(np.float32) * 0.5
        priority += heat_support * 0.5
        priority += density_support * 0.75
        priority -= distances / max(min(self.image_shape), 1)
        if old_seen.any() and (~old_seen).any():
            priority = priority.copy()
            priority[old_seen] = -np.inf
        else:
            priority -= old_seen.astype(np.float32) * 2.0
        priority -= high_seen.astype(np.float32) * 1.0
        target_idx = int(priority.argmax())
        if heatmap_target is not None:
            heat_point, heat_score = heatmap_target
            heat_distance = float(np.linalg.norm(heat_point - roi_center) / max(min(self.image_shape), 1))
            heat_priority = float(heat_score - heat_distance)
            if heat_priority > float(priority[target_idx]):
                return self._action_toward_target(heat_point)
        if priority[target_idx] < -1.5:
            return Action.STOP
        return self._action_toward_target(candidate_centers[target_idx], candidate_boxes[[target_idx]])

    def _heatmap_target(self) -> tuple[np.ndarray, float] | None:
        obj = np.asarray(self.detection.objectness_map, dtype=np.float32)
        if obj.size == 0:
            return None
        grid_size = self.state_cfg.grid_size
        obj = np.nan_to_num(obj.reshape(-1, grid_size, grid_size), nan=0.0, posinf=0.0, neginf=0.0)
        heat = obj.max(axis=0)
        if self.detection_map.shape[0] > 2:
            density = np.clip(self.detection_map[2] * self.state_cfg.count_norm / 10.0, 0.0, 1.0)
            heat = np.maximum(heat * 0.7, density)
        if heat.size == 0:
            return None
        priority = heat.copy()
        previous_map = np.asarray(self.previous_slice_map, dtype=np.float32)
        priority -= 0.6 * previous_map
        priority -= 0.2 * np.asarray(self.history, dtype=np.float32)
        blocked = previous_map >= 0.5
        if blocked.any() and (~blocked).any():
            masked = priority.copy()
            masked[blocked] = -np.inf
            finite = np.isfinite(masked)
            if finite.any() and float(masked[finite].max()) > 0.02:
                priority = masked
        y, x = np.unravel_index(int(priority.argmax()), priority.shape)
        score = float(priority[y, x])
        if score <= 0.02:
            return None
        h, w = self.image_shape
        target = np.array([(x + 0.5) * w / grid_size, (y + 0.5) * h / grid_size], dtype=np.float32)
        return target, score

    def _proposal_seed_target(self) -> np.ndarray | None:
        boxes = self.det_boxes
        scores = self.det_scores
        valid_mask = scores >= self.state_cfg.proposal_min_conf
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        if len(boxes) == 0:
            return None
        image_area = max(float(self.image_shape[0] * self.image_shape[1]), 1.0)
        det_area_ratio = area(boxes) / image_area
        prop_mask = proposal_mask(scores, self.state_cfg)
        small_mask = det_area_ratio <= self.state_cfg.small_area_ratio
        target_mask = prop_mask | (small_mask & (scores < self.env_cfg.high_conf_threshold))
        if not target_mask.any():
            return None
        candidate_boxes = boxes[target_mask]
        candidate_scores = scores[target_mask]
        candidate_centers = centers(candidate_boxes)
        priority = proposal_quality(candidate_scores, self.state_cfg)
        priority += small_mask[target_mask].astype(np.float32) * 0.5
        priority += self._objectness_values_at_points(candidate_centers) * 0.5
        priority += self._proposal_density_values_at_points(candidate_centers) * 0.75
        if len(self.previous_rois) > 0:
            old_seen = self._points_in_previous_rois(candidate_centers)
            if old_seen.any() and (~old_seen).any():
                priority = priority.copy()
                priority[old_seen] = -np.inf
            else:
                priority -= old_seen.astype(np.float32) * 2.0
        if len(priority) == 0 or float(priority.max()) <= 0.0:
            return None
        return candidate_centers[int(priority.argmax())].astype(np.float32)

    def _objectness_values_at_points(self, points: np.ndarray) -> np.ndarray:
        obj = np.asarray(self.detection.objectness_map, dtype=np.float32)
        if obj.size == 0:
            return np.zeros((len(points),), dtype=np.float32)
        grid = obj.reshape(-1, self.state_cfg.grid_size, self.state_cfg.grid_size).max(axis=0)
        return self._grid_values_at_points(grid, points)

    def _proposal_density_values_at_points(self, points: np.ndarray) -> np.ndarray:
        if self.detection_map.shape[0] <= 2:
            return np.zeros((len(points),), dtype=np.float32)
        density = np.clip(self.detection_map[2] * self.state_cfg.count_norm / 10.0, 0.0, 1.0)
        return self._grid_values_at_points(density, points)

    def _grid_values_at_points(self, grid: np.ndarray, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(points) == 0:
            return np.zeros((0,), dtype=np.float32)
        grid = np.asarray(grid, dtype=np.float32).reshape(self.state_cfg.grid_size, self.state_cfg.grid_size)
        h, w = self.image_shape
        xs = np.clip((points[:, 0] / max(w, 1)) * self.state_cfg.grid_size, 0, self.state_cfg.grid_size - 1).astype(int)
        ys = np.clip((points[:, 1] / max(h, 1)) * self.state_cfg.grid_size, 0, self.state_cfg.grid_size - 1).astype(int)
        return grid[ys, xs].astype(np.float32)

    def _points_in_boxes(self, points: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        boxes = as_boxes(boxes)
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(points) == 0 or len(boxes) == 0:
            return np.zeros((len(points),), dtype=bool)
        mask = np.zeros((len(points),), dtype=bool)
        for box in boxes:
            mask |= (
                (points[:, 0] >= box[0])
                & (points[:, 0] <= box[2])
                & (points[:, 1] >= box[1])
                & (points[:, 1] <= box[3])
            )
        return mask

    def _action_toward_target(self, target: np.ndarray, target_box: np.ndarray | None = None) -> Action:
        target = np.asarray(target, dtype=np.float32).reshape(2)
        roi_center = centers(self.roi.reshape(1, 4))[0]
        x1, y1, x2, y2 = self.roi
        inside = x1 <= target[0] <= x2 and y1 <= target[1] <= y2
        if inside:
            if self._roi_area_ratio() > self.env_cfg.max_roi_area_ratio or self._scale_gain() < self.env_cfg.min_scale_gain:
                return Action.ZOOM_IN
            if target_box is not None:
                projected_size = self._projected_sizes(target_box)[0]
                if projected_size < self.env_cfg.target_projected_size:
                    return Action.ZOOM_IN
                if projected_size > self.env_cfg.max_projected_size:
                    return Action.ZOOM_OUT
            min_side, _max_side = self._side_limits()
            if self._roi_side() > min_side * 1.25:
                return Action.ZOOM_IN
            return Action.STOP

        dx = target[0] - roi_center[0]
        dy = target[1] - roi_center[1]
        side = self._roi_side()
        if abs(dx) > side * 0.1 and abs(dy) > side * 0.1:
            if dx < 0 and dy < 0:
                return Action.UP_LEFT
            if dx > 0 and dy < 0:
                return Action.UP_RIGHT
            if dx < 0 and dy > 0:
                return Action.DOWN_LEFT
            return Action.DOWN_RIGHT
        if abs(dx) >= abs(dy):
            return Action.RIGHT if dx > 0 else Action.LEFT
        return Action.DOWN if dy > 0 else Action.UP

    def _overlap_escape_action(self) -> Action | None:
        current_overlap = max(self._old_slice_overlap(), self._attempted_slice_overlap())
        if current_overlap < self.env_cfg.old_slice_overlap_threshold:
            return None
        non_stop_actions = [action for action in Action if action != Action.STOP]
        next_rois = np.stack([self._apply_action(action) for action in non_stop_actions]).astype(np.float32)
        stalled = np.all(np.isclose(next_rois, self.roi.reshape(1, 4), atol=1e-3), axis=1)
        overlaps = np.maximum(self._old_slice_overlaps(next_rois), self._attempted_slice_overlaps(next_rois))
        overlaps = np.where(stalled, np.inf, overlaps)
        best_idx = int(overlaps.argmin())
        if float(overlaps[best_idx]) < current_overlap:
            return non_stop_actions[best_idx]
        return None

    def _initial_roi(self) -> np.ndarray:
        h, w = self.image_shape
        _min_side, max_side = self._side_limits()
        side = min(min(h, w) * self.env_cfg.initial_slice_fraction, max_side)
        heatmap_target = self._heatmap_target()
        target = heatmap_target[0] if heatmap_target is not None else self._proposal_seed_target()
        if target is None:
            target = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        return box_from_center(float(target[0]), float(target[1]), side, self.image_shape)

    def _apply_action(self, action: Action) -> np.ndarray:
        side = self._roi_side()
        step = side * self.env_cfg.move_fraction
        if action == Action.LEFT:
            return translate_box(self.roi, -step, 0.0, self.image_shape)
        if action == Action.RIGHT:
            return translate_box(self.roi, step, 0.0, self.image_shape)
        if action == Action.UP:
            return translate_box(self.roi, 0.0, -step, self.image_shape)
        if action == Action.DOWN:
            return translate_box(self.roi, 0.0, step, self.image_shape)
        diag_step = step / float(np.sqrt(2.0))
        if action == Action.UP_LEFT:
            return translate_box(self.roi, -diag_step, -diag_step, self.image_shape)
        if action == Action.UP_RIGHT:
            return translate_box(self.roi, diag_step, -diag_step, self.image_shape)
        if action == Action.DOWN_LEFT:
            return translate_box(self.roi, -diag_step, diag_step, self.image_shape)
        if action == Action.DOWN_RIGHT:
            return translate_box(self.roi, diag_step, diag_step, self.image_shape)
        min_side, max_side = self._side_limits()
        if action == Action.ZOOM_IN:
            return zoom_box(self.roi, self.env_cfg.zoom_factor, self.image_shape, min_side, max_side)
        if action == Action.ZOOM_OUT:
            return zoom_box(self.roi, 1.0 / self.env_cfg.zoom_factor, self.image_shape, min_side, max_side)
        return self.roi

    def _side_limits(self) -> tuple[float, float]:
        h, w = self.image_shape
        min_side = min(h, w) * self.env_cfg.min_slice_fraction
        max_side_by_fraction = min(h, w) * self.env_cfg.max_slice_fraction
        max_side_by_area = np.sqrt(max(float(h * w) * self.env_cfg.max_roi_area_ratio, 1.0))
        max_side = max(min(max_side_by_fraction, max_side_by_area), min_side)
        return float(min_side), float(max_side)

    def _build_slice_map(self, rois: np.ndarray) -> np.ndarray:
        if len(rois) == 0:
            return np.zeros((self.state_cfg.grid_size, self.state_cfg.grid_size), dtype=np.float32)
        return rasterize_boxes(rois, self.image_shape, self.state_cfg.grid_size)

    def _current_roi_map(self) -> np.ndarray:
        return rasterize_boxes(self.roi.reshape(1, 4), self.image_shape, self.state_cfg.grid_size)

    def _init_previous_covered(self, previous_covered: np.ndarray | None) -> np.ndarray:
        if len(self.hard_boxes) == 0:
            return np.zeros((0,), dtype=bool)
        if previous_covered is not None:
            arr = np.asarray(previous_covered, dtype=bool).reshape(-1)
            if len(arr) != len(self.hard_boxes):
                raise ValueError("previous_covered length must match hard region count")
            return arr.copy()
        covered = np.zeros((len(self.hard_boxes),), dtype=bool)
        for roi in self.previous_rois:
            _scores, hit_mask = self._hard_target_scores(roi)
            covered |= hit_mask
        return covered

    def _points_in_previous_rois(self, points: np.ndarray) -> np.ndarray:
        mask = np.zeros((len(points),), dtype=bool)
        for roi in self.previous_rois:
            mask |= (
                (points[:, 0] >= roi[0])
                & (points[:, 0] <= roi[2])
                & (points[:, 1] >= roi[1])
                & (points[:, 1] <= roi[3])
            )
        return mask

    def _history_overlap_values(self, rois: np.ndarray, history: np.ndarray, history_t: torch.Tensor | None) -> np.ndarray:
        rois = as_boxes(rois)
        if len(rois) == 0:
            return np.zeros((0,), dtype=np.float32)
        if len(history) == 0:
            return np.zeros((len(rois),), dtype=np.float32)
        if self.box_device is not None and history_t is not None:
            rois_t = torch.as_tensor(rois, dtype=torch.float32, device=self.box_device)
            x1 = torch.maximum(rois_t[:, None, 0], history_t[None, :, 0])
            y1 = torch.maximum(rois_t[:, None, 1], history_t[None, :, 1])
            x2 = torch.minimum(rois_t[:, None, 2], history_t[None, :, 2])
            y2 = torch.minimum(rois_t[:, None, 3], history_t[None, :, 3])
            inter = (x2 - x1).clamp_min(0.0) * (y2 - y1).clamp_min(0.0)
            roi_area = ((rois_t[:, 2] - rois_t[:, 0]).clamp_min(0.0) * (rois_t[:, 3] - rois_t[:, 1]).clamp_min(0.0)).clamp_min(1.0)
            values = (inter.max(dim=1).values / roi_area).clamp(0.0, 1.0)
            return values.detach().cpu().numpy().astype(np.float32)
        inter = intersection_matrix(rois, history)
        current_area = np.maximum(area(rois), 1.0)
        return np.clip(inter.max(axis=1) / current_area, 0.0, 1.0).astype(np.float32)

    def _old_slice_overlaps(self, rois: np.ndarray) -> np.ndarray:
        return self._history_overlap_values(rois, self.overlap_rois, self.overlap_rois_t)

    def _attempted_slice_overlaps(self, rois: np.ndarray) -> np.ndarray:
        return self._history_overlap_values(rois, self.previous_rois, self.previous_rois_t)

    def _old_slice_overlap(self, roi: np.ndarray | None = None) -> float:
        roi = self.roi if roi is None else np.asarray(roi, dtype=np.float32).reshape(4)
        return float(self._old_slice_overlaps(roi.reshape(1, 4))[0])

    def _attempted_slice_overlap(self, roi: np.ndarray | None = None) -> float:
        roi = self.roi if roi is None else np.asarray(roi, dtype=np.float32).reshape(4)
        return float(self._attempted_slice_overlaps(roi.reshape(1, 4))[0])

    def _roi_side(self, roi: np.ndarray | None = None) -> float:
        roi = self.roi if roi is None else np.asarray(roi, dtype=np.float32).reshape(4)
        return max(float(roi[2] - roi[0]), float(roi[3] - roi[1]), 1.0)

    def _roi_area_ratio(self, roi: np.ndarray | None = None) -> float:
        roi = self.roi if roi is None else np.asarray(roi, dtype=np.float32).reshape(4)
        image_area = max(float(self.image_shape[0] * self.image_shape[1]), 1.0)
        return float(area(roi.reshape(1, 4))[0] / image_area)

    def _scale_gain(self, roi: np.ndarray | None = None) -> float:
        return float(min(self.image_shape) / self._roi_side(roi))

    def _projected_sizes(self, boxes: np.ndarray, roi: np.ndarray | None = None) -> np.ndarray:
        boxes = as_boxes(boxes)
        if len(boxes) == 0:
            return np.zeros((0,), dtype=np.float32)
        widths = np.maximum(boxes[:, 2] - boxes[:, 0], 1.0)
        heights = np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)
        return (np.maximum(widths, heights) * float(self.env_cfg.reward_imgsz) / self._roi_side(roi)).astype(np.float32)

    def _projected_size_scores(self, boxes: np.ndarray, roi: np.ndarray | None = None) -> np.ndarray:
        projected = self._projected_sizes(boxes, roi)
        if len(projected) == 0:
            return projected
        cfg = self.env_cfg
        below_target = (projected - cfg.min_projected_size) / max(cfg.target_projected_size - cfg.min_projected_size, 1e-6)
        above_target = (cfg.max_projected_size - projected) / max(cfg.max_projected_size - cfg.target_projected_size, 1e-6)
        return np.clip(np.minimum(below_target, above_target), 0.0, 1.0).astype(np.float32)

    def _center_context_mask(self, boxes: np.ndarray, roi: np.ndarray | None = None) -> np.ndarray:
        boxes = as_boxes(boxes)
        if len(boxes) == 0:
            return np.zeros((0,), dtype=bool)
        x1, y1, x2, y2 = self.roi if roi is None else np.asarray(roi, dtype=np.float32).reshape(4)
        width = max(float(x2 - x1), 1.0)
        height = max(float(y2 - y1), 1.0)
        margin_x = width * self.env_cfg.context_margin
        margin_y = height * self.env_cfg.context_margin
        pts = centers(boxes)
        return (
            (pts[:, 0] >= x1 + margin_x)
            & (pts[:, 0] <= x2 - margin_x)
            & (pts[:, 1] >= y1 + margin_y)
            & (pts[:, 1] <= y2 - margin_y)
        )

    def _hard_target_scores(self, roi: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        if len(self.hard_boxes) == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=bool)
        if self.box_device is not None and self.hard_boxes_t is not None:
            return self._hard_target_scores_torch(roi)
        if self._roi_area_ratio(roi) > self.env_cfg.max_roi_area_ratio or self._scale_gain(roi) < self.env_cfg.min_scale_gain:
            return np.zeros((len(self.hard_boxes),), dtype=np.float32), np.zeros((len(self.hard_boxes),), dtype=bool)
        context_mask = self._center_context_mask(self.hard_boxes, roi)
        size_scores = self._projected_size_scores(self.hard_boxes, roi)
        target_scores = np.where(context_mask, size_scores, 0.0).astype(np.float32)
        return target_scores, target_scores > 0.0

    def _hard_target_scores_torch(self, roi: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        assert self.box_device is not None and self.hard_boxes_t is not None
        roi_np = self.roi if roi is None else np.asarray(roi, dtype=np.float32).reshape(4)
        roi_t = torch.as_tensor(roi_np, dtype=torch.float32, device=self.box_device)
        width = (roi_t[2] - roi_t[0]).clamp_min(1.0)
        height = (roi_t[3] - roi_t[1]).clamp_min(1.0)
        side = torch.maximum(width, height).clamp_min(1.0)
        image_area = max(float(self.image_shape[0] * self.image_shape[1]), 1.0)
        roi_area_ratio = ((roi_t[2] - roi_t[0]).clamp_min(0.0) * (roi_t[3] - roi_t[1]).clamp_min(0.0)) / image_area
        scale_gain = float(min(self.image_shape)) / side
        if bool(roi_area_ratio > self.env_cfg.max_roi_area_ratio) or bool(scale_gain < self.env_cfg.min_scale_gain):
            return np.zeros((len(self.hard_boxes),), dtype=np.float32), np.zeros((len(self.hard_boxes),), dtype=bool)

        margin_x = width * self.env_cfg.context_margin
        margin_y = height * self.env_cfg.context_margin
        boxes = self.hard_boxes_t
        pts_x = (boxes[:, 0] + boxes[:, 2]) * 0.5
        pts_y = (boxes[:, 1] + boxes[:, 3]) * 0.5
        context_mask = (
            (pts_x >= roi_t[0] + margin_x)
            & (pts_x <= roi_t[2] - margin_x)
            & (pts_y >= roi_t[1] + margin_y)
            & (pts_y <= roi_t[3] - margin_y)
        )
        box_widths = (boxes[:, 2] - boxes[:, 0]).clamp_min(1.0)
        box_heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(1.0)
        projected = torch.maximum(box_widths, box_heights) * float(self.env_cfg.reward_imgsz) / side
        below_target = (projected - self.env_cfg.min_projected_size) / max(
            self.env_cfg.target_projected_size - self.env_cfg.min_projected_size,
            1e-6,
        )
        above_target = (self.env_cfg.max_projected_size - projected) / max(
            self.env_cfg.max_projected_size - self.env_cfg.target_projected_size,
            1e-6,
        )
        size_scores = torch.minimum(below_target, above_target).clamp(0.0, 1.0)
        target_scores = torch.where(context_mask, size_scores, torch.zeros_like(size_scores))
        target_scores_np = target_scores.detach().cpu().numpy().astype(np.float32)
        return target_scores_np, target_scores_np > 0.0

    def _detected_overlap_score(self) -> float:
        det_mask = self.det_scores >= self.env_cfg.high_conf_threshold
        if not det_mask.any():
            return 0.0
        if self.box_device is not None and self.high_conf_det_boxes_t is not None and len(self.high_conf_det_boxes_t) > 0:
            roi_t = torch.as_tensor(self.roi.reshape(1, 4), dtype=torch.float32, device=self.box_device)
            boxes = self.high_conf_det_boxes_t
            x1 = torch.maximum(roi_t[:, None, 0], boxes[None, :, 0])
            y1 = torch.maximum(roi_t[:, None, 1], boxes[None, :, 1])
            x2 = torch.minimum(roi_t[:, None, 2], boxes[None, :, 2])
            y2 = torch.minimum(roi_t[:, None, 3], boxes[None, :, 3])
            inter = (x2 - x1).clamp_min(0.0) * (y2 - y1).clamp_min(0.0)
            box_area = ((boxes[:, 2] - boxes[:, 0]).clamp_min(0.0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0.0)).clamp_min(1e-6)
            cover = inter[0] / box_area
            return float((cover.sum() / max(int(boxes.shape[0]), 1)).clamp(0.0, 1.0).item())
        det_boxes = self.det_boxes[det_mask]
        det_cover = ioa_matrix(self.roi.reshape(1, 4), det_boxes)[0]
        return float(np.clip(det_cover.sum() / max(len(det_boxes), 1), 0.0, 1.0))

    def _roi_grid_window(self, roi: np.ndarray | None = None) -> tuple[int, int, int, int]:
        roi = self.roi if roi is None else np.asarray(roi, dtype=np.float32).reshape(4)
        h, w = self.image_shape
        grid = self.state_cfg.grid_size
        x1 = int(np.floor(np.clip(roi[0] / max(w, 1), 0.0, 1.0) * grid))
        y1 = int(np.floor(np.clip(roi[1] / max(h, 1), 0.0, 1.0) * grid))
        x2 = int(np.ceil(np.clip(roi[2] / max(w, 1), 0.0, 1.0) * grid))
        y2 = int(np.ceil(np.clip(roi[3] / max(h, 1), 0.0, 1.0) * grid))
        x1 = int(np.clip(x1, 0, grid - 1))
        y1 = int(np.clip(y1, 0, grid - 1))
        x2 = int(np.clip(max(x2, x1 + 1), 1, grid))
        y2 = int(np.clip(max(y2, y1 + 1), 1, grid))
        return y1, y2, x1, x2

    def _objectness_roi_score(self, roi: np.ndarray | None = None) -> float:
        obj = np.asarray(self.detection.objectness_map, dtype=np.float32)
        if obj.size == 0:
            return 0.0
        grid = np.nan_to_num(
            obj.reshape(-1, self.state_cfg.grid_size, self.state_cfg.grid_size).max(axis=0),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        y1, y2, x1, x2 = self._roi_grid_window(roi)
        window = grid[y1:y2, x1:x2]
        return float(np.clip(window.max() if window.size else 0.0, 0.0, 1.0))

    def _observable_target_score(self, roi: np.ndarray | None = None) -> float:
        roi = self.roi if roi is None else np.asarray(roi, dtype=np.float32).reshape(4)
        scores = np.asarray(self.det_scores, dtype=np.float32).reshape(-1)
        boxes = as_boxes(self.det_boxes)
        proposal_score = 0.0
        if len(boxes) > 0:
            image_area = max(float(self.image_shape[0] * self.image_shape[1]), 1.0)
            det_area_ratio = area(boxes) / image_area
            prop_mask = proposal_mask(scores, self.state_cfg)
            small_uncertain = (det_area_ratio <= self.state_cfg.small_area_ratio) & (
                scores < self.env_cfg.high_conf_threshold
            )
            target_mask = prop_mask | small_uncertain
            if target_mask.any():
                candidate_boxes = boxes[target_mask]
                candidate_scores = scores[target_mask]
                center_mask = self._center_context_mask(candidate_boxes, roi)
                if center_mask.any():
                    values = proposal_quality(candidate_scores, self.state_cfg)
                    values += small_uncertain[target_mask].astype(np.float32) * 0.5
                    proposal_score = float(values[center_mask].sum())
                    roi_area_ratio = max(self._roi_area_ratio(roi), 1e-6)
                    density_gain = self.env_cfg.max_roi_area_ratio / roi_area_ratio
                    proposal_score *= float(np.clip(density_gain, 0.25, 2.0))
        objectness_score = self._objectness_roi_score(roi)
        return float(np.clip(proposal_score + objectness_score, 0.0, 4.0))

    def _state(self) -> np.ndarray:
        summary = detection_summary(
            boxes=self.det_boxes,
            scores=self.det_scores,
            roi=self.roi,
            history=self.history,
            previous_slice_map=self.previous_slice_map,
            image_shape=self.image_shape,
            step_index=self.step_index,
            max_steps=self.env_cfg.max_steps,
            old_slice_overlap=self._old_slice_overlap(),
            scale_gain=self._scale_gain(),
            previous_slice_count=len(self.previous_rois),
            cfg=self.state_cfg,
        )
        return build_state_vector(
            self.feature_state,
            self.history,
            self._current_roi_map(),
            self.attempted_slice_map,
            self.accepted_slice_map,
            self.detection_map,
            self.objectness_state,
            self.spatial_feature_state,
            summary,
            static_ready=True,
        )

    def _update_covered(self, commit: bool = True) -> tuple[np.ndarray, np.ndarray]:
        target_scores, hit_mask = self._hard_target_scores()
        if commit:
            self.covered |= hit_mask
        return target_scores, hit_mask

    def _reward(self, action: Action, previous_roi: np.ndarray | None = None) -> tuple[float, dict]:
        if self.env_cfg.use_simplified_reward:
            return self._simplified_reward(action, previous_roi)
        return self._legacy_reward(action, previous_roi)

    def _simplified_reward(self, action: Action, previous_roi: np.ndarray | None = None) -> tuple[float, dict]:
        prev_covered = self.covered.copy()
        commit_hits = action == Action.STOP
        target_scores, hit_mask = self._update_covered(commit=commit_hits)
        candidate_new_mask = hit_mask & ~prev_covered
        new_mask = candidate_new_mask if commit_hits else np.zeros_like(candidate_new_mask, dtype=bool)
        new_hits = int(new_mask.sum())
        candidate_hits = int(candidate_new_mask.sum())
        target_score = float(target_scores[new_mask].sum()) if len(target_scores) else 0.0
        hit_count = int(hit_mask.sum()) if len(hit_mask) else 0
        total_target_score = float(target_scores[hit_mask].sum()) if len(target_scores) else 0.0
        roi_area_ratio = self._roi_area_ratio()
        compactness = 1.0 - min(roi_area_ratio / max(self.env_cfg.max_roi_area_ratio, 1e-6), 1.0)
        compactness_score = total_target_score * compactness
        previous_compactness_score = 0.0
        if previous_roi is not None and len(self.hard_boxes) > 0:
            previous_scores, previous_hit_mask = self._hard_target_scores(previous_roi)
            previous_total_score = (
                float(previous_scores[previous_hit_mask].sum()) if len(previous_scores) else 0.0
            )
            previous_area_ratio = self._roi_area_ratio(previous_roi)
            previous_compactness = 1.0 - min(
                previous_area_ratio / max(self.env_cfg.max_roi_area_ratio, 1e-6),
                1.0,
            )
            previous_compactness_score = previous_total_score * previous_compactness
        compactness_delta = compactness_score - previous_compactness_score
        scale_gain = self._scale_gain()
        old_slice_overlap = self._old_slice_overlap()
        attempted_slice_overlap = self._attempted_slice_overlap()
        observable_score = self._observable_target_score()
        detected_overlap = self._detected_overlap_score()

        cfg = self.env_cfg
        reward = 0.0
        info = {
            "new_hits": new_hits,
            "candidate_hits": candidate_hits,
            "hit_count": hit_count,
            "target_score": target_score,
            "total_target_score": total_target_score,
            "retained_hits": int((hit_mask & prev_covered).sum()) if len(hit_mask) else 0,
            "compactness_score": compactness_score,
            "compactness_delta": compactness_delta,
            "observable_score": observable_score,
            "observable_delta": 0.0,
            "roi_area_ratio": roi_area_ratio,
            "scale_gain": scale_gain,
            "old_slice_overlap": old_slice_overlap,
            "attempted_slice_overlap": attempted_slice_overlap,
            "detected_overlap": detected_overlap,
        }

        if action == Action.STOP and new_hits > 0:
            reward += cfg.target_reward * float(new_hits)
            density = target_score * cfg.max_roi_area_ratio / max(roi_area_ratio, 1e-6)
            reward += cfg.target_reward * 0.3 * float(np.clip(density, 0.0, 3.0))

        step_cost = 0.05 + roi_area_ratio * 0.5
        reward -= cfg.efficiency_weight * step_cost

        constraint_penalty = 0.0
        if roi_area_ratio > cfg.max_roi_area_ratio:
            overflow = roi_area_ratio / max(cfg.max_roi_area_ratio, 1e-6) - 1.0
            constraint_penalty += overflow
        if scale_gain < cfg.min_scale_gain:
            under = cfg.min_scale_gain / max(scale_gain, 1e-6) - 1.0
            constraint_penalty += under
        if old_slice_overlap >= cfg.old_slice_overlap_threshold:
            constraint_penalty += 1.0
        if attempted_slice_overlap >= cfg.old_slice_overlap_threshold:
            constraint_penalty += cfg.attempted_overlap_penalty / max(cfg.constraint_weight, 1e-6)
        reward -= cfg.constraint_weight * constraint_penalty
        reward -= cfg.detected_overlap_penalty * detected_overlap

        if action == Action.STOP:
            if total_target_score > 0.0 and old_slice_overlap < cfg.old_slice_overlap_threshold:
                quality = min(total_target_score, 4.0)
                reward += cfg.stop_bonus_weight * quality
            elif observable_score > 0.3:
                reward += cfg.stop_bonus_weight * 0.3 * min(observable_score, 2.0)
            else:
                reward -= cfg.stop_bonus_weight * 0.5

        return float(reward), info

    def _legacy_reward(self, action: Action, previous_roi: np.ndarray | None = None) -> tuple[float, dict]:
        prev_covered = self.covered.copy()
        commit_hits = action == Action.STOP
        target_scores, hit_mask = self._update_covered(commit=commit_hits)
        candidate_new_mask = hit_mask & ~prev_covered
        new_mask = candidate_new_mask if commit_hits else np.zeros_like(candidate_new_mask, dtype=bool)
        new_hits = int(new_mask.sum())
        candidate_hits = int(candidate_new_mask.sum())
        target_score = float(target_scores[new_mask].sum()) if len(target_scores) else 0.0
        hit_count = int(hit_mask.sum()) if len(hit_mask) else 0
        total_target_score = float(target_scores[hit_mask].sum()) if len(target_scores) else 0.0
        retained_hits = int((hit_mask & prev_covered).sum()) if len(hit_mask) else 0
        roi_area_ratio = self._roi_area_ratio()
        compactness = 1.0 - min(roi_area_ratio / max(self.env_cfg.max_roi_area_ratio, 1e-6), 1.0)
        compactness_score = total_target_score * compactness
        previous_compactness_score = 0.0
        if previous_roi is not None and len(self.hard_boxes) > 0:
            previous_scores, previous_hit_mask = self._hard_target_scores(previous_roi)
            previous_total_score = (
                float(previous_scores[previous_hit_mask].sum()) if len(previous_scores) else 0.0
            )
            previous_area_ratio = self._roi_area_ratio(previous_roi)
            previous_compactness = 1.0 - min(
                previous_area_ratio / max(self.env_cfg.max_roi_area_ratio, 1e-6),
                1.0,
            )
            previous_compactness_score = previous_total_score * previous_compactness
        compactness_delta = compactness_score - previous_compactness_score
        observable_score = self._observable_target_score()
        previous_observable_score = (
            self._observable_target_score(previous_roi) if previous_roi is not None else observable_score
        )
        observable_delta = observable_score - previous_observable_score
        scale_gain = self._scale_gain()
        old_slice_overlap = self._old_slice_overlap()
        attempted_slice_overlap = self._attempted_slice_overlap()

        reward = -self.env_cfg.step_penalty
        info = {
            "new_hits": new_hits,
            "candidate_hits": candidate_hits,
            "hit_count": hit_count,
            "target_score": target_score,
            "total_target_score": total_target_score,
            "retained_hits": retained_hits,
            "compactness_score": compactness_score,
            "compactness_delta": compactness_delta,
            "observable_score": observable_score,
            "observable_delta": observable_delta,
            "roi_area_ratio": roi_area_ratio,
            "scale_gain": scale_gain,
            "old_slice_overlap": old_slice_overlap,
            "attempted_slice_overlap": attempted_slice_overlap,
            "detected_overlap": 0.0,
        }

        if len(self.hard_boxes) > 0:
            if action == Action.STOP and new_hits > 0:
                reward += self.env_cfg.new_hard_reward * new_hits
                density = target_score * self.env_cfg.max_roi_area_ratio / max(roi_area_ratio, 1e-6)
                reward += self.env_cfg.hard_density_reward * float(np.clip(density, 0.0, 4.0))
            if hit_count > 0:
                reward += self.env_cfg.compactness_reward * float(np.clip(compactness_delta, -4.0, 4.0))
                if action != Action.STOP and new_hits == 0 and compactness_delta <= 0.0:
                    reward -= self.env_cfg.continue_target_penalty * min(total_target_score, 4.0)
            else:
                reward -= self.env_cfg.empty_slice_penalty
        elif action != Action.STOP:
            reward -= self.env_cfg.empty_slice_penalty

        if self.env_cfg.observable_target_reward > 0.0:
            reward += self.env_cfg.observable_target_reward * float(np.clip(observable_delta, -2.0, 2.0))

        detected_overlap = self._detected_overlap_score()
        if detected_overlap > 0.0:
            reward -= self.env_cfg.detected_overlap_penalty * detected_overlap
            info["detected_overlap"] = detected_overlap

        reward -= self.env_cfg.area_penalty * roi_area_ratio
        if roi_area_ratio > self.env_cfg.max_roi_area_ratio:
            overflow = roi_area_ratio / max(self.env_cfg.max_roi_area_ratio, 1e-6) - 1.0
            reward -= self.env_cfg.large_roi_penalty * overflow
        if scale_gain < self.env_cfg.min_scale_gain:
            under_scale = self.env_cfg.min_scale_gain / max(scale_gain, 1e-6) - 1.0
            reward -= self.env_cfg.low_scale_penalty * under_scale
        if old_slice_overlap >= self.env_cfg.old_slice_overlap_threshold:
            overflow = old_slice_overlap / max(self.env_cfg.old_slice_overlap_threshold, 1e-6) - 1.0
            reward -= self.env_cfg.old_slice_overlap_penalty * (1.0 + overflow)

        if action == Action.STOP:
            if total_target_score > 0.0 and old_slice_overlap < self.env_cfg.old_slice_overlap_threshold:
                stop_quality = min(total_target_score, 4.0)
                stop_quality += min(total_target_score / max(hit_count, 1), 1.0)
                reward += self.env_cfg.stop_target_reward * stop_quality
            elif observable_score > 0.25 and old_slice_overlap < self.env_cfg.old_slice_overlap_threshold:
                reward += self.env_cfg.stop_observable_target_reward * min(observable_score, 2.0)
            else:
                reward -= self.env_cfg.stop_early_penalty
        return float(reward), info
