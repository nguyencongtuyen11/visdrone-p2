from __future__ import annotations

from dataclasses import dataclass, field

from rl_sahi.common.class_mapping import ClassMapping


@dataclass(slots=True)
class InferenceConfig:
    full_imgsz: int = 640
    slice_imgsz: int = 640
    full_conf: float = 0.01
    output_conf: float = 0.3
    iou: float = 0.7
    merge_iou: float = 0.5
    max_det: int = 3000
    device: str | None = None
    feature_layers: tuple[int, ...] = (10,)
    min_slice_detections: int = 1
    min_slice_utility: float = 0.5
    duplicate_iou: float = 0.5
    max_slice_attempts: int = 0
    target_classes: tuple[int, ...] = (0, 2, 3, 5, 8, 9)
    require_stop_for_acceptance: bool = True
    class_mapping: ClassMapping = field(default_factory=ClassMapping)
