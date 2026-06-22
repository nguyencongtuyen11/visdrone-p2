from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from rl_sahi.common.cache import DetectionCache
from rl_sahi.common.data import read_image_shape
from rl_sahi.common.device import DeviceLike, configure_torch_runtime, configure_ultralytics_for_device
from rl_sahi.detection.features import DetectAuxCollector, FeatureCollector


DEFAULT_AUX_GRID_SIZE = 16
DEFAULT_SPATIAL_FEATURE_CHANNELS = 4


def load_yolo(weights: Path, device: DeviceLike = None) -> YOLO:
    model = YOLO(str(weights))
    resolved_device = configure_torch_runtime(device)
    configure_ultralytics_for_device(resolved_device)
    model.to(resolved_device)
    return model


def detect_one_image(
    model: YOLO,
    image_path: Path,
    imgsz: int = 640,
    conf: float = 0.01,
    iou: float = 0.7,
    max_det: int = 3000,
    device: DeviceLike = None,
    feature_layers: tuple[int, ...] = (10,),
    aux_grid_size: int = DEFAULT_AUX_GRID_SIZE,
    spatial_feature_channels: int = DEFAULT_SPATIAL_FEATURE_CHANNELS,
) -> DetectionCache:
    image_shape = read_image_shape(image_path)
    resolved_device = configure_torch_runtime(device)
    configure_ultralytics_for_device(resolved_device)
    with FeatureCollector(model, feature_layers) as collector, DetectAuxCollector(model) as aux_collector:
        collector.clear()
        aux_collector.clear()
        results = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=resolved_device,
            half=resolved_device.type == "cuda",
            verbose=False,
        )
        feature = collector.vector()
        objectness_map, spatial_feature_map = aux_collector.maps(
            grid_size=aux_grid_size,
            spatial_feature_channels=spatial_feature_channels,
        )
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
        classes = np.zeros((0,), dtype=np.float32)
    else:
        boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
        classes = result.boxes.cls.detach().cpu().numpy().astype(np.float32)
    return DetectionCache(
        image_path=str(image_path),
        image_shape=image_shape,
        boxes=boxes,
        scores=scores,
        classes=classes,
        feature=feature,
        feature_layers=feature_layers,
        objectness_map=objectness_map,
        spatial_feature_map=spatial_feature_map,
    )
