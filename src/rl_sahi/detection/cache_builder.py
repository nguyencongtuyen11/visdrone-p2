from __future__ import annotations

from pathlib import Path

from rl_sahi.common.cache import (
    detection_cache_is_current,
    detection_cache_metadata,
    detection_cache_path,
    save_detection_cache,
)
from rl_sahi.common.data import iter_images
from rl_sahi.detection.yolo import DEFAULT_AUX_GRID_SIZE, DEFAULT_SPATIAL_FEATURE_CHANNELS, detect_one_image, load_yolo


def cache_detections_for_split(
    weights: Path,
    image_root: Path,
    cache_root: Path,
    split: str,
    imgsz: int = 640,
    conf: float = 0.01,
    iou: float = 0.7,
    max_det: int = 3000,
    device: str | None = None,
    feature_layers: tuple[int, ...] = (10,),
    aux_grid_size: int = DEFAULT_AUX_GRID_SIZE,
    spatial_feature_channels: int = DEFAULT_SPATIAL_FEATURE_CHANNELS,
    limit: int | None = None,
    overwrite: bool = False,
) -> int:
    model = load_yolo(weights, device=device)
    images = iter_images(image_root, split=split, limit=limit)
    expected_metadata = detection_cache_metadata(
        weights=weights,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        feature_layers=feature_layers,
        aux_grid_size=aux_grid_size,
        spatial_feature_channels=spatial_feature_channels,
    )
    written = 0
    for index, image_path in enumerate(images, start=1):
        out_path = detection_cache_path(cache_root, split, image_path)
        if out_path.exists() and not overwrite and detection_cache_is_current(out_path, expected_metadata):
            continue
        cache = detect_one_image(
            model=model,
            image_path=image_path,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
            feature_layers=feature_layers,
            aux_grid_size=aux_grid_size,
            spatial_feature_channels=spatial_feature_channels,
        )
        cache.metadata = expected_metadata
        save_detection_cache(out_path, cache)
        written += 1
        if index == 1 or index % 50 == 0:
            print(f"[detect] {split}: {index}/{len(images)} cached -> {out_path}")
    return written
