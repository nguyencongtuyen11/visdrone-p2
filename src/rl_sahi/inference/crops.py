from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from rl_sahi.common.device import DeviceLike, configure_torch_runtime, configure_ultralytics_for_device


def crop_roi(image_path: Path, roi: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    x1, y1, x2, y2 = [int(round(v)) for v in roi]
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, image.shape[1])
    y2 = min(y2, image.shape[0])
    return image[y1:y2, x1:x2].copy(), (x1, y1)


def run_yolo_on_crop(
    model: YOLO,
    image_path: Path,
    roi: np.ndarray,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: DeviceLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return run_yolo_on_crops(
        model,
        [image_path],
        [roi],
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=device,
    )[0]


def run_yolo_on_crops(
    model: YOLO,
    image_paths: list[Path],
    rois: list[np.ndarray],
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: DeviceLike,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if len(image_paths) != len(rois):
        raise ValueError("image_paths and rois must have the same length")
    empty = (
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
    )
    outputs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = [empty for _ in image_paths]
    crops: list[np.ndarray] = []
    offsets: list[tuple[int, int]] = []
    output_indices: list[int] = []
    for index, (image_path, roi) in enumerate(zip(image_paths, rois)):
        crop, offset = crop_roi(Path(image_path), roi)
        if crop.size == 0:
            continue
        crops.append(crop)
        offsets.append(offset)
        output_indices.append(index)

    if not crops:
        return outputs

    resolved_device = configure_torch_runtime(device)
    configure_ultralytics_for_device(resolved_device)
    results = model.predict(
        crops,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        batch=len(crops),
        device=resolved_device,
        half=resolved_device.type == "cuda",
        verbose=False,
    )
    for output_index, offset, result in zip(output_indices, offsets, results):
        if result.boxes is None or len(result.boxes) == 0:
            continue
        boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        boxes[:, [0, 2]] += offset[0]
        boxes[:, [1, 3]] += offset[1]
        scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
        classes = result.boxes.cls.detach().cpu().numpy().astype(np.float32)
        outputs[output_index] = (boxes, scores, classes)
    return outputs
