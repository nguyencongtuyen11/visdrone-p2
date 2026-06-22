from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .boxes import xywhn_to_xyxy


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(image_root: Path, split: str | None = None, limit: int | None = None) -> list[Path]:
    root = Path(image_root)
    search_root = root / split if split else root
    images = sorted(p for p in search_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if limit is not None:
        images = images[:limit]
    return images


def image_id(image_path: Path) -> str:
    return Path(image_path).stem


def image_to_label_path(image_path: Path, image_root: Path, label_root: Path) -> Path:
    image_path = Path(image_path)
    image_root = Path(image_root)
    label_root = Path(label_root)
    rel = image_path.relative_to(image_root)
    return (label_root / rel).with_suffix(".txt")


def read_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return image


def read_image_shape(image_path: Path) -> tuple[int, int]:
    image = read_image(image_path)
    h, w = image.shape[:2]
    return int(h), int(w)


def read_yolo_labels(label_path: Path, image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    label_path = Path(label_path)
    if not label_path.exists():
        return np.zeros((0,), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
    rows: list[list[float]] = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            rows.append([float(x) for x in parts[:5]])
    if not rows:
        return np.zeros((0,), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
    arr = np.asarray(rows, dtype=np.float32)
    classes = arr[:, 0]
    boxes = xywhn_to_xyxy(arr[:, 1:5], image_shape)
    return classes, boxes


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
