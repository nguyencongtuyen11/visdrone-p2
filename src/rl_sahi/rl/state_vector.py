from __future__ import annotations

import numpy as np


def normalize_feature(feature: np.ndarray) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    if feature.size == 0:
        return feature
    feature = np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(feature))
    if norm > 1e-6:
        feature = feature / norm
    return np.clip(feature, -5.0, 5.0).astype(np.float32)


def build_state_vector(
    feature: np.ndarray,
    history: np.ndarray,
    current_roi_map: np.ndarray,
    attempted_slice_map: np.ndarray,
    accepted_slice_map: np.ndarray,
    detection_map: np.ndarray,
    objectness_map: np.ndarray,
    spatial_feature_map: np.ndarray,
    summary: np.ndarray,
    static_ready: bool = False,
) -> np.ndarray:
    if static_ready:
        feature_part = np.asarray(feature, dtype=np.float32).reshape(-1)
        objectness = np.asarray(objectness_map, dtype=np.float32)
        spatial = np.asarray(spatial_feature_map, dtype=np.float32)
    else:
        feature_part = normalize_feature(feature)
        objectness = np.nan_to_num(
            np.asarray(objectness_map, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        spatial = np.nan_to_num(
            np.asarray(spatial_feature_map, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    return np.concatenate(
        [
            feature_part,
            np.asarray(history, dtype=np.float32).reshape(-1),
            np.asarray(current_roi_map, dtype=np.float32).reshape(-1),
            np.asarray(attempted_slice_map, dtype=np.float32).reshape(-1),
            np.asarray(accepted_slice_map, dtype=np.float32).reshape(-1),
            np.asarray(detection_map, dtype=np.float32).reshape(-1),
            objectness.reshape(-1),
            spatial.reshape(-1),
            np.asarray(summary, dtype=np.float32).reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)
