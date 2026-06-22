from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Iterable

import numpy as np
import torch
from torch.nn import functional as F
from ultralytics import YOLO


class FeatureCollector(AbstractContextManager["FeatureCollector"]):
    def __init__(self, yolo: YOLO, layers: Iterable[int]) -> None:
        self.yolo = yolo
        self.layers = tuple(int(x) for x in layers)
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.features: list[np.ndarray] = []

    def __enter__(self) -> "FeatureCollector":
        modules = self.yolo.model.model
        for idx in self.layers:
            if idx < 0 or idx >= len(modules):
                raise ValueError(f"Feature layer {idx} is out of range; model has {len(modules)} modules")
            self.handles.append(modules[idx].register_forward_hook(self._hook))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def clear(self) -> None:
        self.features.clear()

    def _hook(self, _module, _inputs, output) -> None:
        self.features.append(_summarize_tensor_output(output))

    def vector(self) -> np.ndarray:
        if not self.features:
            return np.zeros((0,), dtype=np.float32)
        features = self.features[-len(self.layers) :]
        return np.concatenate(features, axis=0).astype(np.float32)


class DetectAuxCollector(AbstractContextManager["DetectAuxCollector"]):
    def __init__(self, yolo: YOLO) -> None:
        self.yolo = yolo
        self.handle: torch.utils.hooks.RemovableHandle | None = None
        self.output = None

    def __enter__(self) -> "DetectAuxCollector":
        self.handle = self.yolo.model.model[-1].register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def clear(self) -> None:
        self.output = None

    def _hook(self, _module, _inputs, output) -> None:
        self.output = output

    def maps(self, grid_size: int, spatial_feature_channels: int) -> tuple[np.ndarray, np.ndarray]:
        return _extract_detect_aux(self.output, grid_size, spatial_feature_channels)


def _summarize_tensor_output(output) -> np.ndarray:
    if isinstance(output, (list, tuple)):
        tensors = [x for x in output if torch.is_tensor(x)]
        if not tensors:
            return np.zeros((0,), dtype=np.float32)
        output = tensors[0]
    if not torch.is_tensor(output):
        return np.zeros((0,), dtype=np.float32)
    x = output.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
        mean = x.mean(dim=(1, 2))
        std = x.std(dim=(1, 2), unbiased=False)
        return torch.cat([mean, std], dim=0).numpy().astype(np.float32)
    if x.ndim == 3:
        mean = x.mean(dim=(1, 2))
        std = x.std(dim=(1, 2), unbiased=False)
        return torch.cat([mean, std], dim=0).numpy().astype(np.float32)
    return x.reshape(-1).numpy().astype(np.float32)


def _resize_tensor_maps(maps: torch.Tensor, grid_size: int) -> torch.Tensor:
    if maps.ndim == 2:
        maps = maps.unsqueeze(0)
    return F.interpolate(
        maps.unsqueeze(0),
        size=(grid_size, grid_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _normalize_spatial_maps(maps: torch.Tensor) -> torch.Tensor:
    if maps.numel() == 0:
        return maps
    mean = maps.mean(dim=(1, 2), keepdim=True)
    std = maps.std(dim=(1, 2), unbiased=False, keepdim=True).clamp_min(1e-6)
    return ((maps - mean) / std).clamp(-3.0, 3.0) / 3.0


def _compress_feature_level(feature: torch.Tensor, out_channels: int, grid_size: int) -> torch.Tensor:
    if out_channels <= 0:
        return torch.zeros((0, grid_size, grid_size), dtype=torch.float32)
    x = feature.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.ndim != 3:
        return torch.zeros((0, grid_size, grid_size), dtype=torch.float32)
    chunks = torch.chunk(x, min(out_channels, x.shape[0]), dim=0)
    maps = torch.stack([chunk.mean(dim=0) for chunk in chunks], dim=0)
    if maps.shape[0] < out_channels:
        pad = torch.zeros((out_channels - maps.shape[0], maps.shape[1], maps.shape[2]), dtype=maps.dtype)
        maps = torch.cat([maps, pad], dim=0)
    maps = _resize_tensor_maps(maps, grid_size)
    return _normalize_spatial_maps(maps)


def _extract_detect_aux(output, grid_size: int, spatial_feature_channels: int) -> tuple[np.ndarray, np.ndarray]:
    objectness_map = np.zeros((1, grid_size, grid_size), dtype=np.float32)
    spatial_feature_map = np.zeros((0, grid_size, grid_size), dtype=np.float32)
    if not isinstance(output, (tuple, list)) or len(output) < 2 or not isinstance(output[1], dict):
        return objectness_map, spatial_feature_map

    aux = output[1]
    scores = aux.get("scores")
    feats = aux.get("feats")
    if not torch.is_tensor(scores) or not isinstance(feats, list) or not feats:
        return objectness_map, spatial_feature_map

    score_tensor = scores.detach().float().cpu()
    if score_tensor.ndim == 3:
        score_tensor = score_tensor[0]
    if score_tensor.ndim != 2:
        return objectness_map, spatial_feature_map

    level_maps: list[torch.Tensor] = []
    start = 0
    for feature in feats:
        if not torch.is_tensor(feature) or feature.ndim < 3:
            continue
        h, w = int(feature.shape[-2]), int(feature.shape[-1])
        count = h * w
        end = start + count
        if end > score_tensor.shape[1]:
            break
        # YOLO11 has no separate objectness channel; max class logit before threshold/NMS
        # is used as an objectness-like heatmap.
        level_score = torch.sigmoid(score_tensor[:, start:end].max(dim=0).values).reshape(h, w)
        level_maps.append(_resize_tensor_maps(level_score, grid_size)[0])
        start = end
    if level_maps:
        objectness_map = torch.stack(level_maps, dim=0).max(dim=0).values.numpy().astype(np.float32)[None, :, :]

    spatial_levels = [
        _compress_feature_level(feature, spatial_feature_channels, grid_size)
        for feature in feats
        if torch.is_tensor(feature)
    ]
    if spatial_levels:
        spatial_feature_map = torch.cat(spatial_levels, dim=0).numpy().astype(np.float32)
    return objectness_map, spatial_feature_map
