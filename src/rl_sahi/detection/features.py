# Cho phép import các annotation nâng cao từ tương lai
from __future__ import annotations

from contextlib import AbstractContextManager  # Lớp cơ sở dùng để định nghĩa các Context Managers (quản lý ngữ cảnh bằng `with`)
from typing import Iterable                   # Kiểu dữ liệu có thể lặp (Iterable) phục vụ type hints

import numpy as np                            # Thư viện tính toán mảng số học NumPy
import torch                                  # Thư viện tính toán Tensor PyTorch
from torch.nn import functional as F          # Các hàm toán học chức năng của PyTorch (như nội suy interpolation)
from ultralytics import YOLO                  # Thư viện mô hình YOLO của Ultralytics


class FeatureCollector(AbstractContextManager["FeatureCollector"]):
    """
    Context Manager dùng để đăng ký PyTorch forward hook và thu thập các đặc trưng trung gian (backbone features)
    từ các layer chỉ định trong lúc mô hình YOLO chạy suy luận (forward pass).
    """
    def __init__(self, yolo: YOLO, layers: Iterable[int]) -> None:
        self.yolo = yolo
        self.layers = tuple(int(x) for x in layers) # Các index của layer cần lấy đặc trưng
        self.handles: list[torch.utils.hooks.RemovableHandle] = [] # Lưu các handle của hook để gỡ ra khi thoát context
        self.features: list[np.ndarray] = []        # Chứa dữ liệu đặc trưng thu được từ hook

    def __enter__(self) -> "FeatureCollector":
        # Truy cập danh sách các submodule của mô hình YOLO
        modules = self.yolo.model.model
        for idx in self.layers:
            if idx < 0 or idx >= len(modules):
                raise ValueError(f"Feature layer {idx} is out of range; model has {len(modules)} modules")
            # Đăng ký forward hook cho layer chỉ định
            self.handles.append(modules[idx].register_forward_hook(self._hook))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # Gỡ bỏ toàn bộ forward hooks đã đăng ký khi thoát khỏi khối lệnh `with`
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def clear(self) -> None:
        """Xóa sạch các đặc trưng đã thu thập trước đó."""
        self.features.clear()

    def _hook(self, _module, _inputs, output) -> None:
        # Hàm callback của hook, được tự động gọi khi kết thúc lan truyền xuôi qua layer tương ứng
        # Rút trích và tóm tắt Tensor đầu ra của layer rồi lưu vào danh sách
        self.features.append(_summarize_tensor_output(output))

    def vector(self) -> np.ndarray:
        """
        Nối (concatenate) tất cả các đặc trưng thu được thành một vector 1D duy nhất đại diện cho trạng thái ảnh.
        """
        if not self.features:
            return np.zeros((0,), dtype=np.float32)
        # Chỉ lấy số lượng đặc trưng tương ứng với các layers đăng ký gần nhất
        features = self.features[-len(self.layers) :]
        return np.concatenate(features, axis=0).astype(np.float32)


class DetectAuxCollector(AbstractContextManager["DetectAuxCollector"]):
    """
    Context Manager thu thập dữ liệu phụ (auxiliary outputs) trực tiếp từ layer phát hiện cuối cùng (detection head)
    của YOLO để xây dựng bản đồ trạng thái không gian (objectness map và spatial feature map) phục vụ DQN State.
    """
    def __init__(self, yolo: YOLO) -> None:
        self.yolo = yolo
        self.handle: torch.utils.hooks.RemovableHandle | None = None
        self.output = None

    def __enter__(self) -> "DetectAuxCollector":
        # Đăng ký hook vào module cuối cùng của mô hình (thường là module Detect/Segment head, index là -1)
        self.handle = self.yolo.model.model[-1].register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # Gỡ bỏ hook khi thoát context
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def clear(self) -> None:
        self.output = None

    def _hook(self, _module, _inputs, output) -> None:
        self.output = output

    def maps(self, grid_size: int, spatial_feature_channels: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Rút trích bản đồ objectness và bản đồ đặc trưng không gian từ output thu thập được.
        """
        return _extract_detect_aux(self.output, grid_size, spatial_feature_channels)


def _summarize_tensor_output(output) -> np.ndarray:
    """
    Tóm tắt đặc trưng của Tensor đầu ra bằng cách tính trung bình (mean) và độ lệch chuẩn (std)
    trên các chiều không gian (spatial dimensions - width, height) của mỗi kênh (channel).
    Giúp giảm số lượng chiều dữ liệu mà vẫn giữ được thông tin thống kê cốt lõi.
    """
    if isinstance(output, (list, tuple)):
        # Nếu output là một list các tensor, chỉ lấy tensor thực sự
        tensors = [x for x in output if torch.is_tensor(x)]
        if not tensors:
            return np.zeros((0,), dtype=np.float32)
        output = tensors[0]
    if not torch.is_tensor(output):
        return np.zeros((0,), dtype=np.float32)
    x = output.detach().float().cpu()
    
    # Định dạng tensor 4D thường là (batch_size, channels, height, width)
    if x.ndim == 4:
        x = x[0] # Lấy phần tử batch đầu tiên
        mean = x.mean(dim=(1, 2))  # Tính trung bình trên kích thước không gian (W, H) cho mỗi channel
        std = x.std(dim=(1, 2), unbiased=False) # Tính độ lệch chuẩn tương ứng
        return torch.cat([mean, std], dim=0).numpy().astype(np.float32)
    if x.ndim == 3:
        mean = x.mean(dim=(1, 2))
        std = x.std(dim=(1, 2), unbiased=False)
        return torch.cat([mean, std], dim=0).numpy().astype(np.float32)
    # Nếu không phải 3D/4D, biến đổi dẹt thành mảng 1D trực tiếp
    return x.reshape(-1).numpy().astype(np.float32)


def _resize_tensor_maps(maps: torch.Tensor, grid_size: int) -> torch.Tensor:
    """
    Thay đổi kích thước (resize) bản đồ đặc trưng về kích thước lưới grid_size x grid_size
    bằng phương pháp nội suy song tuyến tính (bilinear interpolation).
    """
    if maps.ndim == 2:
        maps = maps.unsqueeze(0)
    return F.interpolate(
        maps.unsqueeze(0),
        size=(grid_size, grid_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _normalize_spatial_maps(maps: torch.Tensor) -> torch.Tensor:
    """
    Chuẩn hóa bản đồ đặc trưng không gian về khoảng giá trị mong muốn (Z-score normalization và cắt biên).
    """
    if maps.numel() == 0:
        return maps
    mean = maps.mean(dim=(1, 2), keepdim=True)
    std = maps.std(dim=(1, 2), unbiased=False, keepdim=True).clamp_min(1e-6)
    return ((maps - mean) / std).clamp(-3.0, 3.0) / 3.0


def _compress_feature_level(feature: torch.Tensor, out_channels: int, grid_size: int) -> torch.Tensor:
    """
    Nén số kênh (channels) của bản đồ đặc trưng của một layer về số kênh đích out_channels.
    Sử dụng việc chia nhỏ các kênh thành các nhóm (chunks) và lấy trung bình của mỗi nhóm để nén thông tin.
    """
    if out_channels <= 0:
        return torch.zeros((0, grid_size, grid_size), dtype=torch.float32)
    x = feature.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.ndim != 3:
        return torch.zeros((0, grid_size, grid_size), dtype=torch.float32)
        
    # Chia nhỏ tensor dọc theo trục channels thành các phần có kích thước xấp xỉ nhau
    chunks = torch.chunk(x, min(out_channels, x.shape[0]), dim=0)
    # Lấy trung bình cộng cho mỗi phần
    maps = torch.stack([chunk.mean(dim=0) for chunk in chunks], dim=0)
    
    # Nếu số lượng bản đồ thu được nhỏ hơn số kênh đầu ra yêu cầu, bổ sung đệm bằng 0
    if maps.shape[0] < out_channels:
        pad = torch.zeros((out_channels - maps.shape[0], maps.shape[1], maps.shape[2]), dtype=maps.dtype)
        maps = torch.cat([maps, pad], dim=0)
        
    # Thay đổi kích thước bản đồ lưới và thực hiện chuẩn hóa
    maps = _resize_tensor_maps(maps, grid_size)
    return _normalize_spatial_maps(maps)


def _extract_detect_aux(output, grid_size: int, spatial_feature_channels: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Thực hiện rút trích bản đồ objectness (bản đồ độ vật thể) và bản đồ đặc trưng không gian từ đầu ra của YOLO Detect head.
    Bản đồ objectness thu được bằng cách lấy điểm xác suất (sigmoid) lớn nhất của các class trên mỗi cell lưới.
    Bản đồ đặc trưng không gian được nén từ các activation maps trung gian trong YOLO head.
    """
    objectness_map = np.zeros((1, grid_size, grid_size), dtype=np.float32)
    spatial_feature_map = np.zeros((0, grid_size, grid_size), dtype=np.float32)
    
    # Kiểm tra tính hợp lệ của output đầu ra YOLO (output[1] chứa thông tin auxiliary dạng dictionary)
    if not isinstance(output, (tuple, list)) or len(output) < 2 or not isinstance(output[1], dict):
        return objectness_map, spatial_feature_map

    aux = output[1]
    scores = aux.get("scores")  # Điểm số dự đoán trước NMS
    feats = aux.get("feats")    # Các đặc trưng trung gian (feature maps) của các nhánh dự đoán
    if not torch.is_tensor(scores) or not isinstance(feats, list) or not feats:
        return objectness_map, spatial_feature_map

    score_tensor = scores.detach().float().cpu()
    if score_tensor.ndim == 3:
        score_tensor = score_tensor[0]
    if score_tensor.ndim != 2:
        return objectness_map, spatial_feature_map

    level_maps: list[torch.Tensor] = []
    start = 0
    # Phân tích qua từng cấp phân giải đặc trưng (feature pyramid levels)
    for feature in feats:
        if not torch.is_tensor(feature) or feature.ndim < 3:
            continue
        h, w = int(feature.shape[-2]), int(feature.shape[-1])
        count = h * w
        end = start + count
        if end > score_tensor.shape[1]:
            break
            
        # YOLO11 không có kênh xác định độ vật thể (objectness) độc lập như YOLOv5.
        # Ở đây ta sử dụng logit lớn nhất trong số toàn bộ các class trước ngưỡng chặn/NMS
        # đi qua hàm sigmoid để tạo ra một bản đồ tương tự bản đồ nhiệt phân bố vật thể (objectness-like heatmap).
        level_score = torch.sigmoid(score_tensor[:, start:end].max(dim=0).values).reshape(h, w)
        level_maps.append(_resize_tensor_maps(level_score, grid_size)[0])
        start = end
        
    # Tạo bản đồ objectness tổng hợp bằng cách lấy giá trị cực đại (max) trên mọi cấp phân giải lưới
    if level_maps:
        objectness_map = torch.stack(level_maps, dim=0).max(dim=0).values.numpy().astype(np.float32)[None, :, :]

    # Nén và chuẩn hóa các mức đặc trưng không gian cho từng cấp đặc trưng
    spatial_levels = [
        _compress_feature_level(feature, spatial_feature_channels, grid_size)
        for feature in feats
        if torch.is_tensor(feature)
    ]
    if spatial_levels:
        spatial_feature_map = torch.cat(spatial_levels, dim=0).numpy().astype(np.float32)
        
    return objectness_map, spatial_feature_map

