# Cho phép import các annotation nâng cao từ tương lai
from __future__ import annotations

from functools import wraps      # Tiện ích giữ nguyên thông tin (metadata) của hàm gốc khi viết decorator
from typing import TypeAlias    # Dùng định nghĩa bí danh kiểu dữ liệu (type alias)

import torch                    # Thư viện tính toán học sâu PyTorch


# Định nghĩa bí danh kiểu dữ liệu biểu thị các giá trị cấu hình thiết bị hợp lệ
DeviceLike: TypeAlias = torch.device | str | None


def _directml_device() -> torch.device | None:
    """
    Thử import thư viện torch_directml và lấy thiết bị DirectML (cho card đồ họa tích hợp iGPU hoặc AMD trên Windows).
    Trả về None nếu không cài đặt thư viện này.
    """
    try:
        import torch_directml
    except ImportError:
        return None
    try:
        return torch_directml.device()
    except Exception:
        return None


def resolve_torch_device(device: DeviceLike = None) -> torch.device:
    """
    Giải quyết và xác định chính xác thiết bị tính toán PyTorch (CUDA GPU, DirectML iGPU, hoặc CPU).
    Nếu truyền vào "auto" hoặc None, tự động chọn thiết bị tốt nhất có sẵn.
    """
    if isinstance(device, torch.device):
        return device

    if device is not None:
        value = str(device).strip()
        normalized = value.lower()
        if normalized in {"", "auto"}:
            device = None
        elif normalized in {"directml", "dml", "igpu"}:
            directml = _directml_device()
            if directml is None:
                raise RuntimeError("DirectML device requested, but torch-directml is not available.")
            return directml
        elif normalized.isdigit():
            # Chap nhan kieu ultralytics: "0","1"... -> "cuda:0" (neu co CUDA), nguoc lai CPU
            return torch.device(f"cuda:{normalized}" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(value)

    # Nếu không chỉ định thiết bị, ưu tiên chọn CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Tiếp theo ưu tiên chọn DirectML (Windows iGPU/AMD) nếu có sẵn
    directml = _directml_device()
    if directml is not None:
        return directml

    # Cuối cùng mặc định là dùng CPU
    return torch.device("cpu")


def configure_torch_runtime(device: DeviceLike = None) -> torch.device:
    """
    Cấu hình các tối ưu hóa runtime cho PyTorch dựa trên thiết bị chạy.
    Bật chế độ cudnn benchmark và TF32 (TensorFloat32) nếu chạy trên CUDA để tăng hiệu năng tính toán.
    """
    resolved = resolve_torch_device(device)
    if resolved.type == "cuda":
        torch.backends.cudnn.benchmark = True # Tối ưu hóa việc chọn thuật toán convolution phù hợp nhất cho GPU
        try:
            # Cho phép tính toán độ chính xác TF32 (TensorFloat32) để tăng tốc độ trên kiến trúc Ampere trở lên
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        try:
            # Thiết lập độ chính xác cho phép nhân ma trận float32 sang High (độ chính xác cao, cân bằng tốc độ)
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    return resolved


def is_directml_device(device: DeviceLike) -> bool:
    """
    Kiểm tra xem thiết bị tính toán có phải là DirectML (kiểu thiết bị PyTorch nội bộ là 'privateuseone') hay không.
    """
    resolved = resolve_torch_device(device)
    return resolved.type == "privateuseone"


def device_description(device: DeviceLike = None) -> str:
    """
    Trả về chuỗi mô tả chi tiết, dễ hiểu về thiết bị hiện tại (ví dụ: tên GPU CUDA).
    """
    resolved = resolve_torch_device(device)
    if resolved.type == "cuda":
        name = torch.cuda.get_device_name(resolved)
        return f"cuda/GPU ({resolved}, {name})"
    if resolved.type == "privateuseone":
        return f"directml/iGPU ({resolved})"
    return str(resolved)


def print_device_info(prefix: str, device: DeviceLike = None) -> None:
    """
    In thông tin thiết bị đang chạy ra màn hình console với tiền tố xác định.
    """
    print(f"[{prefix}] device: {device_description(device)}")


def _directml_tensors_to_cpu(value):
    """
    Hàm đệ quy chuyển các Tensor PyTorch từ thiết bị DirectML về CPU.
    Cần thiết vì một số toán tử của Ultralytics (như NMS) không được DirectML hỗ trợ trực tiếp.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu() if value.device.type == "privateuseone" else value
    if isinstance(value, list):
        return [_directml_tensors_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_directml_tensors_to_cpu(item) for item in value)
    return value


def configure_ultralytics_for_device(device: DeviceLike) -> None:
    """
    Chắp vá (patch) và cấu hình lại thư viện Ultralytics (YOLO) để tương thích hoàn toàn với DirectML.
    Vì DirectML không hỗ trợ chế độ tính toán gradient lúc suy luận và thiếu các kernel tính toán cho thuật toán NMS,
    hàm này sẽ chuyển đổi Tensor sang CPU trước khi chạy non_max_suppression và bao bọc hàm suy luận bằng torch.no_grad().
    """
    if not is_directml_device(device):
        return

    from ultralytics.engine.predictor import BasePredictor
    import ultralytics.nn.autobackend as autobackend
    import ultralytics.utils.nms as nms_module

    # Patch 1: Ép hàm stream_inference chạy ở chế độ no_grad để tránh lỗi DirectML gradient
    stream_inference = BasePredictor.stream_inference
    if not getattr(stream_inference, "_rl_sahi_directml_no_grad", False):
        original_stream_inference = getattr(stream_inference, "__wrapped__", stream_inference)
        patched_stream_inference = torch.no_grad()(original_stream_inference)
        setattr(patched_stream_inference, "_rl_sahi_directml_no_grad", True)
        BasePredictor.stream_inference = patched_stream_inference

    # Patch 2: Ép hàm non_max_suppression chuyển dữ liệu về CPU để thực thi an toàn
    non_max_suppression = nms_module.non_max_suppression
    if not getattr(non_max_suppression, "_rl_sahi_directml_cpu", False):

        @wraps(non_max_suppression)
        def directml_safe_nms(prediction, *args, **kwargs):
            return non_max_suppression(_directml_tensors_to_cpu(prediction), *args, **kwargs)

        setattr(directml_safe_nms, "_rl_sahi_directml_cpu", True)
        nms_module.non_max_suppression = directml_safe_nms
        autobackend.non_max_suppression = directml_safe_nms
    else:
        autobackend.non_max_suppression = non_max_suppression

