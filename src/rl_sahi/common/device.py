from __future__ import annotations

from functools import wraps
from typing import TypeAlias

import torch


DeviceLike: TypeAlias = torch.device | str | None


def _directml_device() -> torch.device | None:
    try:
        import torch_directml
    except ImportError:
        return None
    try:
        return torch_directml.device()
    except Exception:
        return None


def resolve_torch_device(device: DeviceLike = None) -> torch.device:
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
        else:
            return torch.device(value)

    if torch.cuda.is_available():
        return torch.device("cuda")

    directml = _directml_device()
    if directml is not None:
        return directml

    return torch.device("cpu")


def configure_torch_runtime(device: DeviceLike = None) -> torch.device:
    resolved = resolve_torch_device(device)
    if resolved.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    return resolved


def is_directml_device(device: DeviceLike) -> bool:
    resolved = resolve_torch_device(device)
    return resolved.type == "privateuseone"


def device_description(device: DeviceLike = None) -> str:
    resolved = resolve_torch_device(device)
    if resolved.type == "cuda":
        name = torch.cuda.get_device_name(resolved)
        return f"cuda/GPU ({resolved}, {name})"
    if resolved.type == "privateuseone":
        return f"directml/iGPU ({resolved})"
    return str(resolved)


def print_device_info(prefix: str, device: DeviceLike = None) -> None:
    print(f"[{prefix}] device: {device_description(device)}")


def _directml_tensors_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu() if value.device.type == "privateuseone" else value
    if isinstance(value, list):
        return [_directml_tensors_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_directml_tensors_to_cpu(item) for item in value)
    return value


def configure_ultralytics_for_device(device: DeviceLike) -> None:
    if not is_directml_device(device):
        return

    from ultralytics.engine.predictor import BasePredictor
    import ultralytics.nn.autobackend as autobackend
    import ultralytics.utils.nms as nms_module

    stream_inference = BasePredictor.stream_inference
    if not getattr(stream_inference, "_rl_sahi_directml_no_grad", False):
        original_stream_inference = getattr(stream_inference, "__wrapped__", stream_inference)
        patched_stream_inference = torch.no_grad()(original_stream_inference)
        setattr(patched_stream_inference, "_rl_sahi_directml_no_grad", True)
        BasePredictor.stream_inference = patched_stream_inference

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
