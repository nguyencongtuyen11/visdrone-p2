import torch
import torch_directml

device = torch_directml.device()
print(f"Đang dùng thiết bị: {device}")

x = torch.tensor([1.0, 2.0]).to(device)
print(type(x), x)