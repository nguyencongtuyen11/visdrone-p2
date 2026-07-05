# giải thích: Nhập thư viện học máy PyTorch
import torch
# giải thích: Nhập thư viện hỗ trợ DirectML để sử dụng GPU trên các loại phần cứng khác nhau (AMD, Intel, Nvidia)
import torch_directml

# giải thích: Khởi tạo và lấy đối tượng thiết bị DirectML khả dụng
device = torch_directml.device()
# giải thích: In thông tin thiết bị phần cứng đang được sử dụng ra màn hình
print(f"Đang dùng thiết bị: {device}")

# giải thích: Tạo một tensor PyTorch trên bộ nhớ CPU rồi chuyển sang thiết bị DirectML để kiểm tra tính tương thích
x = torch.tensor([1.0, 2.0]).to(device)
# giải thích: In kiểu của đối tượng và giá trị của tensor sau khi chuyển sang thiết bị DirectML
print(type(x), x)