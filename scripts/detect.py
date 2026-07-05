# Cho phép import các annotation kiểu type hint nâng cao từ tương lai
from __future__ import annotations

import argparse  # Thư viện để phân tích cú pháp các đối số dòng lệnh
import sys       # Thư viện quản lý các biến và hàm tương tác với Python interpreter
from pathlib import Path  # Thư viện xử lý đường dẫn tệp tin một cách hướng đối tượng

# Định nghĩa đường dẫn gốc (ROOT) của dự án bằng cách lấy thư mục cha của thư mục chứa file này (parents[1] tức là nhảy lên 2 cấp từ file detect.py)
ROOT = Path(__file__).resolve().parents[1]
# Thêm đường dẫn thư mục nguồn "src" vào hệ thống tìm kiếm module của Python
sys.path.insert(0, str(ROOT / "src"))

# Import các hàm và module từ thư viện nội bộ rl_sahi
from rl_sahi.common.config import load_default_config  # Hàm load cấu hình mặc định
from rl_sahi.common.device import print_device_info    # Hàm in thông tin thiết bị chạy (CPU/GPU)
from rl_sahi.detection.cache_builder import cache_detections_for_split  # Hàm thực hiện dự đoán YOLO và lưu cache toàn bộ các ảnh


def main() -> None:
    # Thiết lập bộ phân tích cú pháp tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Cache full-image YOLO boxes and backbone features.")
    # Đường dẫn tới file config cấu hình
    parser.add_argument("--config", type=Path, default=None)
    # Tập dữ liệu cần xử lý (train, val, test)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    # Giới hạn số lượng ảnh cần cache (để chạy thử nhanh)
    parser.add_argument("--limit", type=int, default=None)
    # Ghi đè lên cache cũ nếu đã tồn tại
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Tải thông tin cấu hình từ tệp tin config được chỉ định hoặc mặc định
    cfg = load_default_config(args.config, ROOT)
    # Lấy phân vùng cấu hình liên quan đến việc detect của YOLO
    detect_cfg = cfg.section("detect")
    # Lấy cấu hình biểu diễn trạng thái của RL (state)
    state_cfg = cfg.section("state")
    # Lấy cấu hình thiết bị chạy từ phần detect
    device = cfg.optional_str("detect", "device")
    print_device_info("detect", device)

    # Chạy YOLO trên toàn bộ các ảnh trong phân vùng split và lưu kết quả (hộp giới hạn + các đặc trưng backbone) vào cache
    written = cache_detections_for_split(
        weights=cfg.path_value("weights"),           # Đường dẫn trọng số YOLO (.pt)
        image_root=cfg.path_value("image_root"),       # Thư mục chứa các ảnh gốc
        cache_root=cfg.path_value("cache_root"),       # Thư mục để lưu trữ cache đầu ra
        split=args.split,                            # Tập dữ liệu cần chạy (train/val/test)
        imgsz=int(detect_cfg["imgsz"]),              # Kích thước ảnh đầu vào cho YOLO
        conf=float(detect_cfg["conf"]),              # Confidence threshold của YOLO
        iou=float(detect_cfg["iou"]),                # IoU threshold cho thuật toán NMS của YOLO
        max_det=int(detect_cfg["max_det"]),          # Số lượng vật thể tối đa phát hiện được trên mỗi ảnh
        device=device,                               # Thiết bị chạy (CPU/CUDA)
        feature_layers=cfg.feature_layers("detect"),  # Các lớp mạng (layer indices) cần trích xuất đặc trưng backbone
        aux_grid_size=int(state_cfg["grid_size"]),   # Kích thước lưới bổ trợ phục vụ cho RL State
        spatial_feature_channels=int(state_cfg.get("spatial_feature_channels", 4)),  # Số kênh đặc trưng không gian
        limit=args.limit,                            # Giới hạn số ảnh xử lý
        overwrite=args.overwrite,                    # Ghi đè cache cũ hay không
    )
    # In kết quả đã lưu cache ra màn hình
    print(f"[detect] wrote {written} caches under {cfg.path_value('cache_root') / 'detections' / args.split}")


# Điểm khởi chạy của chương trình
if __name__ == "__main__":
    main()

