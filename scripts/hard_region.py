# Cho phép import các annotation nâng cao từ tương lai
from __future__ import annotations

import argparse  # Thư viện để phân tích cú pháp các đối số dòng lệnh
import sys       # Thư viện quản lý các biến hệ thống và đường dẫn module của Python
from pathlib import Path  # Thư viện xử lý đường dẫn hướng đối tượng

# Định nghĩa đường dẫn gốc (ROOT) của dự án
ROOT = Path(__file__).resolve().parents[1]
# Thêm đường dẫn thư mục nguồn "src" vào hệ thống tìm kiếm module của Python
sys.path.insert(0, str(ROOT / "src"))

# Import các lớp và hàm nội bộ của dự án
from rl_sahi.common.cache import detection_cache_metadata  # Lớp tạo metadata định danh cho cache YOLO
from rl_sahi.common.class_mapping import ClassMapping      # Lớp quản lý ánh xạ nhãn class đối tượng
from rl_sahi.common.config import load_default_config      # Hàm load cấu hình mặc định
from rl_sahi.common.device import print_device_info        # Hàm in thông tin thiết bị chạy
from rl_sahi.hard_region.cache_builder import cache_hard_regions_for_split  # Hàm thực hiện lọc và lưu cache các vùng khó phát hiện


def main() -> None:
    # Thiết lập phân tích tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Cache small GT boxes that full-image YOLO misses or scores low.")
    # Đường dẫn tới file config cấu hình
    parser.add_argument("--config", type=Path, default=None)
    # Tập dữ liệu cần xử lý (mặc định là train)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    # Giới hạn số lượng ảnh chạy thử (để test nhanh)
    parser.add_argument("--limit", type=int, default=None)
    # Ghi đè lên cache cũ nếu đã tồn tại
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Load cấu hình
    cfg = load_default_config(args.config, ROOT)
    # Lấy phân vùng cấu hình liên quan đến việc phát hiện đối tượng (detect)
    detect_cfg = cfg.section("detect")
    print_device_info("hard", cfg.optional_str("detect", "device"))
    
    # Lấy cấu hình vùng khó (hard_region) và cấu hình trạng thái của RL (state)
    hard_cfg = cfg.section("hard_region")
    state_cfg = cfg.section("state")
    
    # Phân tích danh sách các class mục tiêu từ file cấu hình
    target_raw = hard_cfg.get("target_classes", [])
    if isinstance(target_raw, str):
        # Nếu là chuỗi, tách bằng dấu phẩy và chuyển sang tuple số nguyên
        target_classes = tuple(int(x.strip()) for x in target_raw.split(",") if x.strip())
    else:
        # Nếu đã là list/tuple số, chuyển đổi trực tiếp sang tuple số nguyên
        target_classes = tuple(int(x) for x in target_raw)

    # Chạy quy trình tìm kiếm và lưu cache các vùng khó phát hiện trên toàn bộ các ảnh trong phân vùng split
    # Vùng khó (hard region) được định nghĩa là các vật thể nhỏ (ground truth) mà YOLO chạy trên ảnh gốc bị bỏ sót hoặc có điểm tự tin (score) thấp
    written = cache_hard_regions_for_split(
        image_root=cfg.path_value("image_root"),  # Thư mục ảnh gốc
        label_root=cfg.path_value("label_root"),  # Thư mục nhãn gốc (ground truth)
        cache_root=cfg.path_value("cache_root"),  # Thư mục lưu cache kết quả
        split=args.split,                       # Tập dữ liệu (train/val/test)
        small_area_ratio=float(hard_cfg["small_area_ratio"]), # Tỷ lệ diện tích tối đa để coi là một vật thể "nhỏ"
        small_area_percentile=(
            None
            if hard_cfg.get("small_area_percentile") in (None, "")
            else float(hard_cfg["small_area_percentile"])
        ),                                       # Bách phân vị diện tích của ảnh để xác định ngưỡng vật thể nhỏ (nếu có)
        match_iou=float(hard_cfg["match_iou"]),  # IoU tối thiểu để coi hộp phát hiện khớp với nhãn gốc (ground truth)
        min_detect_score=float(hard_cfg["min_detect_score"]), # Ngưỡng điểm tự tin tối thiểu (nếu YOLO phát hiện dưới ngưỡng này thì vẫn coi là bỏ sót/vùng khó)
        target_classes=target_classes,           # Các lớp đối tượng cần quan tâm
        class_mapping=ClassMapping.from_config(cfg.section("classes")), # Ánh xạ lớp tương ứng
        # Thiết lập siêu dữ liệu YOLO để đối chiếu và kiểm tra tính hợp lệ của cache
        detection_metadata=detection_cache_metadata(
            weights=cfg.path_value("weights"),
            imgsz=int(detect_cfg["imgsz"]),
            conf=float(detect_cfg["conf"]),
            iou=float(detect_cfg["iou"]),
            max_det=int(detect_cfg["max_det"]),
            feature_layers=cfg.feature_layers("detect"),
            aux_grid_size=int(state_cfg["grid_size"]),
            spatial_feature_channels=int(state_cfg.get("spatial_feature_channels", 4)),
        ),
        limit=args.limit,
        overwrite=args.overwrite,
    )
    # In thông tin đã lưu trữ cache vùng khó
    print(f"[hard] wrote {written} caches under {cfg.path_value('cache_root') / 'hard_regions' / args.split}")


if __name__ == "__main__":
    main()

