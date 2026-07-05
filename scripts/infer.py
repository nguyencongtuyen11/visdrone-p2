# Cho phép import các annotation nâng cao từ tương lai
from __future__ import annotations

import argparse  # Thư viện để phân tích cú pháp các đối số dòng lệnh
import sys       # Thư viện quản lý các biến và đường dẫn module của Python
from pathlib import Path  # Thư viện xử lý đường dẫn hướng đối tượng

# Định nghĩa đường dẫn gốc (ROOT) của dự án
ROOT = Path(__file__).resolve().parents[1]
# Thêm đường dẫn thư mục nguồn "src" vào hệ thống tìm kiếm module của Python
sys.path.insert(0, str(ROOT / "src"))

# Import các thành phần tiện ích và mô hình suy luận từ rl_sahi
from rl_sahi.common.config import load_default_config  # Hàm load cấu hình mặc định
from rl_sahi.common.class_mapping import ClassMapping  # Lớp ánh xạ nhãn lớp đối tượng
from rl_sahi.common.data import iter_images            # Hàm lặp để quét các ảnh trong thư mục
from rl_sahi.common.device import print_device_info    # Hàm in thông tin thiết bị
from rl_sahi.inference.config import InferenceConfig   # Cấu hình tham số cho suy luận SAHI
from rl_sahi.inference.pipeline import AdaptiveSahiInferencer  # Lớp thực hiện suy luận SAHI thích ứng dựa trên RL


def main() -> None:
    # Thiết lập bộ phân tích đối số dòng lệnh
    parser = argparse.ArgumentParser(description="Run adaptive-slice inference and save boxes plus slice visualization.")
    # Đường dẫn tới file config cấu hình
    parser.add_argument("--config", type=Path, default=None)
    # Đường dẫn tới một ảnh cụ thể (nếu chỉ muốn suy luận trên 1 ảnh)
    parser.add_argument("--image", type=Path, default=None)
    # Tập dữ liệu cần xử lý (train, val, test) nếu chạy trên toàn bộ tập dữ liệu
    parser.add_argument("--split", default=None, choices=["train", "val", "test"])
    # Đường dẫn checkpoint mô hình RL
    parser.add_argument("--checkpoint", type=Path, default=None)
    # Giới hạn số lượng ảnh chạy
    parser.add_argument("--limit", type=int, default=None)
    # Tùy chọn không sử dụng cache phát hiện YOLO
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    # Load cấu hình mặc định
    cfg = load_default_config(args.config, ROOT)
    # Lấy phân vùng cấu hình liên quan đến suy luận (infer)
    infer_cfg = cfg.section("infer")
    # Xác định thiết bị chạy (CPU/GPU)
    device = cfg.optional_str("infer", "device")
    print_device_info("infer", device)
    
    # Xác định các lớp đối tượng cần quan tâm từ cấu hình
    target_raw = infer_cfg.get("target_classes", [0, 2, 3, 5, 8, 9])
    if isinstance(target_raw, str):
        # Nếu là chuỗi, tách dấu phẩy thành tuple số nguyên
        target_classes = tuple(int(x.strip()) for x in target_raw.split(",") if x.strip())
    else:
        # Nếu là list/tuple số, chuyển sang tuple số nguyên
        target_classes = tuple(int(x) for x in target_raw)

    # Chuẩn bị danh sách ảnh đầu vào
    if args.image is not None:
        # Nếu người dùng chỉ định một ảnh cụ thể bằng --image
        image_path = args.image if args.image.is_absolute() else ROOT / args.image
        images = [image_path]
        split = args.split
    else:
        # Nếu chạy trên toàn bộ thư mục dữ liệu, bắt buộc phải cung cấp --split
        if args.split is None:
            raise ValueError("Use --image for one image or --split train/val/test for a dataset split.")
        images = iter_images(cfg.path_value("image_root"), split=args.split, limit=args.limit)
        split = args.split

    # Xác định đường dẫn file checkpoint cho mô hình RL
    if args.checkpoint is None:
        checkpoint = cfg.path_value("checkpoint")
    else:
        checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
        
    # Khởi tạo đối tượng suy luận thích ứng (AdaptiveSahiInferencer)
    inferencer = AdaptiveSahiInferencer(
        weights=cfg.path_value("weights"),  # Trọng số YOLO (.pt)
        checkpoint=checkpoint,              # Checkpoint mô hình RL (DQN)
        # Thiết lập cấu hình tham số suy luận
        cfg=InferenceConfig(
            full_imgsz=int(infer_cfg["full_imgsz"]),
            slice_imgsz=int(infer_cfg["slice_imgsz"]),
            full_conf=float(infer_cfg["full_conf"]),
            output_conf=float(infer_cfg["output_conf"]),
            iou=float(infer_cfg["iou"]),
            merge_iou=float(infer_cfg["merge_iou"]),
            max_det=int(infer_cfg["max_det"]),
            device=device,
            feature_layers=cfg.feature_layers("infer"),
            min_slice_detections=int(infer_cfg.get("min_slice_detections", 1)),
            min_slice_utility=float(infer_cfg.get("min_slice_utility", 0.5)),
            duplicate_iou=float(infer_cfg.get("duplicate_iou", infer_cfg.get("merge_iou", 0.5))),
            max_slice_attempts=int(infer_cfg.get("max_slice_attempts", 0)),
            target_classes=target_classes,
            require_stop_for_acceptance=bool(infer_cfg.get("require_stop_for_acceptance", True)),
            class_mapping=ClassMapping.from_config(cfg.section("classes")),
        ),
    )
    
    # Thực hiện suy luận trên từng ảnh và lưu kết quả
    for image_path in images:
        meta = inferencer.infer_image(
            image_path=image_path,
            out_dir=cfg.path_value("infer_out_dir"),  # Thư mục lưu file nhãn đầu ra và ảnh minh họa lát cắt
            cache_root=cfg.path_value("cache_root") if split is not None else None, # Thư mục chứa cache YOLO (để tăng tốc nếu có)
            split=split,
            use_cache=bool(infer_cfg["use_cache"]) and not args.no_cache,
        )
        # In tóm tắt kết quả phát hiện của ảnh hiện tại
        print(f"[infer] {image_path.name}: {meta['detections']} boxes, slices={meta['num_slices']}")


if __name__ == "__main__":
    main()

