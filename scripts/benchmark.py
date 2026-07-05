# Cho phép import các annotation kiểu type hint nâng cao (ví dụ: Union, List) từ tương lai
from __future__ import annotations

import argparse  # Thư viện để phân tích cú pháp các đối số dòng lệnh
import sys       # Thư viện quản lý các biến và hàm tương tác với Python interpreter
from pathlib import Path  # Thư viện xử lý đường dẫn tệp tin một cách hướng đối tượng

# Định nghĩa đường dẫn gốc (ROOT) của dự án bằng cách lấy thư mục cha của thư mục chứa file này (parents[1] tức là nhảy lên 2 cấp từ file benchmark.py)
ROOT = Path(__file__).resolve().parents[1]
# Thêm đường dẫn thư mục nguồn "src" vào hệ thống tìm kiếm module của Python
sys.path.insert(0, str(ROOT / "src"))

# Import các lớp và hàm cấu hình từ thư viện nội bộ rl_sahi
from rl_sahi.common.class_mapping import ClassMapping  # Ánh xạ tên lớp đối tượng và ID lớp
from rl_sahi.common.config import load_default_config  # Hàm load cấu hình mặc định từ file cấu hình (.yaml/.ini...)
from rl_sahi.common.device import print_device_info    # Hàm hiển thị thông tin thiết bị chạy (CPU, CUDA GPU...)
from rl_sahi.eval.benchmark import BenchmarkConfig, benchmark_split  # Cấu hình và hàm thực thi đánh giá hiệu năng (benchmark)
from rl_sahi.inference.config import InferenceConfig  # Cấu hình cho tiến trình suy luận (inference)


def _int_tuple(value) -> tuple[int, ...]:
    """
    Hàm phụ trợ để chuyển đổi một chuỗi các số phân tách bằng dấu phẩy thành một tuple chứa các số nguyên.
    Ví dụ: "0,2,3" -> (0, 2, 3)
    """
    if isinstance(value, str):
        # Nếu là chuỗi, tách bằng dấu phẩy, loại bỏ khoảng trắng và chuyển đổi từng phần tử thành int
        return tuple(int(x.strip()) for x in value.split(",") if x.strip())
    # Nếu không phải chuỗi (đã là danh sách/tuple sẵn), chuyển đổi trực tiếp từng phần tử thành int
    return tuple(int(x) for x in value)


def main() -> None:
    # Thiết lập bộ phân tích cú pháp tham số dòng lệnh (Command-line arguments)
    parser = argparse.ArgumentParser(description="Benchmark YOLO full, fixed-grid SAHI, and RL-SAHI.")
    # Đường dẫn tới file config
    parser.add_argument("--config", type=Path, default=None)
    # Tập dữ liệu cần đánh giá (mặc định là val - validation)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    # Đường dẫn tới file checkpoint của mô hình Reinforcement Learning (DQN)
    parser.add_argument("--checkpoint", type=Path, default=None)
    # Giới hạn số lượng ảnh chạy thử (nếu muốn chạy thử nhanh trên một số ít ảnh)
    parser.add_argument("--limit", type=int, default=None)
    # Thư mục đầu ra để lưu kết quả đánh giá
    parser.add_argument("--out-dir", type=Path, default=None)
    # Tùy chọn không sử dụng dữ liệu phát hiện YOLO được cache sẵn
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    # Tải cấu hình mặc định từ đường dẫn được cung cấp hoặc cấu hình mặc định của dự án
    cfg = load_default_config(args.config, ROOT)
    # Lấy phân vùng cấu hình liên quan đến suy luận (inference)
    infer_cfg = cfg.section("infer")
    # Lấy thông tin thiết bị chạy (CPU hoặc GPU CUDA) từ cấu hình
    device = cfg.optional_str("infer", "device")
    print_device_info("benchmark", device)
    
    # Lấy phân vùng cấu hình cho benchmark
    benchmark_cfg = cfg.section("benchmark")
    # Danh sách các class đối tượng mục tiêu cần đánh giá (ví dụ: các class nhỏ cần phát hiện trong VisDrone)
    target_classes = _int_tuple(infer_cfg.get("target_classes", [0, 2, 3, 5, 8, 9]))
    # Thiết lập ánh xạ class (Class Mapping) từ phần cấu hình "classes"
    class_mapping = ClassMapping.from_config(cfg.section("classes"))
    
    # Xác định đường dẫn file checkpoint cho mô hình RL
    checkpoint = cfg.path_value("checkpoint") if args.checkpoint is None else args.checkpoint
    if not checkpoint.is_absolute():
        # Nếu đường dẫn tương đối, chuyển thành đường dẫn tuyệt đối dựa trên thư mục ROOT
        checkpoint = ROOT / checkpoint
        
    # Xác định thư mục lưu kết quả đánh giá (benchmark.csv)
    out_dir = args.out_dir if args.out_dir is not None else ROOT / "runs" / "benchmark" / args.split
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    # Thực thi đánh giá hiệu năng (benchmark) trên tập dữ liệu đã chọn
    rows = benchmark_split(
        weights=cfg.path_value("weights"),  # Đường dẫn trọng số YOLO (.pt)
        checkpoint=checkpoint,              # Đường dẫn checkpoint mô hình RL
        image_root=cfg.path_value("image_root"),  # Thư mục chứa ảnh gốc
        label_root=cfg.path_value("label_root"),  # Thư mục chứa nhãn gốc (ground truth)
        cache_root=cfg.path_value("cache_root"),  # Thư mục chứa cache YOLO boxes và features
        split=args.split,                   # Tập dữ liệu (train/val/test)
        # Thiết lập cấu hình suy luận cho SAHI và RL-SAHI
        infer_cfg=InferenceConfig(
            full_imgsz=int(infer_cfg["full_imgsz"]),        # Kích thước ảnh đầy đủ
            slice_imgsz=int(infer_cfg["slice_imgsz"]),      # Kích thước của lát cắt (slice)
            full_conf=float(infer_cfg["full_conf"]),        # Confidence threshold khi chạy trên ảnh gốc
            output_conf=float(infer_cfg["output_conf"]),    # Confidence threshold đầu ra cuối cùng
            iou=float(infer_cfg["iou"]),                    # IoU threshold cho việc lọc box
            merge_iou=float(infer_cfg["merge_iou"]),        # IoU threshold dùng để gộp các lát cắt (NMS merge)
            max_det=int(infer_cfg["max_det"]),              # Số lượng vật thể phát hiện tối đa
            device=device,
            feature_layers=cfg.feature_layers("infer"),     # Các lớp trích xuất đặc trưng phục vụ cho RL state
            min_slice_detections=int(infer_cfg.get("min_slice_detections", 1)),
            min_slice_utility=float(infer_cfg.get("min_slice_utility", 0.5)),
            duplicate_iou=float(infer_cfg.get("duplicate_iou", infer_cfg.get("merge_iou", 0.5))),
            max_slice_attempts=int(infer_cfg.get("max_slice_attempts", 0)),
            target_classes=target_classes,
            require_stop_for_acceptance=bool(infer_cfg.get("require_stop_for_acceptance", True)),
            class_mapping=class_mapping,
        ),
        # Thiết lập cấu hình đánh giá benchmark cụ thể
        bench_cfg=BenchmarkConfig(
            iou_threshold=float(benchmark_cfg.get("iou_threshold", 0.5)),  # IoU threshold để tính True Positive
            fixed_slice_fraction=float(benchmark_cfg.get("fixed_slice_fraction", 0.35)),  # Tỷ lệ kích thước lát cắt cố định của SAHI thường
            fixed_overlap=float(benchmark_cfg.get("fixed_overlap", 0.2)),                  # Tỷ lệ chồng lấn (overlap) cố định của SAHI thường
            small_area_percentile=float(benchmark_cfg.get("small_area_percentile", 40.0)),  # Bách phân vị diện tích để định nghĩa vật thể "nhỏ"
            topk_slices=int(benchmark_cfg.get("topk_slices", 6)),  # Số slice của baseline heuristic top-K (không RL)
            topk_slice_fraction=float(benchmark_cfg.get("topk_slice_fraction", 0.35)),  # Tỷ lệ cạnh slice cho baseline top-K
            topk_min_separation=int(benchmark_cfg.get("topk_min_separation", 2)),  # Khoảng cách tối thiểu (ô lưới) giữa các đỉnh được chọn
            target_classes=target_classes,
            class_mapping=class_mapping,
        ),
        out_dir=out_dir,
        limit=args.limit,
        use_cache=bool(infer_cfg.get("use_cache", True)) and not args.no_cache,  # Có sử dụng cache hay không
    )
    
    # In kết quả đánh giá ra màn hình console cho từng phương pháp (YOLO gốc, SAHI cố định, RL-SAHI)
    for row in rows:
        print(
            f"[benchmark] {row['method']}: mAP50={row['mAP50']:.4f} "
            f"small_recall={row['small_recall']:.4f} fp/image={row['fp_per_image']:.2f} "
            f"crops/image={row['crops_per_image']:.2f} latency={row['latency_ms_per_image']:.1f}ms"
        )
    print(f"[benchmark] wrote {out_dir / 'benchmark.csv'}")


# Điểm khởi chạy của chương trình Python
if __name__ == "__main__":
    main()

