# Cho phép import các annotation nâng cao từ tương lai
from __future__ import annotations

import argparse  # Thư viện phân tích cú pháp tham số dòng lệnh
import sys       # Quản lý các biến và hàm hệ thống
import time      # Thư viện quản lý thời gian
from datetime import datetime, timedelta  # Lớp định dạng ngày giờ và khoảng thời gian
from pathlib import Path  # Xử lý đường dẫn file hướng đối tượng

# Xác định thư mục ROOT (cha của thư mục chứa train.py, đi lên 2 cấp)
ROOT = Path(__file__).resolve().parents[1]
# Thêm đường dẫn src vào danh sách tìm kiếm module của Python
sys.path.insert(0, str(ROOT / "src"))

# Import các lớp cấu hình và chức năng huấn luyện từ rl_sahi
from rl_sahi.common.cache import detection_cache_metadata  # Lấy metadata cache YOLO
from rl_sahi.common.class_mapping import ClassMapping      # Quản lý ánh xạ class đối tượng
from rl_sahi.common.config import load_default_config      # Load cấu hình mặc định
from rl_sahi.common.device import print_device_info        # In thông tin thiết bị chạy
from rl_sahi.eval.benchmark import BenchmarkConfig         # Cấu hình tham số đánh giá benchmark
from rl_sahi.inference.config import InferenceConfig       # Cấu hình tham số suy luận SAHI
from rl_sahi.rl.env_config import EnvConfig                 # Cấu hình tham số môi trường RL
from rl_sahi.rl.state_config import StateConfig             # Cấu hình vector biểu diễn trạng thái (state)
from rl_sahi.rl.trainer import TrainConfig                 # Cấu hình tham số huấn luyện dqn
from rl_sahi.rl.batched_trainer import batched_train_dqn   # Hàm chính chạy huấn luyện DQN theo lô (batched train)


def _int_tuple(value) -> tuple[int, ...]:
    """
    Hàm phụ trợ chuyển đổi danh sách class phân tách bởi dấu phẩy thành tuple số nguyên.
    """
    if isinstance(value, str):
        return tuple(int(x.strip()) for x in value.split(",") if x.strip())
    return tuple(int(x) for x in value)


def main() -> None:
    # Khởi tạo bộ phân tích tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Train DQN to choose one adaptive slice from cached YOLO state.")
    # Đường dẫn tới file config cấu hình
    parser.add_argument("--config", type=Path, default=None)
    # Phân vùng dữ liệu để huấn luyện (train, val, test)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    # Giới hạn số lượng episode huấn luyện (nếu chỉ định sẽ đè lên cấu hình file config)
    parser.add_argument("--episodes", type=int, default=None)
    # Giới hạn số lượng ảnh đầu vào (để chạy thử nhanh)
    parser.add_argument("--limit", type=int, default=None)
    # Chỉ định thiết bị chạy (ví dụ: cuda:0 hoặc cpu)
    parser.add_argument("--device", default=None)
    # Cho phép tiếp tục huấn luyện từ checkpoint cũ (resume) hay không
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    # Tải cấu hình từ config file
    cfg = load_default_config(args.config, ROOT)
    # Ánh xạ các phân vùng cấu hình vào các dataclass tương ứng
    train_cfg = cfg.dataclass_instance("train", TrainConfig)
    env_cfg = cfg.dataclass_instance("env", EnvConfig)
    state_cfg = cfg.dataclass_instance("state", StateConfig)
    
    detect_cfg = cfg.section("detect")
    hard_cfg = cfg.section("hard_region")
    infer_cfg = cfg.section("infer")
    benchmark_cfg = cfg.section("benchmark")
    
    # Xác định danh sách class mục tiêu từ cấu hình hard_region và infer
    target_classes = _int_tuple(hard_cfg.get("target_classes", ()))
    infer_target_classes = _int_tuple(infer_cfg.get("target_classes", ()))
    
    # Ràng buộc kiểm tra: các class mục tiêu huấn luyện vùng khó và các class suy luận phải khớp nhau
    if target_classes and infer_target_classes and target_classes != infer_target_classes:
        raise ValueError(
            "hard_region.target_classes must match infer.target_classes for train/inference alignment: "
            f"hard_region={target_classes}, infer={infer_target_classes}"
        )
    if not target_classes:
        target_classes = infer_target_classes
        
    # Tạo ánh xạ class đối tượng
    class_mapping = ClassMapping.from_config(cfg.section("classes"))
    
    # Cập nhật số tập (episodes) huấn luyện nếu được truyền qua dòng lệnh
    if args.episodes is not None:
        train_cfg.episodes = args.episodes
    # Cập nhật trạng thái resume nếu được truyền qua dòng lệnh
    if args.resume is not None:
        train_cfg.resume = bool(args.resume)
        
    # Xác định thiết bị tính toán
    device_name = args.device or cfg.optional_str("train", "device")
    print_device_info("train", device_name)

    # --- Bắt đầu đo lường thời gian huấn luyện ---
    start_wall = time.perf_counter()
    start_stamp = datetime.now()
    print(f"[train] started at {start_stamp:%Y-%m-%d %H:%M:%S}")

    # Gọi hàm thực thi huấn luyện mô hình DQN theo batch
    checkpoint = batched_train_dqn(
        image_root=cfg.path_value("image_root"),   # Đường dẫn thư mục chứa ảnh
        cache_root=cfg.path_value("cache_root"),   # Thư mục lưu cache YOLO và vùng khó
        split=args.split,                        # Phân vùng dữ liệu (train/val)
        out_dir=cfg.path_value("dqn_out_dir"),     # Thư mục lưu kết quả checkpoint DQN và log
        cfg=train_cfg,                           # Cấu hình DQN Trainer
        env_cfg=env_cfg,                         # Cấu hình Môi trường RL
        state_cfg=state_cfg,                     # Cấu hình Trạng thái RL
        limit=args.limit,                        # Giới hạn số lượng ảnh chạy
        device_name=device_name,                 # Thiết bị tính toán (CPU/GPU)
        # Siêu dữ liệu cache phát hiện đối tượng YOLO
        detection_metadata=detection_cache_metadata(
            weights=cfg.path_value("weights"),
            imgsz=int(detect_cfg["imgsz"]),
            conf=float(detect_cfg["conf"]),
            iou=float(detect_cfg["iou"]),
            max_det=int(detect_cfg["max_det"]),
            feature_layers=cfg.feature_layers("detect"),
            aux_grid_size=int(state_cfg.grid_size),
            spatial_feature_channels=int(state_cfg.spatial_feature_channels),
        ),
        target_classes=target_classes,
        class_mapping=class_mapping,
        label_root=cfg.path_value("label_root"),   # Thư mục nhãn gốc (ground truth)
        eval_weights=cfg.path_value("weights"),    # Trọng số YOLO dùng để đánh giá trong lúc train
        # Thiết lập cấu hình suy luận cho SAHI chạy trong quá trình đánh giá (evaluation)
        infer_cfg=InferenceConfig(
            full_imgsz=int(infer_cfg["full_imgsz"]),
            slice_imgsz=int(infer_cfg["slice_imgsz"]),
            full_conf=float(infer_cfg["full_conf"]),
            output_conf=float(infer_cfg["output_conf"]),
            iou=float(infer_cfg["iou"]),
            merge_iou=float(infer_cfg["merge_iou"]),
            max_det=int(infer_cfg["max_det"]),
            device=device_name or cfg.optional_str("infer", "device"),
            feature_layers=cfg.feature_layers("infer"),
            min_slice_detections=int(infer_cfg.get("min_slice_detections", 1)),
            min_slice_utility=float(infer_cfg.get("min_slice_utility", 0.5)),
            duplicate_iou=float(infer_cfg.get("duplicate_iou", infer_cfg.get("merge_iou", 0.5))),
            max_slice_attempts=int(infer_cfg.get("max_slice_attempts", 0)),
            target_classes=target_classes,
            require_stop_for_acceptance=bool(infer_cfg.get("require_stop_for_acceptance", True)),
            class_mapping=class_mapping,
        ),
        # Thiết lập cấu hình đánh giá benchmark trong quá trình huấn luyện
        bench_cfg=BenchmarkConfig(
            iou_threshold=float(benchmark_cfg.get("iou_threshold", 0.5)),
            fixed_slice_fraction=float(benchmark_cfg.get("fixed_slice_fraction", 0.35)),
            fixed_overlap=float(benchmark_cfg.get("fixed_overlap", 0.2)),
            small_area_percentile=float(benchmark_cfg.get("small_area_percentile", 40.0)),
            topk_slices=int(benchmark_cfg.get("topk_slices", 6)),
            topk_slice_fraction=float(benchmark_cfg.get("topk_slice_fraction", 0.35)),
            topk_min_separation=int(benchmark_cfg.get("topk_min_separation", 2)),
            target_classes=target_classes,
            class_mapping=class_mapping,
        ),
        eval_use_cache=bool(infer_cfg.get("use_cache", True)),
    )

    # --- Kết thúc đo lường thời gian huấn luyện và ghi log thời gian ---
    elapsed = time.perf_counter() - start_wall
    end_stamp = datetime.now()
    elapsed_str = str(timedelta(seconds=round(elapsed)))
    episodes = train_cfg.episodes
    per_ep = elapsed / episodes if episodes else 0.0
    summary = (
        f"started:  {start_stamp:%Y-%m-%d %H:%M:%S}\n"
        f"finished: {end_stamp:%Y-%m-%d %H:%M:%S}\n"
        f"elapsed:  {elapsed_str}  ({elapsed:.1f} s)\n"
        f"episodes: {episodes}\n"
        f"avg/episode: {per_ep:.3f} s\n"
        f"device: {device_name or 'auto'}\n"
        f"split: {args.split}  limit: {args.limit}\n"
    )
    time_path = cfg.path_value("dqn_out_dir") / "train_time.txt"
    # Đảm bảo thư mục lưu log thời gian tồn tại
    time_path.parent.mkdir(parents=True, exist_ok=True)
    # Ghi đè hoặc nối thêm thông tin thống kê thời gian huấn luyện vào file train_time.txt
    with time_path.open("a", encoding="utf-8") as fh:
        fh.write(summary + ("-" * 40) + "\n")
        
    print("[train] ===== training time =====")
    print(summary, end="")
    print(f"[train] time log appended to: {time_path}")
    print(f"[train] best checkpoint: {checkpoint}")


# Điểm khởi chạy của chương trình
if __name__ == "__main__":
    main()

