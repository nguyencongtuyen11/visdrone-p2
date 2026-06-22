from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.cache import detection_cache_metadata
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.config import load_default_config
from rl_sahi.common.device import print_device_info
from rl_sahi.eval.benchmark import BenchmarkConfig
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.rl.env_config import EnvConfig
from rl_sahi.rl.state_config import StateConfig
from rl_sahi.rl.trainer import TrainConfig
from rl_sahi.rl.batched_trainer import batched_train_dqn

def _int_tuple(value) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(x.strip()) for x in value.split(",") if x.strip())
    return tuple(int(x) for x in value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN to choose one adaptive slice from cached YOLO state.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    cfg = load_default_config(args.config, ROOT)
    train_cfg = cfg.dataclass_instance("train", TrainConfig)
    env_cfg = cfg.dataclass_instance("env", EnvConfig)
    state_cfg = cfg.dataclass_instance("state", StateConfig)
    detect_cfg = cfg.section("detect")
    hard_cfg = cfg.section("hard_region")
    infer_cfg = cfg.section("infer")
    benchmark_cfg = cfg.section("benchmark")
    target_classes = _int_tuple(hard_cfg.get("target_classes", ()))
    infer_target_classes = _int_tuple(infer_cfg.get("target_classes", ()))
    if target_classes and infer_target_classes and target_classes != infer_target_classes:
        raise ValueError(
            "hard_region.target_classes must match infer.target_classes for train/inference alignment: "
            f"hard_region={target_classes}, infer={infer_target_classes}"
        )
    if not target_classes:
        target_classes = infer_target_classes
    class_mapping = ClassMapping.from_config(cfg.section("classes"))
    if args.episodes is not None:
        train_cfg.episodes = args.episodes
    if args.resume is not None:
        train_cfg.resume = bool(args.resume)
    device_name = args.device or cfg.optional_str("train", "device")
    print_device_info("train", device_name)

    checkpoint = batched_train_dqn(
        image_root=cfg.path_value("image_root"),
        cache_root=cfg.path_value("cache_root"),
        split=args.split,
        out_dir=cfg.path_value("dqn_out_dir"),
        cfg=train_cfg,
        env_cfg=env_cfg,
        state_cfg=state_cfg,
        limit=args.limit,
        device_name=device_name,
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
        label_root=cfg.path_value("label_root"),
        eval_weights=cfg.path_value("weights"),
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
        bench_cfg=BenchmarkConfig(
            iou_threshold=float(benchmark_cfg.get("iou_threshold", 0.5)),
            fixed_slice_fraction=float(benchmark_cfg.get("fixed_slice_fraction", 0.35)),
            fixed_overlap=float(benchmark_cfg.get("fixed_overlap", 0.2)),
            small_area_percentile=float(benchmark_cfg.get("small_area_percentile", 40.0)),
            target_classes=target_classes,
            class_mapping=class_mapping,
        ),
        eval_use_cache=bool(infer_cfg.get("use_cache", True)),
    )
    print(f"[train] best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
