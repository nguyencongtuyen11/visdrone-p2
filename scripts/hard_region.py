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
from rl_sahi.hard_region.cache_builder import cache_hard_regions_for_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache small GT boxes that full-image YOLO misses or scores low.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_default_config(args.config, ROOT)
    detect_cfg = cfg.section("detect")
    print_device_info("hard", cfg.optional_str("detect", "device"))
    hard_cfg = cfg.section("hard_region")
    state_cfg = cfg.section("state")
    target_raw = hard_cfg.get("target_classes", [])
    if isinstance(target_raw, str):
        target_classes = tuple(int(x.strip()) for x in target_raw.split(",") if x.strip())
    else:
        target_classes = tuple(int(x) for x in target_raw)

    written = cache_hard_regions_for_split(
        image_root=cfg.path_value("image_root"),
        label_root=cfg.path_value("label_root"),
        cache_root=cfg.path_value("cache_root"),
        split=args.split,
        small_area_ratio=float(hard_cfg["small_area_ratio"]),
        small_area_percentile=(
            None
            if hard_cfg.get("small_area_percentile") in (None, "")
            else float(hard_cfg["small_area_percentile"])
        ),
        match_iou=float(hard_cfg["match_iou"]),
        min_detect_score=float(hard_cfg["min_detect_score"]),
        target_classes=target_classes,
        class_mapping=ClassMapping.from_config(cfg.section("classes")),
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
    print(f"[hard] wrote {written} caches under {cfg.path_value('cache_root') / 'hard_regions' / args.split}")


if __name__ == "__main__":
    main()
