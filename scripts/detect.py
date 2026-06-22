from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.config import load_default_config
from rl_sahi.common.device import print_device_info
from rl_sahi.detection.cache_builder import cache_detections_for_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache full-image YOLO boxes and backbone features.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_default_config(args.config, ROOT)
    detect_cfg = cfg.section("detect")
    state_cfg = cfg.section("state")
    device = cfg.optional_str("detect", "device")
    print_device_info("detect", device)

    written = cache_detections_for_split(
        weights=cfg.path_value("weights"),
        image_root=cfg.path_value("image_root"),
        cache_root=cfg.path_value("cache_root"),
        split=args.split,
        imgsz=int(detect_cfg["imgsz"]),
        conf=float(detect_cfg["conf"]),
        iou=float(detect_cfg["iou"]),
        max_det=int(detect_cfg["max_det"]),
        device=device,
        feature_layers=cfg.feature_layers("detect"),
        aux_grid_size=int(state_cfg["grid_size"]),
        spatial_feature_channels=int(state_cfg.get("spatial_feature_channels", 4)),
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(f"[detect] wrote {written} caches under {cfg.path_value('cache_root') / 'detections' / args.split}")


if __name__ == "__main__":
    main()
