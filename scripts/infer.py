from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.config import load_default_config
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.data import iter_images
from rl_sahi.common.device import print_device_info
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.inference.pipeline import AdaptiveSahiInferencer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adaptive-slice inference and save boxes plus slice visualization.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--split", default=None, choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    cfg = load_default_config(args.config, ROOT)
    infer_cfg = cfg.section("infer")
    device = cfg.optional_str("infer", "device")
    print_device_info("infer", device)
    target_raw = infer_cfg.get("target_classes", [0, 2, 3, 5, 8, 9])
    if isinstance(target_raw, str):
        target_classes = tuple(int(x.strip()) for x in target_raw.split(",") if x.strip())
    else:
        target_classes = tuple(int(x) for x in target_raw)

    if args.image is not None:
        image_path = args.image if args.image.is_absolute() else ROOT / args.image
        images = [image_path]
        split = args.split
    else:
        if args.split is None:
            raise ValueError("Use --image for one image or --split train/val/test for a dataset split.")
        images = iter_images(cfg.path_value("image_root"), split=args.split, limit=args.limit)
        split = args.split

    if args.checkpoint is None:
        checkpoint = cfg.path_value("checkpoint")
    else:
        checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    inferencer = AdaptiveSahiInferencer(
        weights=cfg.path_value("weights"),
        checkpoint=checkpoint,
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
    for image_path in images:
        meta = inferencer.infer_image(
            image_path=image_path,
            out_dir=cfg.path_value("infer_out_dir"),
            cache_root=cfg.path_value("cache_root") if split is not None else None,
            split=split,
            use_cache=bool(infer_cfg["use_cache"]) and not args.no_cache,
        )
        print(f"[infer] {image_path.name}: {meta['detections']} boxes, slices={meta['num_slices']}")


if __name__ == "__main__":
    main()
