"""Benchmark tác nhân TILE-CODING trên 1 split (small_recall/mAP/FP/crops) để so với Deep-DQN."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.config import load_default_config
from rl_sahi.common.data import iter_images
from rl_sahi.detection.yolo import load_yolo
from rl_sahi.eval.benchmark import _evaluate_method, _read_gt, _small_area_threshold, _image_shape
from rl_sahi.inference.pipeline import get_initial_detection
from rl_sahi.rl.env_config import EnvConfig
from rl_sahi.rl.state_config import StateConfig
from rl_sahi.rl.tile_coding import TileQAgent
from rl_sahi.rl.tile_infer import build_infer_cfg, predict_tile_sahi

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=150)
    args = ap.parse_args()

    cfg = load_default_config(args.config, ROOT)
    device = cfg.optional_str("infer", "device")
    tc = tuple(int(x) for x in cfg.section("infer").get("target_classes", (0, 2, 3, 5, 8, 9)))
    cm = ClassMapping.from_config(cfg.section("classes"))
    infer_cfg = build_infer_cfg(cfg, device, tc, cm)
    env_cfg = cfg.dataclass_instance("env", EnvConfig)
    state_cfg = cfg.dataclass_instance("state", StateConfig)
    bc = cfg.section("benchmark")
    iou_thr = float(bc.get("iou_threshold", 0.5))
    pct = float(bc.get("small_area_percentile", 40.0))

    agent = TileQAgent.load(args.checkpoint)
    model = load_yolo(cfg.path_value("weights"), device=device)
    image_root, label_root, cache_root = cfg.path_value("image_root"), cfg.path_value("label_root"), cfg.path_value("cache_root")
    sc = cfg.section("state")

    images = iter_images(image_root, split=args.split, limit=args.limit)
    small_thr = _small_area_threshold(images, image_root, label_root, tc, pct, cm)
    gt, preds, crops = {}, {}, []
    for img in images:
        iid = img.stem
        gb, gc = _read_gt(img, image_root, label_root, tc, cm)
        gt[iid] = (gb, gc, _image_shape(img))
        det = get_initial_detection(model=model, weights=cfg.path_value("weights"), image_path=img,
            weights_imgsz=infer_cfg.full_imgsz, full_conf=infer_cfg.full_conf, full_iou=infer_cfg.iou,
            max_det=infer_cfg.max_det, device=device, feature_layers=infer_cfg.feature_layers,
            aux_grid_size=int(sc["grid_size"]), spatial_feature_channels=int(sc.get("spatial_feature_channels", 4)),
            cache_root=cache_root, split=args.split, use_cache=True)
        b, s, c, crop_count, _, _ = predict_tile_sahi(agent, model, img, det, infer_cfg, env_cfg, state_cfg)
        preds[iid] = (b, s, c); crops.append(crop_count)
    m = _evaluate_method(preds, gt, tc, iou_thr, small_thr)
    print(f"[bench_tile] output_conf={infer_cfg.output_conf} rl_tile: "
          f"mAP50={m['mAP50']:.4f} small_recall={m['small_recall']:.4f} "
          f"fp/image={m['fp_per_image']:.2f} crops/image={float(np.mean(crops)):.2f}")


if __name__ == "__main__":
    main()
