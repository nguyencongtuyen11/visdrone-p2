"""Benchmark 3 phương pháp KHÔNG-RL (full / fixed-grid SAHI / objectness-topk) với 1 detector.

Dùng để xem nhanh bức tranh khi ĐỔI detector (vd fine-tune VisDrone) mà chưa cần train lại RL.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.config import load_default_config
from rl_sahi.common.data import iter_images
from rl_sahi.detection.yolo import load_yolo
from rl_sahi.eval.benchmark import (
    BenchmarkConfig, _evaluate_method, _full_predictions, _image_shape,
    _predict_fixed_sahi, _predict_objectness_topk, _read_gt, _small_area_threshold,
)
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.inference.pipeline import get_initial_detection


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=150)
    args = ap.parse_args()

    cfg = load_default_config(args.config, ROOT)
    device = cfg.optional_str("infer", "device")
    ic, bc, sc = cfg.section("infer"), cfg.section("benchmark"), cfg.section("state")
    tc = tuple(int(x) for x in ic.get("target_classes", (0, 2, 3, 5, 8, 9)))
    cm = ClassMapping.from_config(cfg.section("classes"))
    infer_cfg = InferenceConfig(full_imgsz=int(ic["full_imgsz"]), slice_imgsz=int(ic["slice_imgsz"]),
        full_conf=float(ic["full_conf"]), output_conf=float(ic["output_conf"]), iou=float(ic["iou"]),
        merge_iou=float(ic["merge_iou"]), max_det=int(ic["max_det"]), device=device,
        feature_layers=cfg.feature_layers("infer"), target_classes=tc, class_mapping=cm)
    bench_cfg = BenchmarkConfig(iou_threshold=float(bc.get("iou_threshold", 0.5)),
        fixed_slice_fraction=float(bc.get("fixed_slice_fraction", 0.35)), fixed_overlap=float(bc.get("fixed_overlap", 0.2)),
        small_area_percentile=float(bc.get("small_area_percentile", 40.0)), topk_slices=int(bc.get("topk_slices", 14)),
        topk_slice_fraction=float(bc.get("topk_slice_fraction", 0.35)), topk_min_separation=int(bc.get("topk_min_separation", 2)),
        target_classes=tc, class_mapping=cm)

    model = load_yolo(cfg.path_value("weights"), device=device)
    image_root, label_root, cache_root = cfg.path_value("image_root"), cfg.path_value("label_root"), cfg.path_value("cache_root")
    images = iter_images(image_root, split=args.split, limit=args.limit)
    small_thr = _small_area_threshold(images, image_root, label_root, tc, bench_cfg.small_area_percentile, cm)

    gt = {}
    preds = {"yolo_full": {}, "fixed_grid_sahi": {}, "objectness_topk": {}}
    crops = {k: [] for k in preds}
    for img in images:
        iid = img.stem
        gb, gc = _read_gt(img, image_root, label_root, tc, cm)
        gt[iid] = (gb, gc, _image_shape(img))
        det = get_initial_detection(model=model, weights=cfg.path_value("weights"), image_path=img,
            weights_imgsz=infer_cfg.full_imgsz, full_conf=infer_cfg.full_conf, full_iou=infer_cfg.iou,
            max_det=infer_cfg.max_det, device=device, feature_layers=infer_cfg.feature_layers,
            aux_grid_size=int(sc["grid_size"]), spatial_feature_channels=int(sc.get("spatial_feature_channels", 4)),
            cache_root=cache_root, split=args.split, use_cache=True)
        preds["yolo_full"][iid] = _full_predictions(det, infer_cfg); crops["yolo_full"].append(0)
        b, s, c, n = _predict_fixed_sahi(model, img, det, infer_cfg, bench_cfg); preds["fixed_grid_sahi"][iid] = (b, s, c); crops["fixed_grid_sahi"].append(n)
        b, s, c, n = _predict_objectness_topk(model, img, det, infer_cfg, bench_cfg); preds["objectness_topk"][iid] = (b, s, c); crops["objectness_topk"].append(n)

    for method in preds:
        m = _evaluate_method(preds[method], gt, tc, bench_cfg.iou_threshold, small_thr)
        print(f"[bench_det] {method}: mAP50={m['mAP50']:.4f} small_recall={m['small_recall']:.4f} "
              f"fp/image={m['fp_per_image']:.2f} crops/image={float(np.mean(crops[method])):.2f}")


if __name__ == "__main__":
    main()
