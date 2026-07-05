"""HYBRID coverage+focus: lưới THÔ phủ toàn ảnh (đảm bảo coverage như SAHI nhưng ít lát)
+ lát MỊN thích ứng đặt vào các cụm objectness dày (zoom sâu đúng chỗ vật nhỏ).

Mục tiêu: recall >= fixed-grid SAHI (0.603 @28 crop) với ÍT crop hơn.
Deterministic (không cần train RL) -> dễ lặp tham số nhanh.
"""
from __future__ import annotations

import argparse
import os
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
    _evaluate_method,
    _fixed_grid_rois,
    _full_predictions,
    _image_shape,
    _merge_predictions,
    _objectness_grid,
    _read_gt,
    _small_area_threshold,
    _topk_peak_rois,
)
from rl_sahi.eval.benchmark import _predict_rl_sahi
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.inference.crops import run_yolo_on_crop
from rl_sahi.inference.pipeline import _filter_classes, get_initial_detection
from rl_sahi.rl.checkpoint import load_policy


def predict_hybrid(
    model,
    image_path: Path,
    det,
    cfg: InferenceConfig,
    coarse_frac: float,
    coarse_overlap: float,
    fine_k: int,
    fine_frac: float,
    fine_sep: int,
    slice_imgsz: int,
):
    full_b, full_s, full_c = _full_predictions(det, cfg)
    rois = []
    if coarse_frac > 0:
        rois += _fixed_grid_rois(det.image_shape, coarse_frac, coarse_overlap)
    if fine_k > 0:
        rois += _topk_peak_rois(_objectness_grid(det), det.image_shape, fine_k, fine_frac, fine_sep)
    bp, sp, cp = [full_b], [full_s], [full_c]
    for roi in rois:
        b, s, c = run_yolo_on_crop(
            model, image_path, roi,
            imgsz=slice_imgsz, conf=cfg.output_conf, iou=cfg.iou,
            max_det=cfg.max_det, device=cfg.device,
        )
        c = cfg.class_mapping.map_model_classes(c)
        b, s, c = _filter_classes(b, s, c, cfg.target_classes)
        bp.append(b); sp.append(s); cp.append(c)
    b, s, c = _merge_predictions(det.image_shape, cfg.merge_iou, bp, sp, cp)
    return b, s, c, len(rois)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--coarse-frac", type=float, default=0.6)
    ap.add_argument("--coarse-overlap", type=float, default=0.15)
    ap.add_argument("--fine-k", type=int, default=8)
    ap.add_argument("--fine-frac", type=float, default=0.25)
    ap.add_argument("--fine-sep", type=int, default=2)
    ap.add_argument("--slice-imgsz", type=int, default=0, help="0 = dùng infer.slice_imgsz")
    ap.add_argument("--fine-mode", choices=["topk", "rl"], default="topk",
                    help="rl: lát mịn do RL agent chọn (cần --checkpoint) thay vì topk heuristic")
    ap.add_argument("--checkpoint", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_default_config(args.config, ROOT)
    device = cfg.optional_str("infer", "device")
    ic, bc, sc = cfg.section("infer"), cfg.section("benchmark"), cfg.section("state")
    tc = tuple(int(x) for x in ic.get("target_classes", (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)))
    cm = ClassMapping.from_config(cfg.section("classes"))
    infer_cfg = InferenceConfig(
        full_imgsz=int(ic["full_imgsz"]), slice_imgsz=int(ic["slice_imgsz"]),
        full_conf=float(ic["full_conf"]), output_conf=float(ic["output_conf"]),
        iou=float(ic["iou"]), merge_iou=float(ic["merge_iou"]), max_det=int(ic["max_det"]),
        device=device, feature_layers=cfg.feature_layers("infer"),
        target_classes=tc, class_mapping=cm,
    )
    slice_imgsz = args.slice_imgsz or infer_cfg.slice_imgsz
    iou_thr = float(bc.get("iou_threshold", 0.5))
    pct = float(bc.get("small_area_percentile", 40.0))

    model = load_yolo(cfg.path_value("weights"), device=device)
    image_root, label_root, cache_root = cfg.path_value("image_root"), cfg.path_value("label_root"), cfg.path_value("cache_root")
    images = iter_images(image_root, split=args.split, limit=args.limit)
    small_thr = _small_area_threshold(images, image_root, label_root, tc, pct, cm)

    policy = env_cfg_rl = state_cfg_rl = device_t = None
    if args.fine_mode == "rl":
        from rl_sahi.common.device import resolve_torch_device
        device_t = resolve_torch_device(device)
        policy, ckpt = load_policy(args.checkpoint, device_t)
        env_cfg_rl = ckpt["env_cfg_obj"]
        env_cfg_rl.max_slices = int(args.fine_k)  # RL đóng vai bộ chọn lát MỊN, cap = fine_k
        from rl_sahi.rl.state_config import StateConfig as _SC
        state_cfg_rl = ckpt.get("state_cfg_obj", _SC())

    gt, preds, crops = {}, {}, []
    for img in images:
        iid = img.stem
        gb, gc = _read_gt(img, image_root, label_root, tc, cm)
        gt[iid] = (gb, gc, _image_shape(img))
        det = get_initial_detection(
            model=model, weights=cfg.path_value("weights"), image_path=img,
            weights_imgsz=infer_cfg.full_imgsz, full_conf=infer_cfg.full_conf, full_iou=infer_cfg.iou,
            max_det=infer_cfg.max_det, device=device, feature_layers=infer_cfg.feature_layers,
            aux_grid_size=int(sc["grid_size"]),
            spatial_feature_channels=int(sc.get("spatial_feature_channels", 4)),
            cache_root=cache_root, split=args.split, use_cache=True,
        )
        if args.fine_mode == "rl":
            # RL chọn lát mịn (gồm cả full preds đã merge) + lưới thô coverage, merge lần cuối
            rb, rs, rc, rl_n = _predict_rl_sahi(
                model, policy, device_t, img, det, infer_cfg, env_cfg_rl, state_cfg_rl,
            )
            bp, sp, cp = [rb], [rs], [rc]
            coarse = _fixed_grid_rois(det.image_shape, args.coarse_frac, args.coarse_overlap)
            for roi in coarse:
                b2, s2, c2 = run_yolo_on_crop(
                    model, img, roi, imgsz=slice_imgsz, conf=infer_cfg.output_conf,
                    iou=infer_cfg.iou, max_det=infer_cfg.max_det, device=infer_cfg.device,
                )
                c2 = infer_cfg.class_mapping.map_model_classes(c2)
                b2, s2, c2 = _filter_classes(b2, s2, c2, infer_cfg.target_classes)
                bp.append(b2); sp.append(s2); cp.append(c2)
            b, s, c = _merge_predictions(det.image_shape, infer_cfg.merge_iou, bp, sp, cp)
            n = rl_n + len(coarse)
        else:
            b, s, c, n = predict_hybrid(
                model, img, det, infer_cfg,
                args.coarse_frac, args.coarse_overlap, args.fine_k, args.fine_frac, args.fine_sep,
                slice_imgsz,
            )
        preds[iid] = (b, s, c); crops.append(n)
        if os.environ.get("HYBRID_DEBUG"):
            print(f"[dbg] {iid} rois={n} merged_boxes={len(b)}", file=sys.stderr, flush=True)

    dump = os.environ.get("HYBRID_DUMP")
    if dump:
        np.savez_compressed(dump, small_thr=small_thr,
                            **{f"{iid}__b": p[0] for iid, p in preds.items()},
                            **{f"{iid}__s": p[1] for iid, p in preds.items()},
                            **{f"{iid}__c": p[2] for iid, p in preds.items()})
        print(f"[dump] small_thr={small_thr:.8f} -> {dump}", file=sys.stderr)
    m = _evaluate_method(preds, gt, tc, iou_thr, small_thr)
    print(
        f"[hybrid] mode={args.fine_mode} coarse={args.coarse_frac}/{args.coarse_overlap} fine_k={args.fine_k}@{args.fine_frac} "
        f"imgsz={slice_imgsz} conf={infer_cfg.output_conf} classes={list(tc)} | "
        f"mAP50={m['mAP50']:.4f} small_recall={m['small_recall']:.4f} "
        f"fp/image={m['fp_per_image']:.2f} crops/image={float(np.mean(crops)):.2f}"
    )


if __name__ == "__main__":
    main()
