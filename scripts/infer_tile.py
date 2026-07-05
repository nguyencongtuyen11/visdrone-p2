"""Infer 1 ảnh bằng tile-agent -> xuất detections.txt + metadata.json (định dạng như infer gốc)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.config import load_default_config
from rl_sahi.common.data import image_id
from rl_sahi.detection.yolo import load_yolo
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.inference.merge import save_prediction_txt
from rl_sahi.inference.pipeline import get_initial_detection
from rl_sahi.rl.env_config import EnvConfig
from rl_sahi.rl.state_config import StateConfig
from rl_sahi.rl.tile_coding import TileQAgent
from rl_sahi.rl.tile_infer import build_infer_cfg, predict_tile_sahi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    cfg = load_default_config(args.config, ROOT)
    device = cfg.optional_str("infer", "device")
    tc = tuple(int(x) for x in cfg.section("infer").get("target_classes", (0, 2, 3, 5, 8, 9)))
    cm = ClassMapping.from_config(cfg.section("classes"))
    infer_cfg = build_infer_cfg(cfg, device, tc, cm)
    env_cfg = cfg.dataclass_instance("env", EnvConfig)
    state_cfg = cfg.dataclass_instance("state", StateConfig)

    agent = TileQAgent.load(args.checkpoint)
    model = load_yolo(cfg.path_value("weights"), device=device)
    img = args.image if args.image.is_absolute() else ROOT / args.image
    sc = cfg.section("state")
    det = get_initial_detection(model=model, weights=cfg.path_value("weights"), image_path=img,
        weights_imgsz=infer_cfg.full_imgsz, full_conf=infer_cfg.full_conf, full_iou=infer_cfg.iou,
        max_det=infer_cfg.max_det, device=device, feature_layers=infer_cfg.feature_layers,
        aux_grid_size=int(sc["grid_size"]), spatial_feature_channels=int(sc.get("spatial_feature_channels", 4)),
        cache_root=cfg.path_value("cache_root"), split=args.split, use_cache=True)

    boxes, scores, classes, crop_count, slices_meta, sources = predict_tile_sahi(
        agent, model, img, det, infer_cfg, env_cfg, state_cfg, record=True)

    iid = image_id(img)
    out = cfg.path_value("infer_out_dir")
    det_path = out / "detections" / f"{iid}.txt"
    save_prediction_txt(det_path, boxes, scores, classes, sources)
    meta_path = out / "metadata" / f"{iid}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    rejected = sum(1 for s in slices_meta if not s["accepted"])
    meta_path.write_text(json.dumps({"image": str(img), "num_slices": int(crop_count),
        "num_rejected_slices": int(rejected), "slices": slices_meta, "detections": int(len(boxes))}, indent=2),
        encoding="utf-8")
    print(f"[infer_tile] {iid}: {len(boxes)} boxes, slices={crop_count}, rejected={rejected}")


if __name__ == "__main__":
    main()
