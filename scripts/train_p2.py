"""Train YOLO11s-P2 @1280 tren VisDrone cho SMALL-OBJECT detection.

Y tuong: thay vi cat anh (SAHI/RL-SAHI, chay YOLO 15-29 lan/anh, ~2000ms),
day mot dau P2/4 (stride-4) vao detector de no NHIN duoc vat ~4px trong 1 LUOT.
Warm-start backbone + head P3 tu best_visdrone.pt (khop 50% layer -> hoi tu nhanh).

Chay:
  python scripts/train_p2.py                        # 80 epoch, batch 8, imgsz 1280, device 0
  python scripts/train_p2.py --epochs 60 --batch 4  # neu OOM (T4 16GB thuong du batch 8)
  python scripts/train_p2.py --imgsz 1024           # ha phan giai neu VRAM cang

Ket qua: runs/p2/p2_visdrone_1280/weights/best.pt  (keo ve local de benchmark vs SAHI)
"""
import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent          # .../Test
DATA_ROOT = ROOT / "data" / "raw"
VISDRONE_NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
                  "tricycle", "awning-tricycle", "bus", "motor"]


def write_abs_data_yaml() -> Path:
    """Sinh data yaml voi duong dan TUYET DOI (tranh gotcha datasets_dir cua Ultralytics)."""
    d = {
        "path": str(DATA_ROOT).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(VISDRONE_NAMES)},
    }
    out = ROOT / "configs" / "data_visdrone.autogen.yaml"
    out.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def parse_batch(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return float(v)          # 0.80 = dung 80% VRAM (AutoBatch fraction)
        except (TypeError, ValueError):
            return 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "configs" / "yolo11s-p2.yaml"))
    ap.add_argument("--base", default=str(ROOT / "best_visdrone.pt"),
                    help="weight warm-start (backbone+P3). Dat '' de bo qua.")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", default="8", help="int | -1 (auto) | 0.80 (dung 80% VRAM)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", default="p2_visdrone_1280")
    ap.add_argument("--project", default=str(ROOT / "runs" / "p2"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if not DATA_ROOT.exists():
        sys.exit(f"[train_p2] KHONG thay {DATA_ROOT}\n"
                 f"          Chay truoc: python scripts/download_visdrone.py")

    data = write_abs_data_yaml()
    print(f"[train_p2] data      = {data}")
    print(f"[train_p2] imgsz={args.imgsz} epochs={args.epochs} batch={args.batch} device={args.device}")

    model = YOLO(args.model)
    if args.base and Path(args.base).exists():
        model.load(args.base)                       # intersect_dicts: backbone+P3 khop ~50%
        print(f"[train_p2] warm-start backbone+P3 tu {args.base}")
    else:
        print("[train_p2] KHONG warm-start (train tu init).")

    model.train(
        data=str(data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=parse_batch(args.batch),
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        patience=20,
        cos_lr=True,
        optimizer="auto",
        # augment than thien vat nho
        mosaic=1.0,
        close_mosaic=10,
        scale=0.5,
        val=True,
        plots=True,
    )
    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n[train_p2] XONG. best.pt -> {best}")
    print("[train_p2] Keo file nay ve local, benchmark vs full-640/SAHI/RL-SAHI.")


if __name__ == "__main__":
    main()
