"""Cat train VisDrone thanh slice (tile) + chinh nhan -> de fine-tune YOLO tren crop.

Doc {root}/images/{split} + {root}/labels/{split}, cat luoi tile x tile (overlap),
chi giu box nam >= min-vis trong tile, ghi ra {root}/images/{split}_crops + labels.
Sinh san configs/visdrone_crops.yaml (train = crop + full TRON, val = val goc -> khong leak).

Standalone (chi cv2/numpy). Chay tren Lightning:
  python scripts/make_crops.py --data-root /teamspace/studios/this_studio/AIP391/AIP391/datasets/VisDrone
Roi fine-tune:
  yolo train model=best_visdrone.pt data=configs/visdrone_crops.yaml imgsz=640 epochs=30 batch=16 lr0=0.001
"""
import argparse
from pathlib import Path
import numpy as np, cv2

CLASS_NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
               "tricycle", "awning-tricycle", "bus", "motor"]
ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--data-root", type=Path, required=True, help="thu muc VisDrone (co images/train, labels/train, images/val)")
ap.add_argument("--split", default="train")
ap.add_argument("--tile", type=int, default=640)
ap.add_argument("--overlap", type=float, default=0.2)
ap.add_argument("--min-vis", type=float, default=0.3, help="giu box neu >= ti le nay dien tich nam trong tile")
ap.add_argument("--min-box", type=int, default=4, help="bo box < so px sau khi cat")
ap.add_argument("--limit", type=int, default=0, help="0 = het anh")
ap.add_argument("--keep-empty-frac", type=float, default=0.0, help="ti le tile khong co box van giu (background, giam FP)")
ap.add_argument("--yaml", type=Path, default=ROOT / "configs" / "visdrone_crops.yaml")
args = ap.parse_args()

root = args.data_root
img_dir = root / "images" / args.split
lbl_dir = root / "labels" / args.split
if not img_dir.exists():
    raise SystemExit(f"[crops] khong thay {img_dir} — chinh --data-root cho dung")
out_img = root / "images" / f"{args.split}_crops"
out_lbl = root / "labels" / f"{args.split}_crops"
out_img.mkdir(parents=True, exist_ok=True)
out_lbl.mkdir(parents=True, exist_ok=True)

def tiles(W, H, t, ov):
    step = max(1, int(round(t * (1 - ov))))
    def axis(L):
        if L <= t:
            return [0]
        xs = list(range(0, L - t + 1, step))
        if xs[-1] != L - t:
            xs.append(L - t)
        return xs
    xs, ys = axis(W), axis(H)
    return [(x, y, min(x + t, W), min(y + t, H)) for y in ys for x in xs]

imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
if args.limit:
    imgs = imgs[:args.limit]
if not imgs:
    raise SystemExit(f"[crops] khong co anh o {img_dir}")

rng = np.random.RandomState(0)
n_tiles = n_boxes = n_empty_kept = 0
for i, ip in enumerate(imgs):
    if i % 200 == 0 and i:
        print(f"  {i}/{len(imgs)} anh | {n_tiles} tile | {n_boxes} box", flush=True)
    im = cv2.imread(str(ip))
    if im is None:
        continue
    H, W = im.shape[:2]
    lp = lbl_dir / f"{ip.stem}.txt"
    gt = []  # (cls, x0,y0,x1,y1) pixel
    if lp.exists():
        for line in lp.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0])); cx, cy, w, h = (float(v) for v in p[1:5])
            x0 = (cx - w / 2) * W; y0 = (cy - h / 2) * H
            x1 = (cx + w / 2) * W; y1 = (cy + h / 2) * H
            gt.append((c, x0, y0, x1, y1))
    for (tx0, ty0, tx1, ty1) in tiles(W, H, args.tile, args.overlap):
        tw, th = tx1 - tx0, ty1 - ty0
        lines = []
        for (c, bx0, by0, bx1, by1) in gt:
            ix0, iy0 = max(bx0, tx0), max(by0, ty0)
            ix1, iy1 = min(bx1, tx1), min(by1, ty1)
            iw, ih = ix1 - ix0, iy1 - iy0
            if iw <= 0 or ih <= 0:
                continue
            oa = max((bx1 - bx0) * (by1 - by0), 1e-6)
            if (iw * ih) / oa < args.min_vis:
                continue
            if iw < args.min_box or ih < args.min_box:
                continue
            ncx = (ix0 + ix1) / 2 - tx0; ncy = (iy0 + iy1) / 2 - ty0
            lines.append(f"{c} {ncx / tw:.6f} {ncy / th:.6f} {iw / tw:.6f} {ih / th:.6f}")
        if not lines:
            if rng.rand() >= args.keep_empty_frac:
                continue
            n_empty_kept += 1
        sub = im[ty0:ty1, tx0:tx1]
        name = f"{ip.stem}_{tx0}_{ty0}"
        cv2.imwrite(str(out_img / f"{name}.jpg"), sub, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out_lbl / f"{name}.txt").write_text("\n".join(lines))
        n_tiles += 1; n_boxes += len(lines)

# --- sinh data.yaml: train = crop + full TRON (giu vat to), val = val goc (khong leak) ---
yaml = f"""# tao boi make_crops.py — fine-tune tren crop (train) + full, val goc
path: {root.as_posix()}
train:
  - images/{args.split}_crops
  - images/{args.split}
val: images/val
names:
""" + "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)) + "\n"
args.yaml.parent.mkdir(parents=True, exist_ok=True)
args.yaml.write_text(yaml)

print(f"\n[crops] XONG: {n_tiles} tile ({n_boxes} box, {n_empty_kept} tile rong) -> {out_img}")
print(f"[crops] data.yaml -> {args.yaml}")
print("[crops] Fine-tune (tron crop + full, val goc):")
print(f"  yolo train model=best_visdrone.pt data={args.yaml.as_posix()} imgsz=640 epochs=30 batch=16 lr0=0.001 name=ft_crop")
print("[crops] Xong train -> eval lai bang scripts/eval_coco_map.py voi weight moi (doi WEIGHTS).")
