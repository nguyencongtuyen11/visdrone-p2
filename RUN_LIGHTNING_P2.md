# Train YOLO11s-P2 @1280 tren Lightning (qua GitHub)

Muc tieu: detector **nhin vat nho trong 1 luot** (dau P2/4 stride-4), thay cho cat anh
(SAHI/RL-SAHI chay YOLO 15-29 lan/anh ~2000ms). Ky vong: recall vat nho cao hon,
mAP cao hon, va **nhanh ~10-13x** (1 luot ~50-70ms tren T4).

Kien truc da validate: strides = [4, 8, 16, 32], 9.58M params (chi hon yolo11s 0.12M),
warm-start 50% layer tu best_visdrone.pt, forward @1280 OK.

---

## 0) Chuan bi 1 lan: dua code len GitHub (chay o LOCAL, trong thu muc Test/)

Chua co repo git nen phai khoi tao. Mo terminal tai `...\do-an-moi\Test`:

```bash
git init
git add -A
git status          # KIEM TRA: KHONG duoc thay data/ hay runs/; PHAI thay best_visdrone.pt + yolo11s.pt
git commit -m "P2 small-object detector (yolo11s-p2 @1280) + configs cho Lightning"
```

Tao repo tren GitHub roi push (chon **Private** de khong lo weight/code do an):

**Cach A - co GitHub CLI (`gh`):**
```bash
gh repo create do-an-visdrone-p2 --private --source=. --push
```

**Cach B - khong co gh:** vao github.com -> New repository -> dat ten (VD `do-an-visdrone-p2`),
chon **Private**, KHONG tich "Add README". Xong chay:
```bash
git remote add origin https://github.com/<TEN_GITHUB>/do-an-visdrone-p2.git
git branch -M main
git push -u origin main
```

> Lan sau sua code: `git add -A && git commit -m "..." && git push`

---

## 1) Tren Lightning: pull ve va train

```bash
git clone https://github.com/<TEN_GITHUB>/do-an-visdrone-p2.git
cd do-an-visdrone-p2

pip install -U ultralytics                 # keo torch CUDA + deps
python scripts/download_visdrone.py        # tai VisDrone -> data/raw (~vai phut)

python scripts/train_p2.py --epochs 80 --batch 8 --device 0
```

Neu bao **OOM** (het VRAM):
```bash
python scripts/train_p2.py --epochs 80 --batch 4 --device 0      # giam batch
# hoac ha phan giai (van hon han 640):
python scripts/train_p2.py --epochs 80 --imgsz 1024 --device 0
```

Thoi gian uoc: T4 16GB, batch 8 @1280, ~80 epoch => vai gio (co early-stop patience=20).
Vi backbone da warm-start tu VisDrone nen hoi tu nhanh hon train tu dau.

Ket qua: `runs/p2/p2_visdrone_1280/weights/best.pt` + duong cong `results.png`.

---

## 2) Keo weight ve local, benchmark vs SAHI/RL-SAHI

Tren Lightning tai file `runs/p2/p2_visdrone_1280/weights/best.pt` ve, dat ten `best_p2.pt`
vao thu muc `Test/`. Sau do o LOCAL do lai tren cung tap test:

```bash
# so 1-luot P2@1280 vs full-640 / fixed-grid SAHI (dung script da co)
python scripts/benchmark_detector.py --weights best_p2.pt --imgsz 1280
```
(Bao minh khi weight ve — minh gan dung co lenh benchmark + bang so sanh cho khop bao cao.)

---

## Cau chuyen do an (khung bao cao)
- Baseline: full@640 (goc) · SAHI-28 · RL-SAHI · full@1280 (1 luot).
- **Dong gop: dau P2 @1280 — dạy detector nhin vat nho, thay vi cat anh luc chay.**
- Ket qua ky vong: recall vat nho >= SAHI, mAP cao hon, **nhanh hon ~10x** (1 luot).
- SAHI/RL-SAHI tro thanh "meo luc inference" de doi chung -> lam noi bat huong train detector.

## Loi thuong gap
- `Dataset ... not found` / tim nham `~/datasets`: dung `scripts/train_p2.py` (no tu ghi
  duong dan tuyet doi vao `configs/data_visdrone.autogen.yaml`), dung sua tay `data_visdrone.yaml`.
- OOM: giam `--batch` (8->4->2) hoac `--imgsz 1024`.
- Train lai tiep tu cho dut: them `--resume`.
