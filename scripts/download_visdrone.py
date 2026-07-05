"""Tải tập dữ liệu VisDrone-DET và chuyển đổi sang định dạng YOLO tại ./data/raw (chạy từ thư mục gốc của dự án)."""
# Cho phép import các annotation kiểu nâng cao từ tương lai
from __future__ import annotations

import shutil  # Thư viện dùng để thực hiện các thao tác tệp cấp cao (như xóa thư mục cây rmtree)
from pathlib import Path  # Thư viện xử lý đường dẫn hướng đối tượng

# Import các hàm tải xuống và hiển thị thanh tiến trình từ ultralytics
from ultralytics.utils.downloads import download
from ultralytics.utils import ASSETS_URL, TQDM
from PIL import Image  # Thư viện gối xử lý hình ảnh PIL để đọc kích thước ảnh

# Định nghĩa đường dẫn gốc (ROOT) của dự án
ROOT = Path(__file__).resolve().parents[1]
# Thư mục tạm thời để tải file zip của tập dữ liệu
DL = ROOT / "data" / "_dl"
# Thư mục đích chứa các ảnh sau khi chuyển đổi định dạng
IMG = ROOT / "data" / "raw" / "images"
# Thư mục đích chứa các file nhãn định dạng YOLO
LBL = ROOT / "data" / "raw" / "labels"
# Hằng số ký tự xuống dòng (Newline character)
NL = chr(10)


def visdrone2yolo(split: str, src: str) -> None:
    """
    Chuyển đổi dữ liệu chú thích (annotations) của VisDrone sang định dạng YOLO.
    VisDrone format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
    YOLO format: <class_id> <x_center> <y_center> <width> <height> (đã chuẩn hóa về khoảng [0, 1])
    """
    sd = DL / src       # Thư mục chứa dữ liệu đã giải nén
    im = IMG / split    # Thư mục lưu ảnh cho tập hiện tại (train/val/test)
    lb = LBL / split    # Thư mục lưu nhãn cho tập hiện tại (train/val/test)
    
    # Tạo thư mục nếu chưa tồn tại
    im.mkdir(parents=True, exist_ok=True)
    lb.mkdir(parents=True, exist_ok=True)
    
    # Di chuyển các file ảnh từ thư mục đã giải nén sang thư mục đích
    if (sd / "images").exists():
        for f in (sd / "images").glob("*.jpg"):
            f.rename(im / f.name)
            
    # Duyệt qua từng file chú thích txt với thanh tiến trình TQDM
    for f in TQDM((sd / "annotations").glob("*.txt"), desc=f"convert {split}"):
        # Đọc kích thước ảnh tương ứng để chuẩn hóa các bounding box
        w, h = Image.open(im / f.with_suffix(".jpg").name).size
        dw, dh = 1.0 / w, 1.0 / h # Hệ số chuyển đổi sang tọa độ chuẩn hóa [0, 1]
        out = []
        
        # Đọc từng dòng nhãn trong file txt và tách theo dấu phẩy
        for r in [x.split(",") for x in f.read_text(encoding="utf-8").strip().splitlines()]:
            # Cột thứ 5 (index 4) là score hoặc filter flag (ở đây "0" nghĩa là vùng bị bỏ qua/ignored regions)
            if r[4] != "0":
                x, y, bw, bh = map(int, r[:4]) # Lấy thông tin tọa độ góc trái trên, chiều rộng, chiều cao
                c = int(r[5]) - 1 # Chuyển class_category sang 0-indexed cho YOLO (trong VisDrone class chạy từ 1 đến 11)
                
                # Tính toán tọa độ tâm x, tâm y, chiều rộng, chiều cao và chuẩn hóa về khoảng [0, 1]
                x_center = (x + bw / 2) * dw
                y_center = (y + bh / 2) * dh
                norm_w = bw * dw
                norm_h = bh * dh
                
                out.append(f"{c} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}{NL}")
                
        # Ghi kết quả nhãn định dạng YOLO mới vào file nhãn tương ứng ở thư mục đích
        (lb / f.name).write_text("".join(out), encoding="utf-8")


def main() -> None:
    # Tạo thư mục tạm thời tải xuống
    DL.mkdir(parents=True, exist_ok=True)
    # Tạo danh sách các đường dẫn tải xuống VisDrone (train, val, test-dev) từ máy chủ lưu trữ ASSETS_URL của Ultralytics
    urls = [f"{ASSETS_URL}/VisDrone2019-DET-{s}.zip" for s in ("train", "val", "test-dev")]
    
    # Tải các file zip với 4 luồng song song (threads=4)
    download(urls, dir=DL, threads=4)
    
    # Tiến hành chuyển đổi từng tập dữ liệu tương ứng
    for folder, split in {
        "VisDrone2019-DET-train": "train",
        "VisDrone2019-DET-val": "val",
        "VisDrone2019-DET-test-dev": "test",
    }.items():
        visdrone2yolo(split, folder)
        
    # Xóa thư mục tạm tải xuống để giải phóng bộ nhớ
    shutil.rmtree(DL, ignore_errors=True)
    
    # In ra số lượng ảnh đã xử lý thành công
    for s in ("train", "val", "test"):
        print(f"[download] {s}: {len(list((IMG / s).glob('*.jpg')))} images")


if __name__ == "__main__":
    main()

