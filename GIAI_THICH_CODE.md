# 📖 GIẢI THÍCH CHI TIẾT TOÀN BỘ CODE DỰ ÁN RL-SAHI

## 📋 Mục Lục

1. [Tổng Quan Dự Án](#1-tổng-quan-dự-án)
2. [Cấu Trúc Thư Mục](#2-cấu-trúc-thư-mục)
3. [Luồng Hoạt Động Chính](#3-luồng-hoạt-động-chính)
4. [Module `common` — Tiện Ích Chung](#4-module-common--tiện-ích-chung)
5. [Module `detection` — Phát Hiện YOLO](#5-module-detection--phát-hiện-yolo)
6. [Module `hard_region` — Vùng Khó Phát Hiện](#6-module-hard_region--vùng-khó-phát-hiện)
7. [Module `inference` — Suy Luận Thích Ứng](#7-module-inference--suy-luận-thích-ứng)
8. [Module `rl` — Học Tăng Cường DQN](#8-module-rl--học-tăng-cường-dqn)
9. [Scripts — Các Lệnh Chạy Chính](#9-scripts--các-lệnh-chạy-chính)
10. [File Cấu Hình YAML](#10-file-cấu-hình-yaml)
11. [Thuật Ngữ Chính](#11-thuật-ngữ-chính)

---

## 1. Tổng Quan Dự Án

### Vấn Đề Cần Giải Quyết

Khi sử dụng mô hình phát hiện vật thể **YOLO** trên các bức ảnh có độ phân giải cao (ví dụ: ảnh chụp từ drone — VisDrone dataset), YOLO thường **bỏ sót các vật thể nhỏ** vì chúng chiếm diện tích quá nhỏ so với toàn bộ ảnh.

### Giải Pháp: RL-SAHI (Reinforcement Learning + Slicing Aided Hyper Inference)

Ý tưởng cốt lõi:

1. **Bước 1**: Chạy YOLO trên **toàn bộ ảnh gốc** để lấy kết quả phát hiện ban đầu (full-image detection).
2. **Bước 2**: Sử dụng một **tác tử học tăng cường (RL Agent — DQN)** để **tự động quyết định** vùng nào trên ảnh cần được **cắt lát (slice/crop)** và chạy YOLO lại với độ phân giải cao hơn.
3. **Bước 3**: **Gộp (merge)** kết quả phát hiện từ ảnh gốc và các lát cắt bằng thuật toán NMS (Non-Maximum Suppression) để ra kết quả cuối cùng.

### So Sánh Với SAHI Truyền Thống

| Đặc điểm | SAHI truyền thống | RL-SAHI (dự án này) |
|---|---|---|
| Cách chia lát cắt | Cố định theo lưới đều | Thông minh bằng AI (DQN) |
| Số lượng lát cắt | Nhiều (~28-29 crop/ảnh) | Ít hơn (~15 crop/ảnh) |
| Tốc độ | Chậm (T4: ~760ms/ảnh) | ⚠️ **Hiện CHẬM hơn** (T4: ~1636ms/ảnh — dù ít crop hơn, tốn thêm rollout ~22% + crop RL rơi vào vùng đông vật nên mỗi crop YOLO xử lý lâu hơn) |
| Chất lượng (đo T4, test-100) | recall 0.620 / mAP 0.254 / FP 202 | ✅ **recall 0.691 / mAP 0.302 / FP 179 — thắng cả 3** |

> ⚠️ **Đính chính theo số đo thật (2026-07-05, Tesla T4):** điểm mạnh của RL-SAHI là **RECALL** (đúng mục tiêu đồ án), KHÔNG phải tốc độ. Profile cho thấy 75% thời gian là chạy-lại-YOLO-trên-crop (cả SAHI lẫn RL đều dính), 22% là RL rollout. Xem chi tiết cuối file.

### Kiến Trúc Tổng Thể

```
┌──────────────────────────────────────────────────────────┐
│                    ẢNH ĐẦU VÀO                          │
│                 (ảnh độ phân giải cao)                    │
└─────────────────────┬────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────┐
│            BƯỚC 1: YOLO Full-Image Detection             │
│  • Chạy YOLO trên toàn bộ ảnh → boxes, scores, classes  │
│  • Trích xuất backbone features + objectness map         │
│  • Lưu cache (.npz) để tái sử dụng                      │
└─────────────────────┬────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────┐
│            BƯỚC 2: DQN Agent Chọn Vùng Cắt               │
│  • Xây dựng State Vector từ detections + features        │
│  • DQN dự đoán Q-values cho 11 hành động                 │
│    (trái/phải/trên/dưới/zoom_in/zoom_out/stop/chéo)     │
│  • Agent di chuyển ROI đến vùng tối ưu → chọn STOP      │
└─────────────────────┬────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────┐
│         BƯỚC 3: YOLO Chạy Trên Vùng Cắt (Crop)          │
│  • Cắt ảnh theo ROI được DQN chọn                        │
│  • Chạy YOLO trên crop → phát hiện thêm vật thể nhỏ    │
│  • Lặp lại BƯỚC 2-3 cho đến khi hết budget              │
└─────────────────────┬────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────┐
│           BƯỚC 4: Gộp Kết Quả (Merge + NMS)             │
│  • Nối tất cả boxes từ ảnh gốc + các lát cắt           │
│  • Class-aware NMS để loại bỏ trùng lặp                  │
│  • Xuất kết quả cuối cùng                                │
└──────────────────────────────────────────────────────────┘
```

---

## 2. Cấu Trúc Thư Mục

```
Test/
├── configs/                    # Các file cấu hình YAML
│   ├── default.yaml            # Cấu hình gốc mặc định
│   ├── rl.yaml                 # Cấu hình RL (môi trường, huấn luyện)
│   ├── detection.yaml          # Cấu hình YOLO detection
│   ├── inference.yaml          # Cấu hình suy luận SAHI
│   ├── ft.yaml, ft_rl.yaml    # Fine-tuning configs
│   └── ...                     # Các config sweep/experiment
│
├── scripts/                    # Scripts chạy chính
│   ├── detect.py               # Cache YOLO detections cho toàn bộ ảnh
│   ├── hard_region.py          # Cache vùng khó phát hiện
│   ├── train.py                # Huấn luyện DQN Agent
│   ├── benchmark.py            # Đánh giá so sánh các phương pháp
│   ├── infer.py                # Suy luận RL-SAHI trên ảnh mới
│   └── ...
│
├── src/rl_sahi/                # Mã nguồn chính (package Python)
│   ├── __init__.py             # Khai báo package
│   ├── common/                 # Module tiện ích chung
│   │   ├── actions.py          # Định nghĩa không gian hành động
│   │   ├── box_types.py        # Hằng số và chuẩn hóa boxes
│   │   ├── box_geometry.py     # Hàm hình học (IoU, diện tích, tâm)
│   │   ├── box_transforms.py   # Biến đổi tọa độ (clip, zoom, translate)
│   │   ├── boxes.py            # Re-export tất cả hàm box
│   │   ├── cache.py            # Lưu/tải cache phát hiện (.npz)
│   │   ├── class_mapping.py    # Ánh xạ nhãn lớp đối tượng
│   │   ├── config.py           # Tải/gộp cấu hình YAML
│   │   ├── data.py             # Đọc ảnh, nhãn YOLO
│   │   ├── device.py           # Quản lý thiết bị (CUDA/DirectML/CPU)
│   │   ├── nms.py              # Non-Maximum Suppression
│   │   └── raster.py           # Rasterize boxes → lưới 2D
│   │
│   ├── detection/              # Module phát hiện YOLO
│   │   ├── yolo.py             # Tải YOLO và suy luận 1 ảnh
│   │   ├── features.py         # Thu thập đặc trưng backbone
│   │   ├── cache_builder.py    # Xây dựng cache YOLO cho toàn split
│   │   └── yolo_cache.py       # Re-export
│   │
│   ├── hard_region/            # Module phân tích vùng khó
│   │   ├── regions.py          # Logic lọc vùng khó (hard boxes)
│   │   ├── cache_builder.py    # Cache vùng khó cho toàn split
│   │   └── builder.py          # Re-export
│   │
│   ├── inference/              # Module suy luận thích ứng
│   │   ├── config.py           # Cấu hình InferenceConfig
│   │   ├── crops.py            # Cắt ROI và chạy YOLO trên crop
│   │   ├── merge.py            # Gộp kết quả + Class-aware NMS
│   │   ├── pipeline.py         # Pipeline suy luận chính
│   │   ├── rollout.py          # Rollout DQN cho 1 lát cắt
│   │   ├── visualize.py        # Vẽ kết quả lên ảnh
│   │   └── runner.py           # Re-export
│   │
│   ├── eval/                   # Module đánh giá hiệu năng
│   │   └── benchmark.py        # Tính mAP50, small_recall, so sánh
│   │
│   └── rl/                     # Module học tăng cường DQN
│       ├── env_config.py       # Cấu hình môi trường (EnvConfig)
│       ├── slice_env.py        # Môi trường SliceEnv (core)
│       ├── network.py          # Kiến trúc mạng Q (Dueling DQN)
│       ├── replay.py           # Bộ nhớ trải nghiệm (Replay Buffer)
│       ├── trainer.py          # Huấn luyện DQN (single)
│       ├── batched_trainer.py  # Huấn luyện DQN (batched)
│       ├── checkpoint.py       # Lưu/tải checkpoint
│       ├── crop_outcome.py     # Đánh giá kết quả crop
│       ├── dataset.py          # Quản lý mẫu huấn luyện
│       ├── state_config.py     # Cấu hình không gian trạng thái
│       ├── state_layout.py     # Bố cục vector trạng thái
│       ├── state_maps.py       # Xây dựng bản đồ 2D
│       ├── state_summary.py    # Vector tóm tắt 28 chiều
│       ├── state_vector.py     # Hợp nhất thành vector đầu vào
│       ├── tile_coding.py      # Mã hóa tile (tile coding)
│       ├── tile_infer.py       # Suy luận tile
│       └── tile_state.py       # Trạng thái tile
│
├── tests/                      # Unit tests
├── data/                       # Dữ liệu (VisDrone)
├── runs/                       # Kết quả chạy
├── best_visdrone.pt            # Trọng số YOLO đã huấn luyện
├── yolo11s.pt                  # Trọng số YOLO baseline
└── requirements.txt            # Các thư viện phụ thuộc
```

---

## 3. Luồng Hoạt Động Chính

### 3.1 Luồng Huấn Luyện (Training Pipeline)

```
1. detect.py       → Cache YOLO boxes + features cho toàn bộ ảnh
2. hard_region.py   → Phân tích và cache vùng khó phát hiện
3. train.py         → Huấn luyện DQN Agent sử dụng cached data
     ↓
   Trong mỗi episode:
   a. Chọn ngẫu nhiên 1 ảnh từ tập huấn luyện
   b. Tạo môi trường SliceEnv từ cached detection + hard_region
   c. Agent tương tác với SliceEnv:
      - Quan sát trạng thái (state)
      - Chọn hành động (ε-greedy)
      - Nhận phần thưởng (reward) 
      - Lưu transition vào Replay Buffer
   d. Tối ưu mạng Q bằng loss TD (Temporal Difference)
   e. Cập nhật Target Network định kỳ
   f. Đánh giá benchmark định kỳ
```

### 3.2 Luồng Suy Luận (Inference Pipeline)

```
1. infer.py / pipeline.py
     ↓
   a. Chạy YOLO trên ảnh gốc → full detections
   b. VÒNG LẶP:
      i.   Xây dựng SliceEnv từ detection hiện tại
      ii.  Rollout DQN → chọn ROI (vùng cắt)
      iii. Kiểm tra điều kiện dừng/trùng lặp
      iv.  Chạy YOLO trên crop → crop detections
      v.   Đo lường new_detection_gain
      vi.  Accept/Reject lát cắt
   c. Gộp tất cả detections + Class-aware NMS
   d. Lưu kết quả (.txt, .jpg, .json)
```

---

## 4. Module `common` — Tiện Ích Chung

### 4.1 `actions.py` — Không Gian Hành Động

Định nghĩa **11 hành động** mà DQN Agent có thể thực hiện:

| Hành động | Giá trị | Ý nghĩa |
|---|---|---|
| LEFT | 0 | Dịch ROI sang trái |
| RIGHT | 1 | Dịch ROI sang phải |
| UP | 2 | Dịch ROI lên trên |
| DOWN | 3 | Dịch ROI xuống dưới |
| ZOOM_IN | 4 | Thu nhỏ ROI (phóng to chi tiết) |
| ZOOM_OUT | 5 | Phóng to ROI (nhìn rộng hơn) |
| STOP | 6 | Dừng lại — xác nhận ROI hiện tại |
| UP_LEFT | 7 | Dịch chéo trên-trái |
| UP_RIGHT | 8 | Dịch chéo trên-phải |
| DOWN_LEFT | 9 | Dịch chéo dưới-trái |
| DOWN_RIGHT | 10 | Dịch chéo dưới-phải |

### 4.2 `box_types.py` — Hằng Số Và Chuẩn Hóa

- **`EPS = 1e-9`**: Hằng số vô cùng nhỏ, dùng trong mẫu số phép chia để tránh lỗi chia cho 0.
- **`as_boxes()`**: Chuẩn hóa bất kỳ input nào (list, array) về dạng `numpy array (N, 4)` kiểu `float32`.

### 4.3 `box_geometry.py` — Hàm Hình Học Bounding Box

| Hàm | Chức năng |
|---|---|
| `area(boxes)` | Tính diện tích = `(x2 - x1) × (y2 - y1)` |
| `intersection_matrix(a, b)` | Tính ma trận diện tích giao nhau giữa mọi cặp hộp |
| `iou_matrix(a, b)` | Tính **IoU** = `Intersection / Union` (tỷ lệ giao/hợp) |
| `ioa_matrix(a, b)` | Tính **IoA** = `Intersection / Area(b)` (giao/diện tích b) |
| `centers(boxes)` | Tính tọa độ tâm `[cx, cy]` của mỗi hộp |
| `center_inside(roi, boxes)` | Kiểm tra tâm hộp có nằm trong ROI không |
| `normalized_box(box, image_shape)` | Chuẩn hóa tọa độ hộp về `[0, 1]` |

### 4.4 `box_transforms.py` — Biến Đổi Tọa Độ

| Hàm | Chức năng |
|---|---|
| `xywhn_to_xyxy()` | Chuyển `[cx, cy, w, h]` chuẩn hóa → `[x1, y1, x2, y2]` pixel |
| `xyxy_to_xywhn()` | Ngược lại |
| `clip_boxes()` | Giới hạn hộp trong biên ảnh |
| `box_from_center()` | Tạo hộp vuông từ tâm + cạnh |
| `translate_box()` | Dịch chuyển hộp `(dx, dy)` |
| `zoom_box()` | Thu/phóng hộp quanh tâm |

### 4.5 `nms.py` — Non-Maximum Suppression

Thuật toán **NMS** loại bỏ các hộp trùng lặp:

```
1. Sắp xếp hộp theo score giảm dần
2. Lấy hộp có score cao nhất → giữ lại
3. Tính IoU của hộp đó với tất cả hộp còn lại
4. Loại bỏ các hộp có IoU > ngưỡng (bị coi là trùng)
5. Lặp lại bước 2-4 cho đến hết
```

### 4.6 `raster.py` — Rasterize Bounding Boxes

Chuyển đổi danh sách bounding boxes thành **bản đồ lưới 2D** (ví dụ: `16×16`). Mỗi ô lưới chứa giá trị đại diện (score, count) của các vật thể nằm trong ô đó. Rất hữu ích để xây dựng **state maps** cho DQN.

### 4.7 `cache.py` — Hệ Thống Cache

Quản lý lưu/tải dữ liệu phát hiện YOLO vào file `.npz` (nén NumPy):

- **`DetectionCache`**: Chứa boxes, scores, classes, backbone features, objectness_map, spatial_feature_map.
- **`HardRegionCache`**: Chứa thông tin vùng khó (hard_boxes, GT nhỏ, IoU khớp).
- Hệ thống **versioning + metadata fingerprinting** để phát hiện khi nào cache bị lỗi thời.

### 4.8 `config.py` — Quản Lý Cấu Hình YAML

- Hỗ trợ **include/kế thừa** file config (ví dụ: `ft_rl.yaml` kế thừa `ft.yaml` + `rl.yaml`).
- **Deep merge**: Gộp đệ quy các config lồng nhau.
- Tự động chuyển đường dẫn tương đối thành tuyệt đối.
- Khởi tạo dataclass trực tiếp từ section YAML.

### 4.9 `data.py` — Xử Lý Dữ Liệu

| Hàm | Chức năng |
|---|---|
| `iter_images()` | Quét thư mục tìm tất cả file ảnh (.jpg, .png...) |
| `image_id()` | Lấy tên file (không phần mở rộng) làm ID |
| `read_image()` | Đọc ảnh bằng OpenCV (BGR) |
| `read_yolo_labels()` | Đọc file nhãn YOLO (.txt) → boxes (xyxy) + classes |

### 4.10 `device.py` — Quản Lý Thiết Bị Tính Toán

Tự động phát hiện và cấu hình thiết bị tốt nhất: **CUDA GPU → DirectML (AMD/iGPU) → CPU**.

Bao gồm:
- Bật **cuDNN benchmark** + **TF32** cho GPU NVIDIA.
- **Monkey-patching** thư viện Ultralytics để tương thích DirectML (chuyển tensor về CPU trước NMS).

### 4.11 `class_mapping.py` — Ánh Xạ Nhãn Lớp

Xử lý trường hợp ID nhãn đầu ra YOLO khác với ID nhãn ground truth khác với ID dùng tính mAP. Ví dụ: YOLO class 0 → label class 10 → eval class 0.

---

## 5. Module `detection` — Phát Hiện YOLO

### 5.1 `yolo.py` — Tải Và Chạy YOLO

- **`load_yolo(weights, device)`**: Tải mô hình YOLO11 từ file `.pt`, chuyển sang GPU, áp dụng patches DirectML.
- **`detect_one_image()`**: Chạy YOLO predict trên 1 ảnh:
  1. Đăng ký **forward hooks** vào backbone layers để thu đặc trưng.
  2. Chạy predict → lấy boxes, scores, classes.
  3. Rút trích **objectness_map** (bản đồ nhiệt vật thể) và **spatial_feature_map** từ detection head.
  4. Trả về `DetectionCache` hoàn chỉnh.

### 5.2 `features.py` — Thu Thập Đặc Trưng

Hai Context Manager quan trọng:

- **`FeatureCollector`**: Đăng ký hook vào các backbone layers (ví dụ: layer 10) để thu vector đặc trưng trung gian. Mỗi tensor được tóm tắt bằng `mean + std` trên chiều spatial.
  
- **`DetectAuxCollector`**: Đăng ký hook vào detection head (layer cuối) để:
  - Tạo **objectness map**: Lấy `sigmoid(max_class_logit)` trên mỗi cell → resize về lưới `16×16`.
  - Tạo **spatial feature map**: Nén channels bằng chunk-averaging → chuẩn hóa Z-score.

### 5.3 `cache_builder.py` — Xây Dựng Cache Hàng Loạt

Duyệt qua toàn bộ ảnh trong split, chạy YOLO và lưu cache. **Bỏ qua** ảnh đã có cache hợp lệ (kiểm tra version + metadata).

---

## 6. Module `hard_region` — Vùng Khó Phát Hiện

### Mục Đích

Xác định **vùng khó (hard regions)** = các vật thể nhỏ mà YOLO ảnh gốc **không phát hiện được** hoặc phát hiện với **confidence rất thấp**. Dùng làm **tín hiệu thưởng (reward signal)** cho DQN.

### 6.1 `regions.py` — Logic Lọc Vùng Khó

Quy trình cho 1 ảnh:
1. Đọc nhãn ground truth (GT).
2. Lọc GT chỉ giữ **vật thể nhỏ** (`area/image_area ≤ small_area_ratio`).
3. Tính IoU giữa GT nhỏ và YOLO detections (ràng buộc cùng class).
4. Đánh dấu GT nhỏ là **"vùng khó"** nếu:
   - IoU khớp tốt nhất < `match_iou` (YOLO không phát hiện đè lên), HOẶC
   - Score của detection khớp < `min_detect_score` (YOLO thiếu tự tin).

### 6.2 `cache_builder.py` — Cache Vùng Khó

Yêu cầu detection cache phải tồn tại trước. Hỗ trợ **dynamic threshold** bằng percentile (ví dụ: vật thể nhỏ = bottom 40% diện tích).

---

## 7. Module `inference` — Suy Luận Thích Ứng

### 7.1 `config.py` — InferenceConfig

Các tham số suy luận quan trọng:

| Tham số | Default (dataclass) | **Đang chạy** (inference.yaml + ft.yaml) | Ý nghĩa |
|---|---|---|---|
| `full_imgsz` | 640 | 640 | Kích thước ảnh cho YOLO lần đầu |
| `slice_imgsz` | 640 | 640 | Kích thước crop cho YOLO lần sau |
| `full_conf` | 0.01 | 0.01 | Ngưỡng confidence khi chạy trên ảnh gốc (thấp để giữ nhiều hộp) |
| `output_conf` | 0.3 | **0.10** ★ | Ngưỡng confidence đầu ra — ĐÒN BẨY recall↔FP chính |
| `merge_iou` | 0.5 | 0.5 | Ngưỡng IoU cho NMS merge |
| `min_slice_detections` | 1 | 1 | Số phát hiện mới tối thiểu để chấp nhận lát cắt |
| `min_slice_utility` | 0.5 | **0.2** ★ | Tiện ích tối thiểu (hạ để nhận nhiều lát hơn → recall) |
| `max_slice_attempts` | 0 | **24** | Số lần thử lát tối đa |
| `target_classes` | (0,2,3,5,8,9) — nhánh frozen CŨ | **(0..9) đủ 10 lớp** — nhánh fine-tune ft.yaml | Các class mục tiêu (VisDrone) |

> ⚠️ Cột "Mặc định" là default trong dataclass — **giá trị thật đang chạy nằm ở YAML** (cột đậm). Đọc default dễ hiểu nhầm hệ thống.

### 7.2 `crops.py` — Cắt Và Chạy YOLO Trên Crop

- **`crop_roi()`**: Cắt vùng ROI từ ảnh gốc, trả về ảnh crop + offset `(x1, y1)`.
- **`run_yolo_on_crops()`**: Chạy YOLO batch trên nhiều crops, tự động **cộng offset** để đưa tọa độ về ảnh gốc.

### 7.3 `merge.py` — Gộp Kết Quả

- **`class_aware_nms()`**: NMS riêng biệt theo từng class, tránh lọc nhầm cross-class.
- **`merge_predictions()`**: Nối + clip + NMS tất cả kết quả.
- **`new_detection_gain_after_merge()`**: Đếm số hộp **mới** mà crop mang lại (không trùng với kết quả trước).
- **`new_detection_utility_after_merge()`**: Tổng score của các hộp mới.

### 7.4 `pipeline.py` — Pipeline Suy Luận Chính

Lớp **`AdaptiveSahiInferencer`**: Bộ suy luận thích ứng hoàn chỉnh.

Quy trình `_infer_with_loaded()`:
1. Lọc full detections bằng `output_conf`.
2. **Vòng lặp chính** (tối đa `max_attempts` lần):
   - Tạo `SliceEnv` với lịch sử ROI đã quét.
   - DQN rollout → chọn ROI.
   - Kiểm tra **điều kiện từ chối**:
     - ROI trùng lặp với lát cắt cũ (`old_overlap`)
     - ROI trùng lặp với lát cắt đã thử (`attempted_overlap`)
     - Hết bước mà không chọn STOP (`max_steps`)
     - ROI bị kẹt (stalled)
   - Nếu hợp lệ → chạy YOLO trên crop.
   - Đo lường `new_detection_gain` và `new_detection_utility`.
   - **Accept** nếu gain ≥ min_slice_detections VÀ utility ≥ min_slice_utility.
3. Gộp tất cả → NMS → lưu kết quả.

### 7.5 `rollout.py` — DQN Rollout

Chạy 1 episode trên `SliceEnv`:
1. Reset môi trường → lấy state ban đầu.
2. Vòng lặp:
   - DQN dự đoán Q-values.
   - Mask hành động không hợp lệ → chọn argmax.
   - Thực hiện hành động → nhận state mới + done.
3. Trả về ROI cuối cùng + danh sách hành động.

### 7.6 `visualize.py` — Trực Quan Hóa

Vẽ kết quả lên ảnh:
- **Xanh lá**: Hộp từ ảnh gốc.
- **Cam**: Hộp mới từ lát cắt.
- **Đỏ**: Viền lát cắt được chấp nhận.
- **Cam/vàng**: Viền lát cắt bị từ chối.

---

## 8. Module `rl` — Học Tăng Cường DQN

### 8.1 `env_config.py` — Cấu Hình Môi Trường

Lớp **`EnvConfig`** chứa ~40 siêu tham số điều khiển hành vi môi trường:

**Nhóm giới hạn:**
- `max_steps = 20` (default) — **checkpoint đang dùng đã train với `max_steps = 10`** (override qua local.yaml).
- `max_slices = 8` (default) — **rl.yaml + checkpoint đang dùng = 14** (tăng cho recall-max).

**Nhóm ROI:**
- `initial_slice_fraction = 0.28`: Tỷ lệ cạnh ROI ban đầu so với ảnh.
- `move_fraction = 0.30`: Khoảng dịch chuyển mỗi bước.
- `zoom_factor = 0.75`: Hệ số zoom mỗi bước.

**Nhóm thưởng/phạt (reward function):**
- `step_penalty = 0.03`: Phạt mỗi bước đi (khuyến khích hiệu quả).
- `empty_slice_penalty = 0.35`: Phạt khi lát cắt rỗng.
- `area_penalty = 0.35`: Phạt ROI quá lớn.
- `new_hard_reward = 0.5`: Thưởng phát hiện vùng khó mới.
- `stop_target_reward = 0.4`: Thưởng dừng đúng lúc.
- `max_steps_without_stop_penalty = 4.0`: Phạt nặng khi hết bước mà không stop.

**`StepResult`**: Kết quả mỗi bước = (state, reward, done, info).

### 8.2 `state_config.py` — Cấu Hình Trạng Thái

Cấu trúc vector trạng thái gồm 3 phần:

```
State Vector = [Feature Vector] + [Spatial Maps] + [Summary Vector]
                  (backbone)       (lưới 2D)         (28 chiều)
```

- **`SUMMARY_DIM = 28`**: Vector tóm tắt 28 chiều.
- **`DETECTION_MAP_CHANNELS = 4`**: 4 kênh bản đồ phát hiện.
- **`BASE_MAP_CHANNELS = 9`** (không phải 10): 1 (history) + 1 (current ROI) + 1 (attempted) + 1 (accepted) + 4 (detection) + 1 (objectness) = **9**.
- Tổng kênh map = 9 base + số kênh spatial_feature_map **lấy từ cache thực tế** (không phải config!). **Checkpoint đang dùng: 12 kênh spatial → 21 kênh map → state_dim = 256 (feature L16) + 21×16×16 (5376) + 28 = 5660.**

### 8.3 `state_maps.py` — Bản Đồ Không Gian 2D

4 kênh bản đồ phát hiện (`build_detection_map`):

| Kênh | Tên | Nội dung |
|---|---|---|
| 0 | High confidence | Vật thể có score ≥ 0.5 (đã phát hiện tốt) |
| 1 | Proposal quality | Vật thể trong khoảng [0.01, 0.5] (khả nghi, cần kiểm tra) |
| 2 | Proposal density | Mật độ vật thể nghi ngờ (cộng dồn) |
| 3 | Small objects | Vật thể có diện tích nhỏ |

**`mark_history()`**: Cập nhật bản đồ lịch sử khi agent quét qua một vùng.

### 8.4 `state_summary.py` — Vector Tóm Tắt 28 Chiều

28 giá trị thống kê mô tả trạng thái hiện tại:

| Chỉ số | Nội dung |
|---|---|
| 0 | Số lượng vật thể / norm |
| 1 | Score trung bình toàn ảnh |
| 2 | Score lớn nhất toàn ảnh |
| 3 | Số vật thể score thấp |
| 4 | Số vật thể diện tích nhỏ |
| 5 | Diện tích trung bình |
| 6-11 | Thống kê vật thể **trong ROI** (số lượng, score, diện tích) |
| 12-15 | Tọa độ ROI chuẩn hóa (cx, cy, w, h) |
| 16 | Tỷ lệ diện tích ROI / ảnh |
| 17 | Tiến trình bước (step/max_steps) |
| 18 | Trung bình bản đồ lịch sử |
| 19 | Tỷ lệ khung hình ảnh |
| 20 | Trung bình bản đồ lát cắt trước |
| 21 | Overlap với lát cắt cũ |
| 22 | Scale gain (tỷ lệ phóng đại) |
| 23 | Số lát cắt đã hoàn thành |
| 24-27 | Thống kê bổ sung (proposal count, quality) |

### 8.5 `state_vector.py` — Vector Trạng Thái Hoàn Chỉnh

Hàm `build_state_vector()` nối tất cả thành phần theo thứ tự:

```
[normalized_feature]           ← Backbone feature (mean+std của các channels)
[history_map]                  ← Lưới 16×16: vùng đã quét
[current_roi_map]              ← Lưới 16×16: vị trí ROI hiện tại
[attempted_slice_map]          ← Lưới 16×16: tất cả ROI đã thử
[accepted_slice_map]           ← Lưới 16×16: ROI được chấp nhận
[detection_map × 4 kênh]      ← 4 lưới 16×16 về detection
[objectness_map]               ← Lưới 16×16: nhiệt vật thể YOLO
[spatial_feature_map × N kênh] ← N lưới 16×16 đặc trưng không gian
[summary × 28]                 ← Vector tóm tắt 28 chiều
```

### 8.6 `network.py` — Kiến Trúc Mạng DQN

Lớp **`QNetwork`** hỗ trợ 2 chế độ:

**Chế độ MLP (mặc định):**
```
Input (state_dim) → Linear(hidden) → ReLU → Linear(hidden/2) → ReLU
                                                      ↓
                                            ┌─────────┴─────────┐
                                            │  Dueling DQN       │
                                            │  Value Head → V(s) │
                                            │  Advantage → A(s,a)│
                                            │  Q = V + A - mean(A)│
                                            └────────────────────┘
```

**Chế độ Spatial CNN:**
```
Feature + Summary → Linear → ReLU ─────────────────────┐
                                                        │ concat
Maps (C × 16 × 16) → Conv3×3 → ReLU → Conv3×3 → ReLU ─┘
                      → AdaptiveAvgPool(4×4) → Flatten
                                                        ↓
                                              Linear(trunk) → ReLU
                                                        ↓
                                              Dueling Heads (V, A)
```

### 8.7 `replay.py` — Bộ Nhớ Trải Nghiệm

Hai loại buffer:

- **`ReplayBuffer`**: Lấy mẫu ngẫu nhiên đều (uniform). Dùng deque vòng tròn.
- **`PrioritizedReplayBuffer`**: Lấy mẫu có ưu tiên (PER):
  - Xác suất chọn ∝ `priority^α` (α=0.6).
  - **Importance Sampling weights** bù trừ bias: `w = (N × p)^(-β)`.
  - β tăng dần từ 0.4 → 1.0 theo thời gian huấn luyện.
  - Ưu tiên cập nhật bằng **TD error** (lỗi chênh lệch thời gian).

### 8.8 `checkpoint.py` — Lưu/Tải Checkpoint

**`save_checkpoint()`** lưu:
- Trọng số mạng Q (state_dict).
- Kích thước state_dim.
- Loại mạng (mlp/spatial_cnn) + dueling.
- Toàn bộ cấu hình (TrainConfig, EnvConfig, StateConfig).
- Detection metadata (để kiểm tra tính nhất quán khi inference).

**`load_policy()`** khôi phục:
- Tái tạo kiến trúc QNetwork.
- Nạp trọng số.
- Kiểm tra action space khớp.
- Chuyển sang chế độ eval.

### 8.9 `dataset.py` — Quản Lý Mẫu Huấn Luyện

Lớp **`CachedEpisodeDataset`**:
- Quét ảnh trong split.
- Kiểm tra detection cache + hard_region cache tồn tại và hợp lệ.
- `random_episode()`: Chọn ngẫu nhiên 1 ảnh → tải caches.
- `first_detection()`: Lấy detection đầu tiên để xác định kích thước state.

### 8.10 `crop_outcome.py` — Đánh Giá Kết Quả Crop

Lớp **`CropOutcomeEvaluator`** đánh giá chất lượng crop trong huấn luyện:
- Chạy YOLO trên crop (có cache).
- Tính `new_detection_gain` (số box mới).
- Tính `tp_gain` / `fp_gain` so với ground truth.
- Tính **reward** = `detection_reward × utility + tp_reward × TP - fp_penalty × FP`.

### 8.11 `slice_env.py` — Môi Trường SliceEnv (Core)

File lớn nhất (~1300 dòng), là **trái tim** của hệ thống RL.

**`SliceEnv`** mô phỏng quá trình chọn vùng cắt:

**`reset()`:**
1. Khởi tạo ROI tại tâm ảnh (hoặc vùng có nhiều proposal nhất).
2. Xây dựng bản đồ detection, objectness, spatial.
3. Tính toán state vector ban đầu.

**`step(action)`:**
1. Thực hiện hành động (translate/zoom/stop).
2. Kiểm tra **ràng buộc** (ROI vượt biên, overlap quá cao...).
3. Tính **reward** phức tạp gồm nhiều thành phần:
   - **Hard coverage reward**: Thưởng khi ROI phủ vùng khó.
   - **Efficiency penalty**: Phạt ROI quá lớn.
   - **Step penalty**: Phạt mỗi bước đi.
   - **Overlap penalty**: Phạt trùng lặp với lát cắt cũ.
   - **Stop bonus/penalty**: Thưởng/phạt dừng đúng/sai lúc.
4. Cập nhật state maps.
5. Trả về `StepResult(state, reward, done, info)`.

### 8.12 `trainer.py` và `batched_trainer.py` — Huấn Luyện DQN

**Vòng lặp huấn luyện chính:**

```python
for episode in range(total_episodes):
    # 1. Chọn ngẫu nhiên 1 ảnh
    detection, hard_region = dataset.random_episode()
    
    # 2. Tạo môi trường
    env = SliceEnv(detection, hard_region, env_cfg, state_cfg)
    state = env.reset()
    
    # 3. Tương tác với môi trường
    for step in range(max_steps):
        action = select_action(policy, state, epsilon)  # ε-greedy
        result = env.step(action)
        replay_buffer.push(state, action, result.reward, result.state, result.done)
        
        # 4. Tối ưu hóa mạng Q
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            loss = optimize(policy, target_net, batch, gamma)
        
        state = result.state
        if result.done:
            break
    
    # 5. Cập nhật Target Network
    if episode % target_update_freq == 0:
        target_net.load_state_dict(policy.state_dict())
    
    # 6. Giảm epsilon (ε-decay)
    epsilon = max(epsilon_end, epsilon * decay_factor)
```

**`batched_train_dqn`** bổ sung:
- Huấn luyện theo **batch episodes** (nhiều ảnh song song).
- Đánh giá benchmark định kỳ.
- Lưu checkpoint tốt nhất.
- Hỗ trợ resume từ checkpoint cũ.

---

## 9. Scripts — Các Lệnh Chạy Chính

### 9.1 `detect.py` — Cache YOLO Detections

```bash
python scripts/detect.py --config configs/ft_rl.yaml --split train
```

Chạy YOLO trên toàn bộ ảnh → lưu `.npz` cache (boxes + features + objectness map).

### 9.2 `hard_region.py` — Cache Vùng Khó

```bash
python scripts/hard_region.py --config configs/ft_rl.yaml --split train
```

So sánh YOLO detections vs ground truth → tìm vật thể nhỏ bị bỏ sót.

### 9.3 `train.py` — Huấn Luyện DQN

```bash
python scripts/train.py --config configs/ft_rl.yaml --split train --device cuda
```

Huấn luyện DQN Agent sử dụng cached data. Kết quả: checkpoint `.pt`.

### 9.4 `benchmark.py` — Đánh Giá So Sánh

```bash
python scripts/benchmark.py --config configs/ft_rl.yaml --split val
```

So sánh 4 phương pháp:
1. **YOLO Full**: Chỉ chạy YOLO ảnh gốc.
2. **Fixed Grid SAHI**: SAHI truyền thống (lưới cố định).
3. **Objectness Top-K**: Baseline heuristic (chọn đỉnh objectness).
4. **RL-SAHI**: Phương pháp đề xuất.

Xuất: `benchmark.csv` + `benchmark.json`.

### 9.5 `infer.py` — Suy Luận Trên Ảnh Mới

```bash
python scripts/infer.py --config configs/ft_rl.yaml --image test.jpg
```

Chạy RL-SAHI trên ảnh → xuất: `detections/*.txt`, `visualizations/*.jpg`, `metadata/*.json`.

---

## 10. File Cấu Hình YAML

### Cấu trúc cấu hình kế thừa (include)

```yaml
# ft_rl.yaml
include:
  - ft.yaml        # Kế thừa từ fine-tuning config
  - rl.yaml        # Kế thừa từ RL config

# Override cụ thể
train:
  episodes: 5000
  hidden_dim: 512
```

### Các section chính

| Section | Mô tả |
|---|---|
| `paths` | Đường dẫn weights, data, cache, output |
| `detect` | Cấu hình YOLO detection (imgsz, conf, iou) |
| `hard_region` | Cấu hình lọc vùng khó |
| `env` | Cấu hình môi trường RL |
| `state` | Cấu hình state vector |
| `train` | Cấu hình huấn luyện DQN |
| `infer` | Cấu hình suy luận |
| `benchmark` | Cấu hình đánh giá |
| `classes` | Ánh xạ nhãn lớp |

---

## 11. Thuật Ngữ Chính

| Thuật ngữ | Giải nghĩa |
|---|---|
| **YOLO** | You Only Look Once — mô hình phát hiện vật thể thời gian thực |
| **SAHI** | Slicing Aided Hyper Inference — suy luận trên lát cắt để phát hiện vật thể nhỏ |
| **DQN** | Deep Q-Network — thuật toán học tăng cường sử dụng mạng nơ-ron để ước lượng Q-value |
| **Dueling DQN** | Biến thể DQN tách riêng Value V(s) và Advantage A(s,a) |
| **IoU** | Intersection over Union — tỷ lệ diện tích giao / diện tích hợp |
| **NMS** | Non-Maximum Suppression — loại bỏ hộp trùng lặp |
| **ROI** | Region of Interest — vùng quan tâm (vùng cắt lát) |
| **Bounding Box** | Hộp giới hạn bao quanh vật thể |
| **Backbone** | Phần trích xuất đặc trưng của mạng nơ-ron |
| **Forward Hook** | Cơ chế PyTorch bắt output trung gian khi chạy forward pass |
| **Objectness Map** | Bản đồ nhiệt chỉ xác suất có vật thể tại mỗi vị trí |
| **Replay Buffer** | Bộ nhớ lưu trải nghiệm (state, action, reward) để huấn luyện offline |
| **PER** | Prioritized Experience Replay — replay có ưu tiên theo TD error |
| **ε-greedy** | Chiến lược khám phá: xác suất ε chọn ngẫu nhiên, 1-ε chọn tốt nhất |
| **Target Network** | Bản sao mạng Q dùng tính target, cập nhật chậm để ổn định |
| **TD Error** | Temporal Difference Error — sai số giữa Q dự đoán và Q mục tiêu |
| **mAP50** | Mean Average Precision tại IoU 0.5 — chỉ số đánh giá chính |
| **Small Recall** | Tỷ lệ phát hiện thành công vật thể nhỏ |
| **FP** | False Positive — phát hiện sai (hộp dự đoán không khớp GT) |
| **TP** | True Positive — phát hiện đúng (hộp dự đoán khớp GT) |
| **Ground Truth (GT)** | Nhãn đúng do con người gán |
| **VisDrone** | Dataset ảnh chụp từ drone chứa nhiều vật thể nhỏ |

---

---

## 12. PHỤ LỤC — Số Đo Thật & Đính Chính (cập nhật 2026-07-06)

File gốc do Antigravity sinh tự động từ code; phần này đối chiếu với **checkpoint thật + số đo thật** (những chỗ trên đã sửa inline được đánh ★/⚠️).

### 12.1 Checkpoint đang dùng (`runs/ft_rl/dqn/best.pt`) — nguồn chân lý
| Giá trị | Trong checkpoint | Ghi chú |
|---|---|---|
| `state_dim` | **5660** | = feature 256 + 21 kênh map × 256 + summary 28 |
| kênh map | **21** | 9 base + **12 spatial** (từ cache thực tế, không phải config 4) |
| `max_steps` / `max_slices` | **10 / 14** | khác default code (20 / 8) |
| mạng | spatial_cnn + dueling, hidden 512 | Double DQN + PER |
| reward | clip **100**, TP +3.0, FP −0.75, hard-hit +4.0 | crop-outcome BẬT, train 1000 ep |

### 12.2 Số đo hiệu năng (Tesla T4, test-100, base@640)
| Phương pháp | small_recall | mAP50 | FP/ảnh | crop | ms/ảnh |
|---|---|---|---|---|---|
| YOLO full @640 | 0.402 | 0.277 | 66 | 1 | **45** |
| + SAHI | 0.620 | 0.254 | 202 | 29 | 760 |
| + RL-HYBRID | **0.691** | **0.302** | **179** | 15 | 1636 |

### 12.3 Thời gian đi đâu (GTX 1650, profile tách khâu, ms/ảnh)
| Khâu | RL-HYBRID | SAHI |
|---|---|---|
| base (1 lượt full) | 75 | 75 |
| di chuyển (RL rollout) | 560 (22%) | **0** (lưới cố định không di chuyển) |
| chạy YOLO trên crop | 1906 (**75%**) | 1883 (96%) |
| merge/NMS | 6 | 7 |

**Kết luận:** RL-SAHI = phương pháp **RECALL** (thắng SAHI cả 4 tiêu chí ở recall/mAP/FP/crop); tốc độ là hạn chế cố hữu của mọi phương pháp cắt-lát (75% chi phí = chạy lại YOLO). Muốn nhanh thật phải bỏ "chạy lại detector mỗi crop" (hướng RL-query / P2 — baseline đối chứng).

> **Lưu ý**: Dự án này đã có comment bằng tiếng Việt rất chi tiết trong mã nguồn. File này tổng hợp và giải thích mối quan hệ giữa các module để giúp bạn hiểu toàn bộ kiến trúc một cách hệ thống.
