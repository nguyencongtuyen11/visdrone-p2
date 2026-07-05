# Cho phép import các annotation nâng cao từ tương lai
from __future__ import annotations

import json                  # Thư viện mã hóa/giải mã định dạng JSON
from dataclasses import dataclass  # Decorator tạo các lớp dữ liệu ngắn gọn (dataclasses)
from pathlib import Path     # Thư viện xử lý đường dẫn tệp tin hướng đối tượng
from typing import Any       # Dùng định nghĩa kiểu dữ liệu bất kỳ cho type hinting

import numpy as np           # Thư viện xử lý mảng số học NumPy

# Import hàm phụ trợ lấy ID định danh của ảnh từ tên file
from .data import image_id


# Phiên bản hiện tại của cache phát hiện đối tượng, dùng để xác thực tính hợp lệ khi có thay đổi cấu trúc dữ liệu
DETECTION_CACHE_VERSION = 3


@dataclass(slots=True)
class DetectionCache:
    """
    Lớp lưu trữ dữ liệu cache phát hiện đối tượng của YOLO trên một bức ảnh.
    Sử dụng slots=True để tối ưu hóa bộ nhớ và tăng tốc truy xuất thuộc tính.
    """
    image_path: str                 # Đường dẫn ảnh gốc dưới dạng chuỗi
    image_shape: tuple[int, int]    # Kích thước ảnh (chiều cao, chiều rộng)
    boxes: np.ndarray               # Mảng chứa tọa độ các bounding boxes phát hiện được
    scores: np.ndarray              # Mảng chứa độ tự tin (confidence scores) tương ứng
    classes: np.ndarray             # Mảng chứa nhãn lớp (class labels)
    feature: np.ndarray             # Các đặc trưng mạng nơ-ron thu được từ YOLO backbone
    feature_layers: tuple[int, ...] # Chỉ số các lớp đặc trưng được trích xuất
    objectness_map: np.ndarray      # Bản đồ độ vật thể (objectness map) phục vụ RL state
    spatial_feature_map: np.ndarray # Bản đồ đặc trưng không gian (spatial feature map)
    metadata: dict[str, Any] | None = None  # Siêu dữ liệu bổ sung (như trọng số, kích thước ảnh...)


@dataclass(slots=True)
class HardRegionCache:
    """
    Lớp lưu trữ dữ liệu cache các vùng khó phát hiện (hard regions) trên một bức ảnh.
    Dùng để định vị những đối tượng nhỏ bị YOLO bỏ sót hoặc đạt score thấp nhằm huấn luyện RL.
    """
    image_path: str                 # Đường dẫn ảnh gốc
    image_shape: tuple[int, int]    # Kích thước ảnh
    hard_boxes: np.ndarray          # Các hộp giới hạn của vùng khó
    small_gt_boxes: np.ndarray      # Các hộp nhãn gốc (ground truth) có kích thước nhỏ
    gt_boxes: np.ndarray            # Toàn bộ các hộp nhãn gốc (ground truth) trên ảnh
    matched_iou: np.ndarray         # Ma trận IoU khớp giữa các phát hiện và ground truth
    matched_score: np.ndarray       # Các score khớp tương ứng


def detection_cache_path(cache_root: Path, split: str, image_path: Path) -> Path:
    """
    Tạo đường dẫn tệp tin cache phát hiện đối tượng tương ứng (.npz).
    Ví dụ: <cache_root>/detections/<split>/<image_id>.npz
    """
    return Path(cache_root) / "detections" / split / f"{image_id(image_path)}.npz"


def hard_region_cache_path(cache_root: Path, split: str, image_path: Path) -> Path:
    """
    Tạo đường dẫn tệp tin cache vùng khó tương ứng (.npz).
    Ví dụ: <cache_root>/hard_regions/<split>/<image_id>.npz
    """
    return Path(cache_root) / "hard_regions" / split / f"{image_id(image_path)}.npz"


def _normalize_metadata(value: Any) -> Any:
    """
    Chuẩn hóa các đối tượng phức tạp trong metadata (như Path hay NumPy types)
    về kiểu dữ liệu Python chuẩn để có thể tuần tự hóa sang JSON.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item() # Chuyển đổi NumPy scalar sang Python scalar
    if isinstance(value, dict):
        return {str(key): _normalize_metadata(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_normalize_metadata(item) for item in value]
    return value


def _metadata_json(metadata: dict[str, Any] | None) -> str:
    """
    Chuyển đổi dictionary metadata thành chuỗi JSON đã được sắp xếp key để so sánh nhất quán.
    """
    return json.dumps(_normalize_metadata(metadata or {}), sort_keys=True, separators=(",", ":"))


def _file_fingerprint(path: Path) -> dict[str, Any]:
    """
    Tạo "dấu vân tay" (fingerprint) cho tệp tin (trọng số weights YOLO) gồm đường dẫn tuyệt đối,
    kích thước tệp và thời gian sửa đổi cuối cùng (mtime) dưới dạng nano giây.
    Dùng để phát hiện xem file trọng số có bị thay đổi hay không.
    """
    path = Path(path)
    fingerprint: dict[str, Any] = {"path": str(path.resolve())}
    if not path.exists():
        fingerprint["exists"] = False
        return fingerprint
    stat = path.stat()
    fingerprint["exists"] = True
    fingerprint["size"] = int(stat.st_size)
    fingerprint["mtime_ns"] = int(stat.st_mtime_ns)
    return fingerprint


def detection_cache_metadata(
    weights: Path,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    feature_layers: tuple[int, ...],
    aux_grid_size: int,
    spatial_feature_channels: int,
) -> dict[str, Any]:
    """
    Xây dựng từ điển metadata chứa toàn bộ cấu hình chạy YOLO dùng để so khớp cache.
    """
    return {
        "weights": _file_fingerprint(Path(weights)),
        "imgsz": int(imgsz),
        "conf": float(conf),
        "iou": float(iou),
        "max_det": int(max_det),
        "feature_layers": tuple(int(x) for x in feature_layers),
        "aux_grid_size": int(aux_grid_size),
        "spatial_feature_channels": int(spatial_feature_channels),
    }


def save_detection_cache(path: Path, cache: DetectionCache) -> None:
    """
    Lưu đối tượng DetectionCache vào tệp tin nén định dạng .npz của NumPy.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cache_version=np.asarray(DETECTION_CACHE_VERSION, dtype=np.int32),
        metadata_json=np.asarray(_metadata_json(cache.metadata)),
        image_path=np.asarray(cache.image_path),
        image_shape=np.asarray(cache.image_shape, dtype=np.int32),
        boxes=cache.boxes.astype(np.float32),
        scores=cache.scores.astype(np.float32),
        classes=cache.classes.astype(np.float32),
        feature=cache.feature.astype(np.float32),
        feature_layers=np.asarray(cache.feature_layers, dtype=np.int32),
        objectness_map=cache.objectness_map.astype(np.float32),
        spatial_feature_map=cache.spatial_feature_map.astype(np.float32),
    )


def _compare_metadata(actual_json: str, expected_dict: dict[str, Any]) -> bool:
    """
    So sánh metadata lưu trong cache với cấu hình mong đợi hiện tại.
    Bỏ qua sự khác biệt về đường dẫn tuyệt đối của weights nếu kích thước file và thời gian sửa đổi khớp nhau.
    """
    try:
        actual = json.loads(actual_json)
        if "weights" in actual and "weights" in expected_dict:
            actual_w = actual["weights"]
            expected_w = expected_dict["weights"]
            if isinstance(actual_w, dict) and isinstance(expected_w, dict):
                if "path" in actual_w and "path" in expected_w:
                    actual_path = actual_w["path"]
                    expected_path = expected_w["path"]
                    actual_w_copy = actual_w.copy()
                    expected_w_copy = expected_w.copy()
                    actual_w_copy.pop("path")
                    expected_w_copy.pop("path")
                    # Nếu các thông số kích thước/mtime khớp nhau, đồng bộ hóa đường dẫn để phép so sánh bằng chuỗi JSON khớp
                    if actual_w_copy == expected_w_copy:
                        expected_dict = expected_dict.copy()
                        expected_dict["weights"] = expected_dict["weights"].copy()
                        expected_dict["weights"]["path"] = actual_path
        return actual_json == _metadata_json(expected_dict)
    except Exception:
        return False


def detection_cache_is_current(path: Path, expected_metadata: dict[str, Any] | None = None) -> bool:
    """
    Kiểm tra xem tệp cache hiện tại có khớp phiên bản và cấu hình mong đợi hay không.
    Trả về True nếu cache hợp lệ và có thể tái sử dụng trực tiếp.
    """
    path = Path(path)
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as data:
        if "cache_version" not in data.files:
            return False
        version = int(np.asarray(data["cache_version"]).item())
        return (
            version >= DETECTION_CACHE_VERSION
            and "metadata_json" in data.files
            and "objectness_map" in data.files
            and "spatial_feature_map" in data.files
            and (
                expected_metadata is None
                or _compare_metadata(str(data["metadata_json"].item()), expected_metadata)
            )
        )


def load_detection_cache(path: Path) -> DetectionCache:
    """
    Đọc tệp tin .npz và chuyển đổi ngược lại thành đối tượng DetectionCache.
    """
    with np.load(path, allow_pickle=False) as data:
        shape = data["image_shape"].astype(np.int32).tolist()
        objectness_map = (
            data["objectness_map"].astype(np.float32)
            if "objectness_map" in data.files
            else np.zeros((0,), dtype=np.float32)
        )
        spatial_feature_map = (
            data["spatial_feature_map"].astype(np.float32)
            if "spatial_feature_map" in data.files
            else np.zeros((0,), dtype=np.float32)
        )
        return DetectionCache(
            image_path=str(data["image_path"].item()),
            image_shape=(int(shape[0]), int(shape[1])),
            boxes=data["boxes"].astype(np.float32),
            scores=data["scores"].astype(np.float32),
            classes=data["classes"].astype(np.float32),
            feature=data["feature"].astype(np.float32),
            feature_layers=tuple(int(x) for x in data["feature_layers"].tolist()),
            objectness_map=objectness_map,
            spatial_feature_map=spatial_feature_map,
            metadata=(
                json.loads(str(data["metadata_json"].item()))
                if "metadata_json" in data.files
                else {}
            ),
        )


def save_hard_region_cache(path: Path, cache: HardRegionCache) -> None:
    """
    Lưu đối tượng HardRegionCache vào tệp tin nén .npz.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        image_path=np.asarray(cache.image_path),
        image_shape=np.asarray(cache.image_shape, dtype=np.int32),
        hard_boxes=cache.hard_boxes.astype(np.float32),
        small_gt_boxes=cache.small_gt_boxes.astype(np.float32),
        gt_boxes=cache.gt_boxes.astype(np.float32),
        matched_iou=cache.matched_iou.astype(np.float32),
        matched_score=cache.matched_score.astype(np.float32),
    )


def load_hard_region_cache(path: Path) -> HardRegionCache:
    """
    Tải tệp tin .npz và chuyển đổi ngược thành đối tượng HardRegionCache.
    """
    data = np.load(path, allow_pickle=False)
    shape = data["image_shape"].astype(np.int32).tolist()
    return HardRegionCache(
        image_path=str(data["image_path"].item()),
        image_shape=(int(shape[0]), int(shape[1])),
        hard_boxes=data["hard_boxes"].astype(np.float32),
        small_gt_boxes=data["small_gt_boxes"].astype(np.float32),
        gt_boxes=data["gt_boxes"].astype(np.float32),
        matched_iou=data["matched_iou"].astype(np.float32),
        matched_score=data["matched_score"].astype(np.float32),
    )

