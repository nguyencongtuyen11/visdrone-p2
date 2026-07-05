# giải thích: Cho phép sử dụng các kiểu gợi ý (type hints) nâng cao trong tương lai
from __future__ import annotations

# giải thích: Nhập các thư viện chuẩn của Python để tạo thư mục tạm thời và chạy unit test
import tempfile
import unittest
from pathlib import Path
import sys

# giải thích: Thư viện numpy hỗ trợ tính toán mảng nhiều chiều
import numpy as np

# giải thích: Xác định đường dẫn gốc và thêm thư mục src vào đầu danh sách tìm kiếm module
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# giải thích: Nhập lớp lưu trữ cache và các hàm liên quan để kiểm tra tính hợp lệ, tải/lưu cache
from rl_sahi.common.cache import DetectionCache, detection_cache_is_current, load_detection_cache, save_detection_cache


# giải thích: Tạo một đối tượng cache mẫu (mock cache) với siêu dữ liệu (metadata) tùy chỉnh để phục vụ kiểm thử
def _cache(metadata: dict) -> DetectionCache:
    return DetectionCache(
        image_path="synthetic.jpg",
        image_shape=(32, 32),
        boxes=np.zeros((0, 4), dtype=np.float32),
        scores=np.zeros((0,), dtype=np.float32),
        classes=np.zeros((0,), dtype=np.float32),
        feature=np.zeros((4,), dtype=np.float32),
        feature_layers=(10,),
        objectness_map=np.zeros((1, 16, 16), dtype=np.float32),
        spatial_feature_map=np.zeros((4, 16, 16), dtype=np.float32),
        metadata=metadata,
    )


# giải thích: Định nghĩa lớp kiểm thử cho siêu dữ liệu của cache phát hiện (Detection Cache)
class DetectionCacheMetadataTest(unittest.TestCase):
    # giải thích: Kiểm thử việc phát hiện sai lệch siêu dữ liệu (metadata mismatch) sẽ làm mất hiệu lực của cache
    def test_expected_metadata_mismatch_invalidates_cache(self) -> None:
        metadata = {"imgsz": 640, "feature_layers": (10,), "weights": {"path": "model.pt"}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "det.npz"
            # giải thích: Lưu cache với siêu dữ liệu mẫu vào đường dẫn tạm
            save_detection_cache(path, _cache(metadata))

            # giải thích: Kiểm tra nếu siêu dữ liệu khớp thì cache phải hợp lệ (True)
            self.assertTrue(detection_cache_is_current(path, metadata))
            # giải thích: Kiểm tra nếu thay đổi kích thước ảnh (imgsz) thì cache phải không hợp lệ (False)
            self.assertFalse(detection_cache_is_current(path, {**metadata, "imgsz": 320}))

    # giải thích: Kiểm thử tính chính xác khi ghi và đọc siêu dữ liệu từ file cache (round trip)
    def test_load_round_trips_metadata(self) -> None:
        metadata = {"imgsz": 640, "conf": 0.01}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "det.npz"
            # giải thích: Lưu cache vào file tạm
            save_detection_cache(path, _cache(metadata))

            # giải thích: Tải lại cache từ file tạm
            loaded = load_detection_cache(path)

        # giải thích: Đảm bảo siêu dữ liệu sau khi tải khớp hoàn toàn với siêu dữ liệu ban đầu
        self.assertEqual(loaded.metadata, metadata)


# giải thích: Điểm khởi chạy của chương trình unit test
if __name__ == "__main__":
    unittest.main()

