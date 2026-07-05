# giải thích: Cho phép sử dụng các kiểu gợi ý (type hints) nâng cao trong tương lai
from __future__ import annotations

# giải thích: Nhập các thư viện hệ thống và thư viện chạy unit test
import sys
import unittest
from pathlib import Path

# giải thích: Thư viện numpy hỗ trợ làm việc với mảng số học
import numpy as np

# giải thích: Xác định đường dẫn gốc và thêm thư mục src vào đầu danh sách tìm kiếm module
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# giải thích: Nhập các hàm tính toán lợi ích số lượng hộp phát hiện mới và độ hữu dụng mới sau khi hợp nhất cache
from rl_sahi.inference.merge import new_detection_gain_after_merge, new_detection_utility_after_merge


# giải thích: Định nghĩa lớp unit test để kiểm tra việc tính toán lợi ích tăng thêm khi hợp nhất các hộp phát hiện
class MergeGainTest(unittest.TestCase):
    # giải thích: Kiểm tra trường hợp hộp ứng viên trùng khớp hoàn toàn vị trí và cùng lớp với hộp đã có thì lợi ích (gain) thu được phải bằng 0 (không thêm hộp mới)
    def test_replacement_of_existing_same_class_box_is_not_new(self) -> None:
        gain = new_detection_gain_after_merge(
            image_shape=(100, 100),
            merge_iou=0.5,
            previous_boxes_parts=[np.array([[10.0, 10.0, 30.0, 30.0]], dtype=np.float32)],
            previous_scores_parts=[np.array([0.4], dtype=np.float32)],
            previous_classes_parts=[np.array([0.0], dtype=np.float32)],
            candidate_boxes=np.array([[10.0, 10.0, 30.0, 30.0]], dtype=np.float32),
            candidate_scores=np.array([0.9], dtype=np.float32),
            candidate_classes=np.array([0.0], dtype=np.float32),
        )

        # giải thích: Xác thực gain nhận được là 0 vì hộp mới trùng lặp vị trí và lớp với hộp cũ
        self.assertEqual(gain, 0)

    # giải thích: Kiểm tra trường hợp hộp ứng viên bị dịch chuyển nhẹ so với hộp cũ nhưng IoU vẫn lớn hơn ngưỡng trùng lặp, gain nhận được phải bằng 0
    def test_shifted_replacement_of_existing_same_class_box_is_not_new(self) -> None:
        gain = new_detection_gain_after_merge(
            image_shape=(100, 100),
            merge_iou=0.5,
            previous_boxes_parts=[np.array([[10.0, 10.0, 30.0, 30.0]], dtype=np.float32)],
            previous_scores_parts=[np.array([0.4], dtype=np.float32)],
            previous_classes_parts=[np.array([0.0], dtype=np.float32)],
            candidate_boxes=np.array([[12.0, 12.0, 32.0, 32.0]], dtype=np.float32),
            candidate_scores=np.array([0.9], dtype=np.float32),
            candidate_classes=np.array([0.0], dtype=np.float32),
            duplicate_iou=0.5,
        )

        # giải thích: Xác thực gain nhận được là 0
        self.assertEqual(gain, 0)

    # giải thích: Kiểm tra trường hợp hộp ứng viên có cùng lớp nhưng nằm ở vị trí không gian mới hoàn toàn, gain nhận được phải bằng 1
    def test_spatially_new_same_class_box_counts_as_new(self) -> None:
        gain = new_detection_gain_after_merge(
            image_shape=(100, 100),
            merge_iou=0.5,
            previous_boxes_parts=[np.array([[10.0, 10.0, 30.0, 30.0]], dtype=np.float32)],
            previous_scores_parts=[np.array([0.9], dtype=np.float32)],
            previous_classes_parts=[np.array([0.0], dtype=np.float32)],
            candidate_boxes=np.array([[60.0, 60.0, 80.0, 80.0]], dtype=np.float32),
            candidate_scores=np.array([0.8], dtype=np.float32),
            candidate_classes=np.array([0.0], dtype=np.float32),
        )

        # giải thích: Xác thực gain nhận được là 1 vì hộp mới ở toạ độ hoàn toàn độc lập [60, 60, 80, 80]
        self.assertEqual(gain, 1)

    # giải thích: Kiểm tra độ hữu dụng (utility) tăng thêm đối với hộp mới ở toạ độ mới phải bằng chính độ tin cậy (confidence score) của nó
    def test_spatially_new_utility_uses_candidate_confidence(self) -> None:
        utility = new_detection_utility_after_merge(
            image_shape=(100, 100),
            merge_iou=0.5,
            previous_boxes_parts=[np.array([[10.0, 10.0, 30.0, 30.0]], dtype=np.float32)],
            previous_scores_parts=[np.array([0.9], dtype=np.float32)],
            previous_classes_parts=[np.array([0.0], dtype=np.float32)],
            candidate_boxes=np.array([[60.0, 60.0, 80.0, 80.0]], dtype=np.float32),
            candidate_scores=np.array([0.8], dtype=np.float32),
            candidate_classes=np.array([0.0], dtype=np.float32),
        )

        # giải thích: Xác thực độ hữu dụng tăng thêm bằng 0.8 (bằng độ tin cậy của hộp ứng viên)
        self.assertAlmostEqual(utility, 0.8, places=6)

    # giải thích: Kiểm tra trường hợp hộp ứng viên trùng vị trí địa lý nhưng khác lớp (class) thì vẫn được tính là hộp mới (gain = 1)
    def test_same_location_different_class_counts_as_new(self) -> None:
        gain = new_detection_gain_after_merge(
            image_shape=(100, 100),
            merge_iou=0.5,
            previous_boxes_parts=[np.array([[10.0, 10.0, 30.0, 30.0]], dtype=np.float32)],
            previous_scores_parts=[np.array([0.9], dtype=np.float32)],
            previous_classes_parts=[np.array([0.0], dtype=np.float32)],
            candidate_boxes=np.array([[10.0, 10.0, 30.0, 30.0]], dtype=np.float32),
            candidate_scores=np.array([0.8], dtype=np.float32),
            candidate_classes=np.array([2.0], dtype=np.float32),
        )

        # giải thích: Xác thực gain là 1 do khác lớp phân loại (0.0 so với 2.0)
        self.assertEqual(gain, 1)


# giải thích: Điểm khởi chạy của chương trình unit test
if __name__ == "__main__":
    unittest.main()

