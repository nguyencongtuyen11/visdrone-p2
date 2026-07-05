# giải thích: Cho phép sử dụng các kiểu gợi ý (type hints) nâng cao trong tương lai
from __future__ import annotations

# giải thích: Nhập các thư viện hệ thống và thư viện chạy unit test
import sys
import unittest
from pathlib import Path

# giải thích: Thư viện numpy hỗ trợ tính toán mảng số học
import numpy as np

# giải thích: Xác định đường dẫn gốc và thêm thư mục src vào đầu danh sách tìm kiếm module
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# giải thích: Nhập lớp PrioritizedReplayBuffer để phục vụ việc kiểm thử bộ đệm trải nghiệm ưu tiên
from rl_sahi.rl.replay import PrioritizedReplayBuffer


# giải thích: Định nghĩa lớp unit test cho bộ đệm trải nghiệm ưu tiên (Prioritized Replay Buffer)
class PrioritizedReplayBufferTest(unittest.TestCase):
    # giải thích: Kiểm thử xem phần tử mới được thêm vào có tự động nhận độ ưu tiên lớn nhất hiện tại (max priority) hay không
    def test_new_items_receive_raw_max_priority(self) -> None:
        # giải thích: Khởi tạo bộ đệm với sức chứa 4 và hệ số ưu tiên alpha=0.5
        replay = PrioritizedReplayBuffer(capacity=4, alpha=0.5)
        state = np.array([0.0], dtype=np.float32)
        # giải thích: Thêm phần tử đầu tiên vào bộ đệm
        replay.push(state, 0, 0.0, state, False)
        # giải thích: Cập nhật độ ưu tiên của phần tử index 0 lên mức rất cao (100.0)
        replay.update_priorities(np.array([0]), np.array([100.0], dtype=np.float32))
        # giải thích: Thêm phần tử thứ hai vào bộ đệm
        replay.push(state, 0, 0.0, state, False)

        # giải thích: Xác thực độ ưu tiên của phần tử thứ hai (index 1) bằng với độ ưu tiên lớn nhất hiện tại (_max_priority)
        self.assertAlmostEqual(float(replay._priorities[1]), float(replay._max_priority), places=6)


# giải thích: Điểm khởi chạy của chương trình unit test
if __name__ == "__main__":
    unittest.main()

