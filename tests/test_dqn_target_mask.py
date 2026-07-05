# giải thích: Cho phép sử dụng các kiểu gợi ý (type hints) nâng cao trong tương lai
from __future__ import annotations

# giải thích: Nhập các thư viện chuẩn và hệ thống
import sys
import unittest
from pathlib import Path

# giải thích: Thư viện numpy hỗ trợ tính toán mảng và torch hỗ trợ học sâu
import numpy as np
import torch

# giải thích: Xác định đường dẫn gốc của dự án và thêm thư mục src vào PATH của hệ thống
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# giải thích: Nhập ReplayBuffer và hàm optimize từ module rl để phục vụ việc huấn luyện thử nghiệm
from rl_sahi.rl.replay import ReplayBuffer
from rl_sahi.rl.trainer import optimize


# giải thích: Định nghĩa một mạng Q giả lập (mock Q-network) trả về các giá trị Q-value cố định để phục vụ kiểm thử
class StaticQ(torch.nn.Module):
    # giải thích: Khởi tạo mô hình mạng Q tĩnh với một tham số trọng số (parameter) bằng 0
    def __init__(self) -> None:
        super().__init__()
        self.current = torch.nn.Parameter(torch.zeros(3))

    # giải thích: Hàm lan truyền xuôi (forward pass) trả về Q-value giả định dựa trên ngưỡng đầu vào của trạng thái x
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = []
        for row in x:
            # giải thích: Nếu giá trị trạng thái nhỏ hơn 0.5, trả về giá trị tham số Q hiện tại
            if float(row[0].item()) < 0.5:
                rows.append(self.current)
            # giải thích: Ngược lại, trả về một vector Q-value cố định [0.0, 10.0, 2.0]
            else:
                rows.append(torch.tensor([0.0, 10.0, 2.0], device=x.device) + self.current * 0.0)
        return torch.stack(rows, dim=0)


# giải thích: Lớp kiểm thử chức năng lọc hành động không hợp lệ (Action Masking) của mô hình DQN khi tính toán target
class DqnTargetMaskTest(unittest.TestCase):
    # giải thích: Kiểm thử việc tối ưu hóa DQN lọc chính xác các hành động không hợp lệ ở trạng thái kế tiếp
    def test_optimize_masks_invalid_next_actions(self) -> None:
        # giải thích: Tạo bộ nhớ đệm ReplayBuffer với dung lượng 4
        replay = ReplayBuffer(4)
        # giải thích: Đẩy một bộ chuyển trạng thái (transition) vào bộ nhớ đệm
        # Trạng thái hiện tại: [0.0], Hành động: 0, Phần thưởng: 0.0, Trạng thái kế tiếp: [1.0], Kết thúc: False
        # Mặt nạ hành động kế tiếp: [True, False, True] (nghĩa là hành động index 1 bị cấm/không hợp lệ)
        replay.push(
            np.array([0.0], dtype=np.float32),
            0,
            0.0,
            np.array([1.0], dtype=np.float32),
            False,
            np.array([True, False, True], dtype=bool),
        )
        policy = StaticQ()
        target = StaticQ()
        # giải thích: Bộ tối ưu hóa SGD với tốc độ học lr=0.0 để giữ nguyên trọng số của mạng kiểm thử
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.0)

        # giải thích: Gọi hàm tối ưu hóa để tính toán giá trị loss
        loss = optimize(
            policy,
            target,
            optimizer,
            replay,
            batch_size=1,
            gamma=1.0,
            device=torch.device("cpu"),
            double_dqn=True,
        )

        # giải thích: Xác thực giá trị loss tính toán được khác None
        self.assertIsNotNone(loss)
        # giải thích: Xác thực giá trị loss xấp xỉ bằng 1.5 (giá trị tính toán theo Huber loss/Smooth L1 khi sai lệch là 2.0)
        self.assertAlmostEqual(float(loss), 1.5, places=5)


# giải thích: Điểm khởi chạy của chương trình unit test
if __name__ == "__main__":
    unittest.main()

