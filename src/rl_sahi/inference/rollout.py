# Cho phép import các annotation nâng cao từ tương lai
from __future__ import annotations

import numpy as np  # Thư viện tính toán mảng NumPy
import torch        # Thư viện deep learning PyTorch

# Import các hành động (Action) và tên hành động
from rl_sahi.common.actions import ACTION_NAMES, Action
from rl_sahi.rl.slice_env import SliceEnv


def rollout_one_slice(policy, env: SliceEnv, device: torch.device) -> tuple[np.ndarray, list[str], dict]:
    """
    Thực hiện chạy một đợt rollout duy nhất trên môi trường SliceEnv dưới chính sách điều khiển (policy) của DQN.
    Tác tử (agent) bắt đầu từ vị trí lát cắt khởi tạo, liên tiếp dự báo các hành động dịch chuyển/co giãn (actions)
    cho đến khi chọn hành động dừng (STOP) hoặc đạt giới hạn max_steps.
    
    Trả về:
        env.roi: Tọa độ ROI cuối cùng [x1, y1, x2, y2] sau khi hoàn thành rollout.
        actions: Danh sách tên các hành động mà tác tử đã thực hiện.
        info: Siêu dữ liệu từ bước cuối cùng của môi trường (chứa lý do kết thúc, độ chồng lấn...).
    """
    # Khởi động lại môi trường và lấy trạng thái khởi đầu (state vector)
    state = env.reset()
    
    # Kiểm tra tính khớp kích thước đầu vào của mạng DQN và vector trạng thái
    expected_dim = int(getattr(policy, "input_dim", state.shape[0]))
    if state.shape[0] != expected_dim:
        raise ValueError(
            f"Checkpoint expects state_dim={expected_dim}, but current detection state has {state.shape[0]}. "
            "Regenerate detection caches and retrain the DQN with the current state configuration."
        )
        
    actions: list[str] = []
    info: dict = {}
    
    # Chạy vòng lặp rollout từng bước cho đến giới hạn max_steps
    for _ in range(env.env_cfg.max_steps + 1):
        with torch.no_grad():
            # Tính toán Q-values cho toàn bộ các hành động dựa trên trạng thái hiện tại
            q = policy(torch.from_numpy(state).float().unsqueeze(0).to(device))
            # Lấy danh sách các hành động hợp lệ từ môi trường (ví dụ: không được dịch chuyển vượt biên ảnh)
            valid = torch.from_numpy(env.valid_actions()).bool().to(device)
            # Ép giá trị Q-value của các hành động không hợp lệ thành -vô cùng (để không bao giờ chọn phải)
            q[:, ~valid] = -torch.inf
            # Lọc chọn hành động có Q-value lớn nhất (Greedy Action)
            action = Action(int(q.argmax(dim=1).item()))
            
        actions.append(ACTION_NAMES[action])
        # Đưa hành động vào môi trường để tiến đến trạng thái tiếp theo
        result = env.step(action)
        state = result.state
        info = result.info
        
        # Nếu đạt điều kiện dừng (Done), thoát vòng lặp
        if result.done:
            break
            
    return env.roi.copy(), actions, info

