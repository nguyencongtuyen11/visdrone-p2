from __future__ import annotations

import torch
from torch import nn

from rl_sahi.common.actions import NUM_ACTIONS
from rl_sahi.rl.state_config import StateLayout


# giải thích: Lớp QNetwork kế thừa từ nn.Module, định nghĩa kiến trúc mạng nơ-ron Q cho thuật toán DQN/Dueling DQN
class QNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_actions: int = NUM_ACTIONS,
        layout: StateLayout | None = None,
        use_spatial_cnn: bool = False,
        dueling: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.layout = layout
        self.use_spatial_cnn = bool(use_spatial_cnn and layout is not None)
        self.dueling = bool(dueling)
        self.num_actions = int(num_actions)

        # giải thích: Nếu không sử dụng Spatial CNN, khởi tạo mạng MLP (Multi-Layer Perceptron) đơn giản
        if not self.use_spatial_cnn:
            trunk_dim = hidden_dim // 2
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, trunk_dim),
                nn.ReLU(inplace=True),
            )
            # giải thích: Khởi tạo Dueling Heads (Value & Advantage) hoặc Q Head thông thường
            if self.dueling:
                self.value_head = nn.Linear(trunk_dim, 1)
                self.advantage_head = nn.Linear(trunk_dim, num_actions)
            else:
                self.q_head = nn.Linear(trunk_dim, num_actions)
            return

        # giải thích: Nếu sử dụng Spatial CNN, khởi tạo nhánh CNN xử lý bản đồ 2D và nhánh MLP xử lý vector
        assert layout is not None
        self.spatial = nn.Sequential(
            nn.Conv2d(layout.map_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        spatial_dim = 64 * 4 * 4
        vector_dim = layout.feature_dim + layout.summary_dim
        self.vector = nn.Sequential(
            nn.Linear(vector_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        trunk_dim = hidden_dim
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim + spatial_dim, trunk_dim),
            nn.ReLU(inplace=True),
        )
        # giải thích: Khởi tạo Dueling Heads hoặc Q Head thông thường cho nhánh kết hợp CNN-MLP
        if self.dueling:
            self.value_head = nn.Linear(trunk_dim, 1)
            self.advantage_head = nn.Linear(trunk_dim, num_actions)
        else:
            self.q_head = nn.Linear(trunk_dim, num_actions)

    # giải thích: Hàm gộp nhánh giá trị trạng thái (Value) và nhánh lợi thế hành động (Advantage) trong kiến trúc Dueling DQN
    def _dueling_combine(self, trunk: torch.Tensor) -> torch.Tensor:
        value = self.value_head(trunk)
        advantage = self.advantage_head(trunk)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    # giải thích: Quá trình lan truyền xuôi (forward pass) của mạng nơ-ron
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # giải thích: Trường hợp mạng MLP thuần túy
        if not self.use_spatial_cnn:
            trunk = self.net(x)
            if self.dueling:
                return self._dueling_combine(trunk)
            return self.q_head(trunk)

        # giải thích: Trường hợp mạng kết hợp CNN + MLP: tách vector trạng thái đầu vào thành đặc trưng, bản đồ lưới và phần tóm tắt
        assert self.layout is not None
        feature_end = self.layout.feature_dim
        maps_end = feature_end + self.layout.map_channels * self.layout.grid_size * self.layout.grid_size
        feature = x[:, :feature_end]
        maps = x[:, feature_end:maps_end].reshape(
            -1,
            self.layout.map_channels,
            self.layout.grid_size,
            self.layout.grid_size,
        )
        summary = x[:, maps_end : maps_end + self.layout.summary_dim]
        
        # giải thích: Đưa dữ liệu qua nhánh CNN và MLP, sau đó ghép lại ở phần thân mạng (trunk)
        spatial_out = self.spatial(maps)
        vector_out = self.vector(torch.cat([feature, summary], dim=1))
        trunk = self.trunk(torch.cat([vector_out, spatial_out], dim=1))
        if self.dueling:
            return self._dueling_combine(trunk)
        return self.q_head(trunk)
