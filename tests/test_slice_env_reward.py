# giải thích: Cho phép sử dụng các kiểu gợi ý (type hints) nâng cao trong tương lai
from __future__ import annotations

# giải thích: Nhập thư viện unittest và các thư viện hệ thống
import unittest
from pathlib import Path
import sys

# giải thích: Thư viện numpy hỗ trợ tính toán mảng và ma trận
import numpy as np

# giải thích: Xác định đường dẫn gốc và thêm thư mục src vào PATH của hệ thống
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# giải thích: Nhập các lớp định nghĩa hành động, cache phát hiện, cấu hình môi trường, và lớp môi trường lát cắt
from rl_sahi.common.actions import Action
from rl_sahi.common.cache import DetectionCache, HardRegionCache
from rl_sahi.rl.env_config import EnvConfig
from rl_sahi.rl.slice_env import SliceEnv
from rl_sahi.rl.state_config import BASE_MAP_CHANNELS


# giải thích: Hàm tạo đối tượng cache phát hiện mẫu (trống) với kích thước ảnh 100x100
def _detection_cache() -> DetectionCache:
    return DetectionCache(
        image_path="synthetic.jpg",
        image_shape=(100, 100),
        boxes=np.zeros((0, 4), dtype=np.float32),
        scores=np.zeros((0,), dtype=np.float32),
        classes=np.zeros((0,), dtype=np.float32),
        feature=np.zeros((4,), dtype=np.float32),
        feature_layers=(10,),
        objectness_map=np.zeros((1, 16, 16), dtype=np.float32),
        spatial_feature_map=np.zeros((4, 16, 16), dtype=np.float32),
    )


# giải thích: Hàm tạo đối tượng cache vùng khó mẫu với một hộp vùng khó nằm ở tâm ảnh
def _hard_region_cache() -> HardRegionCache:
    hard_box = np.array([[49.0, 49.0, 51.0, 51.0]], dtype=np.float32)
    return HardRegionCache(
        image_path="synthetic.jpg",
        image_shape=(100, 100),
        hard_boxes=hard_box,
        small_gt_boxes=hard_box.copy(),
        gt_boxes=hard_box.copy(),
        matched_iou=np.zeros((1,), dtype=np.float32),
        matched_score=np.zeros((1,), dtype=np.float32),
    )


# giải thích: Hàm tạo đối tượng cache phát hiện có chứa một hộp phát hiện độ tin cậy cao thuộc lớp chỉ định
def _high_conf_detection_cache(cls: float) -> DetectionCache:
    return DetectionCache(
        image_path="synthetic.jpg",
        image_shape=(100, 100),
        boxes=np.array([[40.0, 40.0, 60.0, 60.0]], dtype=np.float32),
        scores=np.array([0.9], dtype=np.float32),
        classes=np.array([cls], dtype=np.float32),
        feature=np.zeros((4,), dtype=np.float32),
        feature_layers=(10,),
        objectness_map=np.zeros((1, 16, 16), dtype=np.float32),
        spatial_feature_map=np.zeros((4, 16, 16), dtype=np.float32),
    )


# giải thích: Lớp kiểm thử phần thưởng và hành vi của môi trường cắt ảnh SliceEnv
class SliceEnvRewardTest(unittest.TestCase):
    # giải thích: Phương thức tiện ích để khởi tạo nhanh một môi trường SliceEnv với cấu hình mặc định
    def make_env(self) -> SliceEnv:
        return SliceEnv(_detection_cache(), _hard_region_cache(), env_cfg=EnvConfig())

    # giải thích: Kiểm thử việc di chuyển (không STOP) bao phủ mục tiêu không được tích lũy hoặc thưởng ngay lập tức
    def test_non_stop_target_hit_is_not_committed_or_rewarded(self) -> None:
        env = self.make_env()
        env.reset()

        # giải thích: Thực hiện hành động phóng to (ZOOM_IN)
        result = env.step(Action.ZOOM_IN)

        # giải thích: Xác thực số lượng mục tiêu mới được xác nhận (committed) bằng 0
        self.assertEqual(result.info["new_hits"], 0)
        # giải thích: Xác thực số lượng mục tiêu tiềm năng trong ROI hiện tại bằng 1
        self.assertEqual(result.info["candidate_hits"], 1)
        # giải thích: Xác thực mảng đánh dấu các vùng khó đã bao phủ vẫn bằng 0 (chưa commit)
        self.assertEqual(int(env.covered.sum()), 0)
        # giải thích: Xác thực phần thưởng của bước di chuyển này là âm (phạt chi phí bước đi)
        self.assertLess(result.reward, 0.0)

    # giải thích: Kiểm thử việc gọi hành động dừng (STOP) sẽ cam kết (commit) các mục tiêu bao phủ được
    def test_stop_commits_target_hit(self) -> None:
        env = self.make_env()
        env.reset()
        env.step(Action.ZOOM_IN)

        # giải thích: Thực hiện hành động STOP để kết thúc lượt quét vùng quan tâm này
        stop = env.step(Action.STOP)

        # giải thích: Xác thực môi trường đã kết thúc (done = True)
        self.assertTrue(stop.done)
        # giải thích: Xác thực số lượng mục tiêu mới cam kết bằng 1
        self.assertEqual(stop.info["new_hits"], 1)
        # giải thích: Xác thực số lượng mục tiêu cũ được giữ lại từ các lát trước bằng 0
        self.assertEqual(stop.info["retained_hits"], 0)
        # giải thích: Xác thực tổng điểm mục tiêu thu được lớn hơn 0.0
        self.assertGreater(stop.info["total_target_score"], 0.0)
        # giải thích: Xác thực nhận được phần thưởng dương nhờ tìm thấy mục tiêu vùng khó
        self.assertGreater(stop.reward, 0.0)

    # giải thích: Kiểm thử các bước di chuyển thường không cam kết độ bao phủ vùng khó cho đến khi dừng
    def test_non_stop_moves_do_not_commit_hard_coverage(self) -> None:
        env = self.make_env()
        env.reset()
        # giải thích: Thực hiện chuỗi hành động di chuyển mà không dừng
        env.step(Action.ZOOM_IN)
        env.step(Action.ZOOM_IN)
        env.step(Action.ZOOM_OUT)

        # giải thích: Đảm bảo độ bao phủ trên bản đồ vẫn chưa được ghi nhận
        self.assertEqual(int(env.covered.sum()), 0)

        # giải thích: Gọi hành động dừng (STOP)
        stop = env.step(Action.STOP)

        # giải thích: Xác thực mục tiêu đã được ghi nhận thành công và bản đồ bao phủ cập nhật
        self.assertEqual(stop.info["new_hits"], 1)
        self.assertEqual(int(env.covered.sum()), 1)

    # giải thích: Kiểm thử khi đi quá số bước tối đa mà không gọi STOP thì bị phạt kết thúc đột ngột
    def test_max_steps_without_stop_gets_terminal_penalty(self) -> None:
        # giải thích: Tạo môi trường giới hạn tối đa 1 bước đi và đặt mức phạt không STOP là 3.0
        env = SliceEnv(
            _detection_cache(),
            _hard_region_cache(),
            env_cfg=EnvConfig(max_steps=1, max_steps_without_stop_penalty=3.0),
        )
        env.reset()

        # giải thích: Thực hiện hành động di chuyển sang phải (RIGHT) vượt quá số bước cho phép
        result = env.step(Action.RIGHT)

        # giải thích: Xác thực lượt chơi kết thúc và có cờ báo dừng do quá số bước
        self.assertTrue(result.done)
        self.assertTrue(result.info["stop_due_to_max_steps"])
        # giải thích: Phần thưởng nhận được phải nhỏ hơn mức phạt -3.0
        self.assertLess(result.reward, -3.0)

    # giải thích: Kiểm thử khi kích thước lát cắt bị kẹt (không thể thu nhỏ tiếp) mà không dừng thì bị phạt kẹt
    def test_stalled_without_stop_gets_terminal_penalty(self) -> None:
        # giải thích: Cấu hình tỷ lệ ban đầu và tối thiểu bằng nhau (0.35) để gây ra tình trạng kẹt ngay khi zoom in
        env = SliceEnv(
            _detection_cache(),
            None,
            env_cfg=EnvConfig(
                initial_slice_fraction=0.35,
                min_slice_fraction=0.35,
                stalled_without_stop_penalty=2.5,
            ),
        )
        env.reset()

        # giải thích: Cố gắng zoom in khi đã ở kích thước tối thiểu
        result = env.step(Action.ZOOM_IN)

        # giải thích: Xác thực lượt chơi kết thúc do ROI bị kẹt và nhận mức phạt thích đáng
        self.assertTrue(result.done)
        self.assertTrue(result.info["stop_due_to_stalled_roi"])
        self.assertLess(result.reward, -2.5)

    # giải thích: Kiểm thử xem danh sách hành động hợp lệ có chặn ZOOM_IN khi tỷ lệ lát cắt đạt giới hạn nhỏ nhất không
    def test_valid_actions_mask_stalled_zoom(self) -> None:
        env = SliceEnv(
            _detection_cache(),
            None,
            env_cfg=EnvConfig(initial_slice_fraction=0.35, min_slice_fraction=0.35),
        )
        env.reset()

        # giải thích: Lấy mảng mặt nạ hành động hợp lệ
        valid = env.valid_actions()

        # giải thích: Hành động ZOOM_IN phải bị cấm (False) và STOP phải được phép (True)
        self.assertFalse(valid[int(Action.ZOOM_IN)])
        self.assertTrue(valid[int(Action.STOP)])

    # giải thích: Kiểm thử việc cố tình dừng ở vùng bị trùng lặp với lượt trước sẽ bị phạt và vô hiệu hóa STOP
    def test_attempted_overlap_masks_stop_and_penalizes_terminal_stop(self) -> None:
        base_env = SliceEnv(_detection_cache(), None, env_cfg=EnvConfig())
        base_env.reset()
        # giải thích: Khởi tạo môi trường mới chứa thông tin vùng đã duyệt qua trùng với ROI hiện tại
        env = SliceEnv(
            _detection_cache(),
            None,
            env_cfg=EnvConfig(),
            previous_rois=base_env.roi.reshape(1, 4),
            overlap_rois=np.zeros((0, 4), dtype=np.float32),
        )
        env.reset()

        valid = env.valid_actions()
        result = env.step(Action.STOP)

        # giải thích: Xác thực hành động STOP bị vô hiệu hóa vì trùng lặp
        self.assertFalse(valid[int(Action.STOP)])
        # giải thích: Xác thực kết thúc lượt chơi với thông báo trùng lặp và bị phạt điểm
        self.assertTrue(result.info["stop_due_to_attempted_overlap"])
        self.assertLess(result.reward, -2.0)

    # giải thích: Kiểm thử xem hành động di chuyển vào vùng trùng lặp có bị chặn khi vẫn còn hướng đi khác để thoát không
    def test_valid_actions_mask_move_into_attempted_overlap_when_escape_exists(self) -> None:
        base_env = SliceEnv(_detection_cache(), None, env_cfg=EnvConfig())
        base_env.reset()
        # giải thích: Tính toán toạ độ ROI nếu di chuyển sang phải
        attempted = base_env._apply_action(Action.RIGHT).reshape(1, 4)
        # giải thích: Đặt vùng di chuyển sang phải này vào danh sách previous_rois (vùng trùng lặp)
        env = SliceEnv(
            _detection_cache(),
            None,
            env_cfg=EnvConfig(),
            previous_rois=attempted,
            overlap_rois=np.zeros((0, 4), dtype=np.float32),
        )
        env.reset()

        valid = env.valid_actions()

        # giải thích: Hành động đi sang phải (RIGHT) phải bị cấm, trong khi đi sang trái (LEFT) vẫn được phép
        self.assertFalse(valid[int(Action.RIGHT)])
        self.assertTrue(valid[int(Action.LEFT)])

    # giải thích: Kiểm thử hành động đi chéo (ví dụ DOWN_RIGHT) di chuyển đồng thời trên cả hai trục hoành và trục tung
    def test_diagonal_action_moves_on_both_axes(self) -> None:
        env = SliceEnv(_detection_cache(), None, env_cfg=EnvConfig())
        env.reset()
        original = env.roi.copy()

        # giải thích: Di chuyển chéo xuống dưới - sang phải
        env.step(Action.DOWN_RIGHT)

        # giải thích: Xác thực toạ độ x_min và y_min đều tăng so với ban đầu
        self.assertGreater(env.roi[0], original[0])
        self.assertGreater(env.roi[1], original[1])

    # giải thích: Kiểm thử cơ chế lọc lớp mục tiêu (target class filter) loại bỏ các phạt trùng lặp từ lớp không mong muốn
    def test_target_class_filter_excludes_non_target_detection_penalty(self) -> None:
        cfg = EnvConfig(step_penalty=0.0, area_penalty=0.0, detected_overlap_penalty=1.0)
        # Môi trường 1: Chấp nhận tất cả các lớp
        env_all = SliceEnv(_high_conf_detection_cache(99), None, env_cfg=cfg)
        env_all.reset()
        all_result = env_all.step(Action.STOP)

        # Môi trường 2: Chỉ chấp nhận lớp 0 làm mục tiêu
        env_target = SliceEnv(_high_conf_detection_cache(99), None, env_cfg=cfg, target_classes=(0,))
        env_target.reset()
        target_result = env_target.step(Action.STOP)

        # giải thích: Môi trường 1 ghi nhận phạt trùng lặp phát hiện lớn hơn 0
        self.assertGreater(all_result.info["detected_overlap"], 0.0)
        # giải thích: Môi trường 2 ghi nhận phạt trùng lặp phát hiện bằng 0 (do lọc bỏ lớp 99)
        self.assertEqual(target_result.info["detected_overlap"], 0.0)
        # giải thích: Phần thưởng của môi trường lọc lớp mục tiêu tốt hơn môi trường không lọc
        self.assertGreater(target_result.reward, all_result.reward)

    # giải thích: Kiểm thử tính đồng nhất giữa thuật toán tính toán IoU/hộp bằng PyTorch trên GPU so với NumPy trên CPU
    def test_torch_box_ops_match_numpy_reward_backend(self) -> None:
        numpy_cfg = EnvConfig(use_gpu_box_ops=False)
        torch_cfg = EnvConfig(use_gpu_box_ops=True, gpu_box_device="cpu")
        previous = np.array([[20.0, 20.0, 50.0, 50.0]], dtype=np.float32)
        accepted = np.array([[10.0, 10.0, 35.0, 35.0]], dtype=np.float32)
        env_numpy = SliceEnv(
            _high_conf_detection_cache(0),
            _hard_region_cache(),
            env_cfg=numpy_cfg,
            previous_rois=previous,
            overlap_rois=accepted,
        )
        env_torch = SliceEnv(
            _high_conf_detection_cache(0),
            _hard_region_cache(),
            env_cfg=torch_cfg,
            previous_rois=previous,
            overlap_rois=accepted,
        )
        env_numpy.reset()
        env_torch.reset()

        # giải thích: Đảm bảo các hành động hợp lệ từ hai backend hoàn toàn trùng khớp
        np.testing.assert_array_equal(env_torch.valid_actions(), env_numpy.valid_actions())
        result_numpy = env_numpy.step(Action.STOP)
        result_torch = env_torch.step(Action.STOP)

        # giải thích: Xác thực trạng thái kết thúc và giá trị phần thưởng xấp xỉ bằng nhau
        self.assertEqual(result_torch.done, result_numpy.done)
        self.assertAlmostEqual(result_torch.reward, result_numpy.reward, places=5)
        # giải thích: Đối chiếu từng tham số chi tiết trong trường thông tin info
        for key in ("old_slice_overlap", "attempted_slice_overlap", "detected_overlap", "total_target_score"):
            self.assertAlmostEqual(result_torch.info[key], result_numpy.info[key], places=5)

    # giải thích: Kiểm thử xem trạng thái môi trường có chứa các bản đồ biểu diễn ROI hiện tại, ROI đã thử và ROI đã chấp nhận độc lập
    def test_state_has_separate_current_attempted_and_accepted_roi_maps(self) -> None:
        base_env = SliceEnv(_detection_cache(), None, env_cfg=EnvConfig())
        base_env.reset()
        attempted = base_env.roi.reshape(1, 4)
        accepted = base_env._apply_action(Action.RIGHT).reshape(1, 4)
        env = SliceEnv(
            _detection_cache(),
            None,
            env_cfg=EnvConfig(),
            previous_rois=attempted,
            overlap_rois=accepted,
        )

        state = env.reset()
        grid = env.state_cfg.grid_size
        feature_dim = len(env.detection.feature)
        # giải thích: Cắt mảng trạng thái để lấy các bản đồ đặc trưng không gian
        maps = state[feature_dim : feature_dim + BASE_MAP_CHANNELS * grid * grid].reshape(
            BASE_MAP_CHANNELS,
            grid,
            grid,
        )

        # giải thích: Xác thực các kênh bản đồ thứ 1, 2, 3 có tổng giá trị lớn hơn 0 (không rỗng)
        self.assertGreater(float(maps[1].sum()), 0.0)
        self.assertGreater(float(maps[2].sum()), 0.0)
        self.assertGreater(float(maps[3].sum()), 0.0)
        # giải thích: Xác thực bản đồ vùng đã thử và vùng đã chấp nhận không giống hệt nhau
        self.assertFalse(np.array_equal(maps[2], maps[3]))


# giải thích: Điểm khởi chạy của chương trình unit test
if __name__ == "__main__":
    unittest.main()

