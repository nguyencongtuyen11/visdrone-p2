from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.actions import Action
from rl_sahi.common.cache import DetectionCache, HardRegionCache
from rl_sahi.rl.env_config import EnvConfig
from rl_sahi.rl.slice_env import SliceEnv
from rl_sahi.rl.state_config import BASE_MAP_CHANNELS


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


class SliceEnvRewardTest(unittest.TestCase):
    def make_env(self) -> SliceEnv:
        return SliceEnv(_detection_cache(), _hard_region_cache(), env_cfg=EnvConfig())

    def test_non_stop_target_hit_is_not_committed_or_rewarded(self) -> None:
        env = self.make_env()
        env.reset()

        result = env.step(Action.ZOOM_IN)

        self.assertEqual(result.info["new_hits"], 0)
        self.assertEqual(result.info["candidate_hits"], 1)
        self.assertEqual(int(env.covered.sum()), 0)
        self.assertLess(result.reward, 0.0)

    def test_stop_commits_target_hit(self) -> None:
        env = self.make_env()
        env.reset()
        env.step(Action.ZOOM_IN)

        stop = env.step(Action.STOP)

        self.assertTrue(stop.done)
        self.assertEqual(stop.info["new_hits"], 1)
        self.assertEqual(stop.info["retained_hits"], 0)
        self.assertGreater(stop.info["total_target_score"], 0.0)
        self.assertGreater(stop.reward, 0.0)

    def test_non_stop_moves_do_not_commit_hard_coverage(self) -> None:
        env = self.make_env()
        env.reset()
        env.step(Action.ZOOM_IN)
        env.step(Action.ZOOM_IN)
        env.step(Action.ZOOM_OUT)

        self.assertEqual(int(env.covered.sum()), 0)

        stop = env.step(Action.STOP)

        self.assertEqual(stop.info["new_hits"], 1)
        self.assertEqual(int(env.covered.sum()), 1)

    def test_max_steps_without_stop_gets_terminal_penalty(self) -> None:
        env = SliceEnv(
            _detection_cache(),
            _hard_region_cache(),
            env_cfg=EnvConfig(max_steps=1, max_steps_without_stop_penalty=3.0),
        )
        env.reset()

        result = env.step(Action.RIGHT)

        self.assertTrue(result.done)
        self.assertTrue(result.info["stop_due_to_max_steps"])
        self.assertLess(result.reward, -3.0)

    def test_stalled_without_stop_gets_terminal_penalty(self) -> None:
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

        result = env.step(Action.ZOOM_IN)

        self.assertTrue(result.done)
        self.assertTrue(result.info["stop_due_to_stalled_roi"])
        self.assertLess(result.reward, -2.5)

    def test_valid_actions_mask_stalled_zoom(self) -> None:
        env = SliceEnv(
            _detection_cache(),
            None,
            env_cfg=EnvConfig(initial_slice_fraction=0.35, min_slice_fraction=0.35),
        )
        env.reset()

        valid = env.valid_actions()

        self.assertFalse(valid[int(Action.ZOOM_IN)])
        self.assertTrue(valid[int(Action.STOP)])

    def test_attempted_overlap_masks_stop_and_penalizes_terminal_stop(self) -> None:
        base_env = SliceEnv(_detection_cache(), None, env_cfg=EnvConfig())
        base_env.reset()
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

        self.assertFalse(valid[int(Action.STOP)])
        self.assertTrue(result.info["stop_due_to_attempted_overlap"])
        self.assertLess(result.reward, -2.0)

    def test_valid_actions_mask_move_into_attempted_overlap_when_escape_exists(self) -> None:
        base_env = SliceEnv(_detection_cache(), None, env_cfg=EnvConfig())
        base_env.reset()
        attempted = base_env._apply_action(Action.RIGHT).reshape(1, 4)
        env = SliceEnv(
            _detection_cache(),
            None,
            env_cfg=EnvConfig(),
            previous_rois=attempted,
            overlap_rois=np.zeros((0, 4), dtype=np.float32),
        )
        env.reset()

        valid = env.valid_actions()

        self.assertFalse(valid[int(Action.RIGHT)])
        self.assertTrue(valid[int(Action.LEFT)])

    def test_diagonal_action_moves_on_both_axes(self) -> None:
        env = SliceEnv(_detection_cache(), None, env_cfg=EnvConfig())
        env.reset()
        original = env.roi.copy()

        env.step(Action.DOWN_RIGHT)

        self.assertGreater(env.roi[0], original[0])
        self.assertGreater(env.roi[1], original[1])

    def test_target_class_filter_excludes_non_target_detection_penalty(self) -> None:
        cfg = EnvConfig(step_penalty=0.0, area_penalty=0.0, detected_overlap_penalty=1.0)
        env_all = SliceEnv(_high_conf_detection_cache(99), None, env_cfg=cfg)
        env_all.reset()
        all_result = env_all.step(Action.STOP)

        env_target = SliceEnv(_high_conf_detection_cache(99), None, env_cfg=cfg, target_classes=(0,))
        env_target.reset()
        target_result = env_target.step(Action.STOP)

        self.assertGreater(all_result.info["detected_overlap"], 0.0)
        self.assertEqual(target_result.info["detected_overlap"], 0.0)
        self.assertGreater(target_result.reward, all_result.reward)

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

        np.testing.assert_array_equal(env_torch.valid_actions(), env_numpy.valid_actions())
        result_numpy = env_numpy.step(Action.STOP)
        result_torch = env_torch.step(Action.STOP)

        self.assertEqual(result_torch.done, result_numpy.done)
        self.assertAlmostEqual(result_torch.reward, result_numpy.reward, places=5)
        for key in ("old_slice_overlap", "attempted_slice_overlap", "detected_overlap", "total_target_score"):
            self.assertAlmostEqual(result_torch.info[key], result_numpy.info[key], places=5)

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
        maps = state[feature_dim : feature_dim + BASE_MAP_CHANNELS * grid * grid].reshape(
            BASE_MAP_CHANNELS,
            grid,
            grid,
        )

        self.assertGreater(float(maps[1].sum()), 0.0)
        self.assertGreater(float(maps[2].sum()), 0.0)
        self.assertGreater(float(maps[3].sum()), 0.0)
        self.assertFalse(np.array_equal(maps[2], maps[3]))


if __name__ == "__main__":
    unittest.main()
